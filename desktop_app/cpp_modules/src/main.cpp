#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

struct Point3D {
    float x;
    float y;
    float z;
    float doppler;
    int cluster_id = -1;
    bool visited = false;
};

struct ClusterSummary {
    int cluster_id = -1;
    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float doppler_mean = 0.0f;
    int point_count = 0;
};

struct TrackState {
    int id = -1;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    int missing = 0;
};

class DBSCAN {
  public:
    DBSCAN(float eps, int min_pts) : eps_(eps), min_pts_(min_pts) {}

    void fit(std::vector<Point3D>& points) {
        int cluster_id = 0;
        for (auto& point : points) {
            point.cluster_id = -1;
            point.visited = false;
        }

        for (std::size_t index = 0; index < points.size(); ++index) {
            if (points[index].visited) {
                continue;
            }
            points[index].visited = true;
            auto neighbors = region_query(points, static_cast<int>(index));
            if (neighbors.size() < static_cast<std::size_t>(min_pts_)) {
                points[index].cluster_id = -1;
                continue;
            }

            expand_cluster(points, static_cast<int>(index), neighbors, cluster_id);
            cluster_id++;
        }
    }

  private:
    float eps_;
    int min_pts_;

    static float distance_sq(const Point3D& lhs, const Point3D& rhs) {
        const float dx = lhs.x - rhs.x;
        const float dy = lhs.y - rhs.y;
        const float dz = lhs.z - rhs.z;
        return dx * dx + dy * dy + dz * dz;
    }

    std::vector<int> region_query(const std::vector<Point3D>& points, int pivot) const {
        std::vector<int> neighbors;
        const float eps_sq = eps_ * eps_;
        for (std::size_t index = 0; index < points.size(); ++index) {
            if (distance_sq(points[pivot], points[index]) <= eps_sq) {
                neighbors.push_back(static_cast<int>(index));
            }
        }
        return neighbors;
    }

    void expand_cluster(std::vector<Point3D>& points, int seed, std::vector<int>& neighbors, int cluster_id) const {
        points[seed].cluster_id = cluster_id;
        for (std::size_t index = 0; index < neighbors.size(); ++index) {
            const int point_index = neighbors[index];
            if (!points[point_index].visited) {
                points[point_index].visited = true;
                auto secondary_neighbors = region_query(points, point_index);
                if (secondary_neighbors.size() >= static_cast<std::size_t>(min_pts_)) {
                    neighbors.insert(neighbors.end(), secondary_neighbors.begin(), secondary_neighbors.end());
                }
            }
            if (points[point_index].cluster_id == -1) {
                points[point_index].cluster_id = cluster_id;
            }
        }
    }
};

class RadarEngine {
  public:
    RadarEngine(float eps, int min_pts) : dbscan_(eps, min_pts) {}

    py::dict process(py::array_t<float> input_points) {
        auto buffer = input_points.request();
        if (buffer.ndim != 2 || buffer.shape[0] == 0 || buffer.shape[1] < 4) {
            return py::dict();
        }

        const auto point_count = static_cast<int>(buffer.shape[0]);
        const auto stride = static_cast<int>(buffer.shape[1]);
        auto* raw = static_cast<float*>(buffer.ptr);

        std::vector<Point3D> points;
        points.reserve(point_count);
        for (int index = 0; index < point_count; ++index) {
            const float z = raw[index * stride + 2];
            if (z > 2.7f) {
                continue;
            }
            points.push_back({
                raw[index * stride + 0],
                raw[index * stride + 1],
                z,
                raw[index * stride + 3],
            });
        }

        if (points.empty()) {
            return py::dict();
        }

        dbscan_.fit(points);
        auto clusters = build_clusters(points);
        assign_tracks(clusters);

        py::array_t<int> labels(points.size());
        auto label_view = labels.mutable_unchecked<1>();
        py::array_t<int> point_to_track(points.size());
        auto track_view = point_to_track.mutable_unchecked<1>();
        for (std::size_t index = 0; index < points.size(); ++index) {
            label_view(index) = points[index].cluster_id;
            track_view(index) = cluster_to_track_id(points[index].cluster_id);
        }

        py::list output_tracks;
        for (const auto& track : tracks_) {
            py::dict item;
            item["id"] = track.id;
            item["pos"] = py::make_tuple(track.x, track.y, track.z);
            item["vel"] = py::make_tuple(track.vx, track.vy, track.vz);
            output_tracks.append(item);
        }

        py::list output_clusters;
        for (const auto& cluster : clusters) {
            py::dict item;
            item["cluster_id"] = cluster.cluster_id;
            item["track_id"] = cluster_to_track_id(cluster.cluster_id);
            item["centroid"] = py::make_tuple(cluster.cx, cluster.cy, cluster.cz);
            item["dims"] = py::make_tuple(cluster.dx, cluster.dy, cluster.dz);
            item["doppler_mean"] = cluster.doppler_mean;
            item["point_count"] = cluster.point_count;
            output_clusters.append(item);
        }

        py::dict result;
        result["labels"] = labels;
        result["tracks"] = output_tracks;
        result["clusters"] = output_clusters;
        result["point_to_track"] = point_to_track;
        return result;
    }

  private:
    DBSCAN dbscan_;
    std::vector<TrackState> tracks_;
    std::unordered_map<int, int> cluster_track_lookup_;
    int next_track_id_ = 0;

    static float euclidean_distance(float ax, float ay, float az, float bx, float by, float bz) {
        const float dx = ax - bx;
        const float dy = ay - by;
        const float dz = az - bz;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    std::vector<ClusterSummary> build_clusters(const std::vector<Point3D>& points) {
        std::unordered_map<int, std::vector<const Point3D*>> grouped;
        for (const auto& point : points) {
            if (point.cluster_id < 0) {
                continue;
            }
            grouped[point.cluster_id].push_back(&point);
        }

        std::vector<ClusterSummary> clusters;
        for (const auto& entry : grouped) {
            if (entry.second.size() < 4) {
                continue;
            }

            ClusterSummary summary;
            summary.cluster_id = entry.first;
            float min_x = std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float min_z = std::numeric_limits<float>::max();
            float max_x = std::numeric_limits<float>::lowest();
            float max_y = std::numeric_limits<float>::lowest();
            float max_z = std::numeric_limits<float>::lowest();

            for (const auto* point : entry.second) {
                summary.cx += point->x;
                summary.cy += point->y;
                summary.cz += point->z;
                summary.doppler_mean += point->doppler;

                min_x = std::min(min_x, point->x);
                min_y = std::min(min_y, point->y);
                min_z = std::min(min_z, point->z);
                max_x = std::max(max_x, point->x);
                max_y = std::max(max_y, point->y);
                max_z = std::max(max_z, point->z);
            }

            const float count = static_cast<float>(entry.second.size());
            summary.point_count = static_cast<int>(entry.second.size());
            summary.cx /= count;
            summary.cy /= count;
            summary.cz /= count;
            summary.doppler_mean /= count;
            summary.dx = std::max(0.15f, max_x - min_x);
            summary.dy = std::max(0.15f, max_y - min_y);
            summary.dz = std::max(0.15f, max_z - min_z);
            clusters.push_back(summary);
        }

        return clusters;
    }

    void assign_tracks(const std::vector<ClusterSummary>& clusters) {
        cluster_track_lookup_.clear();
        for (auto& track : tracks_) {
            track.x += track.vx * 0.1f;
            track.y += track.vy * 0.1f;
            track.z += track.vz * 0.1f;
            track.missing += 1;
        }

        std::vector<bool> cluster_used(clusters.size(), false);
        for (auto& track : tracks_) {
            float best_distance = 1.5f;
            int best_cluster = -1;
            for (std::size_t index = 0; index < clusters.size(); ++index) {
                if (cluster_used[index]) {
                    continue;
                }
                const auto& cluster = clusters[index];
                const float distance = euclidean_distance(track.x, track.y, track.z, cluster.cx, cluster.cy, cluster.cz);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = static_cast<int>(index);
                }
            }

            if (best_cluster >= 0) {
                const auto& cluster = clusters[best_cluster];
                track.vx = cluster.cx - track.x;
                track.vy = cluster.cy - track.y;
                track.vz = cluster.cz - track.z;
                track.x = cluster.cx;
                track.y = cluster.cy;
                track.z = cluster.cz;
                track.missing = 0;
                cluster_used[best_cluster] = true;
                cluster_track_lookup_[cluster.cluster_id] = track.id;
            }
        }

        for (std::size_t index = 0; index < clusters.size(); ++index) {
            if (cluster_used[index]) {
                continue;
            }
            const auto& cluster = clusters[index];
            TrackState track;
            track.id = next_track_id_++;
            track.x = cluster.cx;
            track.y = cluster.cy;
            track.z = cluster.cz;
            tracks_.push_back(track);
            cluster_track_lookup_[cluster.cluster_id] = track.id;
        }

        tracks_.erase(
            std::remove_if(
                tracks_.begin(),
                tracks_.end(),
                [](const TrackState& track) { return track.missing > 10; }
            ),
            tracks_.end()
        );
    }

    int cluster_to_track_id(int cluster_id) const {
        const auto iterator = cluster_track_lookup_.find(cluster_id);
        if (iterator == cluster_track_lookup_.end()) {
            return -1;
        }
        return iterator->second;
    }
};

PYBIND11_MODULE(radar_engine, module) {
    py::class_<RadarEngine>(module, "RadarEngine")
        .def(py::init<float, int>())
        .def("process", &RadarEngine::process);
}
