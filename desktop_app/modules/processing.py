from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from filterpy.kalman import KalmanFilter as _FilterPyKalmanFilter
except ImportError:  # pragma: no cover - optional dependency
    _FilterPyKalmanFilter = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - optional dependency
    linear_sum_assignment = None

try:
    from sklearn.cluster import DBSCAN as _SklearnDBSCAN
except ImportError:  # pragma: no cover - optional dependency
    _SklearnDBSCAN = None


class _SimpleKalmanFilter:
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1), dtype=np.float32)
        self.F = np.eye(dim_x, dtype=np.float32)
        self.H = np.eye(dim_z, dim_x, dtype=np.float32)
        self.P = np.eye(dim_x, dtype=np.float32)
        self.R = np.eye(dim_z, dtype=np.float32)
        self.Q = np.eye(dim_x, dtype=np.float32) * 0.01

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        measurement = np.asarray(measurement, dtype=np.float32).reshape(self.dim_z, 1)
        innovation = measurement - (self.H @ self.x)
        s_matrix = self.H @ self.P @ self.H.T + self.R
        gain = self.P @ self.H.T @ np.linalg.pinv(s_matrix)
        self.x = self.x + gain @ innovation
        identity = np.eye(self.dim_x, dtype=np.float32)
        self.P = (identity - gain @ self.H) @ self.P


class _SimpleDBSCAN:
    def __init__(self, eps: float, min_samples: int):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        labels = np.full(len(points), -1, dtype=np.int32)
        visited = np.zeros(len(points), dtype=bool)
        cluster_id = 0

        def neighbors(index: int):
            delta = points - points[index]
            return np.where(np.linalg.norm(delta, axis=1) <= self.eps)[0]

        for index in range(len(points)):
            if visited[index]:
                continue
            visited[index] = True
            nearby = list(neighbors(index))
            if len(nearby) < self.min_samples:
                labels[index] = -1
                continue

            labels[index] = cluster_id
            cursor = 0
            while cursor < len(nearby):
                point_index = nearby[cursor]
                if not visited[point_index]:
                    visited[point_index] = True
                    more = list(neighbors(point_index))
                    if len(more) >= self.min_samples:
                        nearby.extend(more)
                if labels[point_index] == -1:
                    labels[point_index] = cluster_id
                cursor += 1
            cluster_id += 1
        return labels


KalmanFilter = _FilterPyKalmanFilter or _SimpleKalmanFilter
DBSCAN = _SklearnDBSCAN or _SimpleDBSCAN


def _linear_sum_assignment(cost_matrix: np.ndarray):
    if linear_sum_assignment is not None:
        return linear_sum_assignment(cost_matrix)

    used_rows = set()
    used_cols = set()
    assignments = []
    flat = []
    for row in range(cost_matrix.shape[0]):
        for col in range(cost_matrix.shape[1]):
            flat.append((float(cost_matrix[row, col]), row, col))
    for _, row, col in sorted(flat, key=lambda item: item[0]):
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        assignments.append((row, col))
    if not assignments:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    rows, cols = zip(*assignments)
    return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)


def _discover_cpp_build_path() -> Optional[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.abspath(os.path.join(current_dir, "..", "cpp_modules", "build"))
    if not os.path.isdir(build_dir):
        return None

    valid_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    try:
        for entry in os.listdir(build_dir):
            if entry.startswith("radar_engine") and entry.endswith(valid_suffixes):
                return build_dir
    except OSError:
        return None
    return None


CPP_BUILD_PATH = _discover_cpp_build_path()
HAS_CPP = False
radar_engine = None

if CPP_BUILD_PATH and CPP_BUILD_PATH not in sys.path:
    sys.path.append(CPP_BUILD_PATH)

if CPP_BUILD_PATH:
    try:
        radar_engine = importlib.import_module("radar_engine")
        HAS_CPP = True
        print(f"[processing] C++ radar engine loaded from {CPP_BUILD_PATH}")
    except ImportError as exc:
        print(f"[processing] C++ radar engine unavailable: {exc}")


@dataclass
class Detection:
    position: np.ndarray
    dims: np.ndarray
    points: np.ndarray
    velocity_hint: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    source: str = "cluster"
    sensor_track_id: Optional[int] = None
    confidence: float = 0.0
    stats: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        self.dims = np.asarray(self.dims, dtype=np.float32).reshape(3)
        self.points = np.asarray(self.points, dtype=np.float32)
        self.velocity_hint = np.asarray(self.velocity_hint, dtype=np.float32).reshape(3)


@dataclass
class ProcessorResult:
    labels: np.ndarray
    detections: List[Detection]
    used_sensor_tracks: bool = False


class PointCloudProcessor:
    DEFAULT_SENSOR_BODY_HEIGHT = 1.60
    MIN_RELIABLE_BODY_HEIGHT = 0.55

    def __init__(self, eps: float = 0.8, min_samples: int = 4,
                 min_point_height: float = -0.3, max_point_height: float = 3.0):
        self.eps = eps
        self.min_samples = min_samples
        self.min_point_height = min_point_height
        self.max_point_height = max_point_height
        self.enable_height_filter = False
        self.engine = None
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        if HAS_CPP and radar_engine is not None:
            try:
                self.engine = radar_engine.RadarEngine(eps, min_samples)
            except Exception as exc:
                print(f"[processing] Failed to initialize C++ engine: {exc}")
                self.engine = None

    def update_params(self, eps: float, min_samples: int,
                      min_point_height: Optional[float] = None,
                      max_point_height: Optional[float] = None,
                      enable_height_filter: Optional[bool] = None) -> None:
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        if min_point_height is not None:
            self.min_point_height = float(min_point_height)
        if max_point_height is not None:
            self.max_point_height = float(max_point_height)
        if enable_height_filter is not None:
            self.enable_height_filter = bool(enable_height_filter)
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)

        if HAS_CPP and radar_engine is not None:
            try:
                self.engine = radar_engine.RadarEngine(self.eps, self.min_samples)
            except Exception as exc:
                print(f"[processing] Failed to refresh C++ engine: {exc}")
                self.engine = None

    def process(
        self,
        points: Optional[np.ndarray],
        sensor_tracks: Optional[Sequence[dict]] = None,
        max_tracks: int = 6,
        point_track_ids: Optional[np.ndarray] = None,
        association_points: Optional[np.ndarray] = None,
        enable_clustering: bool = True,
    ) -> ProcessorResult:
        clean_points = self._sanitize_points(points)
        association_points = clean_points if association_points is None else self._sanitize_points(association_points)
        if point_track_ids is not None:
            point_track_ids = np.asarray(point_track_ids, dtype=np.int32).reshape(-1)
            if association_points.shape[0] != point_track_ids.shape[0]:
                point_track_ids = None
        sensor_tracks = list(sensor_tracks or [])

        if sensor_tracks:
            return self._process_with_sensor_tracks(
                clean_points,
                sensor_tracks,
                max_tracks,
                point_track_ids=point_track_ids,
                association_points=association_points,
                enable_clustering=enable_clustering,
            )
        if enable_clustering:
            return self._process_clusters_only(clean_points, max_tracks)
        else:
            return ProcessorResult(labels=np.full(len(clean_points), -1, dtype=np.int32), detections=[], used_sensor_tracks=False)

    def _sanitize_points(self, points: Optional[np.ndarray]) -> np.ndarray:
        if points is None:
            return np.empty((0, 4), dtype=np.float32)

        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return np.empty((0, 4), dtype=np.float32)

        if arr.shape[1] == 3:
            arr = np.hstack([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)])
        elif arr.shape[1] > 4:
            arr = arr[:, :4]

        valid_mask = np.isfinite(arr).all(axis=1)
        if self.enable_height_filter:
            valid_mask &= (arr[:, 2] >= self.min_point_height)
            valid_mask &= (arr[:, 2] <= self.max_point_height)
        arr = arr[valid_mask]

        if arr.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _cluster_points(self, points: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        if len(points) == 0:
            return np.empty((0,), dtype=np.int32), []

        labels = None
        if self.engine is not None:
            try:
                result = self.engine.process(points.astype(np.float32))
                labels = np.asarray(result.get("labels", []), dtype=np.int32)
            except Exception as exc:
                print(f"[processing] C++ process failed, using DBSCAN fallback: {exc}")

        if labels is None or labels.shape[0] != len(points):
            labels = self.dbscan.fit_predict(points[:, :3]).astype(np.int32)

        clusters = self._build_clusters_from_labels(points, labels)
        return labels, clusters

    def _build_clusters_from_labels(self, points: np.ndarray, labels: np.ndarray) -> List[dict]:
        clusters: List[dict] = []
        for label in np.unique(labels):
            if label < 0:
                continue
            mask = labels == label
            cluster_points = points[mask]
            if len(cluster_points) == 0:
                continue
            if len(cluster_points) < self.min_samples:
                continue

            xyz = cluster_points[:, :3]
            mins = xyz.min(axis=0)
            maxs = xyz.max(axis=0)
            dims = np.maximum(maxs - mins, np.array([0.15, 0.15, 0.15], dtype=np.float32))
            doppler_mean = float(cluster_points[:, 3].mean()) if cluster_points.shape[1] >= 4 else 0.0
            clusters.append(
                {
                    "label": int(label),
                    "points": cluster_points,
                    "centroid": xyz.mean(axis=0).astype(np.float32),
                    "dims": dims.astype(np.float32),
                    "doppler_mean": doppler_mean,
                    "point_count": int(len(cluster_points)),
                    "z_min": float(mins[2]),
                    "z_max": float(maxs[2]),
                }
            )
        return clusters

    def _process_with_sensor_tracks(
        self,
        points: np.ndarray,
        sensor_tracks: Sequence[dict],
        max_tracks: int,
        point_track_ids: Optional[np.ndarray] = None,
        association_points: Optional[np.ndarray] = None,
        enable_clustering: bool = True,
    ) -> ProcessorResult:
        association_points = points if association_points is None else association_points
        if enable_clustering:
            labels, clusters = self._cluster_points(points)
            matched_clusters = self._match_sensor_tracks(sensor_tracks, clusters)
            used_cluster_labels = {
                int(cluster.get("label", -1))
                for cluster in matched_clusters.values()
                if cluster is not None
            }
        else:
            labels = np.full(len(points), -1, dtype=np.int32)
            clusters = []
            matched_clusters = {}
            used_cluster_labels = set()
        
        by_track_id = self._group_points_by_track_id(association_points, point_track_ids)

        detections: List[Detection] = []
        for track in sensor_tracks:
            track_id = int(track.get("id", -1))
            track_confidence = float(np.clip(track.get("confidence", 0.0), 0.0, 1.0))
            exact_points = by_track_id.get(track_id)
            matched_cluster = matched_clusters.get(track_id)
            cluster_dims = matched_cluster["dims"] if matched_cluster is not None else None

            if exact_points is not None and len(exact_points) > 0:
                detection_points = exact_points
                dims = self._dims_from_track(track, detection_points, fallback_dims=cluster_dims)
                confidence = track_confidence or min(1.0, 0.45 + len(detection_points) / 48.0)
                doppler_mean = float(detection_points[:, 3].mean()) if len(detection_points) else 0.0
                stats = {
                    "point_count": float(len(detection_points)),
                    "doppler_mean": doppler_mean,
                    "source_rank": 3.0,
                    "point_assignment_mode": 3.0,
                }
            elif matched_cluster is not None and len(matched_cluster["points"]) > 0:
                detection_points = matched_cluster["points"]
                dims = self._dims_from_track(track, detection_points, fallback_dims=matched_cluster["dims"])
                confidence = max(track_confidence, min(0.85, 0.35 + len(detection_points) / 60.0))
                doppler_mean = float(detection_points[:, 3].mean()) if len(detection_points) else 0.0
                stats = {
                    "point_count": float(len(detection_points)),
                    "doppler_mean": doppler_mean,
                    "source_rank": 2.4,
                    "point_assignment_mode": 2.0,
                    "matched_cluster": 1.0,
                    "cluster_point_count": float(matched_cluster["point_count"]),
                }
            else:
                local_points = self._assign_points_near_track(association_points, track)
                detection_points = local_points
                dims = self._dims_from_track(track, detection_points, fallback_dims=cluster_dims)
                confidence = track_confidence or (0.35 if len(local_points) > 0 else 0.2)
                doppler_mean = float(local_points[:, 3].mean()) if len(local_points) else 0.0
                stats = {
                    "point_count": float(len(local_points)),
                    "doppler_mean": doppler_mean,
                    "source_rank": 1.3,
                    "point_assignment_mode": 1.0,
                }
                if matched_cluster is not None:
                    stats["cluster_point_count"] = float(matched_cluster["point_count"])

            if "height" in track:
                stats["track_height"] = float(track["height"])
            if track_confidence:
                stats["track_confidence"] = track_confidence

            velocity = np.array(
                [
                    float(track.get("vx", 0.0)),
                    float(track.get("vy", 0.0)),
                    float(track.get("vz", 0.0)),
                ],
                dtype=np.float32,
            )
            position = np.array(
                [
                    float(track.get("x", 0.0)),
                    float(track.get("y", 0.0)),
                    float(track.get("z", 0.0)),
                ],
                dtype=np.float32,
            )

            detections.append(
                Detection(
                    position=position,
                    dims=dims,
                    points=detection_points,
                    velocity_hint=velocity,
                    source="sensor_tlv",
                    sensor_track_id=track_id,
                    confidence=confidence,
                    stats=stats,
                )
            )

        for cluster in clusters:
            cluster_label = int(cluster.get("label", -1))
            if cluster_label in used_cluster_labels:
                continue
            if self._cluster_near_sensor_track(cluster, sensor_tracks):
                continue

            velocity_hint = np.array([0.0, cluster["doppler_mean"], 0.0], dtype=np.float32)
            detections.append(
                Detection(
                    position=cluster["centroid"],
                    dims=cluster["dims"],
                    points=cluster["points"],
                    velocity_hint=velocity_hint,
                    source="cluster_gap_fill",
                    confidence=min(0.55, 0.10 + cluster["point_count"] / 64.0),
                    stats={
                        "point_count": float(cluster["point_count"]),
                        "doppler_mean": float(cluster["doppler_mean"]),
                        "source_rank": 0.35,
                        "point_assignment_mode": 0.0,
                        "matched_cluster": 0.0,
                    },
                )
            )

        detections = self._limit_detections(detections, max_tracks)
        return ProcessorResult(labels=labels, detections=detections, used_sensor_tracks=True)

    def _group_points_by_track_id(
        self,
        points: np.ndarray,
        point_track_ids: Optional[np.ndarray],
    ) -> Dict[int, np.ndarray]:
        if point_track_ids is None or len(points) == 0:
            return {}

        valid_mask = (point_track_ids >= 0) & (point_track_ids <= 249)
        if not np.any(valid_mask):
            return {}

        grouped: Dict[int, np.ndarray] = {}
        valid_points = points[valid_mask]
        valid_ids = point_track_ids[valid_mask]
        for track_id in np.unique(valid_ids):
            grouped[int(track_id)] = valid_points[valid_ids == track_id]
        return grouped

    def _process_clusters_only(self, points: np.ndarray, max_tracks: int) -> ProcessorResult:
        labels, clusters = self._cluster_points(points)
        detections: List[Detection] = []

        for cluster in clusters:
            velocity_hint = np.array([0.0, cluster["doppler_mean"], 0.0], dtype=np.float32)
            detections.append(
                Detection(
                    position=cluster["centroid"],
                    dims=cluster["dims"],
                    points=cluster["points"],
                    velocity_hint=velocity_hint,
                    source="cluster",
                    confidence=min(1.0, 0.15 + cluster["point_count"] / 40.0),
                    stats={
                        "point_count": float(cluster["point_count"]),
                        "doppler_mean": float(cluster["doppler_mean"]),
                        "source_rank": 0.0,
                    },
                )
            )

        detections = self._limit_detections(detections, max_tracks)
        return ProcessorResult(labels=labels, detections=detections, used_sensor_tracks=False)

    def _match_sensor_tracks(self, sensor_tracks: Sequence[dict], clusters: Sequence[dict]) -> Dict[int, dict]:
        if not sensor_tracks or not clusters:
            return {}

        cost_matrix = np.full((len(sensor_tracks), len(clusters)), 999.0, dtype=np.float32)
        for row, track in enumerate(sensor_tracks):
            track_pos = self._track_position(track)
            track_dims = self._dims_from_track(track)
            xy_limit = max(0.75, float(max(track_dims[0], track_dims[1])) * 1.4)
            z_limit = max(0.45, float(track_dims[2]) * 0.75)

            for col, cluster in enumerate(clusters):
                centroid = np.asarray(cluster["centroid"], dtype=np.float32)
                dist_xy = float(np.linalg.norm(track_pos[:2] - centroid[:2]))
                z_penalty = abs(float(track_pos[2]) - float(centroid[2]))
                if dist_xy > xy_limit or z_penalty > z_limit:
                    continue

                dims_penalty = float(np.linalg.norm(np.asarray(cluster["dims"], dtype=np.float32) - track_dims)) * 0.25
                point_bonus = min(0.35, float(cluster["point_count"]) / 72.0)
                cost_matrix[row, col] = dist_xy + (0.35 * z_penalty) + dims_penalty - point_bonus

        row_ind, col_ind = _linear_sum_assignment(cost_matrix)
        assignments: Dict[int, dict] = {}
        used_cluster_indexes = set()
        for row, col in zip(row_ind, col_ind):
            cost = float(cost_matrix[row, col])
            if cost >= 999.0 or col in used_cluster_indexes:
                continue
            track_id = int(sensor_tracks[row].get("id", -1))
            assignments[track_id] = clusters[col]
            used_cluster_indexes.add(col)
        return assignments

    def _cluster_near_sensor_track(self, cluster: dict, sensor_tracks: Sequence[dict]) -> bool:
        cluster_pos = np.asarray(cluster["centroid"], dtype=np.float32)
        for track in sensor_tracks:
            track_pos = self._track_position(track)
            track_dims = self._dims_from_track(track)
            xy_limit = max(0.65, float(max(track_dims[0], track_dims[1])) * 1.2)
            z_limit = max(0.4, float(track_dims[2]) * 0.7)
            if np.linalg.norm(track_pos[:2] - cluster_pos[:2]) <= xy_limit and abs(track_pos[2] - cluster_pos[2]) <= z_limit:
                return True
        return False

    def _assign_points_near_track(self, points: np.ndarray, track: dict) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 4), dtype=np.float32)

        center = self._track_position(track)
        dims = self._dims_from_track(track)
        speed = float(np.linalg.norm(self._track_velocity(track)))

        x_radius = max(0.35, float(dims[0]) * 0.9 + 0.2 + min(0.2, speed * 0.1))
        y_radius = max(0.45, float(dims[1]) * 0.85 + 0.35 + min(0.35, speed * 0.2))

        z_min = track.get("z_min")
        z_max = track.get("z_max")
        if z_min is not None and z_max is not None:
            z_low = float(z_min) - 0.12
            z_high = float(z_max) + 0.18
        else:
            half_height = max(0.4, float(dims[2]) * 0.55)
            z_low = float(center[2]) - half_height - 0.1
            z_high = float(center[2]) + half_height + 0.1

        z_center = (z_low + z_high) * 0.5
        z_radius = max(0.35, (z_high - z_low) * 0.5)

        deltas = np.empty((len(points), 3), dtype=np.float32)
        deltas[:, 0] = (points[:, 0] - center[0]) / x_radius
        deltas[:, 1] = (points[:, 1] - center[1]) / y_radius
        deltas[:, 2] = (points[:, 2] - z_center) / z_radius

        mask = np.sum(deltas * deltas, axis=1) <= 1.0
        mask &= points[:, 2] >= z_low
        mask &= points[:, 2] <= z_high
        return points[mask]

    @staticmethod
    def _track_position(track: dict) -> np.ndarray:
        return np.array(
            [
                float(track.get("x", 0.0)),
                float(track.get("y", 0.0)),
                float(track.get("z", 0.0)),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _track_velocity(track: dict) -> np.ndarray:
        return np.array(
            [
                float(track.get("vx", 0.0)),
                float(track.get("vy", 0.0)),
                float(track.get("vz", 0.0)),
            ],
            dtype=np.float32,
        )

    def _dims_from_track(
        self,
        track: dict,
        points: Optional[np.ndarray] = None,
        fallback_dims: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if points is not None and len(points) > 0:
            dims = self._dims_from_points(points)
        elif fallback_dims is not None:
            dims = np.asarray(fallback_dims, dtype=np.float32).copy().reshape(3)
        else:
            dims = np.array([0.6, 0.6, 1.7], dtype=np.float32)

        explicit_dims = track.get("dims")
        if explicit_dims is not None:
            dims = np.asarray(explicit_dims, dtype=np.float32).reshape(3)
        elif track.get("z_min") is not None and track.get("z_max") is not None:
            dims = np.asarray(dims, dtype=np.float32).copy().reshape(3)
            track_height = max(0.0, float(track["z_max"]) - float(track["z_min"]))
            if track_height >= self.MIN_RELIABLE_BODY_HEIGHT:
                dims[2] = track_height
        elif track.get("height") is not None:
            dims = np.asarray(dims, dtype=np.float32).copy().reshape(3)
            track_height = max(0.0, float(track["height"]))
            if track_height >= self.MIN_RELIABLE_BODY_HEIGHT:
                dims[2] = track_height

        # People Tracking point clouds are often sparse around legs or the
        # lower torso. Do not treat a tiny vertical point spread as body height.
        if dims[2] < self.MIN_RELIABLE_BODY_HEIGHT:
            dims = np.asarray(dims, dtype=np.float32).copy().reshape(3)
            dims[2] = self.DEFAULT_SENSOR_BODY_HEIGHT

        return np.maximum(dims, np.array([0.15, 0.15, 0.15], dtype=np.float32)).astype(np.float32)

    def _dims_from_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.array([0.6, 0.6, 1.7], dtype=np.float32)
        xyz = points[:, :3]
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        return np.maximum(maxs - mins, np.array([0.15, 0.15, 0.15], dtype=np.float32)).astype(np.float32)

    def _limit_detections(self, detections: List[Detection], max_tracks: int) -> List[Detection]:
        if max_tracks <= 0 or len(detections) <= max_tracks:
            return detections
        return sorted(
            detections,
            key=lambda item: (
                float(item.stats.get("source_rank", 0.0)),
                float(item.stats.get("point_count", 0.0)),
                float(item.confidence),
            ),
            reverse=True,
        )[:max_tracks]


class Track:
    _id_counter = 0
    DISPLAY_WINDOW_FRAMES = 5
    DISPLAY_POINT_LIMIT = 192
    DISPLAY_LOCAL_XY_LIMIT = 0.7
    DISPLAY_LOCAL_Z_LIMIT = 1.2
    SKELETON_MIN_POINTS = 6
    SKELETON_HIDE_CONFIDENCE = 0.12
    SKELETON_MAX_JOINT_STEP = 0.35
    SKELETON_MAX_EXTREMITY_STEP = 0.25
    SKELETON_MIN_VERTICAL_EXTENT = 0.55
    SKELETON_DEFAULT_HEIGHT = 0.75

    def __init__(
        self,
        initial_pos: np.ndarray,
        initial_dims: Optional[np.ndarray] = None,
        initial_points: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
        sensor_track_id: Optional[int] = None,
        source: str = "cluster",
    ):
        if track_id is None:
            track_id = Track._id_counter
            Track._id_counter += 1
        else:
            Track._id_counter = max(Track._id_counter, int(track_id) + 1)

        self.id = int(track_id)
        self.sensor_track_id = sensor_track_id
        self.source = source
        self.age = 0
        self.missing = 0
        self.state_label = "Normal"
        self.points = np.asarray(initial_points, dtype=np.float32) if initial_points is not None else None
        self.dims = (
            np.asarray(initial_dims, dtype=np.float32).reshape(3)
            if initial_dims is not None
            else np.array([0.6, 0.6, 1.7], dtype=np.float32)
        )
        self.point_count = int(len(self.points)) if self.points is not None else 0
        self.doppler_mean = float(self.points[:, 3].mean()) if self.point_count and self.points.shape[1] >= 4 else 0.0
        self.confidence = 0.0
        self.stats: Dict[str, float] = {}
        self.display_points_local: deque[np.ndarray] = deque(maxlen=self.DISPLAY_WINDOW_FRAMES)
        self.display_points_age: deque[int] = deque(maxlen=self.DISPLAY_WINDOW_FRAMES)
        self._smoothed_skeleton_lines: Optional[np.ndarray] = None

        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = np.zeros((6, 1), dtype=np.float32)
        self.kf.x[:3] = np.asarray(initial_pos, dtype=np.float32).reshape(3, 1)
        self.kf.F = np.eye(6, dtype=np.float32)
        self.kf.H = np.eye(3, 6, dtype=np.float32)
        self.kf.P *= 8.0
        self.kf.R *= 0.15
        self.kf.Q *= 0.02
        self._set_dt(0.1)

    def _set_dt(self, dt: float) -> None:
        self.kf.F = np.eye(6, dtype=np.float32)
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt

    def predict(self, dt: float = 0.1, advance_position: bool = True) -> None:
        self._set_dt(dt if advance_position else 0.0)
        self.kf.predict()
        self.age += 1
        self.missing += 1
        self.points = None
        self.point_count = 0
        if self.display_points_age:
            self.display_points_age = deque((age + 1 for age in self.display_points_age), maxlen=self.DISPLAY_WINDOW_FRAMES)

    def update_from_detection(self, detection: Detection) -> None:
        measurement = detection.position.reshape(3, 1)
        self.kf.update(measurement)
        if detection.sensor_track_id is not None:
            self.sensor_track_id = detection.sensor_track_id
            self.kf.x[3:6] = detection.velocity_hint.reshape(3, 1)
        else:
            self.kf.x[4, 0] = 0.7 * self.kf.x[4, 0] + 0.3 * float(detection.velocity_hint[1])

        self.missing = 0
        self.source = detection.source
        self.confidence = float(detection.confidence)
        self.stats = dict(detection.stats)
        self.points = detection.points if len(detection.points) else None
        self.point_count = int(len(detection.points))

        if detection.points.shape[0] > 0 and detection.points.shape[1] >= 4:
            self.doppler_mean = float(detection.points[:, 3].mean())
        else:
            self.doppler_mean = float(detection.velocity_hint[1])

        if detection.dims is not None:
            dims = np.asarray(detection.dims, dtype=np.float32)
            self.dims = 0.75 * self.dims + 0.25 * dims

        self._append_display_points(detection.points)

    def update_from_sensor_track(self, detection: Detection) -> None:
        self.sensor_track_id = detection.sensor_track_id
        self.source = detection.source
        self.confidence = float(detection.confidence)
        self.stats = dict(detection.stats)
        self.kf.x[:3] = detection.position.reshape(3, 1)
        self.kf.x[3:6] = detection.velocity_hint.reshape(3, 1)
        self.points = detection.points if len(detection.points) else None
        self.point_count = int(len(detection.points))
        self.dims = 0.65 * self.dims + 0.35 * detection.dims

        self.doppler_mean = (
            float(detection.points[:, 3].mean())
            if self.points is not None and self.points.shape[1] >= 4
            else float(detection.velocity_hint[1])
        )
        self.missing = 0
        self._append_display_points(detection.points)

    @property
    def position(self) -> np.ndarray:
        return self.kf.x[:3].flatten()

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.x[3:6].flatten()

    @property
    def height(self) -> float:
        return float(self.dims[2])

    @property
    def width(self) -> float:
        return float(max(self.dims[0], self.dims[1]))

    @property
    def display_points(self) -> np.ndarray:
        if not self.display_points_local:
            return np.empty((0, 4), dtype=np.float32)

        position = self.position.reshape(1, 3).astype(np.float32)
        fused_frames = []
        for frame_points in self.display_points_local:
            if frame_points is None or len(frame_points) == 0:
                continue
            world = frame_points.copy()
            world[:, :3] = world[:, :3] + position
            fused_frames.append(world)

        if not fused_frames:
            return np.empty((0, 4), dtype=np.float32)
        return np.vstack(fused_frames).astype(np.float32)

    def _append_display_points(self, points: np.ndarray) -> None:
        if points is None or len(points) == 0:
            return

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            return

        local = points[:, :4].copy()
        local[:, :3] -= self.position.reshape(1, 3)

        mask = np.abs(local[:, 0]) <= self.DISPLAY_LOCAL_XY_LIMIT
        mask &= np.abs(local[:, 1]) <= self.DISPLAY_LOCAL_XY_LIMIT
        mask &= local[:, 2] >= -self.DISPLAY_LOCAL_Z_LIMIT
        mask &= local[:, 2] <= self.DISPLAY_LOCAL_Z_LIMIT
        local = local[mask]

        if len(local) == 0:
            return

        self.display_points_local.append(local.astype(np.float32))
        self.display_points_age.append(0)
        self._trim_display_cloud()

    def _trim_display_cloud(self) -> None:
        total_points = sum(len(frame_points) for frame_points in self.display_points_local if frame_points is not None)
        while total_points > self.DISPLAY_POINT_LIMIT and self.display_points_local:
            oldest = self.display_points_local[0]
            if oldest is None or len(oldest) == 0:
                self.display_points_local.popleft()
                if self.display_points_age:
                    self.display_points_age.popleft()
                continue

            overflow = total_points - self.DISPLAY_POINT_LIMIT
            if overflow >= len(oldest):
                total_points -= len(oldest)
                self.display_points_local.popleft()
                if self.display_points_age:
                    self.display_points_age.popleft()
            else:
                self.display_points_local[0] = oldest[overflow:]
                total_points -= overflow

    def get_skeleton(self, ratios: Optional[dict] = None, gravity_bias: bool = True) -> np.ndarray:
        points = self._skeleton_source_points()
        if len(points) >= self.SKELETON_MIN_POINTS:
            skeleton = self._get_coarse_skeleton_from_points(points)
        elif self._smoothed_skeleton_lines is not None:
            return self._smoothed_skeleton_lines.copy()
        elif self.confidence < self.SKELETON_HIDE_CONFIDENCE and self.point_count <= 0:
            return np.empty((0, 3), dtype=np.float32)
        else:
            skeleton = self._get_box_heuristic_skeleton()
        skeleton = self._preserve_skeleton_sides(skeleton)
        skeleton = self._limit_skeleton_motion(skeleton)
        return self._smooth_skeleton(skeleton)

    def _skeleton_source_points(self) -> np.ndarray:
        fused = self.display_points
        if fused.shape[0] >= 6:
            points = fused[:, :3]
        elif self.points is not None and len(self.points) >= 6:
            points = np.asarray(self.points, dtype=np.float32)[:, :3]
        else:
            return np.empty((0, 3), dtype=np.float32)

        finite_mask = np.isfinite(points).all(axis=1)
        return np.ascontiguousarray(points[finite_mask], dtype=np.float32)

    def _get_coarse_skeleton_from_points(self, points: np.ndarray) -> np.ndarray:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        extent = np.maximum(maxs - mins, 1e-3)
        horizontal_extent = float(max(extent[0], extent[1], self.width))
        point_vertical_extent = float(extent[2])
        track_height = float(self.height)
        state = str(getattr(self, "state_label", "")).upper()
        confirmed_fall = "FALL" in state
        reliable_low_pose = (
            ("GROUND" in state or "LOW" in state)
            and point_vertical_extent >= self.SKELETON_MIN_VERTICAL_EXTENT
            and horizontal_extent > max(0.6, point_vertical_extent) * 1.25
        )
        flat_body_pose = (
            point_vertical_extent < self.SKELETON_MIN_VERTICAL_EXTENT
            and track_height < self.SKELETON_MIN_VERTICAL_EXTENT
            and horizontal_extent > max(0.8, track_height * 2.0)
        )

        if point_vertical_extent < self.SKELETON_MIN_VERTICAL_EXTENT and not (confirmed_fall or flat_body_pose):
            return self._get_upright_marker_skeleton(points)

        if confirmed_fall or reliable_low_pose or flat_body_pose:
            return self._get_lying_skeleton_from_points(points)
        return self._get_upright_skeleton_from_points(points)

    def _get_upright_marker_skeleton(self, points: Optional[np.ndarray] = None) -> np.ndarray:
        center = np.asarray(self.position, dtype=np.float32).copy()
        if points is not None and len(points) > 0:
            points = np.asarray(points, dtype=np.float32)
            center[:2] = np.mean(points[:, :2], axis=0)
            base_z = float(np.percentile(points[:, 2], 5))
        else:
            base_z = float(center[2] - max(float(self.height), 0.2) / 2.0)

        height = float(self.height)
        if points is not None and len(points) > 0:
            points = np.asarray(points, dtype=np.float32)
            z_span = float(np.percentile(points[:, 2], 95) - np.percentile(points[:, 2], 5))
            xy_span = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
            compact_height = max(height, z_span + 0.20, min(0.85, xy_span * 0.75))
            height = float(np.clip(compact_height, 0.45, 1.10))
        else:
            if height < self.SKELETON_MIN_VERTICAL_EXTENT or height > 2.3:
                height = self.SKELETON_DEFAULT_HEIGHT
            height = float(np.clip(height, 0.45, 1.35))

        if points is not None and len(points) >= 3:
            _forward_xy, lateral_xy = self._horizontal_axes(points)
        else:
            lateral_xy = np.array([1.0, 0.0], dtype=np.float32)

        shoulder = max(0.28, min(0.45, self.width * 0.75))
        foot_spread = max(0.14, min(0.28, self.width * 0.35))

        head = np.array([center[0], center[1], base_z + height], dtype=np.float32)
        neck = np.array([center[0], center[1], base_z + height * 0.82], dtype=np.float32)
        pelvis = np.array([center[0], center[1], base_z + height * 0.48], dtype=np.float32)
        hand_z = base_z + height * 0.58
        left_hand = np.array([center[0], center[1], hand_z], dtype=np.float32)
        right_hand = left_hand.copy()
        left_hand[:2] -= lateral_xy * shoulder
        right_hand[:2] += lateral_xy * shoulder
        left_foot = np.array([center[0], center[1], base_z], dtype=np.float32)
        right_foot = left_foot.copy()
        left_foot[:2] -= lateral_xy * foot_spread
        right_foot[:2] += lateral_xy * foot_spread

        return self._skeleton_lines(head, neck, pelvis, left_hand, right_hand, left_foot, right_foot)

    def _horizontal_axes(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        velocity_xy = np.asarray(self.velocity[:2], dtype=np.float32)
        speed = float(np.linalg.norm(velocity_xy))
        if speed > 0.20:
            forward_xy = velocity_xy / speed
        else:
            centered = points[:, :2] - np.mean(points[:, :2], axis=0, keepdims=True)
            if len(centered) >= 3:
                try:
                    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
                    forward_xy = vh[0].astype(np.float32)
                except np.linalg.LinAlgError:
                    forward_xy = np.array([0.0, 1.0], dtype=np.float32)
            else:
                forward_xy = np.array([0.0, 1.0], dtype=np.float32)

        if np.linalg.norm(forward_xy) < 1e-4:
            forward_xy = np.array([0.0, 1.0], dtype=np.float32)
        forward_xy = forward_xy / np.linalg.norm(forward_xy)
        lateral_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=np.float32)
        return forward_xy.astype(np.float32), lateral_xy.astype(np.float32)

    @staticmethod
    def _mean_or_default(points: np.ndarray, default: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return default.astype(np.float32)
        return np.mean(points, axis=0).astype(np.float32)

    def _split_extreme_points(
        self,
        points: np.ndarray,
        lateral_xy: np.ndarray,
        default_center: np.ndarray,
        default_z: float,
        spread: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(points) < 2:
            left = default_center.copy()
            right = default_center.copy()
            left[:2] -= lateral_xy * spread
            right[:2] += lateral_xy * spread
            left[2] = default_z
            right[2] = default_z
            return left.astype(np.float32), right.astype(np.float32)

        center_xy = np.mean(points[:, :2], axis=0)
        projections = (points[:, :2] - center_xy) @ lateral_xy
        low_cut = np.percentile(projections, 35)
        high_cut = np.percentile(projections, 65)
        left = self._mean_or_default(points[projections <= low_cut], default_center)
        right = self._mean_or_default(points[projections >= high_cut], default_center)
        left[2] = default_z if not np.isfinite(left[2]) else left[2]
        right[2] = default_z if not np.isfinite(right[2]) else right[2]
        return left.astype(np.float32), right.astype(np.float32)

    def _get_upright_skeleton_from_points(self, points: np.ndarray) -> np.ndarray:
        _forward_xy, lateral_xy = self._horizontal_axes(points)
        center = np.asarray(self.position, dtype=np.float32)
        center_xy = center[:2]
        z_low = float(np.percentile(points[:, 2], 5))
        z_high = float(np.percentile(points[:, 2], 95))
        height = max(0.75, z_high - z_low, float(self.height))
        width = max(0.35, float(self.width))

        head_default = np.array([center_xy[0], center_xy[1], z_low + height], dtype=np.float32)
        neck_default = np.array([center_xy[0], center_xy[1], z_low + height * 0.82], dtype=np.float32)
        pelvis_default = np.array([center_xy[0], center_xy[1], z_low + height * 0.45], dtype=np.float32)

        top_mask = points[:, 2] >= np.percentile(points[:, 2], 82)
        torso_mask = (points[:, 2] >= z_low + height * 0.30) & (points[:, 2] <= z_low + height * 0.85)
        lower_mask = points[:, 2] <= z_low + height * 0.25
        arm_mask = (points[:, 2] >= z_low + height * 0.35) & (points[:, 2] <= z_low + height * 0.78)

        head = self._mean_or_default(points[top_mask], head_default)
        torso = self._mean_or_default(points[torso_mask], (neck_default + pelvis_default) * 0.5)
        neck = (head * 0.25 + torso * 0.75).astype(np.float32)
        neck[2] = max(neck[2], z_low + height * 0.72)
        pelvis = (torso * 0.45 + pelvis_default * 0.55).astype(np.float32)
        pelvis[2] = min(pelvis[2], z_low + height * 0.55)

        foot_center = self._mean_or_default(points[lower_mask], np.array([center_xy[0], center_xy[1], z_low], dtype=np.float32))
        foot_z = min(float(foot_center[2]), z_low + height * 0.08)
        left_foot, right_foot = self._split_extreme_points(
            points[lower_mask],
            lateral_xy,
            foot_center,
            foot_z,
            max(0.12, width * 0.22),
        )
        left_foot[2] = foot_z
        right_foot[2] = foot_z

        hand_default = np.array([neck[0], neck[1], z_low + height * 0.55], dtype=np.float32)
        left_hand, right_hand = self._split_extreme_points(
            points[arm_mask],
            lateral_xy,
            hand_default,
            float(hand_default[2]),
            max(0.20, width * 0.55),
        )
        if len(points[arm_mask]) < 4:
            left_hand = hand_default.copy()
            right_hand = hand_default.copy()
            left_hand[:2] -= lateral_xy * max(0.25, width * 0.6)
            right_hand[:2] += lateral_xy * max(0.25, width * 0.6)

        return self._skeleton_lines(head, neck, pelvis, left_hand, right_hand, left_foot, right_foot)

    def _get_lying_skeleton_from_points(self, points: np.ndarray) -> np.ndarray:
        center = self._mean_or_default(points, np.asarray(self.position, dtype=np.float32))
        centered_xy = points[:, :2] - center[:2]
        if len(points) >= 3:
            try:
                _u, _s, vh = np.linalg.svd(centered_xy, full_matrices=False)
                body_xy = vh[0].astype(np.float32)
            except np.linalg.LinAlgError:
                body_xy = np.array([0.0, 1.0], dtype=np.float32)
        else:
            body_xy = np.array([0.0, 1.0], dtype=np.float32)
        if np.linalg.norm(body_xy) < 1e-4:
            body_xy = np.array([0.0, 1.0], dtype=np.float32)
        body_xy = body_xy / np.linalg.norm(body_xy)
        lateral_xy = np.array([-body_xy[1], body_xy[0]], dtype=np.float32)

        projections = centered_xy @ body_xy
        body_min = float(np.percentile(projections, 5))
        body_max = float(np.percentile(projections, 95))
        body_len = max(0.8, body_max - body_min)
        low_end = points[projections <= np.percentile(projections, 20)]
        high_end = points[projections >= np.percentile(projections, 80)]
        low_mean = self._mean_or_default(low_end, center)
        high_mean = self._mean_or_default(high_end, center)

        if float(high_mean[2]) + 0.03 >= float(low_mean[2]):
            head = high_mean
            foot_center = low_mean
        else:
            head = low_mean
            foot_center = high_mean
            body_xy = -body_xy
            lateral_xy = -lateral_xy

        neck = (head * 0.65 + center * 0.35).astype(np.float32)
        pelvis = (center * 0.45 + foot_center * 0.55).astype(np.float32)
        mid_z = float(np.median(points[:, 2]))
        neck[2] = max(float(neck[2]), mid_z)
        pelvis[2] = mid_z

        width = max(0.35, float(self.width), float(np.percentile(np.abs(centered_xy @ lateral_xy), 90)) * 2.0)
        hand_center = (neck * 0.45 + pelvis * 0.55).astype(np.float32)
        left_hand = hand_center.copy()
        right_hand = hand_center.copy()
        left_hand[:2] -= lateral_xy * max(0.25, width * 0.55)
        right_hand[:2] += lateral_xy * max(0.25, width * 0.55)
        left_hand[2] = mid_z
        right_hand[2] = mid_z

        foot_z = float(np.percentile(points[:, 2], 15))
        left_foot = foot_center.copy()
        right_foot = foot_center.copy()
        left_foot[:2] -= lateral_xy * max(0.12, width * 0.25)
        right_foot[:2] += lateral_xy * max(0.12, width * 0.25)
        left_foot[2] = foot_z
        right_foot[2] = foot_z

        # Keep the body stretched even when sparse points cluster near the track center.
        head[:2] = center[:2] + body_xy * (body_len * 0.45)
        pelvis[:2] = center[:2] - body_xy * (body_len * 0.10)
        left_foot[:2] = center[:2] - body_xy * (body_len * 0.45) - lateral_xy * max(0.12, width * 0.25)
        right_foot[:2] = center[:2] - body_xy * (body_len * 0.45) + lateral_xy * max(0.12, width * 0.25)

        return self._skeleton_lines(head, neck, pelvis, left_hand, right_hand, left_foot, right_foot)

    @staticmethod
    def _skeleton_lines(
        head: np.ndarray,
        neck: np.ndarray,
        pelvis: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        left_foot: np.ndarray,
        right_foot: np.ndarray,
    ) -> np.ndarray:
        return np.array(
            [
                head,
                neck,
                neck,
                pelvis,
                neck,
                left_hand,
                neck,
                right_hand,
                pelvis,
                left_foot,
                pelvis,
                right_foot,
            ],
            dtype=np.float32,
        )

    def _smooth_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        skeleton = np.asarray(skeleton, dtype=np.float32)
        if skeleton.shape != (12, 3) or not np.isfinite(skeleton).all():
            return self._get_box_heuristic_skeleton()

        if (
            self._smoothed_skeleton_lines is None
            or self._smoothed_skeleton_lines.shape != skeleton.shape
            or not np.isfinite(self._smoothed_skeleton_lines).all()
        ):
            self._smoothed_skeleton_lines = skeleton.copy()
            return skeleton

        alpha = 0.35
        self._smoothed_skeleton_lines = (
            (1.0 - alpha) * self._smoothed_skeleton_lines + alpha * skeleton
        ).astype(np.float32)
        return self._smoothed_skeleton_lines.copy()

    def _preserve_skeleton_sides(self, skeleton: np.ndarray) -> np.ndarray:
        skeleton = np.asarray(skeleton, dtype=np.float32).copy()
        if self._smoothed_skeleton_lines is None or self._smoothed_skeleton_lines.shape != skeleton.shape:
            return skeleton

        previous = self._smoothed_skeleton_lines
        for left_idx, right_idx in ((4, 6), (8, 10)):
            keep_cost = (
                np.linalg.norm(skeleton[left_idx] - previous[left_idx])
                + np.linalg.norm(skeleton[right_idx] - previous[right_idx])
            )
            swap_cost = (
                np.linalg.norm(skeleton[left_idx] - previous[right_idx])
                + np.linalg.norm(skeleton[right_idx] - previous[left_idx])
            )
            if swap_cost + 0.05 < keep_cost:
                left_pair = skeleton[left_idx : left_idx + 2].copy()
                skeleton[left_idx : left_idx + 2] = skeleton[right_idx : right_idx + 2]
                skeleton[right_idx : right_idx + 2] = left_pair
        return skeleton

    def _limit_skeleton_motion(self, skeleton: np.ndarray) -> np.ndarray:
        skeleton = np.asarray(skeleton, dtype=np.float32).copy()
        if self._smoothed_skeleton_lines is None or self._smoothed_skeleton_lines.shape != skeleton.shape:
            return skeleton

        previous = self._smoothed_skeleton_lines
        for idx in range(len(skeleton)):
            max_step = (
                self.SKELETON_MAX_EXTREMITY_STEP
                if idx in {4, 5, 6, 7, 8, 9, 10, 11}
                else self.SKELETON_MAX_JOINT_STEP
            )
            delta = skeleton[idx] - previous[idx]
            dist = float(np.linalg.norm(delta))
            if dist > max_step and dist > 1e-6:
                skeleton[idx] = previous[idx] + delta * (max_step / dist)
        return skeleton.astype(np.float32)

    def _get_box_heuristic_skeleton(self) -> np.ndarray:
        x, y, z = self.position
        w, d, h = self.dims
        foot_z = z - h / 2
        head_z = z + h / 2

        pelvis = np.array([x, y, foot_z + h * 0.45], dtype=np.float32)
        neck = np.array([x, y, foot_z + h * 0.82], dtype=np.float32)
        head = np.array([x, y, head_z], dtype=np.float32)

        half_arm = max(0.25, w * 0.65)
        half_foot = max(0.12, w * 0.25)
        hand_z = foot_z + h * 0.55
        left_hand = np.array([x - half_arm, y, hand_z], dtype=np.float32)
        right_hand = np.array([x + half_arm, y, hand_z], dtype=np.float32)
        left_foot = np.array([x - half_foot, y, foot_z], dtype=np.float32)
        right_foot = np.array([x + half_foot, y, foot_z], dtype=np.float32)

        return self._skeleton_lines(head, neck, pelvis, left_hand, right_hand, left_foot, right_foot)


class Tracker:
    def __init__(self, max_missing: int = 12, dist_thresh: float = 1.4):
        self.tracks: List[Track] = []
        self.max_missing = max_missing
        self.dist_thresh = dist_thresh
        self.max_tracks = 0
        self.predict_missing_motion_enabled = False

    def update(self, detections: Sequence[Detection], dt: float = 0.1) -> List[Track]:
        detections = list(detections or [])
        for track in self.tracks:
            track.predict(dt, advance_position=self.predict_missing_motion_enabled)

        sensor_detections = [det for det in detections if det.sensor_track_id is not None]
        cluster_detections = [det for det in detections if det.sensor_track_id is None]
        updated_track_ids = set()

        if sensor_detections:
            updated_track_ids = self._apply_sensor_detections(sensor_detections)
        if cluster_detections:
            self._apply_cluster_detections(
                cluster_detections,
                skip_track_ids=updated_track_ids,
            )

        self._drop_expired_tracks()
        return self.tracks

    def _apply_sensor_detections(self, detections: Sequence[Detection]) -> set[int]:
        existing_by_sensor_id = {
            track.sensor_track_id: track
            for track in self.tracks
            if track.sensor_track_id is not None
        }
        updated_track_ids = set()

        for detection in detections:
            track = existing_by_sensor_id.get(detection.sensor_track_id)
            if track is None:
                track = Track(
                    detection.position,
                    detection.dims,
                    detection.points,
                    track_id=detection.sensor_track_id,
                    sensor_track_id=detection.sensor_track_id,
                    source=detection.source,
                )
                self.tracks.append(track)
            Track._id_counter = max(Track._id_counter, int(track.id) + 1)
            track.update_from_sensor_track(detection)
            updated_track_ids.add(track.id)

        return updated_track_ids

    def _apply_cluster_detections(
        self,
        detections: Sequence[Detection],
        skip_track_ids: Optional[set[int]] = None,
    ) -> None:
        skip_track_ids = set(skip_track_ids or set())
        track_pool = [track for track in self.tracks if track.id not in skip_track_ids]

        if not track_pool:
            for detection in detections:
                new_track = Track(
                    detection.position,
                    detection.dims,
                    detection.points,
                    track_id=self._allocate_track_id(),
                    source=detection.source,
                )
                new_track.update_from_detection(detection)
                self.tracks.append(new_track)
            return

        cost_matrix = np.full((len(track_pool), len(detections)), 999.0, dtype=np.float32)
        for t_index, track in enumerate(track_pool):
            for d_index, detection in enumerate(detections):
                euclidean = float(np.linalg.norm(track.position - detection.position))
                if euclidean > self.dist_thresh:
                    continue

                innovation = detection.position.reshape(3, 1) - track.kf.x[:3]
                s_matrix = track.kf.H @ track.kf.P @ track.kf.H.T + track.kf.R
                try:
                    maha_matrix = innovation.T @ np.linalg.inv(s_matrix) @ innovation
                    maha = float(maha_matrix.item()) if hasattr(maha_matrix, "item") else float(maha_matrix[0, 0])
                except np.linalg.LinAlgError:
                    maha = euclidean * euclidean

                velocity_penalty = abs(float(track.velocity[1]) - float(detection.velocity_hint[1])) * 0.35
                height_penalty = abs(float(track.dims[2]) - float(detection.dims[2])) * 0.2
                cost_matrix[t_index, d_index] = maha + velocity_penalty + height_penalty

        row_ind, col_ind = _linear_sum_assignment(cost_matrix)
        assigned_detections = set()

        for t_index, d_index in zip(row_ind, col_ind):
            cost = float(cost_matrix[t_index, d_index])
            if cost >= 12.0:
                continue
            track_pool[t_index].update_from_detection(detections[d_index])
            assigned_detections.add(d_index)

        for d_index, detection in enumerate(detections):
            if d_index in assigned_detections:
                continue
            new_track = Track(
                detection.position,
                detection.dims,
                detection.points,
                track_id=self._allocate_track_id(),
                source=detection.source,
            )
            new_track.update_from_detection(detection)
            self.tracks.append(new_track)

    def _allocate_track_id(self) -> int:
        used_ids = {int(track.id) for track in self.tracks}
        max_tracks = int(getattr(self, "max_tracks", 0) or 0)
        if max_tracks > 0:
            for track_id in range(1, max_tracks + 1):
                if track_id not in used_ids:
                    Track._id_counter = max(Track._id_counter, track_id + 1)
                    return track_id
        track_id = 1
        while track_id in used_ids:
            track_id += 1
        Track._id_counter = max(Track._id_counter, track_id + 1)
        return track_id

    def _drop_expired_tracks(self) -> None:
        self.tracks = [track for track in self.tracks if track.missing <= self.max_missing]
