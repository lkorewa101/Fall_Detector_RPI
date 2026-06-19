import time
import platform
from collections import deque

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtGui import QColor, QFont


class VisualizationManager:
    def __init__(self, main_window):
        self.mw = main_window
        self.render_durations = deque(maxlen=30)
        machine = platform.machine().lower()
        system = platform.system().lower()
        self.rpi_render_safe_mode = system == "linux" and ("arm" in machine or "aarch" in machine)

    def _build_doppler_colors(self, doppler_values):
        doppler_values = np.asarray(doppler_values, dtype=np.float32)
        if len(doppler_values) == 0:
            return np.empty((0, 4), dtype=np.float32)

        doppler_values = np.nan_to_num(doppler_values, nan=0.0, posinf=0.0, neginf=0.0)
        scale = float(getattr(self.mw, "viz_doppler_color_scale", 4.3))
        if not np.isfinite(scale):
            scale = 1.0
        scale = float(np.clip(scale, 0.05, 5.0))

        colors = np.empty((len(doppler_values), 4), dtype=np.float32)
        colors[:, 0] = 0.70
        colors[:, 1] = 0.70
        colors[:, 2] = 0.70
        colors[:, 3] = 0.62

        approach_mask = doppler_values < 0.0
        recede_mask = doppler_values > 0.0
        strength = np.clip(np.abs(doppler_values) / scale, 0.0, 1.0).astype(np.float32)

        if np.any(approach_mask):
            approach_strength = strength[approach_mask, None]
            approach_target = np.array([1.0, 0.08, 0.02], dtype=np.float32)
            colors[approach_mask, :3] = (
                colors[approach_mask, :3] * (1.0 - approach_strength)
                + approach_target * approach_strength
            )

        if np.any(recede_mask):
            recede_strength = strength[recede_mask, None]
            recede_target = np.array([0.02, 0.28, 1.0], dtype=np.float32)
            colors[recede_mask, :3] = (
                colors[recede_mask, :3] * (1.0 - recede_strength)
                + recede_target * recede_strength
            )

        moving_mask = approach_mask | recede_mask
        if np.any(moving_mask):
            colors[moving_mask, 3] = 0.62 + 0.38 * strength[moving_mask]

        return colors

    def _point_doppler_colors(self, points, alpha_offset=0.0, alpha_max=1.0):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 2 and points.shape[1] >= 4 and len(points) > 0:
            colors = self._build_doppler_colors(points[:, 3])
        else:
            colors = self._build_doppler_colors(np.zeros(len(points), dtype=np.float32))
        if len(colors) > 0 and alpha_offset:
            colors[:, 3] = np.clip(colors[:, 3] + float(alpha_offset), 0.0, float(alpha_max))
        elif len(colors) > 0 and alpha_max < 1.0:
            colors[:, 3] = np.minimum(colors[:, 3], float(alpha_max))
        return colors

    @staticmethod
    def _motion_color_from_value(value, static_color=(0.0, 1.0, 0.55, 1.0)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if not np.isfinite(value) or abs(value) < 0.02:
            return static_color

        strength = float(np.clip(abs(value) / 1.5, 0.0, 1.0))
        if value < 0.0:
            return (1.0, 0.15 + (1.0 - strength) * 0.35, 0.05, 0.92 + strength * 0.08)
        return (0.05, 0.35 + (1.0 - strength) * 0.25, 1.0, 0.92 + strength * 0.08)

    @staticmethod
    def _sample_points(points, max_points):
        if points is None:
            return np.empty((0, 4), dtype=np.float32)

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or len(points) == 0:
            return np.empty((0, 4), dtype=np.float32)

        if len(points) <= max_points:
            return points

        indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
        return points[indices]

    @staticmethod
    def _normalize_points(points):
        if points is None:
            return np.empty((0, 4), dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        if points.shape[1] == 3:
            points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=np.float32)])
        elif points.shape[1] > 4:
            points = points[:, :4]
        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        if points.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.ascontiguousarray(points, dtype=np.float32)

    @staticmethod
    def _height_mask(points, min_height, max_height):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        return (points[:, 2] >= float(min_height)) & (points[:, 2] <= float(max_height))

    @staticmethod
    def _cross_lines(points, size):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)
        xyz = points[:, :3]
        offset_x = np.array([size, 0.0, 0.0], dtype=np.float32)
        offset_y = np.array([0.0, size, 0.0], dtype=np.float32)
        lines = np.empty((len(xyz) * 4, 3), dtype=np.float32)
        lines[0::4] = xyz - offset_x
        lines[1::4] = xyz + offset_x
        lines[2::4] = xyz - offset_y
        lines[3::4] = xyz + offset_y
        return lines

    @staticmethod
    def _track_color(track, details=None, motion_color=False):
        details = details or {}
        label = getattr(track, "state_label", "")
        display_label = str(details.get("display_label", ""))
        action_name = str(details.get("action_name", ""))
        label_bundle = f"{label} {display_label} {action_name}".upper()
        if getattr(track, "missing", 0) > 0 or "LOST" in label:
            return (0.4, 0.4, 0.4, 0.35)
        if "FALL" in label_bundle or "FALLING" in label_bundle:
            return (1.0, 0.1, 0.1, 1.0)
        if "CANDIDATE" in label_bundle:
            return (1.0, 0.85, 0.0, 1.0)
        if motion_color:
            motion_value = getattr(track, "doppler_mean", 0.0)
            try:
                velocity = np.asarray(getattr(track, "velocity", []), dtype=np.float32).reshape(-1)
                if velocity.size >= 2 and abs(float(velocity[1])) > abs(float(motion_value)):
                    motion_value = float(velocity[1])
            except Exception:
                pass
            return VisualizationManager._motion_color_from_value(motion_value)
        return (0.0, 1.0, 0.55, 1.0)

    @staticmethod
    def get_box_points(center, dims):
        w, d, h = dims
        x, y, z = center

        p0 = [x - w / 2, y - d / 2, z - h / 2]
        p1 = [x + w / 2, y - d / 2, z - h / 2]
        p2 = [x + w / 2, y + d / 2, z - h / 2]
        p3 = [x - w / 2, y + d / 2, z - h / 2]
        p4 = [x - w / 2, y - d / 2, z + h / 2]
        p5 = [x + w / 2, y - d / 2, z + h / 2]
        p6 = [x + w / 2, y + d / 2, z + h / 2]
        p7 = [x - w / 2, y + d / 2, z + h / 2]

        return np.array(
            [
                p0, p1, p1, p2, p2, p3, p3, p0,
                p4, p5, p5, p6, p6, p7, p7, p4,
                p0, p4, p1, p5, p2, p6, p3, p7,
            ],
            dtype=np.float32,
        )

    def _ensure_dynamic_items(self):
        mw = self.mw
        if self.rpi_render_safe_mode and not getattr(self, "_rpi_gl_options_applied", False):
            for item_name in ("scatter", "trail_scatter"):
                item = getattr(mw, item_name, None)
                if item is not None:
                    try:
                        item.setGLOptions("opaque")
                    except Exception:
                        pass
            self._rpi_gl_options_applied = True

        if not hasattr(self, "fused_scatter"):
            self.fused_scatter = gl.GLScatterPlotItem(pos=np.empty((0, 3)), color=(0, 1, 0.55, 0.85), size=7.0, pxMode=True)
            self.fused_scatter.setGLOptions("opaque" if self.rpi_render_safe_mode else "translucent")
            mw.viewer.addItem(self.fused_scatter)

        if not hasattr(self, "highlight_scatter"):
            self.highlight_scatter = gl.GLScatterPlotItem(pos=np.empty((0, 3)), color=(1, 1, 0, 1), size=7.0, pxMode=True)
            self.highlight_scatter.setGLOptions("opaque" if self.rpi_render_safe_mode else "translucent")
            mw.viewer.addItem(self.highlight_scatter)

        if not hasattr(self, "point_marker_lines"):
            self.point_marker_lines = gl.GLLinePlotItem(pos=np.empty((0, 3), dtype=np.float32), color=(1.0, 0.68, 0.0, 1.0), width=2, mode="lines")
            mw.viewer.addItem(self.point_marker_lines)
            self.point_marker_lines.setVisible(False)

        if not hasattr(self, "height_rejected_lines"):
            self.height_rejected_lines = gl.GLLinePlotItem(pos=np.empty((0, 3), dtype=np.float32), color=(1.0, 0.55, 0.0, 0.9), width=2, mode="lines")
            mw.viewer.addItem(self.height_rejected_lines)
            self.height_rejected_lines.setVisible(False)

    @staticmethod
    def _filter_points_by_cluster_distance(points, tracks, max_dist: float):
        """Filters points that are more than max_dist away from any cluster/track's center."""
        if points is None or len(points) == 0:
            return np.empty((0, 4), dtype=np.float32)

        if not tracks:
            # If no tracks to compare against, keep all points by default
            return points

        centers = np.array([t.position for t in tracks], dtype=np.float32)
        if len(centers) == 0:
            return points

        points_xyz = points[:, :3]
        
        # Calculate distances to all centers
        diffs = points_xyz[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)

        mask = min_dists <= max_dist
        return points[mask]

    @staticmethod
    def _filter_points_near_tracks(points, tracks):
        """Filter points to only keep those near tracked objects (for highlight feature)."""
        if points is None:
            return np.empty((0, 4), dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or len(points) == 0:
            return np.empty((0, 4), dtype=np.float32)
        tracks = list(tracks or [])
        if not tracks:
            return points

        keep_mask = np.zeros(len(points), dtype=bool)
        for track in tracks:
            pos = np.asarray(track.position, dtype=np.float32)
            dims = np.asarray(track.dims, dtype=np.float32)
            x_pad = max(0.45, float(dims[0]) * 0.8)
            y_pad = max(0.55, float(dims[1]) * 1.2)
            z_min = float(pos[2] - dims[2] / 2.0 - 0.05)
            z_max = float(pos[2] + dims[2] / 2.0 + 0.08)
            local = (
                (np.abs(points[:, 0] - pos[0]) <= x_pad)
                & (np.abs(points[:, 1] - pos[1]) <= y_pad)
                & (points[:, 2] >= z_min)
                & (points[:, 2] <= z_max)
            )
            keep_mask |= local

        filtered = points[keep_mask]
        return filtered if len(filtered) > 0 else np.empty((0, 4), dtype=np.float32)

    def refresh_view(self):
        mw = self.mw
        start_time = time.perf_counter()
        try:
            self._ensure_dynamic_items()

            avg_render_time = float(np.mean(self.render_durations)) if self.render_durations else 0.0
            allow_heavy = avg_render_time <= 0.083
            current_size = float(getattr(mw, "viz_point_size", 5.0))
            if self.rpi_render_safe_mode:
                current_size = max(current_size, 9.0)

            raw_points = self._normalize_points(getattr(mw, "latest_points", None))
            render_points = raw_points
            rejected_points = np.empty((0, 4), dtype=np.float32)
            height_filter_enabled = bool(getattr(mw, "point_height_filter_enabled", False))
            if height_filter_enabled and raw_points.shape[0] > 0:
                height_mask = self._height_mask(
                    raw_points,
                    getattr(mw, "pt_min_height", -0.3),
                    getattr(mw, "pt_max_height", 3.0),
                )
                render_points = raw_points[height_mask]
                rejected_points = raw_points[~height_mask]

            max_pts = int(getattr(mw, "viz_max_points", 3000))
            sampled_points = self._sample_points(render_points, max_pts)
            if sampled_points.shape[0] > 0:
                bg_points = np.ascontiguousarray(sampled_points[:, :3], dtype=np.float32)
                bg_colors = np.ascontiguousarray(self._point_doppler_colors(sampled_points), dtype=np.float32)

                mw.scatter.setData(pos=bg_points, color=bg_colors, size=max(4.0, current_size * 0.9))
                if self.rpi_render_safe_mode:
                    marker_source = self._sample_points(sampled_points, 160)
                    marker_points = np.ascontiguousarray(marker_source[:, :3], dtype=np.float32)
                    tick = max(0.035, min(0.12, current_size * 0.006))
                    lower = marker_points.copy()
                    upper = marker_points.copy()
                    lower[:, 2] -= tick
                    upper[:, 2] += tick
                    marker_lines = np.empty((len(marker_points) * 2, 3), dtype=np.float32)
                    marker_lines[0::2] = lower
                    marker_lines[1::2] = upper
                    marker_colors = np.ascontiguousarray(
                        np.repeat(self._point_doppler_colors(marker_source), 2, axis=0),
                        dtype=np.float32,
                    )
                    self.point_marker_lines.setData(pos=marker_lines, color=marker_colors, width=2, mode="lines")
                    self.point_marker_lines.setVisible(True)
                else:
                    self.point_marker_lines.setData(pos=np.empty((0, 3), dtype=np.float32))
                    self.point_marker_lines.setVisible(False)
            else:
                mw.scatter.setData(pos=np.empty((0, 3), dtype=np.float32))
                self.point_marker_lines.setData(pos=np.empty((0, 3), dtype=np.float32))
                self.point_marker_lines.setVisible(False)

            sampled_rejected = self._sample_points(rejected_points, min(max_pts, 1000))
            if height_filter_enabled and sampled_rejected.shape[0] > 0:
                cross_size = max(0.035, min(0.16, current_size * 0.006))
                cross_lines = self._cross_lines(sampled_rejected, cross_size)
                cross_colors = np.ascontiguousarray(
                    np.repeat(self._point_doppler_colors(sampled_rejected, alpha_offset=0.1), 4, axis=0),
                    dtype=np.float32,
                )
                self.height_rejected_lines.setData(pos=cross_lines, color=cross_colors, width=2, mode="lines")
                self.height_rejected_lines.setVisible(True)
            else:
                self.height_rejected_lines.setData(pos=np.empty((0, 3), dtype=np.float32))
                self.height_rejected_lines.setVisible(False)

            fused_pts_list = []
            fused_color_list = []
            render_tracks = list(getattr(mw, "latest_tracks", []) or [])

            if not render_tracks and getattr(mw, "latest_detections", None):
                render_tracks = []
                for index, detection in enumerate(mw.latest_detections):
                    class PreviewTrack:
                        pass
                    preview = PreviewTrack()
                    preview.id = 1000 + index
                    preview.position = detection.position
                    preview.dims = detection.dims
                    preview.state_label = "CANDIDATE"
                    preview.source = detection.source
                    preview.point_count = len(detection.points)
                    preview.display_points = detection.points
                    render_tracks.append(preview)

            for track in render_tracks:
                display_points = getattr(track, "display_points", None)
                if getattr(track, "point_count", 0) <= 0:
                    continue
                if display_points is None or len(display_points) == 0:
                    continue

                display_points = np.asarray(display_points, dtype=np.float32)
                fused_pts_list.append(np.ascontiguousarray(display_points[:, :3], dtype=np.float32))

                fused_color_list.append(self._point_doppler_colors(display_points, alpha_offset=0.05))

            if fused_pts_list:
                fused_points = np.vstack(fused_pts_list)
                fused_colors = np.ascontiguousarray(np.vstack(fused_color_list), dtype=np.float32)
                self.fused_scatter.setData(pos=fused_points, color=fused_colors, size=current_size)
                self.fused_scatter.setVisible(True)
            else:
                self.fused_scatter.setData(pos=np.empty((0, 3), dtype=np.float32))
                self.fused_scatter.setVisible(False)

            # Highlight objects: show points near tracked objects in yellow
            if (
                getattr(mw, "highlight_objects_check", None)
                and mw.highlight_objects_check.isChecked()
                and render_tracks
                and raw_points is not None
                and len(raw_points) > 0
            ):
                highlight_pts = self._filter_points_near_tracks(raw_points, render_tracks)
                if len(highlight_pts) > 0:
                    self.highlight_scatter.setData(
                        pos=np.ascontiguousarray(highlight_pts[:, :3], dtype=np.float32),
                        color=np.ascontiguousarray(
                            self._point_doppler_colors(highlight_pts, alpha_offset=0.05),
                            dtype=np.float32,
                        ),
                        size=max(3.0, current_size * 1.2),
                    )
                    self.highlight_scatter.setVisible(True)
                else:
                    self.highlight_scatter.setData(pos=np.empty((0, 3), dtype=np.float32))
                    self.highlight_scatter.setVisible(False)
            else:
                self.highlight_scatter.setData(pos=np.empty((0, 3), dtype=np.float32))
                self.highlight_scatter.setVisible(False)

            if allow_heavy and mw.viz_trail_length > 0 and len(mw.point_history) > 1:
                all_hist = []
                all_hist_colors = []
                limit = int(mw.viz_trail_length)
                past_frames = list(mw.point_history)[:-1][-limit:]
                for pts in past_frames:
                    # Apply same height filtering to trail history
                    if hasattr(mw, "processor"):
                        pts = mw.processor._sanitize_points(pts)
                    sampled_hist = self._sample_points(pts, 64)
                    if sampled_hist.shape[0] > 0:
                        all_hist.append(sampled_hist[:, :3])
                        all_hist_colors.append(self._point_doppler_colors(sampled_hist, alpha_max=0.3))
                if all_hist:
                    trail_pts = np.ascontiguousarray(np.vstack(all_hist), dtype=np.float32)
                    trail_colors = np.ascontiguousarray(np.vstack(all_hist_colors), dtype=np.float32)
                    mw.trail_scatter.setData(pos=trail_pts, color=trail_colors, size=max(2.0, current_size * 0.5))
                else:
                    mw.trail_scatter.setData(pos=np.empty((0, 3), dtype=np.float32))
            else:
                mw.trail_scatter.setData(pos=np.empty((0, 3), dtype=np.float32))

            requested_style = mw.style_combo.currentIndex()
            effective_style = requested_style if allow_heavy else 0
            if effective_style > 1:
                effective_style = 0
            skeleton_lines = []
            ratios = getattr(mw, "body_ratios", None)
            bias = getattr(mw, "gravity_bias", True)
            active_ids = []
            pose_skeleton = getattr(mw, "latest_pose_skeleton_lines", None)

            for track in render_tracks:
                active_ids.append(track.id)
                if effective_style == 1 and hasattr(track, "get_skeleton"):
                    lines = track.get_skeleton(ratios, bias)
                    if lines is not None and len(lines) > 0:
                        skeleton_lines.append(lines)

            if effective_style == 1 and pose_skeleton is not None and len(pose_skeleton) > 0:
                combined = np.asarray(pose_skeleton, dtype=np.float32)
                mw.skeleton_plot.setData(pos=combined, color=(1.0, 0.55, 0.1, 1), width=4, mode="lines")
            elif effective_style == 1 and len(skeleton_lines) > 0:
                combined = np.vstack(skeleton_lines)
                mw.skeleton_plot.setData(pos=combined, color=(0, 0.8, 1, 1), width=3, mode="lines")
            else:
                mw.skeleton_plot.setData(pos=np.empty((0, 3), dtype=np.float32))

            for tid in list(mw.track_items.keys()):
                if tid in active_ids:
                    continue
                items = mw.track_items.pop(tid)
                for item in items:
                    try:
                        mw.viewer.removeItem(item)
                    except Exception:
                        pass

            for track in render_tracks:
                pos = np.asarray(track.position, dtype=np.float32)
                dims = np.asarray(track.dims, dtype=np.float32)
                details = mw.latest_details.get(track.id, {})
                color = self._track_color(track, details)
                text_pos = (pos[0], pos[1], pos[2] + dims[2] / 2 + 0.2)
                action_name = details.get("action_name", "?")
                action_conf = details.get("action_confidence", 0.0)
                point_count = getattr(track, "point_count", 0)
                fused_count = len(getattr(track, "display_points", [])) if hasattr(track, "display_points") else 0
                vital = getattr(track, "vital", None) or {}
                heart_bpm = float(vital.get("heart_bpm", vital.get("heartRate", 0.0)) or 0.0)
                breath_bpm = float(vital.get("breath_bpm", vital.get("breathingRate", 0.0)) or 0.0)
                heart_text = f"{heart_bpm:.0f}" if heart_bpm > 0.0 else "--"
                breath_text = f"{breath_bpm:.0f}" if breath_bpm > 0.0 else "--"
                info_text = (
                    f"ID:{track.id} | {action_name} ({action_conf:.0%})\n"
                    f"{track.state_label} | pts:{point_count}/{fused_count}\n"
                    f"HR {heart_text} / BR {breath_text}"
                )
                text_color = QColor.fromRgbF(color[0], color[1], color[2], color[3])
                text_font = QFont("Arial", 14, QFont.Bold)

                if track.id not in mw.track_items:
                    t_scatter = gl.GLScatterPlotItem(pos=np.array([pos], dtype=np.float32), color=color, size=12)
                    mw.viewer.addItem(t_scatter)
                    t_text = gl.GLTextItem(pos=text_pos, text=info_text, color=text_color, font=text_font)
                    mw.viewer.addItem(t_text)
                    t_box = gl.GLLinePlotItem()
                    mw.viewer.addItem(t_box)
                    mw.track_items[track.id] = [t_scatter, t_text, t_box]
                else:
                    t_scatter, t_text, t_box = mw.track_items[track.id]
                    t_scatter.setData(pos=np.array([pos], dtype=np.float32), color=color, size=12)
                    try:
                        t_text.setData(pos=text_pos, text=info_text, color=text_color, font=text_font)
                    except AttributeError:
                        t_text.pos = text_pos
                        t_text.text = info_text
                        t_text.color = text_color
                        t_text.setFont(text_font) if hasattr(t_text, "setFont") else None
                        t_text.update()

                t_scatter.setVisible(True)
                t_box.setVisible(True)
                t_box.setData(pos=self.get_box_points(pos, dims), color=color, width=3, mode="lines")

            mw.viewer.update()

        except Exception as e:
            print(f"[ERROR] refresh_view: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.render_durations.append(time.perf_counter() - start_time)
