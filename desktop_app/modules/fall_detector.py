from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .ai_model import FallAI, FallSequenceAI


@dataclass
class TrackFallState:
    state: str = "standing"
    descending_frames: int = 0
    grounded_frames: int = 0
    low_posture_frames: int = 0
    stable_frames: int = 0
    candidate_frames: int = 0
    impact_motion_frames: int = 0
    upright_height_ema: float = 0.0
    upright_center_z_ema: float = 0.0
    last_effective_height: float = 0.0
    last_effective_center_z: float = 0.0
    last_upright_frame: int = -9999
    last_descent_frame: int = -9999
    last_lost_descent_frame: int = -9999
    last_lost_descent_drop: float = 0.0
    last_lost_descent_vz: float = 0.0
    last_fall_frame: int = -9999
    cooldown_until_frame: int = -9999
    fall_latched: bool = False
    lost_descent_latched: bool = False
    alert_sent: bool = False
    last_total_score: float = 0.0
    last_ai_frame: int = -9999
    last_ai_score: float = 0.0
    smoothed_ai_score: float = 0.0
    last_action_name: str = "Unknown"
    last_action_confidence: float = 0.0
    sequence_features: list[np.ndarray] = field(default_factory=list)
    sequence_previous_center_z: float | None = None
    last_sequence_frame: int = -9999
    last_sequence_score: float = 0.0
    smoothed_sequence_score: float = 0.0


class HybridFallDetector:
    def __init__(self) -> None:
        self.ai_model = FallAI()
        self.sequence_model = FallSequenceAI()
        self.rule_weight = 0.70
        self.ai_weight = 0.15
        self.sequence_weight = 0.15
        self.speed_threshold = -0.5
        self.height_threshold = 0.8
        self.low_height_threshold = 0.55
        self.fall_confirm_frames = 6
        self.ai_min_interval_frames = 3
        self.baseline_alpha = 0.2
        self.baseline_timeout_frames = 75
        self.recovery_frames = 4
        self.fall_cooldown_frames = 24
        self.ai_candidate_min_points = 10
        self.ai_smoothing_alpha = 0.45
        self.sequence_smoothing_alpha = 0.35
        self.sequence_history_frames = 180
        self.sequence_min_interval_frames = 3
        self.sequence_frame_delta_seconds = 0.1
        self.sequence_candidate_min_points = 3
        self.ai_assisted_fall_threshold = 0.45
        self.sequence_assisted_fall_threshold = 0.60
        self.use_rule_candidate_filter: bool = True
        self.ai_falling_threshold = 0.5
        self.enable_lost_descent_fall = False
        self.lost_descent_missing_frames = 3
        self.lost_descent_window_frames = 10
        self.lost_descent_min_center_drop = 0.08
        self.lost_descent_vz_threshold = -0.35
        self.state_by_track: Dict[int, TrackFallState] = {}

    @property
    def strict_mode(self) -> bool:
        return self.use_rule_candidate_filter

    @strict_mode.setter
    def strict_mode(self, enabled: bool) -> None:
        self.use_rule_candidate_filter = bool(enabled)

    def set_sensitivity(self, speed_thresh: float, height_thresh: float) -> None:
        self.speed_threshold = float(speed_thresh)
        self.height_threshold = float(height_thresh)
        self.low_height_threshold = min(self.height_threshold, 0.65)

    def detect(self, track, simulate_ai: bool = False, frame_index: Optional[int] = None):
        state = self.state_by_track.setdefault(track.id, TrackFallState())
        frame_index = -1 if frame_index is None else int(frame_index)

        vz = float(track.velocity[2])
        center_z = float(track.position[2])
        width = float(max(track.dims[0], track.dims[1]))
        height = float(track.dims[2])
        point_count = int(getattr(track, "point_count", 0) or 0)
        missing_frames = int(getattr(track, "missing", 0) or 0)
        signal_lost = missing_frames > 0

        display_points = getattr(track, "display_points", None)
        point_vertical_extent = 0.0
        point_low_z = center_z - height / 2.0
        point_high_z = center_z + height / 2.0
        if display_points is not None and len(display_points) >= 3:
            try:
                points = np.asarray(display_points, dtype=np.float32)
                if points.ndim == 2 and points.shape[1] >= 3:
                    z_values = points[:, 2]
                    z_values = z_values[np.isfinite(z_values)]
                    if len(z_values) >= 3:
                        point_low_z = float(np.percentile(z_values, 5))
                        point_high_z = float(np.percentile(z_values, 95))
                        point_vertical_extent = max(0.0, point_high_z - point_low_z)
            except Exception:
                point_vertical_extent = 0.0

        previous_baseline_valid = (
            state.upright_height_ema > 0.0
            and (frame_index < 0 or (frame_index - state.last_upright_frame) <= self.baseline_timeout_frames)
        )
        height_signal = max(height, point_vertical_extent)
        height_reliable = height_signal >= 0.25 or previous_baseline_valid
        effective_height = height_signal if height_signal >= 0.25 else (state.upright_height_ema or 1.60)
        effective_center_z = (point_low_z + point_high_z) * 0.5 if point_vertical_extent >= 0.25 else center_z
        ground_height = point_low_z if point_vertical_extent >= 0.25 else effective_center_z - effective_height / 2.0
        aspect_ratio = width / max(effective_height, 0.1)

        low_posture = height_reliable and (effective_height <= self.low_height_threshold or aspect_ratio >= 1.35)
        grounded = height_reliable and (
            ground_height <= self.height_threshold * 0.35 or effective_center_z <= self.height_threshold
        )
        descending = vz <= self.speed_threshold
        upright_like = (
            height_reliable
            and effective_height >= max(self.low_height_threshold + 0.35, 1.05)
            and aspect_ratio <= 1.1
            and ground_height >= self.height_threshold * 0.2
            and abs(vz) < 0.6
        )

        if upright_like:
            if state.upright_height_ema <= 0.0:
                state.upright_height_ema = effective_height
                state.upright_center_z_ema = effective_center_z
            else:
                alpha = self.baseline_alpha
                state.upright_height_ema = ((1.0 - alpha) * state.upright_height_ema) + (alpha * effective_height)
                state.upright_center_z_ema = ((1.0 - alpha) * state.upright_center_z_ema) + (alpha * effective_center_z)
            state.last_upright_frame = frame_index
            state.stable_frames += 1
        else:
            state.stable_frames = max(0, state.stable_frames - 1)

        baseline_valid = (
            state.upright_height_ema > 0.0
            and (frame_index < 0 or (frame_index - state.last_upright_frame) <= self.baseline_timeout_frames)
        )
        height_drop = baseline_valid and (state.upright_height_ema - effective_height) >= max(0.30, state.upright_height_ema * 0.28)
        center_drop = baseline_valid and (state.upright_center_z_ema - effective_center_z) >= max(0.22, state.upright_height_ema * 0.18)
        has_previous_motion = state.last_effective_height > 0.0
        frame_height_delta = (state.last_effective_height - effective_height) if has_previous_motion else 0.0
        frame_center_delta = (state.last_effective_center_z - effective_center_z) if has_previous_motion else 0.0
        rapid_height_drop = has_previous_motion and frame_height_delta >= max(0.055, (state.upright_height_ema or 1.60) * 0.030)
        rapid_center_drop = has_previous_motion and frame_center_delta >= max(0.045, (state.upright_height_ema or 1.60) * 0.026)
        impact_motion = descending or rapid_height_drop or rapid_center_drop
        descent_event = impact_motion or (height_drop and center_drop and state.impact_motion_frames > 0)

        lost_descent_drop = max(0.0, frame_center_delta, frame_height_delta)
        lost_descent_motion = (
            descending
            or rapid_center_drop
            or rapid_height_drop
            or (has_previous_motion and lost_descent_drop >= self.lost_descent_min_center_drop)
        )
        if not signal_lost and lost_descent_motion:
            state.last_lost_descent_frame = frame_index
            state.last_lost_descent_drop = float(lost_descent_drop)
            state.last_lost_descent_vz = float(vz)
        frames_since_lost_descent = (
            999999 if frame_index < 0 else frame_index - state.last_lost_descent_frame
        )
        recent_lost_descent = 0 <= frames_since_lost_descent <= self.lost_descent_window_frames
        lost_descent_candidate = (
            self.enable_lost_descent_fall
            and signal_lost
            and missing_frames >= self.lost_descent_missing_frames
            and recent_lost_descent
            and (
                baseline_valid
                or state.last_lost_descent_drop >= self.lost_descent_min_center_drop
                or state.last_lost_descent_vz <= self.lost_descent_vz_threshold
            )
        )
        candidate_state = (
            baseline_valid and (descent_event or grounded or low_posture or state.fall_latched)
        ) or lost_descent_candidate
        in_cooldown = frame_index >= 0 and frame_index < state.cooldown_until_frame

        state.descending_frames = state.descending_frames + 1 if descent_event else max(0, state.descending_frames - 1)
        state.impact_motion_frames = state.impact_motion_frames + 1 if impact_motion else max(0, state.impact_motion_frames - 1)
        state.low_posture_frames = state.low_posture_frames + 1 if low_posture else max(0, state.low_posture_frames - 1)
        state.grounded_frames = state.grounded_frames + 1 if grounded else max(0, state.grounded_frames - 1)
        state.candidate_frames = state.candidate_frames + 1 if candidate_state else max(0, state.candidate_frames - 1)
        if impact_motion:
            state.last_descent_frame = frame_index
        recent_descent = frame_index < 0 or (frame_index - state.last_descent_frame) <= (self.fall_confirm_frames + 6)
        state.last_effective_height = float(effective_height)
        state.last_effective_center_z = float(effective_center_z)

        recovered_upright = upright_like and state.stable_frames >= self.recovery_frames
        if state.fall_latched and recovered_upright:
            state.fall_latched = False
            state.lost_descent_latched = False
            state.alert_sent = False
            if frame_index >= 0:
                state.cooldown_until_frame = frame_index + self.fall_cooldown_frames

        if state.fall_latched:
            state.state = "fall_confirmed"
        elif descent_event and baseline_valid:
            state.state = "descending"
        elif grounded and low_posture:
            state.state = "grounded"
        elif low_posture:
            state.state = "low_posture"
        elif recovered_upright or (not low_posture and not grounded and abs(vz) < 0.2):
            state.state = "standing"
        else:
            state.state = "monitoring"

        rule_score = 0.0
        if baseline_valid:
            rule_score += 0.10
        if descent_event:
            rule_score += 0.25
        if impact_motion:
            rule_score += 0.18
        if height_drop:
            rule_score += 0.15
        if center_drop:
            rule_score += 0.10
        if low_posture:
            rule_score += 0.15
        if grounded:
            rule_score += 0.20
        if recent_descent and grounded and low_posture:
            rule_score += 0.10
        if lost_descent_candidate:
            rule_score += 0.45

        should_latch_fall = (
            not state.fall_latched
            and not in_cooldown
            and baseline_valid
            and recent_descent
            and state.grounded_frames >= self.fall_confirm_frames
            and state.low_posture_frames >= self.fall_confirm_frames
        )
        strong_posture_fall = (
            not state.fall_latched
            and not in_cooldown
            and baseline_valid
            and state.grounded_frames >= self.fall_confirm_frames
            and state.low_posture_frames >= self.fall_confirm_frames
            and state.candidate_frames >= self.fall_confirm_frames
            and (height_drop or center_drop or state.impact_motion_frames > 0)
            and rule_score >= 0.70
        )
        lost_descent_fall = (
            not state.fall_latched
            and not in_cooldown
            and lost_descent_candidate
        )
        if should_latch_fall or strong_posture_fall or lost_descent_fall:
            state.fall_latched = True
            state.lost_descent_latched = bool(lost_descent_fall)
            state.state = "fall_confirmed"
            state.last_fall_frame = frame_index
            rule_score = max(rule_score, 1.0)
        elif grounded and low_posture and baseline_valid and recent_descent:
            rule_score = max(rule_score, 0.72)
        elif grounded and low_posture:
            rule_score = max(rule_score, 0.34)
        if state.fall_latched and state.lost_descent_latched:
            rule_score = max(rule_score, 0.92)

        if self.use_rule_candidate_filter:
            should_run_ai = point_count >= self.ai_candidate_min_points
        else:
            should_run_ai = True

        sequence_feature, sequence_center_z = self.sequence_model.extract_frame_feature(
            track,
            state.sequence_previous_center_z,
            self.sequence_frame_delta_seconds,
        )
        if sequence_feature is not None:
            state.sequence_features.append(sequence_feature)
            state.sequence_previous_center_z = sequence_center_z
            if len(state.sequence_features) > self.sequence_history_frames:
                state.sequence_features = state.sequence_features[-self.sequence_history_frames :]
        should_run_sequence = (
            self.sequence_model.enabled
            and len(state.sequence_features) >= self.sequence_model.min_frames
            and (point_count >= self.sequence_candidate_min_points or not self.use_rule_candidate_filter)
        )

        ai_score = state.smoothed_ai_score
        action_name = state.last_action_name
        action_conf = state.last_action_confidence
        sequence_score = state.smoothed_sequence_score

        enough_interval = frame_index < 0 or (frame_index - state.last_ai_frame) >= self.ai_min_interval_frames
        if should_run_ai and enough_interval:
            raw_ai_score, action_name, action_conf = self.ai_model.predict_all(track)
            if state.smoothed_ai_score > 0.0:
                ai_score = ((1.0 - self.ai_smoothing_alpha) * state.smoothed_ai_score) + (self.ai_smoothing_alpha * raw_ai_score)
            else:
                ai_score = float(raw_ai_score)
            state.last_ai_frame = frame_index
            state.last_ai_score = float(raw_ai_score)
            state.smoothed_ai_score = float(ai_score)
            state.last_action_name = action_name
            state.last_action_confidence = float(action_conf)
        elif self.use_rule_candidate_filter and not candidate_state and state.smoothed_ai_score > 0.0:
            state.smoothed_ai_score *= 0.9
            ai_score = state.smoothed_ai_score

        enough_sequence_interval = (
            frame_index < 0 or (frame_index - state.last_sequence_frame) >= self.sequence_min_interval_frames
        )
        if should_run_sequence and enough_sequence_interval:
            raw_sequence_score = self.sequence_model.predict_from_history(
                state.sequence_features,
                self.sequence_frame_delta_seconds,
            )
            if state.smoothed_sequence_score > 0.0:
                sequence_score = (
                    ((1.0 - self.sequence_smoothing_alpha) * state.smoothed_sequence_score)
                    + (self.sequence_smoothing_alpha * raw_sequence_score)
                )
            else:
                sequence_score = float(raw_sequence_score)
            state.last_sequence_frame = frame_index
            state.last_sequence_score = float(raw_sequence_score)
            state.smoothed_sequence_score = float(sequence_score)
        elif self.use_rule_candidate_filter and not candidate_state and state.smoothed_sequence_score > 0.0:
            state.smoothed_sequence_score *= 0.95
            sequence_score = state.smoothed_sequence_score

        if simulate_ai and (candidate_state or not self.use_rule_candidate_filter) and rule_score > 0.5:
            ai_score = 1.0
            state.smoothed_ai_score = 1.0

        model_assisted_fall = (
            self.use_rule_candidate_filter
            and not state.fall_latched
            and not in_cooldown
            and grounded
            and low_posture
            and state.grounded_frames >= self.fall_confirm_frames
            and state.low_posture_frames >= self.fall_confirm_frames
            and (baseline_valid or state.impact_motion_frames > 0 or self.ai_model.enabled)
            and (
                (self.ai_model.enabled and ai_score >= self.ai_assisted_fall_threshold)
                or (self.sequence_model.enabled and sequence_score >= self.sequence_assisted_fall_threshold)
            )
        )
        if model_assisted_fall:
            state.fall_latched = True
            state.lost_descent_latched = False
            state.state = "fall_confirmed"
            state.last_fall_frame = frame_index
            rule_score = max(rule_score, 1.0)

        if self.use_rule_candidate_filter:
            total_score = rule_score
            if self.ai_model.enabled or self.sequence_model.enabled:
                ai_weight = self.ai_weight if self.ai_model.enabled else 0.0
                sequence_weight = self.sequence_weight if self.sequence_model.enabled else 0.0
                rule_weight = max(0.0, 1.0 - ai_weight - sequence_weight)
                total_score = (rule_score * rule_weight) + (ai_score * ai_weight) + (sequence_score * sequence_weight)
        else:
            model_scores = []
            if self.ai_model.enabled:
                model_scores.append(float(ai_score))
            if self.sequence_model.enabled:
                model_scores.append(float(sequence_score))
            total_score = max(model_scores) if model_scores else 0.0

        action_label = str(action_name or "").upper()
        model_label_fall = "FALL" in action_label or "FALLING" in action_label

        if self.use_rule_candidate_filter:
            if model_label_fall and not state.fall_latched and not in_cooldown:
                state.fall_latched = True
                state.lost_descent_latched = False
                state.state = "fall_confirmed"
                state.last_fall_frame = frame_index
                rule_score = max(rule_score, 0.82)
                total_score = max(total_score, ai_score, 0.82)
            is_fall = state.fall_latched and total_score >= 0.60
        else:
            is_fall = (
                model_label_fall
                or (self.ai_model.enabled and ai_score >= self.ai_falling_threshold)
                or (self.sequence_model.enabled and sequence_score >= self.sequence_model.threshold)
            )
            if not is_fall:
                state.alert_sent = False
        state.last_total_score = total_score
        should_alert = is_fall and not state.alert_sent
        if should_alert:
            state.alert_sent = True

        if is_fall:
            display_label = "FALL"
        elif not self.use_rule_candidate_filter:
            display_label = action_name if action_name != "Unknown" else "Normal"
        elif state.state == "grounded":
            display_label = "GROUND"
        elif state.state == "descending":
            display_label = "DESCENDING"
        elif low_posture:
            display_label = "LOW POSTURE"
        else:
            display_label = "Normal"

        details = {
            "rule_score": float(rule_score),
            "ai_score": float(ai_score),
            "sequence_score": float(sequence_score),
            "total_score": float(total_score),
            "vz": vz,
            "z": effective_center_z,
            "ground_height": float(ground_height),
            "dims": track.dims,
            "aspect_ratio": float(aspect_ratio),
            "point_count": point_count,
            "missing_frames": missing_frames,
            "height_reliable": bool(height_reliable),
            "point_vertical_extent": float(point_vertical_extent),
            "baseline_valid": baseline_valid,
            "baseline_height": float(state.upright_height_ema),
            "baseline_center_z": float(state.upright_center_z_ema),
            "height_drop": bool(height_drop),
            "center_drop": bool(center_drop),
            "rapid_height_drop": bool(rapid_height_drop),
            "rapid_center_drop": bool(rapid_center_drop),
            "impact_motion": bool(impact_motion),
            "frame_height_delta": float(frame_height_delta),
            "frame_center_delta": float(frame_center_delta),
            "lost_descent_enabled": bool(self.enable_lost_descent_fall),
            "lost_descent_candidate": bool(lost_descent_candidate),
            "lost_descent_latched": bool(state.lost_descent_latched),
            "lost_descent_recent": bool(recent_lost_descent),
            "lost_descent_frames_since": int(frames_since_lost_descent),
            "lost_descent_last_drop": float(state.last_lost_descent_drop),
            "lost_descent_last_vz": float(state.last_lost_descent_vz),
            "fsm_state": state.state,
            "display_label": display_label,
            "action_name": action_name,
            "action_confidence": action_conf,
            "descending_frames": state.descending_frames,
            "grounded_frames": state.grounded_frames,
            "low_posture_frames": state.low_posture_frames,
            "candidate_frames": state.candidate_frames,
            "impact_motion_frames": state.impact_motion_frames,
            "fall_latched": state.fall_latched,
            "strong_posture_fall": bool(strong_posture_fall),
            "model_assisted_fall": bool(model_assisted_fall),
            "model_label_fall": bool(model_label_fall),
            "candidate_state": bool(candidate_state),
            "should_run_ai": bool(should_run_ai),
            "should_run_sequence": bool(should_run_sequence),
            "sequence_frames": int(len(state.sequence_features)),
            "sequence_model_enabled": bool(self.sequence_model.enabled),
            "use_rule_candidate_filter": bool(self.use_rule_candidate_filter),
            "ai_only_mode": bool(not self.use_rule_candidate_filter),
            "should_alert": should_alert,
        }
        return is_fall, total_score, details
