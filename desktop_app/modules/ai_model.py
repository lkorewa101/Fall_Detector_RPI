from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


_FALL_AI_CACHE: dict[str, dict[str, object]] = {}
_FALL_AI_FAILED_PATHS: set[str] = set()
_FALL_SEQUENCE_AI_CACHE: dict[str, dict[str, object]] = {}


def _safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"))


# Fall-related action names (fallback if model doesn't contain them)
DEFAULT_FALL_ACTIONS = {
    'Death From Right', 'Fall Flat', 'Fall Over',
    'Slipping', 'Stumble Backwards', 'Tripping',
    'fall_forward', 'fall_side', 'slip', 'trip',
}

SEQUENCE_FRAME_FEATURE_NAMES = (
    "center_x",
    "center_y",
    "center_z",
    "range_x",
    "range_y",
    "range_z",
    "std_x",
    "std_y",
    "std_z",
    "min_z",
    "max_z",
    "mean_doppler",
    "std_doppler",
    "point_count",
    "vz",
    "aspect_xy_z",
)

SEQUENCE_AGGREGATIONS = ("mean", "std", "min", "max", "first", "last", "delta")

# Action -> Category mapping
ACTION_TO_CATEGORY = {
    # Sitting
    'Checking Dice':        'Sitting',
    'Shoved':               'Sitting',
    # Sitting → Standing
    'Situp To Idle':        'Sit → Stand',
    'Snatch':               'Sit → Stand',
    # Falling
    'Death From Right':     'Falling',
    'Fall Flat':            'Falling',
    'Fall Over':            'Falling',
    'Slipping':             'Falling',
    'Stumble Backwards':    'Falling',
    'Tripping':             'Falling',
    # Lying
    'Laying':               'Lying',
    # Lying → Standing
    'Standing Up':          'Lie → Stand',
    'Crawling':             'Lie → Stand',
    # Standing → Lying
    'Lying Down':           'Stand → Lie',
    # Standing
    'Left Turn':            'Standing',
    'Look Over Shoulder':   'Standing',
    'Old Man Idle':         'Standing',
    'Right Turn':           'Standing',
    # Walking
    'Drunk Walk':           'Walking',
    'Running':              'Walking',
    'Standard Walk':        'Walking',
    'Wheelbarrow Walk':     'Walking',
    # Synthetic IWR6843ISK collector labels
    'standing':             'Standing',
    'walking':              'Walking',
    'sitting':              'Sitting',
    'lying':                'Lying',
    'stand_up':             'Lie → Stand',
    'fall_forward':         'Falling',
    'fall_side':            'Falling',
    'slip':                 'Falling',
    'trip':                 'Falling',
}


class FallAI:
    def __init__(self) -> None:
        self.model = None
        self.label_encoder = None
        self.fall_actions = DEFAULT_FALL_ACTIONS
        self.enabled = False
        self.is_multiclass = False
        self.model_candidates = self._candidate_model_paths()
        self.model_path = self.model_candidates[0] if self.model_candidates else None
        self.load_model()

    def _is_readable_model(self, candidate: Path) -> bool:
        try:
            return candidate.exists() and candidate.is_file() and candidate.suffix == ".pkl" and os.access(candidate, os.R_OK)
        except OSError as exc:
            _safe_print(f"[ai_model] Skipping unreadable model path: {candidate} ({exc})")
            return False

    def _candidate_model_paths(self) -> list[Path]:
        module_dir = Path(__file__).resolve().parent
        env_path = os.getenv("FALL_MODEL_PATH")
        if env_path:
            candidate = Path(env_path).expanduser().resolve()
            if self._is_readable_model(candidate):
                return [candidate]
            return []

        app_root = module_dir.parent
        candidates = [
            app_root / "models" / "latest" / "fall_model_fallsim2_unity.pkl",
            app_root / "models" / "latest" / "fall_model.pkl",
            module_dir / "fall_model_synthetic9000.pkl",
            module_dir / "fall_model.pkl",
        ]
        return [candidate for candidate in candidates if self._is_readable_model(candidate)]

    def _resolve_model_path(self) -> Optional[Path]:
        candidates = self._candidate_model_paths()
        return candidates[0] if candidates else None

    def set_model_path(self, model_path: str) -> bool:
        candidate = Path(model_path).expanduser().resolve()
        if not candidate.exists() or candidate.suffix.lower() != ".pkl":
            _safe_print(f"[ai_model] Invalid model path: {model_path}")
            return False

        if str(candidate) in _FALL_AI_FAILED_PATHS:
            return False

        if self.model_path is not None and candidate == self.model_path:
            return self.enabled

        previous_state = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "fall_actions": self.fall_actions,
            "enabled": self.enabled,
            "is_multiclass": self.is_multiclass,
            "model_candidates": list(self.model_candidates or []),
            "model_path": self.model_path,
        }
        self.model_candidates = [candidate]
        self.model_path = candidate
        self.load_model()
        if not self.enabled:
            self.model = previous_state["model"]
            self.label_encoder = previous_state["label_encoder"]
            self.fall_actions = previous_state["fall_actions"]
            self.enabled = bool(previous_state["enabled"])
            self.is_multiclass = bool(previous_state["is_multiclass"])
            self.model_candidates = previous_state["model_candidates"]
            self.model_path = previous_state["model_path"]
            _safe_print(f"[ai_model] Keeping previous model after failed selection: {self.model_path}")
        return self.enabled

    def load_model(self) -> None:
        candidates = list(self.model_candidates or [])
        if self.model_path is not None and self.model_path not in candidates:
            candidates.insert(0, self.model_path)

        if not candidates:
            _safe_print("[ai_model] No trusted model path configured. AI scoring disabled.")
            self.model = None
            self.enabled = False
            return

        errors: list[str] = []
        for candidate in candidates:
            if str(candidate) in _FALL_AI_FAILED_PATHS:
                continue
            try:
                cache_key = str(candidate)
                cached = _FALL_AI_CACHE.get(cache_key)
                if cached is not None:
                    self.model = cached["model"]
                    self.label_encoder = cached["label_encoder"]
                    self.fall_actions = cached["fall_actions"]
                    self.is_multiclass = bool(cached["is_multiclass"])
                    self.enabled = bool(cached["enabled"])
                    self.model_path = candidate
                    return

                with candidate.open("rb") as handle:
                    data = pickle.load(handle)

                if isinstance(data, dict) and 'model' in data:
                    # Multi-class model (new format)
                    self.model = data['model']
                    self.label_encoder = data.get('label_encoder', None)
                    self.fall_actions = data.get('fall_actions', DEFAULT_FALL_ACTIONS)
                    self.is_multiclass = self.label_encoder is not None
                    self.enabled = hasattr(self.model, 'predict_proba')
                    self._validate_loaded_model()
                    self.model_path = candidate
                    _FALL_AI_CACHE[cache_key] = {
                        "model": self.model,
                        "label_encoder": self.label_encoder,
                        "fall_actions": self.fall_actions,
                        "is_multiclass": self.is_multiclass,
                        "enabled": self.enabled,
                    }
                    _safe_print(f"[ai_model] Multi-class model loaded from {self.model_path}")
                    if self.label_encoder is not None:
                        _safe_print(f"[ai_model] Classes: {list(self.label_encoder.classes_)}")
                    return
                else:
                    # Legacy binary model
                    self.model = data
                    self.label_encoder = None
                    self.is_multiclass = False
                    self.enabled = hasattr(self.model, "predict_proba")
                    self._validate_loaded_model()
                    self.model_path = candidate
                    _FALL_AI_CACHE[cache_key] = {
                        "model": self.model,
                        "label_encoder": self.label_encoder,
                        "fall_actions": self.fall_actions,
                        "is_multiclass": self.is_multiclass,
                        "enabled": self.enabled,
                    }
                    _safe_print(f"[ai_model] Binary model loaded from {self.model_path}")
                    return

            except Exception as exc:
                _FALL_AI_FAILED_PATHS.add(str(candidate))
                errors.append(f"{candidate}: {exc}")
                _safe_print(f"[ai_model] Failed to load model candidate {candidate}: {exc}")

        _safe_print("[ai_model] No model candidate loaded. AI scoring disabled.")
        for error in errors[:3]:
            _safe_print(f"[ai_model] candidate error: {error}")
        self.model = None
        self.label_encoder = None
        self.enabled = False

    def _validate_loaded_model(self) -> None:
        if not self.enabled or self.model is None or not hasattr(self.model, "predict_proba"):
            raise ValueError("loaded model does not expose predict_proba")
        smoke_features = np.zeros((1, 11), dtype=np.float32)
        probas = self.model.predict_proba(smoke_features)
        if probas is None or len(probas) == 0:
            raise ValueError("loaded model predict_proba returned no probabilities")

    def _extract_features(self, track) -> Optional[np.ndarray]:
        """Extract feature vector from a track object."""
        points = getattr(track, "points", None)
        if points is None or len(points) == 0:
            return None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            return None

        if points.shape[1] >= 4:
            dopplers = points[:, 3]
        else:
            dopplers = np.zeros(len(points), dtype=np.float32)

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        
        std_x = float(np.std(xs)) if len(xs) > 1 else 0.0
        std_y = float(np.std(ys)) if len(ys) > 1 else 0.0
        std_z = float(np.std(zs)) if len(zs) > 1 else 0.0
        range_x = float(np.max(xs) - np.min(xs))
        range_y = float(np.max(ys) - np.min(ys))
        range_z = float(np.max(zs) - np.min(zs))
        
        features = np.array([[
            float(np.mean(zs)),
            float(np.min(zs)),
            std_z,
            std_x,
            std_y,
            range_x,
            range_y,
            range_z,
            float(np.mean(dopplers)),
            float(np.min(dopplers)),
            float(len(points))
        ]], dtype=np.float32)
        
        return features

    def predict_all(self, track) -> Tuple[float, str, float]:
        """Perform a single inference to get both fall probability and action info.
        Returns: (fall_score, action_name, action_confidence)
        """
        if not self.enabled or self.model is None:
            return 0.0, "Unknown", 0.0

        features = self._extract_features(track)
        if features is None:
            return 0.0, "Unknown", 0.0

        try:
            probas = self.model.predict_proba(features)[0]
            classes = self.label_encoder.classes_ if self.is_multiclass else [0, 1]

            if self.is_multiclass:
                # 1. Fall Score (sum of related classes)
                fall_score = sum(probas[i] for i, c in enumerate(classes) if c in self.fall_actions)
                
                # 2. Action Name & Confidence
                cat_scores = {}
                for i, cls_name in enumerate(classes):
                    cat = ACTION_TO_CATEGORY.get(cls_name, cls_name)
                    cat_scores[cat] = cat_scores.get(cat, 0.0) + probas[i]
                best_cat = max(cat_scores, key=cat_scores.get)
                
                return float(fall_score), best_cat, float(cat_scores[best_cat])
            else:
                # Binary fallback
                fall_prob = float(probas[1])
                action = "Falling" if fall_prob >= 0.5 else "Standing"
                return fall_prob, action, max(fall_prob, 1.0 - fall_prob)

        except Exception as exc:
            print(f"[ai_model] Integrated inference error: {exc}")
            return 0.0, "Unknown", 0.0


def sequence_feature_names() -> list[str]:
    names: list[str] = []
    for name in SEQUENCE_FRAME_FEATURE_NAMES:
        for agg in SEQUENCE_AGGREGATIONS:
            names.append(f"{name}_{agg}")
    names.extend(["frames_with_points", "duration_seconds"])
    return names


class FallSequenceAI:
    def __init__(self) -> None:
        self.model = None
        self.enabled = False
        self.feature_names = sequence_feature_names()
        self.threshold = 0.5
        self.min_frames = 8
        self.model_candidates = self._candidate_model_paths()
        self.model_path = self.model_candidates[0] if self.model_candidates else None
        self.load_model()

    def _candidate_model_paths(self) -> list[Path]:
        module_dir = Path(__file__).resolve().parent
        env_path = os.getenv("FALL_SEQUENCE_MODEL_PATH")
        if env_path:
            candidate = Path(env_path).expanduser().resolve()
            if candidate.exists() and candidate.suffix == ".pkl":
                return [candidate]
            return []

        app_root = module_dir.parent
        candidates = [
            app_root / "models" / "latest" / "fall_sequence_model_d_fall.pkl",
        ]
        return [candidate for candidate in candidates if candidate.exists() and candidate.suffix == ".pkl"]

    def load_model(self) -> None:
        if not self.model_candidates:
            self.enabled = False
            self.model = None
            return

        errors: list[str] = []
        for candidate in self.model_candidates:
            try:
                cache_key = str(candidate)
                cached = _FALL_SEQUENCE_AI_CACHE.get(cache_key)
                if cached is not None:
                    self.model = cached["model"]
                    self.feature_names = list(cached["feature_names"])
                    self.threshold = float(cached["threshold"])
                    self.model_path = candidate
                    self.enabled = bool(cached["enabled"])
                    return

                with candidate.open("rb") as handle:
                    data = pickle.load(handle)
                if not isinstance(data, dict) or "model" not in data:
                    raise ValueError("sequence artifact must be a dict with a model")
                model = data["model"]
                if not hasattr(model, "predict_proba"):
                    raise ValueError("sequence model does not expose predict_proba")
                feature_names = data.get("feature_names") or self.feature_names
                if len(feature_names) != len(self.feature_names):
                    raise ValueError(f"sequence feature count mismatch: {len(feature_names)} != {len(self.feature_names)}")

                smoke = np.zeros((1, len(self.feature_names)), dtype=np.float32)
                probas = model.predict_proba(smoke)
                if probas is None or len(probas) == 0:
                    raise ValueError("sequence model predict_proba returned no probabilities")

                self.model = model
                self.feature_names = list(feature_names)
                self.threshold = float(data.get("threshold", 0.5))
                self.model_path = candidate
                self.enabled = True
                _FALL_SEQUENCE_AI_CACHE[cache_key] = {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "threshold": self.threshold,
                    "enabled": self.enabled,
                }
                _safe_print(f"[ai_model] Sequence model loaded from {self.model_path}")
                return
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                _safe_print(f"[ai_model] Failed to load sequence model candidate {candidate}: {exc}")

        self.model = None
        self.enabled = False
        for error in errors[:3]:
            _safe_print(f"[ai_model] sequence candidate error: {error}")

    @staticmethod
    def extract_frame_feature(track, previous_center_z: float | None, frame_delta_seconds: float) -> tuple[np.ndarray | None, float | None]:
        points = getattr(track, "points", None)
        if points is None or len(points) < 3:
            points = getattr(track, "display_points", None)
        if points is None or len(points) < 3:
            return None, previous_center_z

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            return None, previous_center_z

        finite = np.isfinite(points[:, :3]).all(axis=1)
        points = points[finite]
        if len(points) < 3:
            return None, previous_center_z

        xyz = points[:, :3]
        doppler = points[:, 3] if points.shape[1] >= 4 else np.zeros(len(points), dtype=np.float32)
        center = np.mean(xyz, axis=0)
        low = np.percentile(xyz, 5, axis=0)
        high = np.percentile(xyz, 95, axis=0)
        spans = np.maximum(high - low, np.zeros(3, dtype=np.float32))
        stds = np.std(xyz, axis=0) if len(xyz) > 1 else np.zeros(3, dtype=np.float32)
        vz = 0.0 if previous_center_z is None else (float(center[2]) - float(previous_center_z)) / max(frame_delta_seconds, 1e-3)
        aspect = max(float(spans[0]), float(spans[1])) / max(float(spans[2]), 0.05)
        feature = np.array(
            [
                float(center[0]),
                float(center[1]),
                float(center[2]),
                float(spans[0]),
                float(spans[1]),
                float(spans[2]),
                float(stds[0]),
                float(stds[1]),
                float(stds[2]),
                float(np.min(xyz[:, 2])),
                float(np.max(xyz[:, 2])),
                float(np.mean(doppler)),
                float(np.std(doppler)) if len(doppler) > 1 else 0.0,
                float(len(points)),
                float(vz),
                float(aspect),
            ],
            dtype=np.float32,
        )
        return feature, float(center[2])

    def aggregate_features(self, features: np.ndarray, frame_delta_seconds: float) -> np.ndarray | None:
        if features.ndim != 2 or features.shape[0] < self.min_frames:
            return None
        parts = [
            np.mean(features, axis=0),
            np.std(features, axis=0),
            np.min(features, axis=0),
            np.max(features, axis=0),
            features[0],
            features[-1],
            features[-1] - features[0],
        ]
        flat = np.concatenate(parts).astype(np.float32)
        extras = np.array(
            [float(features.shape[0]), float(features.shape[0]) * float(frame_delta_seconds)],
            dtype=np.float32,
        )
        return np.concatenate([flat, extras]).reshape(1, -1).astype(np.float32)

    def predict_from_history(self, frame_features: list[np.ndarray], frame_delta_seconds: float) -> float:
        if not self.enabled or self.model is None:
            return 0.0
        if len(frame_features) < self.min_frames:
            return 0.0
        features = self.aggregate_features(np.vstack(frame_features).astype(np.float32), frame_delta_seconds)
        if features is None:
            return 0.0
        try:
            probas = self.model.predict_proba(features)[0]
            return float(probas[1] if len(probas) > 1 else probas[0])
        except Exception as exc:
            print(f"[ai_model] Sequence inference error: {exc}")
            return 0.0
