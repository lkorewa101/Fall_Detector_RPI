from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


JOINT_NAMES = [
    "head",
    "neck",
    "pelvis",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand",
    "right_hand",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
]

BONE_PAIRS = [
    ("head", "neck"),
    ("neck", "pelvis"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
]

TARGET_SCALE = np.tile(np.array([5.5, 5.5, 2.2], dtype=np.float32), len(JOINT_NAMES)).astype(np.float32)


def _normalize_points(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.empty((0, 4), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if arr.shape[1] == 3:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)])
    elif arr.shape[1] > 4:
        arr = arr[:, :4]
    return np.ascontiguousarray(arr, dtype=np.float32)


def build_pose_features(points: np.ndarray | None) -> np.ndarray:
    pts = _normalize_points(points)
    if len(pts) == 0:
        return np.zeros(38, dtype=np.float32)

    xyz = pts[:, :3]
    doppler = pts[:, 3]
    scale = np.array([5.5, 5.5, 2.2], dtype=np.float32)
    stats = [
        np.array([min(len(pts), 150) / 150.0], dtype=np.float32),
        xyz.mean(axis=0) / scale,
        xyz.std(axis=0) / np.array([2.0, 2.0, 1.2], dtype=np.float32),
        xyz.min(axis=0) / scale,
        xyz.max(axis=0) / scale,
        np.percentile(xyz[:, 2], [5, 25, 50, 75, 95]).astype(np.float32) / 2.2,
        np.array(
            [
                float(np.mean(doppler)),
                float(np.std(doppler)),
                float(np.percentile(doppler, 10)),
                float(np.percentile(doppler, 90)),
            ],
            dtype=np.float32,
        )
        / 4.0,
    ]

    x = np.clip((xyz[:, 0] + 2.0) / 4.0, 0.0, 0.999)
    z = np.clip(xyz[:, 2] / 2.2, 0.0, 0.999)
    xi = np.floor(x * 4).astype(np.int32)
    zi = np.floor(z * 4).astype(np.int32)
    occ = np.zeros((4, 4), dtype=np.float32)
    for a, b in zip(xi, zi):
        occ[int(a), int(b)] += 1.0
    occ = (occ.reshape(-1) / max(1, len(pts))).astype(np.float32)
    return np.concatenate([*stats, occ]).astype(np.float32)


def skeleton_lines_from_joints(joints: dict[str, np.ndarray]) -> np.ndarray:
    lines = []
    for start, end in BONE_PAIRS:
        lines.append(np.asarray(joints[start], dtype=np.float32))
        lines.append(np.asarray(joints[end], dtype=np.float32))
    return np.vstack(lines).astype(np.float32)


def fit_skeleton_lines_to_points(lines: np.ndarray, points: np.ndarray | None) -> np.ndarray:
    lines = np.asarray(lines, dtype=np.float32)
    pts = _normalize_points(points)
    if lines.ndim != 2 or lines.shape[1] != 3 or len(lines) == 0 or len(pts) == 0:
        return lines.astype(np.float32)

    xyz = pts[:, :3]
    finite_points = np.isfinite(xyz).all(axis=1)
    finite_lines = np.isfinite(lines).all(axis=1)
    if not np.any(finite_points) or not np.all(finite_lines):
        return lines.astype(np.float32)

    xyz = xyz[finite_points]
    point_low = np.percentile(xyz, 5, axis=0).astype(np.float32)
    point_high = np.percentile(xyz, 95, axis=0).astype(np.float32)
    point_center = ((point_low + point_high) * 0.5).astype(np.float32)
    allowed_span = np.maximum(point_high - point_low + np.array([0.25, 0.25, 0.20], dtype=np.float32), 0.25)

    line_low = lines.min(axis=0)
    line_high = lines.max(axis=0)
    line_center = ((line_low + line_high) * 0.5).astype(np.float32)
    line_span = np.maximum(line_high - line_low, 1e-3)
    axis_scale = allowed_span / line_span
    scale = float(np.clip(np.min(axis_scale), 0.005, 1.0))
    fitted = (lines - line_center.reshape(1, 3)) * scale + point_center.reshape(1, 3)
    return fitted.astype(np.float32)


def _decode_names(raw_names) -> list[str]:
    names = []
    for value in list(raw_names):
        if isinstance(value, bytes):
            names.append(value.decode("utf-8", errors="ignore"))
        else:
            names.append(str(value))
    return names


def load_sidecar_skeletons(data_path: str | Path) -> dict[int, np.ndarray]:
    path = Path(data_path)
    sidecar = path.parent / "joints.npz"
    if not sidecar.exists():
        return {}
    data = np.load(sidecar, allow_pickle=True)
    try:
        joints = np.asarray(data["joints"], dtype=np.float32)
        frame_numbers = np.asarray(data["frame_numbers"], dtype=np.int32) if "frame_numbers" in data.files else np.arange(len(joints), dtype=np.int32)
        joint_names = _decode_names(data["joint_names"]) if "joint_names" in data.files else JOINT_NAMES
    finally:
        data.close()

    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    out: dict[int, np.ndarray] = {}
    for row, frame_number in enumerate(frame_numbers):
        frame_joints = {}
        for name in JOINT_NAMES:
            idx = name_to_idx.get(name)
            if idx is None or idx >= joints.shape[1]:
                break
            frame_joints[name] = joints[row, idx, :]
        if len(frame_joints) == len(JOINT_NAMES):
            out[int(frame_number)] = skeleton_lines_from_joints(frame_joints)
    return out


if TORCH_AVAILABLE and nn is not None:

    class PoseMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Linear(96, 128),
                nn.ReLU(),
                nn.Linear(128, 96),
                nn.ReLU(),
                nn.Linear(96, output_dim),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

else:
    PoseMLP = None  # type: ignore[assignment]


class PoseEstimator:
    def __init__(self, model_path: str | Path | None = None, log_callback=None) -> None:
        self.log_callback = log_callback
        self.model = None
        self.backend = ""
        self.available = False
        self.model_path: Optional[Path] = None
        self.target_scale = TARGET_SCALE.copy()
        path = Path(model_path) if model_path else self._discover_latest_model()
        if path is not None:
            self.load(path)

    def _log(self, message: str) -> None:
        if self.log_callback:
            try:
                self.log_callback(message)
                return
            except Exception:
                pass
        print(message)

    def _discover_latest_model(self) -> Optional[Path]:
        module_dir = Path(__file__).resolve().parent
        app_root = module_dir.parent
        preferred = [app_root / "models" / "latest" / "pose_model.pkl"]
        if TORCH_AVAILABLE:
            preferred.extend(
                [
                    app_root / "models" / "latest" / "pose_model.pt",
                    module_dir / "pose_model_synthetic9000.pt",
                ]
            )
        for path in preferred:
            if path.exists() and path.is_file():
                return path
        repo_root = Path(__file__).resolve().parents[2]
        collector_root = repo_root / "virtual-iwr6843-data-collector" / "output" / "sessions"
        if not collector_root.exists():
            return None
        candidates = [path for path in collector_root.rglob("pose_model.pt") if path.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def load(self, model_path: str | Path) -> bool:
        path = Path(model_path)
        if path.suffix.lower() == ".pkl":
            return self._load_sklearn_pose(path)
        if not TORCH_AVAILABLE or torch is None or PoseMLP is None:
            self._log(f"[pose] torch unavailable for model: {path}")
            return False
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            input_dim = int(checkpoint.get("input_dim", 38))
            output_dim = int(checkpoint.get("output_dim", len(JOINT_NAMES) * 3))
            model = PoseMLP(input_dim, output_dim)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.model = model
            self.backend = "torch"
            self.target_scale = np.asarray(checkpoint.get("target_scale", TARGET_SCALE), dtype=np.float32).reshape(-1)
            self.available = True
            self.model_path = path
            self._log(f"[pose] loaded model: {path}")
            return True
        except Exception as exc:
            self._log(f"[pose] model load failed: {exc}")
            self.model = None
            self.available = False
            return False

    def _load_sklearn_pose(self, path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            if not hasattr(model, "predict"):
                raise ValueError("loaded pose model does not expose predict")
            self.model = model
            self.backend = "sklearn"
            if isinstance(payload, dict):
                self.target_scale = np.asarray(payload.get("target_scale", TARGET_SCALE), dtype=np.float32).reshape(-1)
            else:
                self.target_scale = TARGET_SCALE.copy()
            self.available = True
            self.model_path = path
            self._log(f"[pose] loaded sklearn model: {path}")
            return True
        except Exception as exc:
            self._log(f"[pose] sklearn model load failed: {exc}")
            self.model = None
            self.backend = ""
            self.available = False
            return False

    def predict_skeleton(self, points: np.ndarray | None) -> Optional[np.ndarray]:
        if not self.available or self.model is None:
            return None
        pts = _normalize_points(points)
        if len(pts) < 8:
            return None
        feature = build_pose_features(pts).reshape(1, -1)
        try:
            if self.backend == "sklearn":
                pred = np.asarray(self.model.predict(feature), dtype=np.float32).reshape(-1)
            else:
                if torch is None:
                    return None
                with torch.no_grad():
                    pred = self.model(torch.tensor(feature, dtype=torch.float32)).cpu().numpy().reshape(-1)
            pred = (pred * self.target_scale).reshape(len(JOINT_NAMES), 3)
            joints = {name: pred[index].astype(np.float32) for index, name in enumerate(JOINT_NAMES)}
            return fit_skeleton_lines_to_points(skeleton_lines_from_joints(joints), pts)
        except Exception:
            return None
