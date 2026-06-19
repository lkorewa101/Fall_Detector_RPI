from __future__ import annotations

import argparse
import json
import pickle
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from modules.csv_stream import CsvStream  # noqa: E402
from modules.pose_estimator import JOINT_NAMES, TARGET_SCALE, build_pose_features  # noqa: E402
from tools.validate_mmwave_dataset import FRAME_COLUMNS, CsvSource, _iter_sources, _normalize_columns, _read_source  # noqa: E402


def _read_dataframe(source: CsvSource, max_rows: int) -> pd.DataFrame:
    if source.zip_member:
        with zipfile.ZipFile(source.path) as archive:
            with archive.open(source.zip_member) as handle:
                return pd.read_csv(handle, nrows=max_rows)
    return _read_source(source, max_rows=max_rows)


def _unit_xy_axis(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-6:
        axis = np.asarray(fallback, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(axis))
    return (axis / max(norm, 1e-6)).astype(np.float32)


def _principal_xy_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(xy) < 3:
        major = np.array([1.0, 0.0], dtype=np.float32)
    else:
        centered = xy - np.mean(xy, axis=0, keepdims=True)
        cov = np.cov(centered.T)
        if np.asarray(cov).shape != (2, 2) or not np.isfinite(cov).all():
            major = np.array([1.0, 0.0], dtype=np.float32)
        else:
            values, vectors = np.linalg.eigh(cov)
            major = vectors[:, int(np.argmax(values))].astype(np.float32)
    major = _unit_xy_axis(major, np.array([1.0, 0.0], dtype=np.float32))
    minor = np.array([-major[1], major[0]], dtype=np.float32)
    return major, minor


def _vec2(axis: np.ndarray, distance: float, z: float = 0.0) -> np.ndarray:
    return np.array([float(axis[0]) * distance, float(axis[1]) * distance, z], dtype=np.float32)


def _with_z(point: np.ndarray, z: float) -> np.ndarray:
    out = np.asarray(point, dtype=np.float32).copy()
    out[2] = float(z)
    return out


def pseudo_joints_from_points(points: np.ndarray) -> dict[str, np.ndarray] | None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 3:
        return None
    xyz = pts[:, :3]
    if not np.isfinite(xyz).all():
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 3:
        return None

    low = np.percentile(xyz, 5, axis=0).astype(np.float32)
    high = np.percentile(xyz, 95, axis=0).astype(np.float32)
    center = np.mean(xyz, axis=0).astype(np.float32)
    span = np.maximum(high - low, np.array([0.05, 0.05, 0.05], dtype=np.float32))
    major_xy, minor_xy = _principal_xy_axes(xyz[:, :2])
    horizontal = float(max(span[0], span[1]))
    vertical = float(span[2])
    lying = horizontal > max(0.75, vertical * 1.35) and vertical < 0.9

    if lying:
        body_axis = major_xy
        lateral_axis = minor_xy
        length = float(np.clip(horizontal, 0.85, 1.9))
        shoulder_width = float(np.clip(horizontal * 0.28, 0.28, 0.55))
        hip_width = float(np.clip(shoulder_width * 0.72, 0.22, 0.42))
        z_mid = float(np.clip(np.percentile(xyz[:, 2], 55), low[2], high[2]))
        z_high = float(np.clip(z_mid + max(0.08, vertical * 0.20), low[2], high[2] + 0.08))

        head = _with_z(center + _vec2(body_axis, length * 0.47), z_high)
        neck = _with_z(center + _vec2(body_axis, length * 0.30), z_mid)
        pelvis = _with_z(center + _vec2(body_axis, -length * 0.05), z_mid)
        left_shoulder = neck + _vec2(lateral_axis, shoulder_width * 0.50)
        right_shoulder = neck + _vec2(lateral_axis, -shoulder_width * 0.50)
        left_elbow = left_shoulder + _vec2(lateral_axis, shoulder_width * 0.35) + _vec2(body_axis, -length * 0.10, -0.03)
        right_elbow = right_shoulder + _vec2(lateral_axis, -shoulder_width * 0.35) + _vec2(body_axis, -length * 0.10, -0.03)
        left_hand = left_shoulder + _vec2(lateral_axis, shoulder_width * 0.55) + _vec2(body_axis, -length * 0.22, -0.05)
        right_hand = right_shoulder + _vec2(lateral_axis, -shoulder_width * 0.55) + _vec2(body_axis, -length * 0.22, -0.05)
        left_hip = pelvis + _vec2(lateral_axis, hip_width * 0.50)
        right_hip = pelvis + _vec2(lateral_axis, -hip_width * 0.50)
        left_knee = left_hip + _vec2(body_axis, -length * 0.25, -0.02)
        right_knee = right_hip + _vec2(body_axis, -length * 0.25, -0.02)
        left_foot = left_hip + _vec2(body_axis, -length * 0.48, -0.04)
        right_foot = right_hip + _vec2(body_axis, -length * 0.48, -0.04)
    else:
        lateral_axis = major_xy
        shoulder_width = float(np.clip(horizontal * 0.60, 0.28, 0.62))
        hip_width = float(np.clip(shoulder_width * 0.68, 0.22, 0.45))
        foot_width = float(np.clip(hip_width * 0.95, 0.20, 0.45))
        z_min = float(low[2])
        z_max = float(high[2])
        z_neck = float(z_min + vertical * 0.78)
        z_pelvis = float(z_min + vertical * 0.46)
        arm_len = float(np.clip(vertical * 0.34, 0.24, 0.65))
        leg_len = float(max(0.18, z_pelvis - z_min))
        center_xy = np.array([center[0], center[1], 0.0], dtype=np.float32)

        head = center_xy + np.array([0.0, 0.0, z_max], dtype=np.float32)
        neck = center_xy + np.array([0.0, 0.0, z_neck], dtype=np.float32)
        pelvis = center_xy + np.array([0.0, 0.0, z_pelvis], dtype=np.float32)
        left_shoulder = neck + _vec2(lateral_axis, shoulder_width * 0.50)
        right_shoulder = neck + _vec2(lateral_axis, -shoulder_width * 0.50)
        left_elbow = left_shoulder + _vec2(lateral_axis, shoulder_width * 0.18, -arm_len * 0.45)
        right_elbow = right_shoulder + _vec2(lateral_axis, -shoulder_width * 0.18, -arm_len * 0.45)
        left_hand = left_shoulder + _vec2(lateral_axis, shoulder_width * 0.30, -arm_len)
        right_hand = right_shoulder + _vec2(lateral_axis, -shoulder_width * 0.30, -arm_len)
        left_hip = pelvis + _vec2(lateral_axis, hip_width * 0.50)
        right_hip = pelvis + _vec2(lateral_axis, -hip_width * 0.50)
        left_knee = left_hip + np.array([0.0, 0.0, -leg_len * 0.55], dtype=np.float32)
        right_knee = right_hip + np.array([0.0, 0.0, -leg_len * 0.55], dtype=np.float32)
        left_foot = left_hip + _vec2(lateral_axis, foot_width * 0.18, -leg_len)
        right_foot = right_hip + _vec2(lateral_axis, -foot_width * 0.18, -leg_len)

    joints = {
        "head": head,
        "neck": neck,
        "pelvis": pelvis,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_foot": left_foot,
        "right_foot": right_foot,
    }
    return {name: np.asarray(joints[name], dtype=np.float32) for name in JOINT_NAMES}


def target_from_joints(joints: dict[str, np.ndarray]) -> np.ndarray:
    flat = np.concatenate([np.asarray(joints[name], dtype=np.float32).reshape(3) for name in JOINT_NAMES])
    return (flat / TARGET_SCALE).astype(np.float32)


def collect_pose_dataset(
    path: Path,
    max_files: int,
    max_rows: int,
    max_frames: int,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    groups: list[str] = []
    skipped: list[dict[str, str]] = []
    for source in list(_iter_sources(path))[:max_files]:
        try:
            df = _normalize_columns(_read_dataframe(source, max_rows=max_rows))
        except Exception as exc:
            skipped.append({"source": source.label, "reason": f"read_error:{exc}"})
            continue
        frame_col = CsvStream._first_column(df, FRAME_COLUMNS)
        if frame_col is None:
            skipped.append({"source": source.label, "reason": "missing_frame_column"})
            continue
        grouped = df.groupby(frame_col)
        added = 0
        for frame_id in sorted(df[frame_col].dropna().unique())[:max_frames]:
            points = CsvStream._points_from_frame(grouped.get_group(frame_id))
            if len(points) < min_points:
                continue
            joints = pseudo_joints_from_points(points)
            if joints is None:
                continue
            x_rows.append(build_pose_features(points))
            y_rows.append(target_from_joints(joints))
            groups.append(source.label)
            added += 1
        if added == 0:
            skipped.append({"source": source.label, "reason": "no_pose_frames"})
    if not x_rows:
        raise ValueError("no pose training frames collected")
    return np.vstack(x_rows).astype(np.float32), np.vstack(y_rows).astype(np.float32), groups, skipped


def grouped_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    unique_groups = np.asarray(sorted(set(groups)), dtype=object)
    if len(unique_groups) < 2:
        raise ValueError("not enough source groups for train/test split")
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
    )
    train_set = set(train_groups.tolist())
    test_set = set(test_groups.tolist())
    train_mask = np.asarray([group in train_set for group in groups], dtype=bool)
    test_mask = np.asarray([group in test_set for group in groups], dtype=bool)
    return x[train_mask], x[test_mask], y[train_mask], y[test_mask], train_groups.tolist(), test_groups.tolist()


def pose_metrics(pred_scaled: np.ndarray, truth_scaled: np.ndarray) -> dict[str, object]:
    pred = pred_scaled * TARGET_SCALE.reshape(1, -1)
    truth = truth_scaled * TARGET_SCALE.reshape(1, -1)
    joint_errors = np.linalg.norm(
        pred.reshape(-1, len(JOINT_NAMES), 3) - truth.reshape(-1, len(JOINT_NAMES), 3),
        axis=2,
    )
    return {
        "mean_joint_error_m": float(np.mean(joint_errors)),
        "median_joint_error_m": float(np.median(joint_errors)),
        "p95_joint_error_m": float(np.percentile(joint_errors, 95)),
        "pck_10cm": float(np.mean(joint_errors <= 0.10)),
        "pck_15cm": float(np.mean(joint_errors <= 0.15)),
    }


def train_pose_model(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    training_source: Path,
    test_size: float,
    random_state: int,
) -> tuple[dict[str, object], dict[str, object]]:
    x_train, x_test, y_train, y_test, train_groups, test_groups = grouped_train_test_split(
        x, y, groups, test_size=test_size, random_state=random_state
    )
    model = ExtraTreesRegressor(
        n_estimators=240,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test).astype(np.float32)
    metrics = {
        "model_type": "ExtraTreesRegressor",
        "label_type": "geometry_pseudo_skeleton",
        "frames": int(len(y)),
        "train_frames": int(len(y_train)),
        "test_frames": int(len(y_test)),
        "train_sources": int(len(train_groups)),
        "test_sources": int(len(test_groups)),
        **pose_metrics(pred, y_test),
    }
    artifact = {
        "model": model,
        "backend": "sklearn",
        "joint_names": JOINT_NAMES,
        "target_scale": TARGET_SCALE,
        "feature_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]),
        "training_source": str(training_source),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "note": "Trained from D: mmWave point clouds using deterministic geometry pseudo-labels, not external mocap labels.",
    }
    return artifact, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a sklearn pose_model.pkl from D: mmWave point clouds")
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    x, y, groups, skipped = collect_pose_dataset(
        args.input,
        max_files=args.max_files,
        max_rows=args.max_rows,
        max_frames=args.max_frames,
        min_points=args.min_points,
    )
    artifact, metrics = train_pose_model(
        x,
        y,
        groups,
        training_source=args.input,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input),
        "max_files": int(args.max_files),
        "max_rows": int(args.max_rows),
        "max_frames": int(args.max_frames),
        "min_points": int(args.min_points),
        "skipped": skipped,
        "metrics": metrics,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.model_output.open("wb") as handle:
        pickle.dump(artifact, handle)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
