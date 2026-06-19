from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from modules.csv_stream import CsvStream  # noqa: E402
from tools.evaluate_fall_dataset import infer_label  # noqa: E402
from tools.validate_mmwave_dataset import FRAME_COLUMNS, CsvSource, _iter_sources, _normalize_columns, _read_source  # noqa: E402


FRAME_FEATURE_NAMES = (
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

AGGREGATIONS = ("mean", "std", "min", "max", "first", "last", "delta")


@dataclass(frozen=True)
class SequenceExample:
    source: str
    label: str
    feature: np.ndarray


def _read_dataframe(source: CsvSource, max_rows: int) -> pd.DataFrame:
    if source.zip_member:
        import zipfile

        with zipfile.ZipFile(source.path) as archive:
            with archive.open(source.zip_member) as handle:
                return pd.read_csv(handle, nrows=max_rows)
    return _read_source(source, max_rows=max_rows)


def frame_features_from_points(points: np.ndarray, previous_center_z: float | None, frame_delta_seconds: float) -> np.ndarray:
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    if points.shape[1] >= 4:
        doppler = np.asarray(points[:, 3], dtype=np.float32)
    else:
        doppler = np.zeros(len(points), dtype=np.float32)

    center = np.mean(xyz, axis=0)
    low = np.percentile(xyz, 5, axis=0)
    high = np.percentile(xyz, 95, axis=0)
    spans = np.maximum(high - low, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    stds = np.std(xyz, axis=0) if len(xyz) > 1 else np.zeros(3, dtype=np.float32)
    vz = 0.0 if previous_center_z is None else (float(center[2]) - float(previous_center_z)) / max(frame_delta_seconds, 1e-3)
    aspect = max(float(spans[0]), float(spans[1])) / max(float(spans[2]), 0.05)
    return np.array(
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


def sequence_feature_names() -> list[str]:
    names: list[str] = []
    for name in FRAME_FEATURE_NAMES:
        for agg in AGGREGATIONS:
            names.append(f"{name}_{agg}")
    names.extend(["frames_with_points", "duration_seconds"])
    return names


def aggregate_frame_features(features: np.ndarray, frame_delta_seconds: float) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("no frame features to aggregate")
    parts: list[np.ndarray] = []
    parts.append(np.mean(features, axis=0))
    parts.append(np.std(features, axis=0))
    parts.append(np.min(features, axis=0))
    parts.append(np.max(features, axis=0))
    parts.append(features[0])
    parts.append(features[-1])
    parts.append(features[-1] - features[0])
    flat = np.concatenate(parts).astype(np.float32)
    extras = np.array(
        [float(features.shape[0]), float(features.shape[0]) * float(frame_delta_seconds)],
        dtype=np.float32,
    )
    return np.concatenate([flat, extras]).astype(np.float32)


def extract_sequence_feature_from_dataframe(
    df: pd.DataFrame,
    max_frames: int,
    min_points: int,
    frame_delta_seconds: float,
) -> np.ndarray | None:
    normalized = _normalize_columns(df)
    frame_col = CsvStream._first_column(normalized, FRAME_COLUMNS)
    if frame_col is None:
        return None
    grouped = normalized.groupby(frame_col)
    frame_ids = sorted(normalized[frame_col].dropna().unique())[:max_frames]
    previous_center_z: float | None = None
    frame_features: list[np.ndarray] = []
    for frame_id in frame_ids:
        points = CsvStream._points_from_frame(grouped.get_group(frame_id))
        if len(points) < min_points:
            continue
        feature = frame_features_from_points(points, previous_center_z, frame_delta_seconds)
        previous_center_z = float(feature[2])
        frame_features.append(feature)
    if not frame_features:
        return None
    return aggregate_frame_features(np.vstack(frame_features), frame_delta_seconds)


def load_examples(
    path: Path,
    max_files: int,
    max_rows: int,
    max_frames: int,
    min_points: int,
    frame_delta_seconds: float,
) -> tuple[list[SequenceExample], list[dict[str, str]]]:
    examples: list[SequenceExample] = []
    skipped: list[dict[str, str]] = []
    for source in list(_iter_sources(path))[:max_files]:
        label = infer_label(source.label)
        if label not in {"fall", "non_fall"}:
            skipped.append({"source": source.label, "reason": "unknown_label"})
            continue
        try:
            df = _read_dataframe(source, max_rows=max_rows)
            feature = extract_sequence_feature_from_dataframe(
                df,
                max_frames=max_frames,
                min_points=min_points,
                frame_delta_seconds=frame_delta_seconds,
            )
        except Exception as exc:
            skipped.append({"source": source.label, "reason": f"error:{exc}"})
            continue
        if feature is None:
            skipped.append({"source": source.label, "reason": "no_features"})
            continue
        examples.append(SequenceExample(source=source.label, label=label, feature=feature))
    return examples, skipped


def train_and_evaluate(
    examples: list[SequenceExample],
    random_state: int,
    test_size: float,
) -> tuple[dict[str, object], dict[str, object]]:
    if len(examples) < 8:
        raise ValueError("not enough labelled examples")
    x = np.vstack([example.feature for example in examples]).astype(np.float32)
    y = np.asarray([1 if example.label == "fall" else 0 for example in examples], dtype=np.int32)
    sources = [example.source for example in examples]
    x_train, x_test, y_train, y_test, src_train, src_test = train_test_split(
        x,
        y,
        sources,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, pred, labels=[1, 0])
    metrics: dict[str, object] = {
        "model_type": "ExtraTreesClassifier",
        "classes": ["non_fall", "fall"],
        "examples": int(len(examples)),
        "train_examples": int(len(y_train)),
        "test_examples": int(len(y_test)),
        "fall_examples": int(np.sum(y == 1)),
        "non_fall_examples": int(np.sum(y == 0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "confusion_labels": ["fall", "non_fall"],
        "confusion_matrix": matrix.astype(int).tolist(),
        "classification_report": classification_report(
            y_test,
            pred,
            target_names=["non_fall", "fall"],
            output_dict=True,
            zero_division=0,
        ),
        "test_predictions": [
            {
                "source": source,
                "label": "fall" if int(true) else "non_fall",
                "predicted": "fall" if int(item_pred) else "non_fall",
                "fall_probability": float(item_prob),
            }
            for source, true, item_pred, item_prob in zip(src_test, y_test, pred, prob)
        ],
    }
    artifact = {
        "model": model,
        "feature_names": sequence_feature_names(),
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "window_size": None,
        "threshold": 0.5,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    return artifact, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a fall/non-fall sequence model from mmWave point-cloud CSV datasets")
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--frame-delta-seconds", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    examples, skipped = load_examples(
        args.input,
        max_files=args.max_files,
        max_rows=args.max_rows,
        max_frames=args.max_frames,
        min_points=args.min_points,
        frame_delta_seconds=args.frame_delta_seconds,
    )
    artifact, metrics = train_and_evaluate(examples, random_state=args.random_state, test_size=args.test_size)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input),
        "max_files": int(args.max_files),
        "max_rows": int(args.max_rows),
        "max_frames": int(args.max_frames),
        "min_points": int(args.min_points),
        "frame_delta_seconds": float(args.frame_delta_seconds),
        "feature_names": sequence_feature_names(),
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
