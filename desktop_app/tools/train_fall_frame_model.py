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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from modules.ai_model import DEFAULT_FALL_ACTIONS  # noqa: E402
from modules.csv_stream import CsvStream  # noqa: E402
from tools.evaluate_fall_dataset import infer_label  # noqa: E402
from tools.validate_mmwave_dataset import FRAME_COLUMNS, CsvSource, _iter_sources, _normalize_columns, _read_source  # noqa: E402


FEATURE_NAMES = [
    "mean_z",
    "min_z",
    "std_z",
    "std_x",
    "std_y",
    "range_x",
    "range_y",
    "range_z",
    "mean_doppler",
    "min_doppler",
    "point_count",
]


def _read_dataframe(source: CsvSource, max_rows: int) -> pd.DataFrame:
    if source.zip_member:
        with zipfile.ZipFile(source.path) as archive:
            with archive.open(source.zip_member) as handle:
                return pd.read_csv(handle, nrows=max_rows)
    return _read_source(source, max_rows=max_rows)


def frame_feature(points: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return None
    dopplers = points[:, 3] if points.shape[1] >= 4 else np.zeros(len(points), dtype=np.float32)
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    return np.array(
        [
            float(np.mean(zs)),
            float(np.min(zs)),
            float(np.std(zs)) if len(zs) > 1 else 0.0,
            float(np.std(xs)) if len(xs) > 1 else 0.0,
            float(np.std(ys)) if len(ys) > 1 else 0.0,
            float(np.max(xs) - np.min(xs)),
            float(np.max(ys) - np.min(ys)),
            float(np.max(zs) - np.min(zs)),
            float(np.mean(dopplers)),
            float(np.min(dopplers)),
            float(len(points)),
        ],
        dtype=np.float32,
    )


def source_frame_features(
    source: CsvSource,
    max_rows: int,
    max_frames: int,
    min_points: int,
) -> tuple[list[np.ndarray], str | None, str | None]:
    label = infer_label(source.label)
    if label not in {"fall", "non_fall"}:
        return [], label, "unknown_label"
    try:
        df = _normalize_columns(_read_dataframe(source, max_rows=max_rows))
    except Exception as exc:
        return [], label, f"read_error:{exc}"
    frame_col = CsvStream._first_column(df, FRAME_COLUMNS)
    if frame_col is None:
        return [], label, "missing_frame_column"
    grouped = df.groupby(frame_col)
    features: list[np.ndarray] = []
    for frame_id in sorted(df[frame_col].dropna().unique())[:max_frames]:
        points = CsvStream._points_from_frame(grouped.get_group(frame_id))
        if len(points) < min_points:
            continue
        feature = frame_feature(points)
        if feature is not None:
            features.append(feature)
    if not features:
        return [], label, "no_features"
    return features, label, None


def collect_dataset(
    path: Path,
    max_files: int,
    max_rows: int,
    max_frames: int,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    groups: list[str] = []
    skipped: list[dict[str, str]] = []
    for source in list(_iter_sources(path))[:max_files]:
        features, label, reason = source_frame_features(source, max_rows=max_rows, max_frames=max_frames, min_points=min_points)
        if reason is not None:
            skipped.append({"source": source.label, "reason": reason})
            continue
        target = 1 if label == "fall" else 0
        x_rows.extend(features)
        y_rows.extend([target] * len(features))
        groups.extend([source.label] * len(features))
    if not x_rows:
        raise ValueError("no frame features collected")
    return np.vstack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.int32), groups, skipped


def grouped_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    unique_groups = np.asarray(sorted(set(groups)), dtype=object)
    group_labels = []
    for group in unique_groups:
        idx = [i for i, item in enumerate(groups) if item == group]
        group_labels.append(int(np.max(y[idx])))
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
        stratify=np.asarray(group_labels, dtype=np.int32),
    )
    train_set = set(train_groups.tolist())
    test_set = set(test_groups.tolist())
    train_mask = np.asarray([group in train_set for group in groups], dtype=bool)
    test_mask = np.asarray([group in test_set for group in groups], dtype=bool)
    return x[train_mask], x[test_mask], y[train_mask], y[test_mask], train_groups.tolist(), test_groups.tolist()


def evaluate_source_predictions(
    model: ExtraTreesClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_groups_per_frame: list[str],
) -> dict[str, object]:
    proba = model.predict_proba(x_test)[:, 1]
    by_group: dict[str, list[float]] = {}
    truth: dict[str, int] = {}
    for group, p, target in zip(test_groups_per_frame, proba, y_test):
        by_group.setdefault(group, []).append(float(p))
        truth[group] = int(target)
    labels = []
    preds = []
    rows = []
    for group, probs in sorted(by_group.items()):
        score = float(np.mean(probs))
        pred = 1 if score >= 0.5 else 0
        labels.append(truth[group])
        preds.append(pred)
        rows.append(
            {
                "source": group,
                "label": "fall" if truth[group] else "non_fall",
                "predicted": "fall" if pred else "non_fall",
                "fall_probability": score,
                "frames": len(probs),
            }
        )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "confusion_labels": ["fall", "non_fall"],
        "confusion_matrix": confusion_matrix(labels, preds, labels=[1, 0]).astype(int).tolist(),
        "predictions": rows,
    }


def train_model(
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
    test_groups_per_frame = [group for group in groups if group in set(test_groups)]
    source_metrics = evaluate_source_predictions(model, x_test, y_test, test_groups_per_frame)
    metrics = {
        "model_type": "ExtraTreesClassifier",
        "feature_names": FEATURE_NAMES,
        "frames": int(len(y)),
        "train_frames": int(len(y_train)),
        "test_frames": int(len(y_test)),
        "train_sources": int(len(train_groups)),
        "test_sources": int(len(test_groups)),
        "fall_frames": int(np.sum(y == 1)),
        "non_fall_frames": int(np.sum(y == 0)),
        "frame_accuracy": float(accuracy_score(y_test, pred)),
        "frame_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "frame_confusion_labels": ["fall", "non_fall"],
        "frame_confusion_matrix": confusion_matrix(y_test, pred, labels=[1, 0]).astype(int).tolist(),
        "frame_classification_report": classification_report(
            y_test,
            pred,
            target_names=["non_fall", "fall"],
            output_dict=True,
            zero_division=0,
        ),
        "source_metrics": source_metrics,
    }
    artifact = {
        "model": model,
        "label_encoder": None,
        "fall_actions": DEFAULT_FALL_ACTIONS,
        "feature_names": FEATURE_NAMES,
        "training_source": str(training_source),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    return artifact, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the app-compatible 11-feature fall_model.pkl from D: mmWave data")
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    x, y, groups, skipped = collect_dataset(
        args.input,
        max_files=args.max_files,
        max_rows=args.max_rows,
        max_frames=args.max_frames,
        min_points=args.min_points,
    )
    artifact, metrics = train_model(
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
