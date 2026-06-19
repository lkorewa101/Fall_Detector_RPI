from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from modules.csv_stream import CsvStream  # noqa: E402
from modules import fall_detector as fall_detector_module  # noqa: E402
from modules.fall_detector import HybridFallDetector  # noqa: E402
from tools.validate_mmwave_dataset import FRAME_COLUMNS, CsvSource, _iter_sources, _normalize_columns, _read_source  # noqa: E402


@dataclass
class ReplayTrack:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    dims: np.ndarray
    point_count: int
    display_points: np.ndarray
    points: np.ndarray


class _DisabledFallAI:
    enabled = False
    model = None

    def predict_all(self, track):
        return 0.0, "Unknown", 0.0


def infer_label(source_label: str) -> str | None:
    label_scope = source_label.split("!", 1)[1] if "!" in source_label else source_label
    normalized = label_scope.lower().replace("\\", "/").replace("_", "-")
    tokens = [token for token in normalized.replace("!", "/").split("/") if token]
    joined = "/" + "/".join(tokens) + "/"

    nonfall_markers = ("nonfall", "non-fall", "notfall", "not-fall", "no-fall", "normal", "adl")
    if any(marker in normalized for marker in nonfall_markers) or any(token in {"not", "nonfall", "non-fall"} for token in tokens):
        return "non_fall"
    if "/fall/" in joined or any(token == "fall" for token in tokens):
        return "fall"
    return None


def _track_from_points(
    points: np.ndarray,
    previous_center: np.ndarray | None,
    frame_delta_seconds: float,
) -> tuple[ReplayTrack, np.ndarray]:
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    low = np.percentile(xyz, 5, axis=0)
    high = np.percentile(xyz, 95, axis=0)
    center = np.mean(xyz, axis=0).astype(np.float32)
    dims = np.maximum(high - low, np.array([0.15, 0.15, 0.15], dtype=np.float32)).astype(np.float32)
    if previous_center is None:
        velocity = np.zeros(3, dtype=np.float32)
    else:
        velocity = ((center - previous_center) / max(frame_delta_seconds, 1e-3)).astype(np.float32)
    track = ReplayTrack(
        id=1,
        position=center,
        velocity=velocity,
        dims=dims,
        point_count=int(len(points)),
        display_points=points,
        points=points,
    )
    return track, center


def evaluate_dataframe(
    df: pd.DataFrame,
    source_label: str,
    max_frames: int,
    min_points: int,
    frame_delta_seconds: float,
    disable_ai: bool = True,
) -> dict[str, object]:
    normalized = _normalize_columns(df)
    frame_col = CsvStream._first_column(normalized, FRAME_COLUMNS)
    label = infer_label(source_label)
    if frame_col is None:
        return {
            "source": source_label,
            "label": label,
            "status": "unsupported",
            "ok": False,
            "error": "missing_frame_column",
        }

    if disable_ai:
        original_fall_ai = fall_detector_module.FallAI
        try:
            fall_detector_module.FallAI = _DisabledFallAI
            detector = HybridFallDetector()
        finally:
            fall_detector_module.FallAI = original_fall_ai
    else:
        detector = HybridFallDetector()

    grouped = normalized.groupby(frame_col)
    frame_ids = sorted(normalized[frame_col].dropna().unique())[:max_frames]
    previous_center: np.ndarray | None = None
    frames_checked = 0
    frames_with_points = 0
    fall_frames: list[int] = []
    max_score = 0.0
    max_rule_score = 0.0
    max_ai_score = 0.0
    max_sequence_score = 0.0
    sequence_eval_frames = 0
    ai_eval_frames = 0
    fall_latched_frames = 0
    max_score_details: dict[str, object] = {}
    last_state = "unknown"

    for local_index, frame_id in enumerate(frame_ids):
        points = CsvStream._points_from_frame(grouped.get_group(frame_id))
        frames_checked += 1
        if len(points) < min_points:
            continue
        frames_with_points += 1
        track, previous_center = _track_from_points(points, previous_center, frame_delta_seconds)
        is_fall, score, details = detector.detect(track, simulate_ai=False, frame_index=local_index)
        score_value = float(score)
        if score_value >= max_score:
            max_score = score_value
            max_score_details = {
                "frame": int(local_index),
                "rule_score": float(details.get("rule_score", 0.0)),
                "ai_score": float(details.get("ai_score", 0.0)),
                "sequence_score": float(details.get("sequence_score", 0.0)),
                "total_score": float(details.get("total_score", score_value)),
                "fsm_state": str(details.get("fsm_state", "")),
                "fall_latched": bool(details.get("fall_latched", False)),
                "candidate_state": bool(details.get("candidate_state", False)),
                "grounded_frames": int(details.get("grounded_frames", 0)),
                "low_posture_frames": int(details.get("low_posture_frames", 0)),
                "candidate_frames": int(details.get("candidate_frames", 0)),
                "impact_motion_frames": int(details.get("impact_motion_frames", 0)),
                "sequence_frames": int(details.get("sequence_frames", 0)),
            }
        max_rule_score = max(max_rule_score, float(details.get("rule_score", 0.0)))
        max_ai_score = max(max_ai_score, float(details.get("ai_score", 0.0)))
        max_sequence_score = max(max_sequence_score, float(details.get("sequence_score", 0.0)))
        if details.get("should_run_sequence", False):
            sequence_eval_frames += 1
        if details.get("should_run_ai", False):
            ai_eval_frames += 1
        if details.get("fall_latched", False):
            fall_latched_frames += 1
        last_state = str(details.get("fsm_state", last_state))
        if is_fall:
            fall_frames.append(int(local_index))

    predicted = "fall" if fall_frames else "non_fall"
    correct = None if label is None else (predicted == label)
    return {
        "source": source_label,
        "label": label,
        "predicted": predicted,
        "correct": correct,
        "status": "ok" if frames_with_points else "empty",
        "ok": bool(frames_with_points),
        "frames_checked": int(frames_checked),
        "frames_with_points": int(frames_with_points),
        "fall_frames": fall_frames[:20],
        "fall_frame_count": int(len(fall_frames)),
        "max_score": float(max_score),
        "max_rule_score": float(max_rule_score),
        "max_ai_score": float(max_ai_score),
        "max_sequence_score": float(max_sequence_score),
        "sequence_eval_frames": int(sequence_eval_frames),
        "ai_eval_frames": int(ai_eval_frames),
        "fall_latched_frames": int(fall_latched_frames),
        "max_score_details": max_score_details,
        "last_state": last_state,
    }


def _read_dataframe(source: CsvSource, max_rows: int) -> pd.DataFrame:
    if source.zip_member:
        with zipfile.ZipFile(source.path) as archive:
            with archive.open(source.zip_member) as handle:
                return pd.read_csv(handle, nrows=max_rows)
    return _read_source(source, max_rows=max_rows)


def evaluate_source(
    source: CsvSource,
    max_rows: int,
    max_frames: int,
    min_points: int,
    frame_delta_seconds: float,
    disable_ai: bool,
) -> dict[str, object]:
    try:
        df = _read_dataframe(source, max_rows=max_rows)
    except Exception as exc:
        return {
            "source": source.label,
            "label": infer_label(source.label),
            "status": "error",
            "ok": False,
            "error": f"read_error:{exc}",
        }
    return evaluate_dataframe(
        df,
        source_label=source.label,
        max_frames=max_frames,
        min_points=min_points,
        frame_delta_seconds=frame_delta_seconds,
        disable_ai=disable_ai,
    )


def confusion(results: Iterable[dict[str, object]]) -> dict[str, int]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "unknown": 0}
    for item in results:
        label = item.get("label")
        predicted = item.get("predicted")
        if label not in {"fall", "non_fall"} or predicted not in {"fall", "non_fall"}:
            counts["unknown"] += 1
        elif label == "fall" and predicted == "fall":
            counts["tp"] += 1
        elif label == "non_fall" and predicted == "non_fall":
            counts["tn"] += 1
        elif label == "non_fall" and predicted == "fall":
            counts["fp"] += 1
        elif label == "fall" and predicted == "non_fall":
            counts["fn"] += 1
    return counts


def evaluate_dataset(
    path: Path,
    max_files: int,
    max_rows: int,
    max_frames: int,
    min_points: int,
    frame_delta_seconds: float,
    disable_ai: bool,
) -> dict[str, object]:
    all_sources = list(_iter_sources(path))
    sources = all_sources[:max_files]
    results = [
        evaluate_source(
            source,
            max_rows=max_rows,
            max_frames=max_frames,
            min_points=min_points,
            frame_delta_seconds=frame_delta_seconds,
            disable_ai=disable_ai,
        )
        for source in sources
    ]
    matrix = confusion(results)
    labelled = matrix["tp"] + matrix["tn"] + matrix["fp"] + matrix["fn"]
    accuracy = ((matrix["tp"] + matrix["tn"]) / labelled) if labelled else None
    fall_recall = (matrix["tp"] / (matrix["tp"] + matrix["fn"])) if (matrix["tp"] + matrix["fn"]) else None
    false_positive_rate = (matrix["fp"] / (matrix["fp"] + matrix["tn"])) if (matrix["fp"] + matrix["tn"]) else None
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(path),
        "sources_found": len(all_sources),
        "sources_checked": len(results),
        "max_files": int(max_files),
        "max_rows": int(max_rows),
        "max_frames": int(max_frames),
        "min_points": int(min_points),
        "frame_delta_seconds": float(frame_delta_seconds),
        "ai_enabled": not disable_ai,
        "confusion": matrix,
        "accuracy": accuracy,
        "fall_recall": fall_recall,
        "false_positive_rate": false_positive_rate,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay mmWave point-cloud CSV datasets through the fall detector")
    parser.add_argument("input", type=Path, help="CSV, directory, or ZIP containing point-cloud CSV files")
    parser.add_argument("--max-files", type=int, default=60)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--frame-delta-seconds", type=float, default=0.1)
    parser.add_argument("--enable-ai", action="store_true", help="Use the runtime frame AI model during replay")
    parser.add_argument("--quiet", action="store_true", help="Print only a compact summary while still writing full JSON")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    report = evaluate_dataset(
        args.input,
        max_files=args.max_files,
        max_rows=args.max_rows,
        max_frames=args.max_frames,
        min_points=args.min_points,
        frame_delta_seconds=args.frame_delta_seconds,
        disable_ai=not args.enable_ai,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.quiet:
        print(
            json.dumps(
                {
                    "sources_checked": report["sources_checked"],
                    "ai_enabled": report["ai_enabled"],
                    "confusion": report["confusion"],
                    "accuracy": report["accuracy"],
                    "fall_recall": report["fall_recall"],
                    "false_positive_rate": report["false_positive_rate"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(text)
    return 0 if report["sources_checked"] and report["confusion"]["unknown"] < report["sources_checked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
