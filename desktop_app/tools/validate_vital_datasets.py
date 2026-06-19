from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


ADC_RE = re.compile(r"^adc_(?P<profile>.+?)_posi(?:ti|t)on(?P<position>\d+)_ \((?P<trial>\d+)\)\.bin$", re.IGNORECASE)
LOG_RE = re.compile(
    r"^log_Target(?P<target>\d+)_(?P<profile>.+?)_posi(?:ti|t)on(?P<position>\d+)_ \((?P<trial>\d+)\)\.csv$",
    re.IGNORECASE,
)
LOG_SAMPLE_INTERVAL_SEC = 0.008
REFERENCE_WINDOW_SAMPLES = 3500


@dataclass(frozen=True, order=True)
class VitalCaseKey:
    condition: str
    position: int
    profile: str
    trial: int


@dataclass
class VitalLog:
    target: int
    path: Path
    size: int


@dataclass
class VitalCase:
    key: VitalCaseKey
    adc_path: Path | None = None
    adc_size: int = 0
    logs: dict[int, VitalLog] = field(default_factory=dict)


def _condition_name(path: Path) -> str:
    for part in path.parts:
        lowered = part.lower()
        if "asymmetricalposition" in lowered:
            return "asymmetrical"
        if "symmetricalposition" in lowered:
            return "symmetrical"
    return "unknown"


def _iter_files(root: Path, suffix: str) -> Iterable[Path]:
    yield from sorted(root.rglob(f"*{suffix}"))


def _case_key(path: Path, match: re.Match[str]) -> VitalCaseKey:
    return VitalCaseKey(
        condition=_condition_name(path),
        position=int(match.group("position")),
        profile=match.group("profile").upper(),
        trial=int(match.group("trial")),
    )


def _discover_mendeley_cases(root: Path) -> dict[VitalCaseKey, VitalCase]:
    cases: dict[VitalCaseKey, VitalCase] = {}
    for adc_path in _iter_files(root, ".bin"):
        match = ADC_RE.match(adc_path.name)
        if not match:
            continue
        key = _case_key(adc_path, match)
        case = cases.setdefault(key, VitalCase(key=key))
        case.adc_path = adc_path
        case.adc_size = adc_path.stat().st_size

    for log_path in _iter_files(root, ".csv"):
        match = LOG_RE.match(log_path.name)
        if not match:
            continue
        key = _case_key(log_path, match)
        target = int(match.group("target"))
        case = cases.setdefault(key, VitalCase(key=key))
        case.logs[target] = VitalLog(target=target, path=log_path, size=log_path.stat().st_size)

    return cases


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(np.float32, copy=False)
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32, copy=False), kernel, mode="same")


def _count_peaks(values: np.ndarray, *, threshold: float | None, min_distance: int) -> int:
    if len(values) < 3:
        return 0
    peaks = []
    last_idx = -min_distance
    for idx in range(1, len(values) - 1):
        if threshold is not None and values[idx] < threshold:
            continue
        if values[idx] <= values[idx - 1] or values[idx] < values[idx + 1]:
            continue
        if idx - last_idx < min_distance:
            if peaks and values[idx] > values[peaks[-1]]:
                peaks[-1] = idx
                last_idx = idx
            continue
        peaks.append(idx)
        last_idx = idx
    return len(peaks)


def _finite_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_reference_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ecg: list[float] = []
    pcg: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            col1 = _finite_float(row.get("Column1", ""))
            col2 = _finite_float(row.get("Column2", ""))
            if col1 is None or col2 is None:
                continue
            ecg.append(-col1)
            pcg.append(col2)
    return np.asarray(ecg, dtype=np.float32), np.asarray(pcg, dtype=np.float32)


def _analyze_reference_log(path: Path) -> dict[str, object]:
    try:
        ecg, pcg = _read_reference_log(path)
    except Exception as exc:
        return {"ok": False, "error": f"read_error:{exc}"}

    samples = min(len(ecg), len(pcg))
    if samples <= 0:
        return {"ok": False, "error": "empty_log", "samples": int(samples)}

    ecg = ecg[-REFERENCE_WINDOW_SAMPLES:]
    pcg = pcg[-REFERENCE_WINDOW_SAMPLES:]
    duration_sec = float(len(ecg) * LOG_SAMPLE_INTERVAL_SEC)

    heart_count = _count_peaks(ecg, threshold=20.0, min_distance=max(1, int(0.30 / LOG_SAMPLE_INTERVAL_SEC)))
    pcg_smooth = _moving_average(pcg, max(3, int(0.40 / LOG_SAMPLE_INTERVAL_SEC)))
    pcg_smooth = _moving_average(pcg_smooth, max(3, int(0.80 / LOG_SAMPLE_INTERVAL_SEC)))
    breath_threshold = float(np.percentile(pcg_smooth, 60.0)) if len(pcg_smooth) else None
    breath_count = _count_peaks(
        pcg_smooth,
        threshold=breath_threshold,
        min_distance=max(1, int(1.40 / LOG_SAMPLE_INTERVAL_SEC)),
    )

    heart_bpm = (60.0 * heart_count / duration_sec) if duration_sec > 0 else 0.0
    breath_bpm = (60.0 * breath_count / duration_sec) if duration_sec > 0 else 0.0
    return {
        "ok": True,
        "samples": int(samples),
        "window_samples": int(len(ecg)),
        "window_duration_sec": duration_sec,
        "heart_count": int(heart_count),
        "breath_count": int(breath_count),
        "heart_bpm": float(heart_bpm),
        "breath_bpm": float(breath_bpm),
        "ecg_min": float(np.min(ecg)),
        "ecg_max": float(np.max(ecg)),
        "pcg_min": float(np.min(pcg)),
        "pcg_max": float(np.max(pcg)),
    }


def _case_to_json(case: VitalCase, analyze_logs: bool) -> dict[str, object]:
    targets: dict[str, object] = {}
    for target, log in sorted(case.logs.items()):
        entry: dict[str, object] = {"path": str(log.path), "size": int(log.size)}
        if analyze_logs:
            entry["reference"] = _analyze_reference_log(log.path)
        targets[str(target)] = entry

    missing_targets = [target for target in (1, 2) if target not in case.logs]
    return {
        "condition": case.key.condition,
        "position": case.key.position,
        "profile": case.key.profile,
        "trial": case.key.trial,
        "complete": bool(case.adc_path and not missing_targets),
        "adc": None
        if case.adc_path is None
        else {
            "path": str(case.adc_path),
            "size": int(case.adc_size),
            "size_mod_iq16": int(case.adc_size % 4),
        },
        "targets": targets,
        "missing_targets": missing_targets,
    }


def _summarize_rates(cases: list[dict[str, object]]) -> dict[str, object]:
    heart_rates: list[float] = []
    breath_rates: list[float] = []
    for case in cases:
        targets = case.get("targets", {})
        if not isinstance(targets, dict):
            continue
        for target in targets.values():
            if not isinstance(target, dict):
                continue
            ref = target.get("reference", {})
            if not isinstance(ref, dict) or not ref.get("ok"):
                continue
            heart_rates.append(float(ref.get("heart_bpm", 0.0)))
            breath_rates.append(float(ref.get("breath_bpm", 0.0)))

    def stats(values: list[float]) -> dict[str, object]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    return {"heart_bpm": stats(heart_rates), "breath_bpm": stats(breath_rates)}


def validate_mendeley_vital_dataset(root: Path, analyze_logs: bool = True) -> dict[str, object]:
    cases_by_key = _discover_mendeley_cases(root)
    case_reports = [_case_to_json(case, analyze_logs=analyze_logs) for case in cases_by_key.values()]
    case_reports.sort(key=lambda item: (str(item["condition"]), int(item["position"]), str(item["profile"]), int(item["trial"])))

    adc_count = sum(1 for case in cases_by_key.values() if case.adc_path is not None)
    log_count = sum(len(case.logs) for case in cases_by_key.values())
    complete_count = sum(1 for case in case_reports if case["complete"])
    profiles = sorted({str(case["profile"]) for case in case_reports})
    conditions = sorted({str(case["condition"]) for case in case_reports})
    positions = sorted({int(case["position"]) for case in case_reports})
    errors = []
    for case in case_reports:
        if case["adc"] is None:
            errors.append({"case": case, "error": "missing_adc"})
        if case["missing_targets"]:
            errors.append({"case": case, "error": "missing_target_logs"})

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(root),
        "dataset": "mendeley_multi_person_iwr6843isk",
        "cases_found": len(case_reports),
        "complete_cases": complete_count,
        "adc_files": adc_count,
        "reference_log_files": log_count,
        "conditions": conditions,
        "positions": positions,
        "profiles": profiles,
        "errors": errors,
        "reference_rate_summary": _summarize_rates(case_reports) if analyze_logs else {},
        "cases": case_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Mendeley multi-person IWR6843ISK vital dataset pairing")
    parser.add_argument("input", type=Path, help="mendeley_multi_person_iwr6843isk root directory")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--skip-log-analysis", action="store_true", help="Only validate file pairing, without ECG/PCG rates")
    args = parser.parse_args()

    report = validate_mendeley_vital_dataset(args.input, analyze_logs=not args.skip_log_analysis)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {key: report[key] for key in (
        "created_at",
        "input",
        "cases_found",
        "complete_cases",
        "adc_files",
        "reference_log_files",
        "conditions",
        "positions",
        "profiles",
        "reference_rate_summary",
    )}
    summary["error_count"] = len(report["errors"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["complete_cases"] > 0 and not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
