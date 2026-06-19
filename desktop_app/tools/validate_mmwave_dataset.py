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


POINTCLOUD_MARKERS = ("tlv_1020_pointcloud", "pointcloud", "point_cloud")
FRAME_COLUMNS = ("frame", "frameno", "framenum", "framenumber", "frame_number", "frameid", "frame_id")


@dataclass(frozen=True)
class CsvSource:
    label: str
    path: Path
    zip_member: str = ""
    size: int = 0


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()
    return out


def _iter_sources(path: Path) -> Iterable[CsvSource]:
    if path.is_dir():
        for csv_path in sorted(path.rglob("*.csv")):
            lowered = csv_path.name.lower()
            if any(marker in lowered for marker in POINTCLOUD_MARKERS):
                yield CsvSource(label=str(csv_path), path=csv_path, size=csv_path.stat().st_size)
        for zip_path in sorted(path.rglob("*.zip")):
            yield from _iter_zip_sources(zip_path)
        return

    if path.suffix.lower() == ".zip":
        yield from _iter_zip_sources(path)
        return

    if path.suffix.lower() == ".csv":
        yield CsvSource(label=str(path), path=path, size=path.stat().st_size)


def _iter_zip_sources(path: Path) -> Iterable[CsvSource]:
    try:
        with zipfile.ZipFile(path) as archive:
            for info in CsvStream._rank_zip_csv_candidates(archive):
                yield CsvSource(
                    label=f"{path}!{info.filename}",
                    path=path,
                    zip_member=info.filename,
                    size=info.file_size,
                )
    except zipfile.BadZipFile:
        return


def _read_source(source: CsvSource, max_rows: int) -> pd.DataFrame:
    if source.zip_member:
        with zipfile.ZipFile(source.path) as archive:
            with archive.open(source.zip_member) as handle:
                return pd.read_csv(handle, nrows=max_rows)
    return pd.read_csv(source.path, nrows=max_rows)


def _detect_format(df: pd.DataFrame) -> str:
    columns = set(df.columns)
    if (
        CsvStream._first_column(df, CsvStream.X_COLUMNS)
        and CsvStream._first_column(df, CsvStream.Y_COLUMNS)
        and CsvStream._first_column(df, CsvStream.Z_COLUMNS)
    ):
        return "cartesian"
    if CsvStream._first_column(df, CsvStream.RANGE_COLUMNS) and CsvStream._first_column(df, CsvStream.AZIMUTH_COLUMNS):
        return "spherical_tall"
    if any(column.startswith("range_") for column in columns) and any(column.startswith("azimuth_") for column in columns):
        return "spherical_wide_list"
    return "unknown"


def _analyze_source(source: CsvSource, max_rows: int, max_frames: int) -> dict[str, object]:
    try:
        df = _normalize_columns(_read_source(source, max_rows=max_rows))
    except Exception as exc:
        return {
            "source": source.label,
            "size": source.size,
            "status": "error",
            "ok": False,
            "error": f"read_error:{exc}",
        }

    frame_col = CsvStream._first_column(df, FRAME_COLUMNS)
    if frame_col is None:
        return {
            "source": source.label,
            "size": source.size,
            "status": "unsupported",
            "ok": False,
            "format": _detect_format(df),
            "rows_read": int(len(df)),
            "error": "missing_frame_column",
        }

    point_counts: list[int] = []
    z_min: list[float] = []
    z_max: list[float] = []
    frame_ids = sorted(df[frame_col].dropna().unique())[:max_frames]
    grouped = df.groupby(frame_col)
    for frame_id in frame_ids:
        points = CsvStream._points_from_frame(grouped.get_group(frame_id))
        point_counts.append(int(len(points)))
        if len(points):
            z_min.append(float(np.min(points[:, 2])))
            z_max.append(float(np.max(points[:, 2])))

    counts = np.asarray(point_counts, dtype=np.float32)
    nonzero = counts[counts > 0]
    status = "ok" if len(nonzero) > 0 else "empty"
    return {
        "source": source.label,
        "size": source.size,
        "status": status,
        "ok": bool(len(nonzero) > 0),
        "format": _detect_format(df),
        "rows_read": int(len(df)),
        "frame_column": frame_col,
        "frames_checked": int(len(point_counts)),
        "frames_with_points": int(len(nonzero)),
        "points_total": int(np.sum(counts)) if len(counts) else 0,
        "points_mean": float(np.mean(nonzero)) if len(nonzero) else 0.0,
        "points_median": float(np.median(nonzero)) if len(nonzero) else 0.0,
        "points_max": int(np.max(nonzero)) if len(nonzero) else 0,
        "z_min": float(np.min(z_min)) if z_min else None,
        "z_max": float(np.max(z_max)) if z_max else None,
    }


def validate_dataset(path: Path, max_files: int, max_rows: int, max_frames: int) -> dict[str, object]:
    all_sources = list(_iter_sources(path))
    sources = all_sources[:max_files]
    results = [_analyze_source(source, max_rows=max_rows, max_frames=max_frames) for source in sources]
    ok_count = sum(1 for item in results if item.get("ok"))
    empty_count = sum(1 for item in results if item.get("status") == "empty")
    error_count = sum(1 for item in results if item.get("status") == "error")
    unsupported_count = sum(1 for item in results if item.get("status") == "unsupported")
    formats: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for item in results:
        key = str(item.get("format", "unknown"))
        formats[key] = formats.get(key, 0) + 1
        status = str(item.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(path),
        "sources_found": len(all_sources),
        "sources_checked": len(results),
        "sources_ok": ok_count,
        "sources_empty": empty_count,
        "sources_error": error_count,
        "sources_unsupported": unsupported_count,
        "formats": dict(sorted(formats.items())),
        "statuses": dict(sorted(statuses.items())),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate mmWave point-cloud CSV datasets against the desktop app parser")
    parser.add_argument("input", type=Path, help="CSV, directory, or ZIP containing point-cloud CSV files")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    report = validate_dataset(args.input, max_files=args.max_files, max_rows=args.max_rows, max_frames=args.max_frames)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if int(report["sources_ok"]) > 0 and int(report["sources_error"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
