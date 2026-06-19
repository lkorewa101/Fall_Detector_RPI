
import threading
import time
import ast
import zipfile
import pandas as pd
import numpy as np
from .tlv_parser import ParsedFrame
from .pose_estimator import load_sidecar_skeletons

class CsvStream(threading.Thread):
    FRAME_COLUMNS = ("frame", "frameno", "framenum", "framenumber", "frame_number", "frameid", "frame_id")
    X_COLUMNS = ("x", "posx", "x_m", "xmeter")
    Y_COLUMNS = ("y", "posy", "y_m", "ymeter")
    Z_COLUMNS = ("z", "posz", "z_m", "zmeter")
    RANGE_COLUMNS = ("range", "rangem", "range_m", "r")
    AZIMUTH_COLUMNS = ("azimuth", "azimuthangle", "azimuth_deg", "azimuthrad", "az")
    ELEVATION_COLUMNS = ("elevation", "elevationangle", "elevation_deg", "elevationrad", "el")
    VELOCITY_COLUMNS = ("velocity", "doppler", "dopplervelocity", "radialvelocity", "v", "vel")

    def __init__(self, csv_path, callback, log_callback=None, fps=30, progress_callback=None):
        super().__init__()
        self.csv_path = csv_path
        self.callback = callback
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.fps = fps
        self.running = False
        self.paused = False
        self.current_idx = 0
        self.total_frames = 0
        self.speed_multiplier = 1.0
        self.daemon = True
        self.sidecar_skeletons = {}
        self.loaded_source = str(csv_path)

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def toggle_pause(self):
        self.paused = not self.paused

    def seek(self, percent):
        if self.total_frames > 0:
            self.current_idx = int((percent / 100.0) * (self.total_frames - 1))
            self.current_idx = max(0, min(self.current_idx, self.total_frames - 1))

    def set_speed(self, speed):
        self.speed_multiplier = speed

    @staticmethod
    def _first_column(df, names):
        for name in names:
            if name in df.columns:
                return name
        return None

    @staticmethod
    def _angle_to_radians(values):
        arr = np.asarray(values, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if len(finite) and float(np.nanmax(np.abs(finite))) > (2.0 * np.pi + 0.25):
            arr = np.deg2rad(arr)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _sanitize_output_points(points):
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 4 or len(arr) == 0:
            return np.empty((0, 4), dtype=np.float32)
        arr = arr[:, :4]
        valid = np.isfinite(arr).all(axis=1)
        valid &= (arr[:, 0] >= -50.0) & (arr[:, 0] <= 50.0)
        valid &= (arr[:, 1] >= -50.0) & (arr[:, 1] <= 50.0)
        valid &= (arr[:, 2] >= -50.0) & (arr[:, 2] <= 50.0)
        valid &= (arr[:, 3] >= -100.0) & (arr[:, 3] <= 100.0)
        return arr[valid].astype(np.float32, copy=False)

    @staticmethod
    def _cell_to_float_array(value):
        if value is None:
            return np.empty((0,), dtype=np.float32)
        if isinstance(value, float) and np.isnan(value):
            return np.empty((0,), dtype=np.float32)
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "none"}:
                return np.empty((0,), dtype=np.float32)
            try:
                value = ast.literal_eval(text) if text.startswith(("[", "(", "{")) else float(text)
            except (ValueError, SyntaxError):
                return np.empty((0,), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr[np.isfinite(arr)].astype(np.float32, copy=False)

    @staticmethod
    def _match_suffixed_columns(frame_data, prefix):
        marker = f"{prefix}_"
        matches = {}
        for column in frame_data.columns:
            if column.startswith(marker):
                suffix = column[len(marker):]
                if suffix.isdigit():
                    matches[int(suffix)] = column
        return matches

    @staticmethod
    def _is_pointcloud_member(name):
        lowered = str(name).lower()
        return (
            lowered.endswith(".csv")
            and not lowered.startswith("__macosx/")
            and (
                "tlv_1020_pointcloud" in lowered
                or "pointcloud" in lowered
                or "point_cloud" in lowered
            )
        )

    @classmethod
    def _score_point_csv_columns(cls, columns, source_name=""):
        normalized = {str(column).strip().lower() for column in columns}
        has_frame = any(column in normalized for column in cls.FRAME_COLUMNS)
        has_cartesian = (
            any(column in normalized for column in cls.X_COLUMNS)
            and any(column in normalized for column in cls.Y_COLUMNS)
            and any(column in normalized for column in cls.Z_COLUMNS)
        )
        has_spherical = (
            any(column in normalized for column in cls.RANGE_COLUMNS)
            and any(column in normalized for column in cls.AZIMUTH_COLUMNS)
        )
        has_wide_spherical = (
            any(column.startswith("range_") for column in normalized)
            and any(column.startswith("azimuth_") for column in normalized)
        )
        if not has_frame or not (has_cartesian or has_spherical or has_wide_spherical):
            return 0

        score = 10
        if has_cartesian:
            score += 30
        if has_spherical:
            score += 25
        if has_wide_spherical:
            score += 20
        if cls._is_pointcloud_member(source_name):
            score += 50
        return score

    @classmethod
    def _rank_zip_csv_candidates(cls, archive):
        infos = [
            info
            for info in archive.infolist()
            if (
                not info.is_dir()
                and info.file_size > 0
                and info.filename.lower().endswith(".csv")
                and not info.filename.lower().startswith("__macosx/")
            )
        ]
        preferred = [info for info in infos if cls._is_pointcloud_member(info.filename)]
        scan_infos = preferred or infos
        ranked = []
        for info in scan_infos:
            try:
                with archive.open(info) as handle:
                    sample = pd.read_csv(handle, nrows=16)
            except Exception:
                continue
            sample.columns = sample.columns.str.strip().str.lower()
            score = cls._score_point_csv_columns(sample.columns, info.filename)
            if score > 0:
                ranked.append((score, info.file_size <= 8192, info.filename, info))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in ranked]

    @classmethod
    def _read_csv_source(cls, csv_path):
        path_text = str(csv_path)
        if ".zip!" in path_text.lower():
            zip_path_text, member = path_text.split("!", 1)
            with zipfile.ZipFile(zip_path_text) as archive:
                with archive.open(member) as handle:
                    return pd.read_csv(handle), f"{zip_path_text}!{member}"

        if path_text.lower().endswith(".zip"):
            with zipfile.ZipFile(path_text) as archive:
                candidates = cls._rank_zip_csv_candidates(archive)
                last_error = None
                for info in candidates:
                    try:
                        with archive.open(info) as handle:
                            df = pd.read_csv(handle)
                        if len(df) > 0:
                            return df, f"{path_text}!{info.filename}"
                    except Exception as exc:
                        last_error = exc
                if last_error is not None:
                    raise last_error
                raise ValueError("ZIP does not contain a non-empty point-cloud CSV")

        return pd.read_csv(path_text), path_text

    @classmethod
    def _points_from_wide_spherical_frame(cls, frame_data):
        range_cols = cls._match_suffixed_columns(frame_data, "range")
        if not range_cols:
            return np.empty((0, 4), dtype=np.float32)

        rows = []
        az_cols = cls._match_suffixed_columns(frame_data, "azimuth")
        el_cols = cls._match_suffixed_columns(frame_data, "elevation")
        doppler_cols = cls._match_suffixed_columns(frame_data, "doppler")

        for _, row in frame_data.iterrows():
            for suffix, range_col in sorted(range_cols.items()):
                rng = cls._cell_to_float_array(row.get(range_col))
                if len(rng) == 0:
                    continue
                az = cls._cell_to_float_array(row.get(az_cols.get(suffix)))
                if len(az) == 0:
                    continue
                el = cls._cell_to_float_array(row.get(el_cols.get(suffix))) if suffix in el_cols else np.zeros_like(rng)
                if len(el) == 0:
                    el = np.zeros_like(rng)
                doppler = (
                    cls._cell_to_float_array(row.get(doppler_cols.get(suffix)))
                    if suffix in doppler_cols
                    else np.zeros_like(rng)
                )
                if len(doppler) == 0:
                    doppler = np.zeros_like(rng)
                count = min(len(rng), len(az), len(el), len(doppler))
                if count <= 0:
                    continue
                rng = rng[:count]
                az = cls._angle_to_radians(az[:count])
                el = cls._angle_to_radians(el[:count])
                doppler = doppler[:count]
                cos_el = np.cos(el)
                rows.append(
                    np.column_stack(
                        [
                            rng * cos_el * np.sin(az),
                            rng * cos_el * np.cos(az),
                            rng * np.sin(el),
                            doppler,
                        ]
                    )
                )

        if not rows:
            return np.empty((0, 4), dtype=np.float32)
        return cls._sanitize_output_points(np.vstack(rows))

    @classmethod
    def _points_from_frame(cls, frame_data):
        x_col = cls._first_column(frame_data, cls.X_COLUMNS)
        y_col = cls._first_column(frame_data, cls.Y_COLUMNS)
        z_col = cls._first_column(frame_data, cls.Z_COLUMNS)
        vel_col = cls._first_column(frame_data, cls.VELOCITY_COLUMNS)

        if x_col and y_col and z_col:
            velocity = (
                frame_data[vel_col].to_numpy(dtype=np.float32)
                if vel_col
                else np.zeros((len(frame_data),), dtype=np.float32)
            )
            return cls._sanitize_output_points(np.column_stack(
                [
                    frame_data[x_col].to_numpy(dtype=np.float32),
                    frame_data[y_col].to_numpy(dtype=np.float32),
                    frame_data[z_col].to_numpy(dtype=np.float32),
                    velocity,
                ]
            ))

        range_col = cls._first_column(frame_data, cls.RANGE_COLUMNS)
        az_col = cls._first_column(frame_data, cls.AZIMUTH_COLUMNS)
        el_col = cls._first_column(frame_data, cls.ELEVATION_COLUMNS)
        if range_col and az_col:
            rng = frame_data[range_col].to_numpy(dtype=np.float32)
            az = cls._angle_to_radians(frame_data[az_col].to_numpy(dtype=np.float32))
            if el_col:
                el = cls._angle_to_radians(frame_data[el_col].to_numpy(dtype=np.float32))
            else:
                el = np.zeros_like(rng, dtype=np.float32)
            velocity = (
                frame_data[vel_col].to_numpy(dtype=np.float32)
                if vel_col
                else np.zeros((len(frame_data),), dtype=np.float32)
            )
            cos_el = np.cos(el)
            return cls._sanitize_output_points(np.column_stack(
                [
                    rng * cos_el * np.sin(az),
                    rng * cos_el * np.cos(az),
                    rng * np.sin(el),
                    velocity,
                ]
            ))

        wide_points = cls._points_from_wide_spherical_frame(frame_data)
        if len(wide_points):
            return wide_points

        return np.empty((0, 4), dtype=np.float32)

    def run(self):
        try:
            self.sidecar_skeletons = load_sidecar_skeletons(self.csv_path)
            if self.sidecar_skeletons:
                self.log(f"[CsvStream] Loaded sidecar joints.npz skeletons: {len(self.sidecar_skeletons)} frames")
            df, self.loaded_source = self._read_csv_source(self.csv_path)
            self.log(f"[CsvStream] Loaded {len(df)} rows from {self.loaded_source}")
        except Exception as e:
            self.log(f"[CsvStream] Error loading CSV: {e}")
            return

        # Convert columns to lowercase to handle both 'Frame' and 'frame', 'X' and 'x', etc.
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle 'doppler' vs 'velocity' naming
        if 'doppler' in df.columns and 'velocity' not in df.columns:
             df.rename(columns={'doppler': 'velocity'}, inplace=True)

        # Group by frame
        frame_col = self._first_column(df, self.FRAME_COLUMNS)
        if frame_col is None:
             self.log("[CsvStream] CSV missing frame column")
             return

        frames = df.groupby(frame_col)
        unique_frames = sorted(df[frame_col].unique())

        self.total_frames = len(unique_frames)
        self.current_idx = 0
        self.running = True
        self.log(f"[CsvStream] Starting playback: {self.total_frames} frames at {self.fps} FPS")

        while self.running:
            if self.current_idx >= self.total_frames:
                if not self.paused:
                    self.paused = True
                    if self.progress_callback:
                        self.progress_callback(self.total_frames, self.total_frames)
                time.sleep(0.1)
                continue

            if self.paused:
                time.sleep(0.1)
                continue
            
            start_time = time.time()
            frame_idx = unique_frames[self.current_idx]
            
            frame_data = frames.get_group(frame_idx)
            
            points = self._points_from_frame(frame_data)
            
            # Create ParsedFrame
            parsed = ParsedFrame(
                ts=time.time(),
                points=points,
                tracks=[], # CSV doesn't have track info by default, or we ignore it for now
                raw=b"",
                skeleton_lines=self.sidecar_skeletons.get(int(frame_idx)),
            )
            
            self.callback(parsed)
            
            if self.progress_callback:
                self.progress_callback(self.current_idx + 1, self.total_frames)
            
            self.current_idx += 1
            
            elapsed = time.time() - start_time
            current_interval = 1.0 / (self.fps * self.speed_multiplier)
            sleep_time = max(0, current_interval - elapsed)
            time.sleep(sleep_time)

        self.running = False
        self.log("[CsvStream] Playback finished")

    def stop(self):
        self.running = False
