from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
LOCAL_PY_PKGS = Path.home() / "py310_pkgs"
if LOCAL_PY_PKGS.exists() and str(LOCAL_PY_PKGS) not in sys.path:
    sys.path.insert(0, str(LOCAL_PY_PKGS))

import numpy as np

from modules.ai_model import FallAI, FallSequenceAI
from modules.pose_estimator import PoseEstimator


def main() -> int:
    points = np.array(
        [
            [-0.30, 2.0, 0.40, 0.0],
            [-0.20, 2.0, 0.80, 0.0],
            [-0.10, 2.0, 1.20, 0.0],
            [0.00, 2.0, 1.60, 0.0],
            [0.10, 2.0, 1.80, 0.0],
            [0.20, 2.0, 1.40, 0.0],
            [0.30, 2.0, 1.00, 0.0],
            [0.40, 2.0, 0.60, 0.0],
        ],
        dtype=np.float32,
    )

    fall_ai = FallAI()
    fall_prediction = fall_ai.predict_all(SimpleNamespace(points=points))
    sequence_ai = FallSequenceAI()
    sequence_history = [
        FallSequenceAI.extract_frame_feature(
            SimpleNamespace(points=points + np.array([0.0, 0.0, -0.03 * frame, 0.0], dtype=np.float32)),
            None if frame == 0 else 1.10 - (0.03 * (frame - 1)),
            0.1,
        )[0]
        for frame in range(max(8, sequence_ai.min_frames))
    ]
    sequence_score = sequence_ai.predict_from_history(
        [feature for feature in sequence_history if feature is not None],
        0.1,
    )
    pose = PoseEstimator()
    skeleton = pose.predict_skeleton(points)

    print(f"fall_enabled={fall_ai.enabled}")
    print(f"fall_model={fall_ai.model_path}")
    print(f"fall_prediction={fall_prediction}")
    print(f"sequence_enabled={sequence_ai.enabled}")
    print(f"sequence_model={sequence_ai.model_path}")
    print(f"sequence_score={sequence_score:.6f}")
    print(f"pose_available={pose.available}")
    print(f"pose_backend={pose.backend}")
    print(f"pose_model={pose.model_path}")
    print(f"skeleton_shape={None if skeleton is None else skeleton.shape}")
    return 0 if fall_ai.enabled and sequence_ai.enabled and pose.available and skeleton is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
