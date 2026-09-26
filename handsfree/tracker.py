from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from handsfree.config import TrackerConfig

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

WRIST = 0
THUMB_TIP = 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


@dataclass
class Hand:
    landmarks: np.ndarray  # (21, 3) normalized x, y in [0, 1], z relative depth
    pixels: np.ndarray  # (21, 2) pixel coordinates in the processed frame
    handedness: str  # "Left" / "Right" as reported by MediaPipe
    score: float


class HandTracker:
    def __init__(self, cfg: TrackerConfig):
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Run `python tools/download_model.py` first."
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        # VIDEO mode rejects non-increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        h, w = frame_bgr.shape[:2]
        hands = []
        for lms, handedness in zip(result.hand_landmarks, result.handedness):
            landmarks = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            pixels = landmarks[:, :2] * np.array([w, h], dtype=np.float32)
            category = handedness[0]
            hands.append(Hand(landmarks, pixels, category.category_name, category.score))
        return hands

    def close(self):
        self._landmarker.close()


def fingers_up(hand: Hand) -> list[bool]:
    """[thumb, index, middle, ring, pinky]; assumes an upright hand facing the camera."""
    p = hand.pixels
    thumb_out = abs(p[THUMB_TIP, 0] - p[17, 0]) > abs(p[3, 0] - p[17, 0])
    others = [p[tip, 1] < p[pip, 1] for tip, pip in (
        (INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)
    )]
    return [bool(thumb_out), *others]
