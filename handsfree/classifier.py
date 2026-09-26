from pathlib import Path

import joblib
import numpy as np

from handsfree.tracker import WRIST, Hand

INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
FEATURE_VERSION = 2

# "none" = anything that isn't a shortcut, including the mouse poses (point, pinch, two fingers).
DEFAULT_LABELS = ["none", "fist", "open_palm", "thumbs_up", "rock", "call"]


def hand_features(pixels: np.ndarray, handedness: str | None = None) -> np.ndarray:
    """42-d pose descriptor: 2D landmarks in a canonical hand frame.

    Wrist at the origin, palm length (wrist → middle knuckle) as the unit, palm axis rotated
    to point up, and the hand mirrored so the index knuckle is left of the pinky knuckle.
    This removes position, distance to the camera, in-plane rotation, and which hand or side
    (palm/back) is showing, all of which vary a lot between recording sessions. MediaPipe's
    handedness label is ignored because it flips when the hand turns over.
    Consequence: poses that differ only by orientation (thumbs up vs down) are not separable.
    """
    p = pixels[:, :2].astype(np.float64) - pixels[WRIST, :2]
    palm = max(float(np.linalg.norm(p[MIDDLE_MCP])), 1e-6)
    p /= palm

    x, y = p[MIDDLE_MCP]
    angle = -np.arctan2(x, -y)  # rotate so the palm axis points to -y (up on screen)
    c, s = np.cos(angle), np.sin(angle)
    p = p @ np.array([[c, s], [-s, c]])

    if p[INDEX_MCP, 0] > p[PINKY_MCP, 0]:
        p[:, 0] = -p[:, 0]
    return p.reshape(-1)


class GestureClassifier:
    """Wraps the trained scikit-learn pipeline; smooths class probabilities over time."""

    def __init__(self, model_path: str | Path, smoothing: float = 0.6):
        bundle = joblib.load(model_path)
        if bundle.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"{model_path} was trained with different features; retrain it.")
        self.pipeline = bundle["pipeline"]
        self.labels: list[str] = list(bundle["labels"])
        self.smoothing = smoothing
        self._probs: np.ndarray | None = None

    def predict(self, hand: Hand | None) -> tuple[str | None, float]:
        if hand is None:
            self._probs = None
            return None, 0.0
        x = hand_features(hand.pixels, hand.handedness)[None]
        probs = self.pipeline.predict_proba(x)[0]
        if self._probs is None:
            self._probs = probs
        else:
            self._probs = self.smoothing * self._probs + (1 - self.smoothing) * probs
        i = int(np.argmax(self._probs))
        return self.labels[i], float(self._probs[i])
