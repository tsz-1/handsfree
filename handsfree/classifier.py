from pathlib import Path

import joblib
import numpy as np

from handsfree.tracker import WRIST, Hand

MIDDLE_MCP = 9
FEATURE_VERSION = 1

# "none" = anything that isn't a shortcut, including the mouse poses (point, pinch, two fingers).
DEFAULT_LABELS = ["none", "fist", "open_palm", "thumbs_up", "rock", "call"]


def hand_features(pixels: np.ndarray, handedness: str) -> np.ndarray:
    """42-d pose descriptor: 2D landmarks relative to the wrist, in palm lengths.

    Translation and scale are removed so distance to the camera doesn't matter. Left hands are
    mirrored onto right hands. Rotation is kept on purpose: it separates e.g. thumbs up/down.
    """
    p = pixels[:, :2].astype(np.float64) - pixels[WRIST, :2]
    palm = max(float(np.linalg.norm(p[MIDDLE_MCP])), 1e-6)
    p /= palm
    if handedness == "Left":
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
