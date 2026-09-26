"""Hard-negative mining: save the frames that led to a false shortcut trigger as `none`."""

import time
from collections import deque
from pathlib import Path

import numpy as np

from handsfree.tracker import Hand


class HardNegativeRecorder:
    def __init__(self, data_dir: str | Path = "data/gestures", window_s: float = 1.5):
        self.window_s = window_s
        self.path = Path(data_dir) / f"hardneg-{time.strftime('%Y%m%d-%H%M%S')}.npz"
        self._recent: deque[tuple[float, Hand]] = deque()
        self._saved: list[tuple[float, Hand]] = []
        self.frame_size = (0, 0)

    def push(self, hand: Hand | None, t: float, frame_size: tuple[int, int]):
        self.frame_size = frame_size
        if hand is not None:
            self._recent.append((t, hand))
        while self._recent and t - self._recent[0][0] > self.window_s:
            self._recent.popleft()

    def save_recent(self) -> int:
        """Append the buffered frames as `none` samples and write the session file."""
        new = list(self._recent)
        self._recent.clear()
        if not new:
            return 0
        self._saved += new
        hands = [h for _, h in self._saved]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            pixels=np.stack([h.pixels for h in hands]),
            landmarks=np.stack([h.landmarks for h in hands]),
            handedness=np.array([h.handedness for h in hands]),
            labels=np.array(["none"] * len(hands)),
            timestamps=np.array([t for t, _ in self._saved]),
            frame_size=np.array(self.frame_size),
        )
        return len(new)
