from dataclasses import dataclass

import numpy as np

from handsfree.config import ShortcutConfig
from handsfree.tracker import WRIST, Hand

MIDDLE_MCP = 9


def hand_in_frame(hand: Hand, width: int, height: int) -> bool:
    """False while the hand is entering or leaving: MediaPipe guesses off-screen landmarks,
    and a half-visible hand often looks like a fist."""
    x, y = hand.pixels[:, 0], hand.pixels[:, 1]
    return bool((x >= 0).all() and (y >= 0).all() and (x < width).all() and (y < height).all())


class MotionGate:
    """Tracks wrist speed in palm lengths per second; deliberate poses are held still."""

    def __init__(self, max_speed: float):
        self.max_speed = max_speed
        self.speed = 0.0
        self._prev: tuple[np.ndarray, float] | None = None

    def update(self, hand: Hand | None, t: float) -> bool:
        """Returns True if the hand is still enough for a pose to count."""
        if hand is None:
            self._prev = None
            self.speed = 0.0
            return False
        wrist = hand.pixels[WRIST].astype(float)
        palm = max(float(np.linalg.norm(hand.pixels[MIDDLE_MCP] - hand.pixels[WRIST])), 1e-6)
        if self._prev is not None and t > self._prev[1]:
            raw = float(np.linalg.norm(wrist - self._prev[0])) / palm / (t - self._prev[1])
            self.speed = 0.5 * self.speed + 0.5 * raw
        self._prev = (wrist, t)
        return self.speed <= self.max_speed


@dataclass
class ShortcutEvent:
    gesture: str
    binding: dict


class ShortcutEngine:
    """Fires a bound gesture once it has been held confidently for `hold_s`.

    Each hold fires at most once: the gesture must change (or the hand leave) before it can
    fire again, and a per-gesture cooldown guards against rapid re-triggers.
    """

    def __init__(self, cfg: ShortcutConfig):
        self.cfg = cfg
        self.candidate: str | None = None
        self._since = 0.0
        self._fired = False
        self._last_fire: dict[str, float] = {}

    def progress(self, t: float) -> float:
        """0..1 of the hold time elapsed for the current candidate (for the UI)."""
        if self.candidate is None or self._fired:
            return 0.0
        return min((t - self._since) / self.cfg.hold_s, 1.0) if self.cfg.hold_s > 0 else 1.0

    def update(self, label: str | None, confidence: float, t: float) -> list[ShortcutEvent]:
        bound = label in self.cfg.bindings and confidence >= self.cfg.min_confidence
        candidate = label if bound else None

        if candidate != self.candidate:
            self.candidate, self._since, self._fired = candidate, t, False
        if candidate is None or self._fired or t - self._since < self.cfg.hold_s:
            return []

        if t - self._last_fire.get(candidate, -float("inf")) < self.cfg.cooldown_s:
            return []  # Keep holding: fires as soon as the cooldown ends.
        self._fired = True
        self._last_fire[candidate] = t
        return [ShortcutEvent(candidate, self.cfg.bindings[candidate])]
