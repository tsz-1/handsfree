from dataclasses import dataclass
from enum import Enum

import numpy as np

from handsfree.config import GestureConfig
from handsfree.tracker import INDEX_TIP, THUMB_TIP, Hand, fingers_up


class IntentKind(Enum):
    MOVE = "move"
    CLICK = "click"


@dataclass
class Intent:
    kind: IntentKind
    x: float = 0.0  # normalized frame coordinates of the pointer
    y: float = 0.0


@dataclass
class GestureState:
    pinch_distance: float = 0.0
    pinching: bool = False


class GestureInterpreter:
    """Turns tracked hands into pointer intents: index finger moves, thumb-index pinch clicks."""

    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self.state = GestureState()
        self._last_click_t = -float("inf")

    def update(self, hand: Hand | None, t: float) -> list[Intent]:
        if hand is None:
            self.state = GestureState()
            return []

        pinch_distance = float(np.linalg.norm(hand.pixels[INDEX_TIP] - hand.pixels[THUMB_TIP]))
        pinching = pinch_distance < self.cfg.pinch_threshold_px
        self.state = GestureState(pinch_distance, pinching)

        tip_x, tip_y = hand.landmarks[INDEX_TIP, :2]
        if pinching:
            if t - self._last_click_t >= self.cfg.click_cooldown_s:
                self._last_click_t = t
                return [Intent(IntentKind.CLICK, tip_x, tip_y)]
            return []
        if fingers_up(hand)[1]:
            return [Intent(IntentKind.MOVE, tip_x, tip_y)]
        return []
