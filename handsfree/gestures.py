from dataclasses import dataclass
from enum import Enum

import numpy as np

from handsfree.config import GestureConfig
from handsfree.tracker import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, WRIST, Hand, fingers_up

MIDDLE_MCP = 9


class IntentKind(Enum):
    MOVE = "move"
    CLICK = "click"
    RIGHT_CLICK = "right_click"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    SCROLL = "scroll"


@dataclass
class Intent:
    kind: IntentKind
    x: float = 0.0  # normalized frame coordinates of the pointer (MOVE)
    y: float = 0.0
    at: float | None = None  # when the gesture started; clicks land where the cursor was then
    amount: float = 0.0  # scroll lines, positive = up


class Mode(Enum):
    IDLE = "idle"
    PINCH = "pinch"
    DRAG = "drag"
    RIGHT_PINCH = "right pinch"
    SCROLL = "scroll"


class PinchDetector:
    """Thresholding with hysteresis: enter below `enter`, release only above `exit`."""

    def __init__(self, enter: float, exit: float):
        self.enter, self.exit = enter, exit
        self.active = False

    def update(self, ratio: float, may_enter: bool = True) -> bool:
        if self.active:
            self.active = ratio < self.exit
        else:
            self.active = may_enter and ratio < self.enter
        return self.active

    def reset(self):
        self.active = False


@dataclass
class HandMetrics:
    palm: float  # wrist → middle-finger knuckle, in pixels; the scale reference
    left_ratio: float  # thumb–index tip distance / palm
    right_ratio: float  # thumb–middle tip distance / palm
    index_ext: float  # index tip → wrist / palm; ~2 extended, <1 curled into a fist
    middle_ext: float
    fingers: list[bool]


def measure(hand: Hand) -> HandMetrics:
    p = hand.pixels
    palm = max(float(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST])), 1e-6)

    def d(a, b):
        return float(np.linalg.norm(p[a] - p[b])) / palm

    return HandMetrics(
        palm=palm,
        left_ratio=d(THUMB_TIP, INDEX_TIP),
        right_ratio=d(THUMB_TIP, MIDDLE_TIP),
        index_ext=d(INDEX_TIP, WRIST),
        middle_ext=d(MIDDLE_TIP, WRIST),
        fingers=fingers_up(hand),
    )


@dataclass
class GestureState:
    mode: Mode = Mode.IDLE
    metrics: HandMetrics | None = None
    scroll_anchor_y: float | None = None  # pixel y of the index tip when scrolling began
    scroll_offset: float = 0.0  # palm sizes above (+) or below (-) the anchor


class GestureInterpreter:
    """Finite-state machine turning tracked hands into mouse intents.

    IDLE        index finger up moves the pointer
    PINCH       thumb+index closed; release quickly → click, hold → DRAG
    DRAG        mouse button held, pointer follows the hand until release
    RIGHT_PINCH thumb+middle closed; release → right click
    SCROLL      index+middle up; vertical offset from where it started sets scroll speed
    """

    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self.left = PinchDetector(cfg.pinch_enter, cfg.pinch_exit)
        self.right = PinchDetector(cfg.pinch_enter, cfg.pinch_exit)
        self.state = GestureState()
        self._onset = 0.0
        self._last_t: float | None = None
        self._scroll_lost_at: float | None = None

    def update(self, hand: Hand | None, t: float) -> list[Intent]:
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t

        if hand is None:
            if self._hold_scroll(t):
                return []
            intents = [Intent(IntentKind.MOUSE_UP)] if self.state.mode is Mode.DRAG else []
            self.left.reset()
            self.right.reset()
            self.state = GestureState()
            return intents

        m = measure(hand)
        min_ext = self.cfg.min_finger_extension
        left = self.left.update(m.left_ratio, may_enter=m.index_ext > min_ext)
        right = self.right.update(m.right_ratio, may_enter=m.middle_ext > min_ext)
        self.state.metrics = m

        x, y = (float(v) for v in hand.landmarks[INDEX_TIP, :2])
        mode = self.state.mode

        if mode is Mode.PINCH:
            if not left:
                return self._to(Mode.IDLE, Intent(IntentKind.CLICK, at=self._onset))
            if t - self._onset >= self.cfg.hold_s:
                return self._to(Mode.DRAG, Intent(IntentKind.MOUSE_DOWN, at=self._onset))
            return []

        if mode is Mode.DRAG:
            if not left:
                return self._to(Mode.IDLE, Intent(IntentKind.MOUSE_UP))
            return [Intent(IntentKind.MOVE, x, y)]

        if mode is Mode.RIGHT_PINCH:
            if not right:
                return self._to(Mode.IDLE, Intent(IntentKind.RIGHT_CLICK, at=self._onset))
            return []

        # IDLE or SCROLL: look for the start of a new gesture.
        if left and (not right or m.left_ratio <= m.right_ratio):
            self.right.reset()
            self._onset = t
            return self._to(Mode.PINCH)
        if right:
            self.left.reset()
            self._onset = t
            return self._to(Mode.RIGHT_PINCH)

        _, index_up, middle_up, ring_up, pinky_up = m.fingers
        if index_up and middle_up and not ring_up and not pinky_up:
            tip_y_px = float(hand.pixels[INDEX_TIP, 1])
            self._scroll_lost_at = None
            if mode is not Mode.SCROLL:
                self._to(Mode.SCROLL)
                self.state.scroll_anchor_y = tip_y_px
                return []
            return self._scroll(tip_y_px, m.palm, dt)

        if self._hold_scroll(t):
            return []

        self._to(Mode.IDLE)
        return [Intent(IntentKind.MOVE, x, y)] if index_up else []

    def _hold_scroll(self, t: float) -> bool:
        """Keep scroll mode (and its anchor) through brief posture or tracking dropouts.

        Otherwise a one-frame glitch re-anchors at the current, already-offset position,
        and moving back toward the original neutral point reverses the scroll direction.
        """
        if self.state.mode is not Mode.SCROLL:
            return False
        if self._scroll_lost_at is None:
            self._scroll_lost_at = t
        return t - self._scroll_lost_at < self.cfg.scroll_grace_s

    def _to(self, mode: Mode, *intents: Intent) -> list[Intent]:
        self.state.mode = mode
        if mode is not Mode.SCROLL:
            self.state.scroll_anchor_y = None
            self.state.scroll_offset = 0.0
            self._scroll_lost_at = None
        return list(intents)

    def _scroll(self, y_px: float, palm: float, dt: float) -> list[Intent]:
        # Joystick-style: offset from the anchor (in palm sizes) beyond a dead zone sets speed.
        offset = (self.state.scroll_anchor_y - y_px) / palm
        self.state.scroll_offset = offset
        excess = abs(offset) - self.cfg.scroll_deadzone
        if excess <= 0 or dt <= 0:
            return []
        speed = np.sign(offset) * excess * self.cfg.scroll_speed
        return [Intent(IntentKind.SCROLL, amount=float(speed * dt))]
