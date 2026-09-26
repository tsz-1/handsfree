import sys
from collections import deque

import numpy as np
import pyautogui

from handsfree.config import MouseConfig
from handsfree.filters import OneEuroFilter
from handsfree.gestures import Intent, IntentKind

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

HISTORY_S = 1.0

# pyautogui's public press() silently ignores these on macOS; its keyboard map lacks them.
MEDIA_KEYS = {
    "play": "KEYTYPE_PLAY",
    "next": "KEYTYPE_NEXT",
    "previous": "KEYTYPE_PREVIOUS",
    "mute": "KEYTYPE_MUTE",
    "volume_up": "KEYTYPE_SOUND_UP",
    "volume_down": "KEYTYPE_SOUND_DOWN",
}
ACTIONS = {"toggle_pause"}


def validate_binding(gesture: str, binding: dict):
    kinds = [k for k in ("action", "media", "keys") if k in binding]
    if len(kinds) != 1 or len(binding) != 1:
        raise ValueError(f"Binding for {gesture!r} needs exactly one of action/media/keys")
    if "action" in binding and binding["action"] not in ACTIONS:
        raise ValueError(f"Unknown action {binding['action']!r} for {gesture!r}")
    if "media" in binding and binding["media"] not in MEDIA_KEYS:
        raise ValueError(f"Unknown media key {binding['media']!r} for {gesture!r}")
    if "keys" in binding:
        bad = [k for k in binding["keys"] if str(k) not in pyautogui.KEYBOARD_KEYS]
        if bad:
            raise ValueError(f"Unknown keys {bad} for {gesture!r}")


def click_at(x: float, y: float, count: int = 1):
    """Left-click with an explicit click count.

    macOS does not infer double-clicks from timing: the event itself carries the count, and
    pyautogui never sets it, so its clicks can't open a file. Post the events ourselves.
    """
    if sys.platform != "darwin":
        pyautogui.click(x, y)  # Windows and X11 derive the count from timing.
        return
    import Quartz

    for kind in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
        event = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, count)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def press_keys(keys: list):
    pyautogui.hotkey(*[str(k) for k in keys])


def press_media(name: str):
    if sys.platform != "darwin":
        raise NotImplementedError("Media keys are only implemented for macOS")
    from pyautogui import _pyautogui_osx

    key = MEDIA_KEYS[name]
    _pyautogui_osx._specialKeyEvent(key, "down")
    _pyautogui_osx._specialKeyEvent(key, "up")


class MouseController:
    def __init__(self, cfg: MouseConfig):
        self.cfg = cfg
        self.screen_w, self.screen_h = pyautogui.size()
        self._filter = OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._button_down = False
        self._scroll_remainder = 0.0
        self._last_click: tuple[float, np.ndarray, int] | None = None  # time, pos, count

    def to_screen(self, x: float, y: float) -> np.ndarray:
        """Map a point in the active region of the frame (normalized) to screen pixels."""
        mx, my = self.cfg.margin_x, self.cfg.margin_y
        sx = np.interp(x, (mx, 1 - mx), (0, self.screen_w - 1))
        sy = np.interp(y, (my, 1 - my), (0, self.screen_h - 1))
        return np.array([sx, sy])

    def position_at(self, t: float) -> np.ndarray | None:
        """Latest recorded cursor position at or before time t."""
        best = None
        for ts, pos in self._history:
            if ts > t:
                break
            best = pos
        return best if best is not None else (self._history[0][1] if self._history else None)

    def execute(self, intents: list[Intent], t: float):
        for intent in intents:
            kind = intent.kind
            if kind is IntentKind.MOVE:
                self._move(intent, t)
            elif kind is IntentKind.CLICK:
                self._click(intent, t)
            elif kind is IntentKind.RIGHT_CLICK:
                pyautogui.rightClick(*self._press_position(intent, t))
            elif kind is IntentKind.MOUSE_DOWN:
                pyautogui.mouseDown(*self._press_position(intent, t))
                self._button_down = True
            elif kind is IntentKind.MOUSE_UP:
                self.release()
            elif kind is IntentKind.SCROLL:
                self._scroll(intent.amount)

    def release(self):
        if self._button_down:
            pyautogui.mouseUp()
            self._button_down = False

    def _click(self, intent: Intent, t: float):
        pos = self._press_position(intent, t)
        if not pos:
            return
        pos = np.array(pos)
        count = 1
        if self._last_click is not None:
            last_t, last_pos, last_count = self._last_click
            if (t - last_t <= self.cfg.double_click_s
                    and np.linalg.norm(pos - last_pos) <= self.cfg.double_click_px):
                # Second pinch lands on the same target as the first, even if the hand drifted.
                pos, count = last_pos, last_count + 1
        click_at(float(pos[0]), float(pos[1]), count)
        self._last_click = (t, pos, count)

    def _press_position(self, intent: Intent, t: float) -> tuple:
        # Closing the pinch drags the fingertip; press where the cursor was just before.
        start = intent.at if intent.at is not None else t
        pos = self.position_at(start - self.cfg.click_rewind_s)
        return () if pos is None else (float(pos[0]), float(pos[1]))

    def _move(self, intent: Intent, t: float):
        if self._history and t - self._history[-1][0] > self.cfg.reset_after_s:
            self._filter.reset()
            self._history.clear()

        pos = self._filter(self.to_screen(intent.x, intent.y), t)
        self._history.append((t, pos))
        while self._history and t - self._history[0][0] > HISTORY_S:
            self._history.popleft()
        if self._button_down:
            # macOS only delivers drag events (e.g. moving windows) via dragTo. The button
            # must be explicit: with mouseDownUp=False pyautogui skips the step that turns
            # its default "primary" into "left", and the macOS backend then asserts.
            pyautogui.dragTo(*pos, button="left", mouseDownUp=False)
        else:
            pyautogui.moveTo(*pos)

    def _scroll(self, amount: float):
        self._scroll_remainder += amount
        lines = int(self._scroll_remainder)
        if lines:
            pyautogui.scroll(lines)
            self._scroll_remainder -= lines
