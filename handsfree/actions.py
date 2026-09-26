from collections import deque

import numpy as np
import pyautogui

from handsfree.config import MouseConfig
from handsfree.filters import OneEuroFilter
from handsfree.gestures import Intent, IntentKind

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

HISTORY_S = 1.0


class MouseController:
    def __init__(self, cfg: MouseConfig):
        self.cfg = cfg
        self.screen_w, self.screen_h = pyautogui.size()
        self._filter = OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
        self._history: deque[tuple[float, np.ndarray]] = deque()

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
            if intent.kind is IntentKind.MOVE:
                self._move(intent, t)
            elif intent.kind is IntentKind.CLICK:
                self._click(t)

    def _move(self, intent: Intent, t: float):
        if self._history and t - self._history[-1][0] > self.cfg.reset_after_s:
            self._filter.reset()
            self._history.clear()

        pos = self._filter(self.to_screen(intent.x, intent.y), t)
        self._history.append((t, pos))
        while self._history and t - self._history[0][0] > HISTORY_S:
            self._history.popleft()
        pyautogui.moveTo(*pos)

    def _click(self, t: float):
        # Closing the pinch drags the fingertip; click where the cursor was just before.
        pos = self.position_at(t - self.cfg.click_rewind_s)
        if pos is not None:
            pyautogui.click(*pos)
        else:
            pyautogui.click()
