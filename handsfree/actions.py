import numpy as np
import pyautogui

from handsfree.config import MouseConfig
from handsfree.gestures import Intent, IntentKind

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


class MouseController:
    def __init__(self, cfg: MouseConfig):
        self.cfg = cfg
        self.screen_w, self.screen_h = pyautogui.size()
        self._pos: np.ndarray | None = None

    def to_screen(self, x: float, y: float) -> np.ndarray:
        """Map a point in the active region of the frame (normalized) to screen pixels."""
        mx, my = self.cfg.margin_x, self.cfg.margin_y
        sx = np.interp(x, (mx, 1 - mx), (0, self.screen_w - 1))
        sy = np.interp(y, (my, 1 - my), (0, self.screen_h - 1))
        return np.array([sx, sy])

    def execute(self, intents: list[Intent]):
        for intent in intents:
            if intent.kind is IntentKind.MOVE:
                target = self.to_screen(intent.x, intent.y)
                if self._pos is None:
                    self._pos = target
                else:
                    self._pos = self._pos + (target - self._pos) / self.cfg.smoothing
                pyautogui.moveTo(*self._pos)
            elif intent.kind is IntentKind.CLICK:
                pyautogui.click()
