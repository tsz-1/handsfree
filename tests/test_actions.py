import numpy as np
import pytest

from handsfree import actions
from handsfree.config import MouseConfig
from handsfree.gestures import Intent, IntentKind


@pytest.fixture
def mouse(monkeypatch):
    calls = {"move": [], "click": []}
    monkeypatch.setattr(actions.pyautogui, "size", lambda: (1000, 1000))
    monkeypatch.setattr(actions.pyautogui, "moveTo", lambda x, y: calls["move"].append((x, y)))
    monkeypatch.setattr(actions.pyautogui, "click", lambda *p: calls["click"].append(p))
    cfg = MouseConfig(margin_x=0.0, margin_y=0.0, min_cutoff=1000.0, beta=0.0)  # ~no smoothing
    ctrl = actions.MouseController(cfg)
    ctrl.calls = calls
    return ctrl


def test_maps_active_region_to_full_screen():
    ctrl = actions.MouseController.__new__(actions.MouseController)
    ctrl.cfg = MouseConfig(margin_x=0.25, margin_y=0.25)
    ctrl.screen_w, ctrl.screen_h = 1001, 1001
    np.testing.assert_allclose(ctrl.to_screen(0.25, 0.75), [0, 1000])
    np.testing.assert_allclose(ctrl.to_screen(0.0, 1.0), [0, 1000])  # clamped


def test_click_rewinds_past_pinch_drift(mouse):
    # Steady at x=0.2, then the fingertip drifts to x=0.4 while the pinch closes.
    dt = 1 / 30
    drift_frames = int(mouse.cfg.click_rewind_s / dt) - 1
    t = 0.0
    for _ in range(10):
        mouse.execute([Intent(IntentKind.MOVE, 0.2, 0.5)], t)
        t += dt
    for _ in range(drift_frames):
        mouse.execute([Intent(IntentKind.MOVE, 0.4, 0.5)], t)
        t += dt
    mouse.execute([Intent(IntentKind.CLICK)], t)

    (x, _), = mouse.calls["click"]
    assert x == pytest.approx(0.2 * 999, abs=5)


def test_filter_resets_after_hand_lost(mouse):
    mouse.execute([Intent(IntentKind.MOVE, 0.0, 0.0)], 0.0)
    mouse.execute([Intent(IntentKind.MOVE, 1.0, 1.0)], 5.0)
    assert mouse.calls["move"][-1] == pytest.approx((999, 999))
