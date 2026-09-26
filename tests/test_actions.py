import numpy as np
import pytest

from handsfree import actions
from handsfree.config import MouseConfig
from handsfree.gestures import Intent, IntentKind


@pytest.fixture
def mouse(monkeypatch):
    calls = {k: [] for k in ("move", "drag", "click", "right", "down", "up", "scroll")}
    pag = actions.pyautogui
    monkeypatch.setattr(pag, "size", lambda: (1000, 1000))
    monkeypatch.setattr(pag, "moveTo", lambda x, y: calls["move"].append((x, y)))
    monkeypatch.setattr(pag, "dragTo", lambda x, y, **kw: calls["drag"].append((x, y)))
    monkeypatch.setattr(pag, "click", lambda *p: calls["click"].append(p))
    monkeypatch.setattr(pag, "rightClick", lambda *p: calls["right"].append(p))
    monkeypatch.setattr(pag, "mouseDown", lambda *p: calls["down"].append(p))
    monkeypatch.setattr(pag, "mouseUp", lambda *p: calls["up"].append(p))
    monkeypatch.setattr(pag, "scroll", lambda n: calls["scroll"].append(n))
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


def test_press_uses_gesture_onset_not_release_time(mouse):
    for i in range(10):
        mouse.execute([Intent(IntentKind.MOVE, 0.2, 0.5)], i / 30)
    # No MOVEs while pinched; the click arrives on release, well after the rewind window.
    mouse.execute([Intent(IntentKind.CLICK, at=10 / 30)], 1.0)
    (x, _), = mouse.calls["click"]
    assert x == pytest.approx(0.2 * 999, abs=5)


def test_drag_uses_drag_events_and_releases(mouse):
    mouse.execute([Intent(IntentKind.MOVE, 0.5, 0.5)], 0.0)
    mouse.execute([Intent(IntentKind.MOUSE_DOWN, at=0.0)], 0.1)
    mouse.execute([Intent(IntentKind.MOVE, 0.6, 0.5)], 0.2)
    mouse.execute([Intent(IntentKind.MOUSE_UP)], 0.3)
    mouse.release()
    assert len(mouse.calls["down"]) == 1 and len(mouse.calls["drag"]) == 1
    assert len(mouse.calls["up"]) == 1


def test_scroll_accumulates_fractional_lines(mouse):
    for _ in range(14):
        mouse.execute([Intent(IntentKind.SCROLL, amount=0.25)], 0.0)
    assert mouse.calls["scroll"] == [1, 1, 1]


def test_filter_resets_after_hand_lost(mouse):
    mouse.execute([Intent(IntentKind.MOVE, 0.0, 0.0)], 0.0)
    mouse.execute([Intent(IntentKind.MOVE, 1.0, 1.0)], 5.0)
    assert mouse.calls["move"][-1] == pytest.approx((999, 999))
