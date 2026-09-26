import numpy as np
import pytest

from handsfree.config import GestureConfig
from handsfree.gestures import GestureInterpreter, IntentKind, Mode, PinchDetector
from handsfree.tracker import Hand

W, H = 640, 480
DT = 1 / 30

# Finger joints (MCP, PIP, DIP, TIP) by x position; palm size (wrist → middle MCP) is 100 px.
FINGERS = {"index": (5, 290), "middle": (9, 320), "ring": (13, 345), "pinky": (17, 368)}


def make_hand(up=("index",), pinch=None, gap=5.0, scale=1.0, dy=0.0):
    """Upright hand in pixel space. `pinch` = "index" | "middle" puts the thumb tip on that
    finger's (bent) tip, `gap` px apart. `scale` shrinks the hand around the wrist."""
    p = np.zeros((21, 2))
    wrist = np.array([320.0, 400.0])
    p[0] = wrist
    for name, (mcp, x) in FINGERS.items():
        if name == pinch:
            ys = (300, 270, 255, 240)  # bent toward the thumb, tip still ~1.6 palms from wrist
        elif name in up:
            ys = (300, 260, 230, 200)
        else:
            ys = (300, 320, 335, 330)  # curled into the palm
        for j, y in enumerate(ys):
            p[mcp + j] = (x, y)
    p[1:5] = [(300, 370), (280, 350), (265, 330), (250, 320)]  # thumb away from fingers
    if pinch:
        tip = FINGERS[pinch][0] + 3
        p[4] = p[tip] + (gap, 0)
    p = wrist + (p - wrist) * scale + (0, dy)
    return Hand(np.c_[p / (W, H), np.zeros(21)].astype(np.float32), p.astype(np.float32),
                "Right", 1.0)


def fist(scale=1.0):
    hand = make_hand(up=(), scale=scale)
    thumb_on_index = hand.pixels[8] + (8 * scale, 0)  # thumb resting next to curled index tip
    hand.pixels[4] = thumb_on_index
    hand.landmarks[4, :2] = thumb_on_index / (W, H)
    return hand


class Clock:
    def __init__(self, g):
        self.g, self.t = g, 0.0

    def feed(self, hand, frames=1):
        out = []
        for _ in range(frames):
            out += self.g.update(hand, self.t)
            self.t += DT
        return out


@pytest.fixture
def clock():
    return Clock(GestureInterpreter(GestureConfig()))


def kinds(intents):
    return [i.kind for i in intents if i.kind is not IntentKind.MOVE]


def test_pointing_moves_cursor(clock):
    intents = clock.feed(make_hand())
    assert [i.kind for i in intents] == [IntentKind.MOVE]


def test_quick_pinch_clicks_once_on_release(clock):
    clock.feed(make_hand(), 5)
    onset = clock.t
    during = clock.feed(make_hand(pinch="index"), 4)
    release = clock.feed(make_hand())
    assert kinds(during) == []
    assert kinds(release) == [IntentKind.CLICK]
    assert release[0].at == pytest.approx(onset)


def test_long_pinch_drags_instead_of_repeat_clicking(clock):
    clock.feed(make_hand(), 3)
    held = clock.feed(make_hand(pinch="index"), 30)  # 1 s
    assert kinds(held) == [IntentKind.MOUSE_DOWN]
    assert clock.g.state.mode is Mode.DRAG
    assert any(i.kind is IntentKind.MOVE for i in held)
    assert kinds(clock.feed(make_hand())) == [IntentKind.MOUSE_UP]


def test_hysteresis_holds_pinch_between_thresholds(clock):
    clock.feed(make_hand(pinch="index", gap=5), 2)  # enter (ratio ~0.05)
    wobble = clock.feed(make_hand(pinch="index", gap=35), 3)  # 0.35: between enter and exit
    assert clock.g.state.mode is Mode.PINCH
    assert kinds(wobble) == []


def test_gap_between_thresholds_does_not_start_pinch(clock):
    clock.feed(make_hand(pinch="index", gap=35), 10)
    assert clock.g.state.mode is Mode.IDLE


def test_pinch_is_scale_invariant(clock):
    clock.feed(make_hand(scale=0.5), 3)
    clock.feed(make_hand(pinch="index", scale=0.5), 3)
    assert kinds(clock.feed(make_hand(scale=0.5))) == [IntentKind.CLICK]


def test_fist_does_not_click(clock):
    intents = clock.feed(make_hand(), 3) + clock.feed(fist(), 10) + clock.feed(make_hand(), 3)
    assert kinds(intents) == []


def test_thumb_middle_pinch_right_clicks(clock):
    clock.feed(make_hand(up=("index",)), 3)
    clock.feed(make_hand(up=("index",), pinch="middle"), 4)
    assert kinds(clock.feed(make_hand())) == [IntentKind.RIGHT_CLICK]


def test_two_fingers_scroll_with_dead_zone(clock):
    two = ("index", "middle")
    assert kinds(clock.feed(make_hand(up=two), 5)) == []
    assert kinds(clock.feed(make_hand(up=two, dy=-10), 5)) == []  # 0.1 palm: inside dead zone
    up = clock.feed(make_hand(up=two, dy=-50), 10)
    assert up and all(i.kind is IntentKind.SCROLL and i.amount > 0 for i in up)
    down = clock.feed(make_hand(up=two, dy=50), 10)
    assert down and all(i.amount < 0 for i in down)


def test_losing_hand_mid_drag_releases_button(clock):
    clock.feed(make_hand(pinch="index"), 20)
    assert clock.g.state.mode is Mode.DRAG
    assert kinds(clock.feed(None)) == [IntentKind.MOUSE_UP]
    assert clock.g.state.mode is Mode.IDLE


def test_pinch_detector_hysteresis():
    d = PinchDetector(enter=0.3, exit=0.4)
    assert [d.update(r) for r in (0.35, 0.25, 0.35, 0.39, 0.41, 0.35)] == [
        False, True, True, True, False, False
    ]
    assert d.update(0.1, may_enter=False) is False
