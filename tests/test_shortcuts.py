import pytest

from handsfree.config import ShortcutConfig
from handsfree.shortcuts import ShortcutEngine

DT = 1 / 30
BINDINGS = {"fist": {"action": "toggle_pause"}, "rock": {"keys": ["command", "shift", "3"]}}


@pytest.fixture
def engine():
    return ShortcutEngine(ShortcutConfig(min_confidence=0.8, hold_s=0.4, cooldown_s=1.0,
                                         bindings=BINDINGS))


def feed(engine, label, conf, seconds, t0):
    events, t = [], t0
    for _ in range(round(seconds / DT)):
        events += engine.update(label, conf, t)
        t += DT
    return events, t


def test_fires_once_after_hold(engine):
    events, _ = feed(engine, "rock", 0.95, 2.0, 0.0)
    assert [e.gesture for e in events] == ["rock"]


def test_short_pose_does_not_fire(engine):
    events, _ = feed(engine, "rock", 0.95, 0.3, 0.0)
    assert events == []


def test_low_confidence_or_unbound_does_not_fire(engine):
    assert feed(engine, "rock", 0.6, 2.0, 0.0)[0] == []
    assert feed(engine, "open_palm", 0.99, 2.0, 0.0)[0] == []


def test_must_release_and_respect_cooldown(engine):
    _, t = feed(engine, "rock", 0.95, 0.6, 0.0)  # fires at ~0.4 s
    _, t = feed(engine, None, 0.0, 0.1, t)
    again, t = feed(engine, "rock", 0.95, 0.6, t)  # within 1 s cooldown
    assert again == []
    _, t = feed(engine, None, 0.0, 0.5, t)
    later, _ = feed(engine, "rock", 0.95, 0.6, t)
    assert [e.gesture for e in later] == ["rock"]


def test_hold_through_cooldown_fires_when_it_ends(engine):
    _, t = feed(engine, "fist", 0.95, 0.5, 0.0)  # fires at ~0.4 s
    _, t = feed(engine, None, 0.0, 0.2, t)
    events, _ = feed(engine, "fist", 0.95, 1.5, t)
    assert [e.gesture for e in events] == ["fist"]


def test_switching_gesture_restarts_hold(engine):
    _, t = feed(engine, "rock", 0.95, 0.3, 0.0)
    events, _ = feed(engine, "fist", 0.95, 0.3, t)
    assert events == []
    assert engine.progress(t + 0.3) > 0
