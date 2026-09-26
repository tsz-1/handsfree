import numpy as np
from handgen import H, W, make_hand

from handsfree.hardneg import HardNegativeRecorder
from handsfree.shortcuts import MotionGate, hand_in_frame


def test_hand_in_frame_rejects_partially_visible_hand():
    assert hand_in_frame(make_hand(), W, H)
    assert not hand_in_frame(make_hand(dx=-300), W, H)  # thumb pushed off the left edge
    assert not hand_in_frame(make_hand(dy=100), W, H)  # wrist below the bottom edge


def test_motion_gate_blocks_fast_hand_and_recovers():
    gate = MotionGate(max_speed=1.5)
    assert not gate.update(None, 0.0)
    gate.update(make_hand(), 0.0)
    # 60 px per frame at 30 fps with a 100 px palm = 18 palm lengths/s
    moving = [gate.update(make_hand(dx=60 * i), i / 30) for i in range(1, 6)]
    assert not any(moving)
    resting = [gate.update(make_hand(dx=300), 0.2 + i / 30) for i in range(1, 15)]
    assert resting[-1]


def test_hard_negative_recorder_saves_window_as_none(tmp_path):
    rec = HardNegativeRecorder(tmp_path, window_s=1.0)
    for i in range(60):  # 2 s of frames; only the last second should be kept
        rec.push(make_hand(dx=i), i / 30, (W, H))
    rec.push(None, 2.0, (W, H))
    assert rec.save_recent() == 30
    d = np.load(rec.path)
    assert set(d["labels"]) == {"none"} and d["pixels"].shape == (30, 21, 2)
    assert rec.save_recent() == 0  # buffer was consumed
    rec.push(make_hand(), 3.0, (W, H))
    assert rec.save_recent() == 1
    assert len(np.load(rec.path)["labels"]) == 31
