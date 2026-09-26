"""Synthetic hands for tests: an upright right hand built in pixel space."""

import numpy as np

from handsfree.tracker import Hand

W, H = 640, 480

# Finger joints (MCP, PIP, DIP, TIP) by x position; palm size (wrist → middle MCP) is 100 px.
FINGERS = {"index": (5, 290), "middle": (9, 320), "ring": (13, 345), "pinky": (17, 368)}


def make_hand(up=("index",), pinch=None, gap=5.0, scale=1.0, dy=0.0, dx=0.0,
              handedness="Right"):
    """`up` lists extended fingers. `pinch` = "index" | "middle" puts the thumb tip on that
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
    p = wrist + (p - wrist) * scale + (dx, dy)
    return Hand(np.c_[p / (W, H), np.zeros(21)].astype(np.float32), p.astype(np.float32),
                handedness, 1.0)


def fist(scale=1.0):
    hand = make_hand(up=(), scale=scale)
    thumb_on_index = hand.pixels[8] + (8 * scale, 0)  # thumb resting next to curled index tip
    hand.pixels[4] = thumb_on_index
    hand.landmarks[4, :2] = thumb_on_index / (W, H)
    return hand


def mirrored(hand: Hand) -> Hand:
    """The same pose as seen for the other hand (mirrored about the wrist)."""
    p = hand.pixels.copy()
    p[:, 0] = 2 * p[0, 0] - p[:, 0]
    other = "Left" if hand.handedness == "Right" else "Right"
    return Hand(np.c_[p / (W, H), np.zeros(21)].astype(np.float32), p, other, 1.0)
