"""Record labeled hand landmarks for training the gesture classifier.

Keys (in the preview window):
  0-9    start a clip of the gesture with that number; form the pose, then hold it
         while moving the hand around (10-15 s per clip, 3-4 clips per gesture)
  space  stop the clip (the session is saved every time you stop)
  u      undo: discard the clip being recorded, or the last one (repeatable)
  q      quit

Each run is one session file in data/gestures/. Record at least two sessions (e.g. different
lighting, seat, or day) so tools/train.py can test on a session it never trained on.
"""

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np

from handsfree.app import open_camera
from handsfree.classifier import DEFAULT_LABELS
from handsfree.config import load_config
from handsfree.tracker import HAND_CONNECTIONS, HandTracker

DATA_DIR = Path("data/gestures")
WINDOW = "HandsFree recorder"
TIPS = {
    "none": "point, pinch, two fingers, relaxed hand, transitions",
    "fist": "tight fist, knuckles to camera",
    "open_palm": "all five fingers spread",
    "thumbs_up": "thumb up, other fingers curled",
    "rock": "index + pinky up (horns)",
    "call": "thumb + pinky out (shaka)",
}


def save(path: Path, samples: list[dict], frame_size: tuple[int, int]):
    if not samples:
        if path.exists():
            path.unlink()
            print(f"No samples left; removed {path}")
        return
    np.savez_compressed(
        path,
        pixels=np.stack([s["pixels"] for s in samples]),
        landmarks=np.stack([s["landmarks"] for s in samples]),
        handedness=np.array([s["handedness"] for s in samples]),
        labels=np.array([s["label"] for s in samples]),
        timestamps=np.array([s["t"] for s in samples]),
        frame_size=np.array(frame_size),
    )
    print(f"Saved {len(samples)} samples to {path}")


def end_clip(samples: list[dict], start: int, t0: float, t1: float, trim: float):
    """Drop the frames where the hand was still forming or leaving the pose.

    Transitions are only unwanted in real gestures; for "none" they are useful negatives.
    """
    clip = samples[start:]
    if not clip or clip[0]["label"] == "none" or trim <= 0:
        return
    kept = [s for s in clip if t0 + trim <= s["t"] <= t1 - trim]
    samples[start:] = kept
    print(f"Clip {clip[0]['label']}: kept {len(kept)}/{len(clip)} samples after trimming")


def draw(frame, hand, labels, counts, current, recording, clip_s):
    h, w = frame.shape[:2]
    if hand is not None:
        pts = hand.pixels.astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 1)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, (255, 0, 255), cv2.FILLED)

    for i, label in enumerate(labels):
        active = label == current and recording
        color = (0, 0, 255) if active else (255, 255, 255)
        cv2.putText(frame, f"{i} {label}: {counts.get(label, 0)}", (w - 210, 30 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if recording:
        cv2.circle(frame, (25, 30), 10, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, f"REC {current} {clip_s:4.1f}s", (45, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, TIPS.get(current, ""), (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "Vary angle, distance and position", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if hand is None:
            cv2.putText(frame, "No hand", (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "0-9 record, space stop, u undo, q quit", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--every", type=int, default=2,
                        help="Keep every Nth frame; consecutive frames are nearly identical")
    parser.add_argument("--trim", type=float, default=0.3,
                        help="Seconds dropped at the start and end of each clip (not for 'none')")
    args = parser.parse_args()
    if len(args.labels) > 10:
        parser.error("At most 10 labels (keys 0-9)")

    cfg = load_config(args.config)
    cap = open_camera(cfg)
    tracker = HandTracker(cfg.tracker)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.npz"

    samples: list[dict] = []
    clip_starts: list[int] = []  # index in `samples` where each recorded clip begins
    counts: dict[str, int] = {}
    current, recording, frame_i, clip_t0 = args.labels[0], False, 0, 0.0
    frame_size = (cfg.camera.width, cfg.camera.height)
    print(__doc__)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if cfg.camera.mirror:
                frame = cv2.flip(frame, 1)
            frame_size = (frame.shape[1], frame.shape[0])
            now = time.perf_counter()
            hands = tracker.detect(frame, int(now * 1000))
            hand = hands[0] if hands else None

            frame_i += 1
            if recording and hand is not None and frame_i % args.every == 0:
                samples.append({"pixels": hand.pixels, "landmarks": hand.landmarks,
                                "handedness": hand.handedness, "label": current, "t": now})
                counts[current] = counts.get(current, 0) + 1

            draw(frame, hand, args.labels, counts, current, recording, now - clip_t0)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and recording:
                recording = False
                end_clip(samples, clip_starts[-1], clip_t0, now, args.trim)
                counts = collections.Counter(s["label"] for s in samples)
                save(path, samples, frame_size)
            elif key == ord("u") and clip_starts:
                start = clip_starts.pop()
                dropped = len(samples) - start
                del samples[start:]
                counts = collections.Counter(s["label"] for s in samples)
                recording = False
                print(f"Undo: discarded {dropped} samples")
                save(path, samples, frame_size)
            elif ord("0") <= key <= ord("9") and key - ord("0") < len(args.labels):
                if recording:
                    end_clip(samples, clip_starts[-1], clip_t0, now, args.trim)
                    counts = collections.Counter(s["label"] for s in samples)
                current, recording, clip_t0 = args.labels[key - ord("0")], True, now
                clip_starts.append(len(samples))
    finally:
        if recording:
            end_clip(samples, clip_starts[-1], clip_t0, time.perf_counter(), args.trim)
        save(path, samples, frame_size)
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
