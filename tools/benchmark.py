"""Measure end-to-end latency, FPS, and cursor jitter with the live camera.

Phase 1: use the system normally for a while; per-frame stage timings are recorded.
Phase 2: hold your index finger still; the raw fingertip trajectory is mapped to screen pixels
         and smoothed three ways on identical input to compare jitter.

Nothing controls the mouse during the benchmark. Results go to reports/benchmark.json and are
printed as a Markdown table for the README.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from handsfree.actions import MouseController
from handsfree.app import draw_overlay, load_classifier, open_camera
from handsfree.config import load_config
from handsfree.filters import OneEuroFilter
from handsfree.gestures import GestureInterpreter
from handsfree.tracker import INDEX_TIP, HandTracker

WINDOW = "HandsFree benchmark"
GREEN, RED, WHITE = (0, 255, 0), (0, 0, 255), (255, 255, 255)


def banner(frame, lines, color):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 80 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def ema(points, divisor):
    out, p = [], points[0].copy()
    for x in points:
        p = p + (x - p) / divisor
        out.append(p.copy())
    return np.array(out)


def one_euro(points, ts, cfg):
    f = OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
    return np.array([f(x, t) for x, t in zip(points, ts)])


def jitter(points):
    """RMS deviation from the mean position, in pixels."""
    return float(np.sqrt(((points - points.mean(0)) ** 2).sum(1).mean()))


def synthetic_lag(cfg, speed_px_s=1800.0, fps=30):
    """Lag behind a constant-velocity target after 2 s (no camera involved)."""
    n = 2 * fps
    ts = np.arange(n) / fps
    ramp = np.stack([ts * speed_px_s, np.zeros(n)], 1)
    return {
        "ema_div8_px": float(abs(ema(ramp, 8)[-1, 0] - ramp[-1, 0])),
        "one_euro_px": float(abs(one_euro(ramp, ts, cfg)[-1, 0] - ramp[-1, 0])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seconds", type=float, default=20, help="Phase 1 duration")
    parser.add_argument("--still", type=float, default=5, help="Phase 2 hold duration")
    parser.add_argument("--out", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    cap = open_camera(cfg)
    tracker = HandTracker(cfg.tracker)
    gestures = GestureInterpreter(cfg.gestures)
    classifier = load_classifier(cfg)
    mouse = MouseController(cfg.mouse)  # only used for the screen mapping

    stages = {"capture": [], "detect": [], "logic": [], "total": []}
    tips, tip_ts = [], []
    phase, phase_start = 1, time.perf_counter()
    hand_frames = 0
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                continue
            if cfg.camera.mirror:
                frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame, int(t1 * 1000))
            t2 = time.perf_counter()
            hand = hands[0] if hands else None
            gestures.update(hand, t2)
            if classifier:
                classifier.predict(hand)
            if hand is not None:
                mouse._filter(mouse.to_screen(*hand.landmarks[INDEX_TIP, :2]), t2)
            t3 = time.perf_counter()

            elapsed = t3 - phase_start
            if phase == 1:
                stages["capture"].append(t1 - t0)
                stages["detect"].append(t2 - t1)
                stages["logic"].append(t3 - t2)
                stages["total"].append(t3 - t0)
                hand_frames += hand is not None
                draw_overlay(frame, cfg, hand, gestures.state, 1 / max(t3 - t0, 1e-6))
                banner(frame, [f"Phase 1/2: use it normally  {args.seconds - elapsed:4.1f}s"],
                       GREEN)
                if elapsed >= args.seconds:
                    phase, phase_start = 2, t3
            elif phase == 2:
                remaining = 3 - elapsed
                if remaining > 0:
                    banner(frame, ["Phase 2/2: point your index finger at the camera",
                                   f"and hold it STILL... {remaining:3.1f}"], RED)
                else:
                    banner(frame, [f"HOLD STILL  {args.still - (elapsed - 3):4.1f}s"], RED)
                    if hand is not None:
                        tips.append(mouse.to_screen(*hand.landmarks[INDEX_TIP, :2]))
                        tip_ts.append(t3)
                    if elapsed - 3 >= args.still:
                        break
            cv2.imshow(WINDOW, frame)
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    if len(stages["total"]) < 30 or len(tips) < 30:
        raise SystemExit("Not enough frames captured; run the benchmark again.")

    ms = {k: {"p50": float(np.percentile(v, 50) * 1000), "p95": float(np.percentile(v, 95) * 1000)}
          for k, v in stages.items()}
    fps = len(stages["total"]) / sum(stages["total"])

    pts, ts = np.array(tips), np.array(tip_ts)
    warm = ts > ts[0] + 1.0  # let the filters settle
    jit = {
        "raw_px": jitter(pts[warm]),
        "ema_div8_px": jitter(ema(pts, 8)[warm]),
        "one_euro_px": jitter(one_euro(pts, ts, cfg.mouse)[warm]),
    }
    lag = synthetic_lag(cfg.mouse)

    result = {
        "frames": len(stages["total"]),
        "hand_visible_fraction": hand_frames / len(stages["total"]),
        "fps": fps,
        "latency_ms": ms,
        "jitter": jit,
        "lag_at_1800px_s": lag,
        "screen": [mouse.screen_w, mouse.screen_h],
        "camera": [cfg.camera.width, cfg.camera.height],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n{len(stages['total'])} frames, {fps:.1f} FPS, hand visible "
          f"{100 * result['hand_visible_fraction']:.0f}% of the time\n")
    print("| Stage | p50 (ms) | p95 (ms) |\n| --- | --- | --- |")
    for k, v in ms.items():
        print(f"| {k} | {v['p50']:.1f} | {v['p95']:.1f} |")
    print("\n| Smoothing | Jitter, hand still (px RMS) | Lag at 1800 px/s (px, synthetic) |")
    print("| --- | --- | --- |")
    print(f"| None | {jit['raw_px']:.1f} | 0 |")
    print(f"| Exponential (÷8, prototype) | {jit['ema_div8_px']:.1f} | {lag['ema_div8_px']:.0f} |")
    print(f"| One Euro (this project) | {jit['one_euro_px']:.1f} | {lag['one_euro_px']:.0f} |")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
