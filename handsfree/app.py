import argparse
import time

import cv2

from handsfree.actions import MouseController
from handsfree.config import Config, load_config
from handsfree.gestures import GestureInterpreter, GestureState, Intent, IntentKind, Mode
from handsfree.tracker import (
    HAND_CONNECTIONS,
    INDEX_TIP,
    MIDDLE_TIP,
    THUMB_TIP,
    Hand,
    HandTracker,
)

WINDOW = "HandsFree"
MAGENTA, GREEN, WHITE = (255, 0, 255), (0, 255, 0), (255, 255, 255)
YELLOW, GRAY = (0, 255, 255), (150, 150, 150)


def open_camera(cfg: Config, max_index: int = 4) -> cv2.VideoCapture:
    candidates = [cfg.camera.index] + [i for i in range(max_index) if i != cfg.camera.index]
    for index in candidates:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
            ok, _ = cap.read()
            if ok:
                if index != cfg.camera.index:
                    print(f"Camera {cfg.camera.index} unavailable, using camera {index} instead.")
                return cap
        cap.release()
    raise RuntimeError(
        f"No working camera found (tried indices {candidates}). On macOS, allow Camera access "
        "for your terminal app in System Settings → Privacy & Security → Camera, then restart it."
    )


def scroll_label(state: GestureState, cfg: Config) -> str:
    if state.mode is not Mode.SCROLL:
        return ""
    offset = state.scroll_offset
    if abs(offset) <= cfg.gestures.scroll_deadzone:
        direction = "HOLD"
    else:
        direction = "UP" if (offset > 0) == (cfg.gestures.scroll_speed > 0) else "DOWN"
    return f" {direction} ({offset:+.2f})"


def draw_overlay(frame, cfg: Config, hand: Hand | None, state: GestureState, fps: float):
    h, w = frame.shape[:2]
    mx, my = int(cfg.mouse.margin_x * w), int(cfg.mouse.margin_y * h)
    cv2.rectangle(frame, (mx, my), (w - mx, h - my), MAGENTA, 2)

    if hand is not None:
        pts = hand.pixels.astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), WHITE, 1)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, MAGENTA, cv2.FILLED)

        thumb = pts[THUMB_TIP]
        for tip, active_modes in ((INDEX_TIP, (Mode.PINCH, Mode.DRAG)),
                                  (MIDDLE_TIP, (Mode.RIGHT_PINCH,))):
            active = state.mode in active_modes
            cv2.line(frame, tuple(thumb), tuple(pts[tip]), GREEN if active else MAGENTA, 2)
            if active:
                cv2.circle(frame, tuple((thumb + pts[tip]) // 2), 12, GREEN, cv2.FILLED)

    m = state.metrics
    if state.scroll_anchor_y is not None and m is not None:
        ay = int(state.scroll_anchor_y)
        band = int(cfg.gestures.scroll_deadzone * m.palm)
        cv2.line(frame, (0, ay), (w, ay), YELLOW, 1)
        for edge in (ay - band, ay + band):
            cv2.line(frame, (0, edge), (w, edge), GRAY, 1)
        if hand is not None:
            tip = tuple(hand.pixels[INDEX_TIP].astype(int))
            cv2.line(frame, (tip[0], ay), tip, YELLOW, 2)

    lines = [f"Mode: {state.mode.value}{scroll_label(state, cfg)}"]
    if hand is not None and m is not None:
        lines += [
            f"L pinch {m.left_ratio:.2f}  R pinch {m.right_ratio:.2f}",
            f"Index ext {m.index_ext:.2f}  Middle ext {m.middle_ext:.2f}",
        ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 35 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAGENTA, 2)

    cv2.putText(frame, f"FPS: {fps:.0f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)


def log_intents(intents: list[Intent], t: float):
    for intent in intents:
        if intent.kind is IntentKind.MOVE:
            continue
        extra = f" {intent.amount:+.2f}" if intent.kind is IntentKind.SCROLL else ""
        print(f"[{t:9.3f}] {intent.kind.value}{extra}")


def run(cfg: Config, dry_run: bool = False, verbose: bool = False):
    cap = open_camera(cfg)
    tracker = HandTracker(cfg.tracker)
    gestures = GestureInterpreter(cfg.gestures)
    mouse = MouseController(cfg.mouse)

    fps, last_t = 0.0, time.perf_counter()
    last_mode = Mode.IDLE
    print("HandsFree running. Press 'q' in the preview window to quit.")
    if dry_run:
        print("Dry run: gestures are recognized but the mouse is not controlled.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: cannot read from camera. Check the camera index and permissions.")
                time.sleep(1)
                continue
            if cfg.camera.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            hands = tracker.detect(frame, int(now * 1000))
            hand = hands[0] if hands else None
            intents = gestures.update(hand, now)
            if verbose:
                if gestures.state.mode is not last_mode:
                    last_mode = gestures.state.mode
                    print(f"[{now:9.3f}] -> {last_mode.value}")
                log_intents(intents, now)
            if not dry_run:
                mouse.execute(intents, now)

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt

            if cfg.ui.show_window:
                draw_overlay(frame, cfg, hand, gestures.state, fps)
                cv2.imshow(WINDOW, frame)
                if cfg.ui.always_on_top:
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        mouse.release()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Touchless computer control with hand gestures")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config")
    parser.add_argument("--camera", type=int, help="Override the camera index")
    parser.add_argument("--dry-run", action="store_true",
                        help="Recognize gestures without controlling the mouse (implies --verbose)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print click, drag, and scroll events to the terminal")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg.camera.index = args.camera
    run(cfg, dry_run=args.dry_run, verbose=args.verbose or args.dry_run)


if __name__ == "__main__":
    main()
