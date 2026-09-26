import argparse
import time

import cv2

from handsfree.actions import MouseController
from handsfree.config import Config, load_config
from handsfree.gestures import GestureInterpreter, GestureState
from handsfree.tracker import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP, Hand, HandTracker

WINDOW = "HandsFree"
MAGENTA, GREEN, WHITE = (255, 0, 255), (0, 255, 0), (255, 255, 255)


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

        thumb, index = pts[THUMB_TIP], pts[INDEX_TIP]
        cv2.line(frame, tuple(thumb), tuple(index), MAGENTA, 2)
        if state.pinching:
            mid = (thumb + index) // 2
            cv2.circle(frame, tuple(mid), 12, GREEN, cv2.FILLED)
        cv2.putText(frame, f"Pinch: {state.pinch_distance:.0f}px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, MAGENTA, 2)

    cv2.putText(frame, f"FPS: {fps:.0f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)


def run(cfg: Config):
    cap = open_camera(cfg)
    tracker = HandTracker(cfg.tracker)
    gestures = GestureInterpreter(cfg.gestures)
    mouse = MouseController(cfg.mouse)

    fps, last_t = 0.0, time.perf_counter()
    print("HandsFree running. Press 'q' in the preview window to quit.")
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
            mouse.execute(gestures.update(hand, now))

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
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Touchless computer control with hand gestures")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config")
    parser.add_argument("--camera", type=int, help="Override the camera index")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg.camera.index = args.camera
    run(cfg)


if __name__ == "__main__":
    main()
