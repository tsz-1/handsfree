# HandsFree

Touchless computer control with hand gestures, using a regular webcam.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
python tools/download_model.py
python -m handsfree            # or: python -m handsfree --camera 0
```

On macOS, grant your terminal **Camera** and **Accessibility** permissions
(System Settings → Privacy & Security), otherwise the camera or mouse control won't work.

## Gestures

| Gesture | Action |
| --- | --- |
| Index finger up | Move cursor |
| Thumb–index pinch, quick release | Left click (twice for double-click) |
| Thumb–index pinch, hold > 0.3 s | Drag; release to drop |
| Thumb–middle pinch | Right click |
| Index + middle up, move hand up/down | Scroll (farther from start = faster) |

### Gesture shortcuts

A classifier trained on your own recordings recognizes static poses and maps them to actions
(configurable in `config.yaml`):

| Pose | Default action |
| --- | --- |
| Fist (hold) | Pause / resume control |
| Thumbs up | Play / pause media |
| Rock 🤘 | Screenshot |
| Call 🤙 | Mission Control |

```bash
python tools/record.py   # record labeled poses (do at least two sessions)
python tools/train.py    # compare models with leave-one-session-out CV, save the best
```

Press `q` in the preview window to quit. Settings live in `config.yaml`.
Use `--dry-run` to see recognized gestures without controlling the mouse.
See [docs/TUNING.zh-CN.md](docs/TUNING.zh-CN.md) for a tuning guide.
