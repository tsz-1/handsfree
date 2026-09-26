from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True


@dataclass
class TrackerConfig:
    model_path: str = "models/hand_landmarker.task"
    num_hands: int = 1
    min_detection_confidence: float = 0.8
    min_tracking_confidence: float = 0.5


@dataclass
class MouseConfig:
    margin_x: float = 0.15
    margin_y: float = 0.20
    min_cutoff: float = 0.5
    beta: float = 0.005
    d_cutoff: float = 1.0
    click_rewind_s: float = 0.15
    reset_after_s: float = 0.5
    double_click_s: float = 0.8
    double_click_px: float = 40.0


@dataclass
class GestureConfig:
    pinch_enter: float = 0.30
    pinch_exit: float = 0.40
    min_finger_extension: float = 1.2
    hold_s: float = 0.3
    scroll_deadzone: float = 0.15
    scroll_speed: float = 40.0
    scroll_grace_s: float = 0.3


@dataclass
class ShortcutConfig:
    enabled: bool = True
    model_path: str = "models/gesture_classifier.joblib"
    smoothing: float = 0.6
    min_confidence: float = 0.8
    hold_s: float = 0.4
    cooldown_s: float = 1.0
    max_speed: float = 1.5
    bindings: dict = field(default_factory=dict)


@dataclass
class UIConfig:
    show_window: bool = True
    always_on_top: bool = True


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    shortcuts: ShortcutConfig = field(default_factory=ShortcutConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def _build(cls, data: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        kwargs[name] = _build(type(default), value or {}) if is_dataclass(default) else value
    return cls(**kwargs)


def load_config(path: str | Path | None) -> Config:
    if path is None or not Path(path).exists():
        return Config()
    with open(path) as f:
        return _build(Config, yaml.safe_load(f) or {})
