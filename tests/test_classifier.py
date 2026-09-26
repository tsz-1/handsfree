import subprocess
import sys
from pathlib import Path

import numpy as np
from handgen import fist, make_hand, mirrored

from handsfree.classifier import GestureClassifier, hand_features

ROOT = Path(__file__).resolve().parents[1]


def features(hand):
    return hand_features(hand.pixels, hand.handedness)


def test_features_ignore_position_and_distance():
    base = features(make_hand(up=("index", "pinky")))
    np.testing.assert_allclose(features(make_hand(up=("index", "pinky"), dx=-80, dy=30)), base,
                               atol=1e-5)
    np.testing.assert_allclose(features(make_hand(up=("index", "pinky"), scale=0.5)), base,
                               atol=1e-5)


def test_left_hand_maps_onto_right_hand():
    hand = make_hand(up=("index", "pinky"))
    np.testing.assert_allclose(features(mirrored(hand)), features(hand), atol=1e-5)


def test_features_ignore_in_plane_rotation():
    hand = make_hand(up=("index", "pinky"))
    base = features(hand)
    for deg in (35, 90, -120):
        a = np.deg2rad(deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        p = (hand.pixels - hand.pixels[0]) @ rot.T + hand.pixels[0]
        np.testing.assert_allclose(hand_features(p.astype(np.float32)), base, atol=1e-4)


def test_different_poses_differ():
    assert np.linalg.norm(features(make_hand(up=())) - features(make_hand())) > 0.5


POSES = {
    "none": lambda: make_hand(),
    "fist": lambda: fist(),
    "open_palm": lambda: make_hand(up=("index", "middle", "ring", "pinky")),
    "rock": lambda: make_hand(up=("index", "pinky")),
}


def write_session(path: Path, rng: np.random.Generator, n: int = 60):
    pixels, labels = [], []
    for label, pose in POSES.items():
        for _ in range(n):
            hand = pose()
            scale, shift = rng.uniform(0.6, 1.2), rng.uniform(-60, 60, 2)
            p = (hand.pixels - hand.pixels[0]) * scale + hand.pixels[0] + shift
            pixels.append(p + rng.normal(0, 2, p.shape))
            labels.append(label)
    np.savez(path, pixels=np.array(pixels), handedness=np.array(["Right"] * len(labels)),
             labels=np.array(labels))


def test_train_script_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    for i in range(2):
        write_session(data / f"s{i}.npz", rng)
    model = tmp_path / "model.joblib"

    subprocess.run(
        [sys.executable, str(ROOT / "tools/train.py"), "--data", str(data), "--out", str(model),
         "--reports", str(tmp_path / "reports"), "--models", "logreg", "knn"],
        check=True, cwd=tmp_path, capture_output=True, text=True,
    )
    assert (tmp_path / "reports/confusion_matrix.png").exists()
    assert (tmp_path / "reports/metrics.json").exists()

    clf = GestureClassifier(model, smoothing=0.0)
    assert clf.predict(make_hand(up=("index", "pinky"), dx=40, scale=0.8))[0] == "rock"
    assert clf.predict(fist())[0] == "fist"
    assert clf.predict(None) == (None, 0.0)
