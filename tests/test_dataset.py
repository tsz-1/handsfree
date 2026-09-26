import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "tools/dataset.py"


def run(cwd, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


def make_session(root: Path, labels):
    d = root / "data/gestures"
    d.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    np.savez_compressed(d / "s1.npz", pixels=np.zeros((n, 21, 2)), landmarks=np.zeros((n, 21, 3)),
                        handedness=np.array(["Right"] * n), labels=np.array(labels),
                        timestamps=np.arange(n, dtype=float), frame_size=np.array([640, 480]))
    return d / "s1.npz"


def test_drop_label_keeps_other_samples(tmp_path):
    path = make_session(tmp_path, ["fist", "rock"])  # n == len(frame_size): must not confuse
    run(tmp_path, "drop", "s1", "--label", "rock")
    d = np.load(path)
    assert list(d["labels"]) == ["fist"]
    assert d["pixels"].shape == (1, 21, 2)
    assert list(d["frame_size"]) == [640, 480]
    assert "fist 1" in run(tmp_path, "list")


def test_trim_drops_clip_edges_but_not_none():
    sys.path.insert(0, str(SCRIPT.parent))
    from dataset import clip_keep_mask

    # fist clip 0-2 s, then a 'none' clip, then another fist clip after a gap.
    t = np.concatenate([np.arange(0, 2.01, 0.1), np.arange(2.1, 3.01, 0.1),
                        np.arange(5, 6.01, 0.1)])
    labels = np.array(["fist"] * 21 + ["none"] * 10 + ["fist"] * 11)
    keep = clip_keep_mask(t, labels, trim=0.3)
    assert keep[labels == "none"].all()
    first = t[:21][keep[:21]]
    assert first.min() >= 0.3 - 1e-9 and first.max() <= 1.7 + 1e-9
    last = t[31:][keep[31:]]
    assert last.min() >= 5.3 - 1e-9 and last.max() <= 5.7 + 1e-9


def test_drop_session_deletes_file(tmp_path):
    path = make_session(tmp_path, ["fist"])
    run(tmp_path, "drop", "s1.npz")
    assert not path.exists()
