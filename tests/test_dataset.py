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


def test_drop_session_deletes_file(tmp_path):
    path = make_session(tmp_path, ["fist"])
    run(tmp_path, "drop", "s1.npz")
    assert not path.exists()
