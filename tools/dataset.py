"""Inspect and clean recorded gesture sessions in data/gestures/.

  python tools/dataset.py list                          # samples per gesture in each session
  python tools/dataset.py drop SESSION --label rock     # remove one gesture from a session
  python tools/dataset.py drop SESSION                  # delete the whole session
  python tools/dataset.py trim SESSION                  # drop transition frames at clip edges

`trim` is for sessions recorded before tools/record.py trimmed clips itself; run it once.

SESSION is a file name (e.g. 20260923-185506.npz) or its stem.
"""

import argparse
import collections
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/gestures")
PER_SAMPLE = {"pixels", "landmarks", "handedness", "labels", "timestamps"}


def resolve(name: str) -> Path:
    path = DATA_DIR / (name if name.endswith(".npz") else f"{name}.npz")
    if not path.exists():
        raise SystemExit(f"No such session: {path}")
    return path


def cmd_list(_args):
    files = sorted(DATA_DIR.glob("*.npz"))
    if not files:
        print(f"No sessions in {DATA_DIR}")
        return
    total = collections.Counter()
    for f in files:
        counts = collections.Counter(str(label) for label in np.load(f)["labels"])
        total += counts
        detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        print(f"{f.name}  ({sum(counts.values())}): {detail}")
    print(f"\nTotal ({sum(total.values())}): "
          + ", ".join(f"{k} {v}" for k, v in sorted(total.items())))


def cmd_drop(args):
    path = resolve(args.session)
    if not args.label:
        path.unlink()
        print(f"Deleted {path}")
        return
    data = dict(np.load(path))
    keep = ~np.isin(data["labels"].astype(str), args.label)
    removed = int((~keep).sum())
    if removed == 0:
        print(f"No samples labeled {args.label} in {path.name}")
        return
    if not keep.any():
        path.unlink()
        print(f"Removed all {removed} samples; deleted {path}")
        return
    np.savez_compressed(path, **{k: v[keep] if k in PER_SAMPLE else v for k, v in data.items()})
    print(f"Removed {removed} samples labeled {args.label} from {path.name}")


def clip_keep_mask(t: np.ndarray, labels: np.ndarray, trim: float, gap: float = 0.5):
    """Split samples into clips (label change or a time gap > `gap`) and drop `trim` seconds
    at both ends of each clip, except for "none" where transitions are wanted."""
    keep = np.ones(len(t), dtype=bool)
    breaks = np.flatnonzero((labels[1:] != labels[:-1]) | (np.diff(t) > gap)) + 1
    for seg in np.split(np.arange(len(t)), breaks):
        if len(seg) == 0 or labels[seg[0]] == "none":
            continue
        ts = t[seg]
        keep[seg] = (ts >= ts[0] + trim) & (ts <= ts[-1] - trim)
    return keep


def cmd_trim(args):
    path = resolve(args.session)
    data = dict(np.load(path))
    keep = clip_keep_mask(data["timestamps"], data["labels"].astype(str), args.seconds)
    removed = int((~keep).sum())
    np.savez_compressed(path, **{k: v[keep] if k in PER_SAMPLE else v for k, v in data.items()})
    print(f"Trimmed {removed} transition samples from {path.name}; {int(keep.sum())} left")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    drop = sub.add_parser("drop")
    drop.add_argument("session")
    drop.add_argument("--label", nargs="+", help="Gesture(s) to remove; omit to delete the file")
    drop.set_defaults(fn=cmd_drop)
    trim = sub.add_parser("trim", help="Drop pose-forming frames at clip edges (run once)")
    trim.add_argument("session")
    trim.add_argument("--seconds", type=float, default=0.3)
    trim.set_defaults(fn=cmd_trim)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
