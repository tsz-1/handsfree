"""Inspect and clean recorded gesture sessions in data/gestures/.

  python tools/dataset.py list                          # samples per gesture in each session
  python tools/dataset.py drop SESSION --label rock     # remove one gesture from a session
  python tools/dataset.py drop SESSION                  # delete the whole session

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
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
