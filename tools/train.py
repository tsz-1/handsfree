"""Train the static gesture classifier from recordings in data/gestures/.

Models are compared with leave-one-session-out cross-validation: every fold tests on a
recording session the model never saw. Consecutive frames are nearly identical, so a random
train/test split would leak and overstate accuracy (the script reports that number too).
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from handsfree.classifier import FEATURE_VERSION, hand_features

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=2000),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    # Single-threaded: at runtime we predict one frame at a time, where thread dispatch dominates.
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
    "mlp": lambda: MLPClassifier((128, 64), max_iter=500, early_stopping=True, random_state=0),
}


def load_sessions(data_dir: Path):
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No recordings in {data_dir}. Run tools/record.py first.")
    X, y, groups = [], [], []
    for gi, f in enumerate(files):
        d = np.load(f)
        X += [hand_features(p, h) for p, h in zip(d["pixels"], d["handedness"])]
        y += [str(label) for label in d["labels"]]
        groups += [gi] * len(d["labels"])
    return np.array(X), np.array(y), np.array(groups), [f.name for f in files]


def augment(X: np.ndarray, y: np.ndarray, copies: int, rng: np.random.Generator):
    """Landmark jitter plus a small rotation (features are already rotation-normalized, so
    this only models noise in the palm-axis estimate)."""
    if copies == 0:
        return X, y
    pts = X.reshape(len(X), 21, 2)
    out = [X]
    for _ in range(copies):
        theta = np.deg2rad(rng.uniform(-5, 5, size=len(X)))
        c, s = np.cos(theta), np.sin(theta)
        rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (N, 2, 2)
        aug = np.einsum("nij,nkj->nki", rot, pts) + rng.normal(0, 0.02, pts.shape)
        out.append(aug.reshape(len(X), -1))
    return np.concatenate(out), np.tile(y, copies + 1)


def make_folds(y, groups):
    if len(np.unique(groups)) >= 2:
        return list(LeaveOneGroupOut().split(y, y, groups)), "leave-one-session-out"
    # One session only: hold out the last 20% of each gesture's frames (in time order).
    train, test = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        cut = int(len(idx) * 0.8)
        train += list(idx[:cut])
        test += list(idx[cut:])
    return [(np.array(train), np.array(test))], "temporal 80/20 split (single session)"


def build(name):
    return make_pipeline(StandardScaler(), MODELS[name]())


def latency_ms(pipeline, x: np.ndarray, n: int = 200) -> float:
    pipeline.predict_proba(x[None])
    start = time.perf_counter()
    for _ in range(n):
        pipeline.predict_proba(x[None])
    return (time.perf_counter() - start) / n * 1000


def evaluate(name, X, y, folds, copies, seed):
    rng = np.random.default_rng(seed)
    y_true, y_pred = [], []
    for train_idx, test_idx in folds:
        Xa, ya = augment(X[train_idx], y[train_idx], copies, rng)
        model = build(name).fit(Xa, ya)
        y_true += list(y[test_idx])
        y_pred += list(model.predict(X[test_idx]))
    return np.array(y_true), np.array(y_pred), model


def plot_confusion(y_true, y_pred, labels, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/gestures"))
    parser.add_argument("--out", type=Path, default=Path("models/gesture_classifier.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--augment", type=int, default=2, help="Augmented copies per sample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y, groups, sessions = load_sessions(args.data)
    labels = sorted({str(v) for v in y})
    counts = {label: int((y == label).sum()) for label in labels}
    print(f"{len(y)} samples, {len(sessions)} session(s), {len(labels)} gestures: {counts}")

    folds, protocol = make_folds(y, groups)
    if len(sessions) < 2:
        print("Warning: only one session; the test split shares lighting and pose with training, "
              "so accuracy will be optimistic. Record another session for an honest number.")
    print(f"Evaluation: {protocol}\n")

    results = {}
    print(f"{'model':<14}{'accuracy':>10}{'macro F1':>10}{'ms/frame':>10}")
    for name in args.models:
        y_true, y_pred, model = evaluate(name, X, y, folds, args.augment, args.seed)
        results[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "latency_ms": latency_ms(model, X[0]),
            "_pred": (y_true, y_pred),
        }
        r = results[name]
        print(f"{name:<14}{r['accuracy']:>10.3f}{r['macro_f1']:>10.3f}{r['latency_ms']:>10.2f}")

    best = max(results, key=lambda n: (round(results[n]["macro_f1"], 3), -results[n]["latency_ms"]))
    print(f"\nBest: {best}")

    leak_acc = None
    if len(sessions) >= 2:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed,
                                              stratify=y)
        leak_acc = accuracy_score(yte, build(best).fit(Xtr, ytr).predict(Xte))
        print(f"For comparison, a random frame-level split gives {leak_acc:.3f} accuracy "
              f"vs {results[best]['accuracy']:.3f} on unseen sessions.")

    args.reports.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = results[best]["_pred"]
    plot_confusion(y_true, y_pred, labels, args.reports / "confusion_matrix.png",
                   f"{best} ({protocol})")
    metrics = {
        "protocol": protocol,
        "sessions": len(sessions),
        "samples": len(y),
        "per_class": counts,
        "best_model": best,
        "random_split_accuracy": leak_acc,
        "models": {n: {k: round(float(v), 4) for k, v in r.items() if not k.startswith("_")}
                   for n, r in results.items()},
    }
    (args.reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {args.reports / 'confusion_matrix.png'} and {args.reports / 'metrics.json'}")

    Xa, ya = augment(X, y, args.augment, np.random.default_rng(args.seed))
    final = build(best).fit(Xa, ya)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "labels": [str(c) for c in final.classes_],
                 "feature_version": FEATURE_VERSION, "model": best}, args.out)
    print(f"Saved {best} to {args.out}")


if __name__ == "__main__":
    main()
