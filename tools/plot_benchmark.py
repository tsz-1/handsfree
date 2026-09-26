"""Plot the jitter/lag trade-off of cursor smoothing from a saved benchmark.

Re-runs the smoothing comparison offline on the hold-still trajectory recorded by
tools/benchmark.py, sweeping the exponential filter's divisor and One Euro's beta, and
writes reports/smoothing_tradeoff.png. No camera needed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import ema, jitter, one_euro, synthetic_lag

from handsfree.config import MouseConfig, load_config


def ema_lag(divisor, speed_px_s=1800.0, fps=30):
    n = 2 * fps
    ts = np.arange(n) / fps
    ramp = np.stack([ts * speed_px_s, np.zeros(n)], 1)
    return float(abs(ema(ramp, divisor)[-1, 0] - ramp[-1, 0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--benchmark", type=Path, default=Path("reports/benchmark.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/smoothing_tradeoff.png"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mouse = load_config(args.config).mouse
    data = json.loads(args.benchmark.read_text())["still_trajectory"]
    ts, pts = np.array(data["t"]), np.array(data["xy"])
    warm = ts > 1.0

    ema_divs = [1.5, 2, 3, 4, 6, 8, 12]
    ema_pts = [(ema_lag(d), jitter(ema(pts, d)[warm])) for d in ema_divs]
    betas = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02]
    oe_pts = []
    for beta in betas:
        cfg = MouseConfig(min_cutoff=mouse.min_cutoff, beta=beta, d_cutoff=mouse.d_cutoff)
        oe_pts.append((synthetic_lag(cfg)["one_euro_px"], jitter(one_euro(pts, ts, cfg)[warm])))
    chosen = (synthetic_lag(mouse)["one_euro_px"], jitter(one_euro(pts, ts, mouse)[warm]))
    raw = jitter(pts[warm])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(*zip(*ema_pts), "o-", color="tab:gray", label="Exponential smoothing (÷1.5 … ÷12)")
    ax.plot(*zip(*oe_pts), "s-", color="tab:blue", label="One Euro (β sweep)")
    ax.plot(*chosen, "*", color="tab:red", markersize=16, label="One Euro, shipped config")
    ax.axhline(raw, color="k", linestyle=":", label=f"No smoothing ({raw:.1f} px/frame)")
    for d, (x, y) in zip(ema_divs, ema_pts):
        ax.annotate(f"÷{d:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    for beta, (x, y) in zip(betas, oe_pts):
        ax.annotate(f"β={beta:g}", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Lag behind a 1800 px/s target (px, lower is better)")
    ax.set_ylabel("Cursor jitter, hand held still (px/frame RMS)")
    ax.set_title("Cursor smoothing: jitter vs lag on a recorded hold-still trajectory")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    print(f"raw {raw:.2f} | shipped One Euro: jitter {chosen[1]:.2f}, lag {chosen[0]:.0f}")


if __name__ == "__main__":
    main()
