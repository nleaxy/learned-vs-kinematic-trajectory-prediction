import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stats import COLORS, FREQS, MARGIN, RESULTS

# one journal column is 83.5 mm wide, text stays at 8 pt
WIDTH = 83.5 / 25.4
HEIGHT = 2.05
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def line(ax, x, rows, scale, **kw):
    y = rows["value"].to_numpy() * scale
    err = [y - rows["lo95"].to_numpy() * scale, rows["hi95"].to_numpy() * scale - y]
    ax.errorbar(x, y, yerr=err, capsize=2, linewidth=1.2, elinewidth=0.8, markersize=4, **kw)


def finish(fig, ax, ylabel, path):
    ax.set_xticks(range(len(FREQS)), [f"{f} Hz" for f in FREQS])
    ax.set_xlabel("History frequency")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, handlelength=2.6)
    fig.savefig(path.with_suffix(".png"), dpi=600)
    # no creation date, so a rerun gives the same file
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    plt.close(fig)


def fig_advantage(summary, path):
    s = summary[(summary["grouping"] == "motion") & (summary["mode"] == "window") & (summary["quantity"] == "adv")]
    x = np.arange(len(FREQS))
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), layout="constrained")
    for (group, label, marker), color in zip([("straight", "Straight", "o"), ("turn", "Turn", "s")], COLORS):
        line(ax, x, s[s["group"] == group].set_index("freq").loc[FREQS], 100, color=color, marker=marker, label=label)
    ax.axhline(MARGIN * 100, color="#6f6e69", linewidth=0.8, linestyle="--")
    ax.text(len(FREQS) - 1, MARGIN * 100 + 0.8, "5% threshold", ha="right", va="bottom", color="#6f6e69")
    finish(fig, ax, "Advantage $A$, %", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/journal")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "stats_summary.csv")
    fig_advantage(summary, out / "Fig01")
    print(f"saved Fig01 to {out}")


if __name__ == "__main__":
    main()