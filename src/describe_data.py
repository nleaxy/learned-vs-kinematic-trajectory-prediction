from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from groups import LAST_OBS, motion_type, road_user_group
from prepare_data import to_agent_frame

PROC = Path("data/processed")
FIG = Path("figures/qc")
N_PLOT = 100


def load(name):
    z = np.load(PROC / f"{name}.npz")
    return z["ids"], z["xy"], z["heading"], road_user_group(z["object_type"])


def plot_grid(frame, idx, title, path):
    side = int(np.ceil(np.sqrt(len(idx))))
    fig, axes = plt.subplots(side, side, figsize=(side * 1.6, side * 1.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idx):
        ax.plot(frame[i, :LAST_OBS + 1, 0], frame[i, :LAST_OBS + 1, 1], color="tab:blue", lw=1)
        ax.plot(frame[i, LAST_OBS:, 0], frame[i, LAST_OBS:, 1], color="tab:orange", lw=1)
        ax.plot(0, 0, "k.", ms=3)
        ax.set_aspect("equal", "datalim")
        ax.set_title(str(i), fontsize=6)
    fig.suptitle(f"{title}: history blue, future orange, agent frame")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)

    for name in ["train", "val"]:
        _, xy, heading, group = load(name)
        motion, _ = motion_type(xy, heading)
        print(f"\n{name}: motion types")
        print(pd.Series(motion).value_counts().to_string())

    ids, xy, heading, group = load("test")
    motion, dpsi = motion_type(xy, heading)
    table = pd.crosstab(pd.Series(group, name="road user"), pd.Series(motion, name="motion"), margins=True)
    print("\ntest: road user type x motion type")
    print(table.to_string())
    print("\ngroups below 300 test scenarios:")
    small = [(r, c, table.loc[r, c]) for r in table.index for c in table.columns if table.loc[r, c] < 300]
    print("\n".join(f"  {r} / {c}: {n}" for r, c, n in small) or "  none")

    print("\nsensitivity, test motion types for other thresholds")
    for s, t in [(10, 30), (20, 60)]:
        m, _ = motion_type(xy, heading, straight=s, turn=t)
        print(f"  {s}/{t}:", pd.Series(m).value_counts().to_dict())

    # random scenarios of every motion type for the check by eye
    frame = to_agent_frame(xy, heading)
    rng = np.random.default_rng(0)
    for m in sorted(set(motion)):
        idx = np.flatnonzero(motion == m)
        idx = rng.choice(idx, size=min(N_PLOT, len(idx)), replace=False)
        plot_grid(frame, idx, f"test, {m}", FIG / f"{m}.png")
    print(f"\nfigures saved to {FIG}")


if __name__ == "__main__":
    main()