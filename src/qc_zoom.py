from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from groups import LAST_OBS, motion_type, road_user_group
from prepare_data import to_agent_frame

PROC = Path("data/processed")
OUT = Path("figures/qc")

# panels flagged on the overview pictures, numbers are row indices in test.npz
FLAGGED = {
    "turn": [23917, 17895, 16812, 8288, 12054, 16210, 21199, 22071, 24978, 3320],
    "intermediate": [6321, 808, 9521, 1071, 9846, 1862, 10266],
    "straight": [14975, 15795],
    "stationary": [15547, 14101, 8581, 14032, 19793, 7126, 16325, 13566],
    "undefined": [24316, 15037],
}


def history_angle(frame):
    # direction of the last 0.5 s of history relative to the dataset heading, degrees
    v = frame[:, LAST_OBS] - frame[:, LAST_OBS - 5]
    return np.degrees(np.abs(np.arctan2(v[:, 1], v[:, 0]))), np.linalg.norm(v, axis=1)


def main():
    z = np.load(PROC / "test.npz")
    xy, heading = z["xy"], z["heading"]
    road = road_user_group(z["object_type"])
    motion, dpsi = motion_type(xy, heading)
    frame = to_agent_frame(xy, heading)
    path = np.linalg.norm(np.diff(xy[:, LAST_OBS:], axis=1), axis=2).sum(1)
    hist_ang, hist_dist = history_angle(frame)

    # agents that move at t 0 but against the dataset heading
    moving = hist_dist >= 0.5
    against = moving & (hist_ang > 90)
    print("moving at t 0 against the dataset heading, whole test set")
    print(pd.crosstab(motion[moving], against[moving], rownames=["motion"], colnames=["against"]).to_string())
    print(pd.crosstab(road[moving], against[moving], rownames=["road user"], colnames=["against"]).to_string())

    rows = []
    for label, idx in FLAGGED.items():
        for i in idx:
            rows.append(dict(i=i, shown_in=label, label_now=motion[i], road_user=road[i],
                             dpsi=round(dpsi[i], 1), future_path_m=round(path[i], 2),
                             hist_angle=round(hist_ang[i], 1), hist_move_m=round(hist_dist[i], 2)))
    table = pd.DataFrame(rows)
    print("\nflagged panels")
    print(table.to_string(index=False))

    n = len(table)
    cols = 5
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols * 3.4, np.ceil(n / cols) * 3.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, table.itertuples()):
        f = frame[r.i]
        ax.axis("on")
        ax.plot(f[:LAST_OBS + 1, 0], f[:LAST_OBS + 1, 1], color="tab:blue", lw=1, marker=".", ms=2)
        ax.plot(f[LAST_OBS:, 0], f[LAST_OBS:, 1], color="tab:orange", lw=1, marker=".", ms=2)
        ax.plot(f[-1, 0], f[-1, 1], "rx", ms=6)
        ax.plot(0, 0, "k.", ms=6)
        ax.set_aspect("equal", "datalim")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
        ax.set_title(f"{r.i} {r.road_user} {r.label_now}\ndpsi {r.dpsi}, path {r.future_path_m} m", fontsize=7)
    fig.suptitle("flagged panels in meters: history blue, future orange, end red x, agent moves to +x")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "zoom.png", dpi=110)
    table.to_csv(OUT / "zoom.csv", index=False)
    print(f"\nsaved {OUT / 'zoom.png'} and {OUT / 'zoom.csv'}")


if __name__ == "__main__":
    main()