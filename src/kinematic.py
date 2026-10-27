from pathlib import Path

import numpy as np
import pandas as pd

from groups import LAST_OBS, motion_type, road_user_group

PROC = Path("data/processed")
RESULTS = Path("results")
FREQS = {10: 1, 5: 2, 2: 5, 1: 10}
T = np.arange(1, 61) * 0.1


def cv(xy, k):
    dt = k * 0.1
    p0, p1 = xy[:, LAST_OBS], xy[:, LAST_OBS - k]
    v = (p0 - p1) / dt
    return p0[:, None] + v[:, None] * T[None, :, None]


def ca(xy, k):
    # second order backward differences on the last three retained points
    dt = k * 0.1
    p0, p1, p2 = xy[:, LAST_OBS], xy[:, LAST_OBS - k], xy[:, LAST_OBS - 2 * k]
    v = (3 * p0 - 4 * p1 + p2) / (2 * dt)
    a = (p0 - 2 * p1 + p2) / dt ** 2
    t = T[None, :, None]
    return p0[:, None] + v[:, None] * t + 0.5 * a[:, None] * t ** 2


def errors(pred, xy):
    d = np.linalg.norm(pred - xy[:, LAST_OBS + 1:], axis=2)
    return d.mean(1), d[:, -1]


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for split in ["val", "test"]:
        z = np.load(PROC / f"{split}.npz")
        xy, ids = z["xy"], z["ids"]
        group = road_user_group(z["object_type"])
        motion, _ = motion_type(xy, z["heading"])
        for f, k in FREQS.items():
            for name, model in [("cv", cv), ("ca", ca)]:
                ade, fde = errors(model(xy, k), xy)
                rows.append(pd.DataFrame({
                    "scenario_id": ids, "split": split, "freq": f, "model": name,
                    "road_user": group, "motion": motion, "ade": ade, "fde": fde,
                }))
    df = pd.concat(rows, ignore_index=True)
    df["miss"] = df["fde"] > 2.0
    df.to_parquet(RESULTS / "kinematic_errors.parquet", index=False)

    # only validation numbers are printed, test errors stay in the file until the main analysis
    val = df[df["split"] == "val"]
    print("validation, mean over scenarios")
    print(val.groupby(["freq", "model"])[["ade", "fde", "miss"]].mean().round(3).unstack("model").sort_index(ascending=False).to_string())
    print("\nvalidation ADE by motion type")
    print(val.pivot_table(index="motion", columns=["model", "freq"], values="ade").round(2).to_string())
    print("\nvalidation ADE by road user type")
    print(val.pivot_table(index="road_user", columns=["model", "freq"], values="ade").round(2).to_string())
    print(f"\nsaved {len(df)} rows to {RESULTS / 'kinematic_errors.parquet'}")


if __name__ == "__main__":
    main()