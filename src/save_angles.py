from pathlib import Path

import numpy as np
import pandas as pd

from groups import motion_type

PROC = Path("data/processed")
OUT = Path("results/motion_angles.parquet")
THRESHOLDS = [(15, 45), (10, 30), (20, 60)]


def main():
    rows = []
    for split in ["val", "test"]:
        z = np.load(PROC / f"{split}.npz")
        motion, dpsi = motion_type(z["xy"], z["heading"])
        rows.append(pd.DataFrame({"scenario_id": z["ids"], "split": split, "motion": motion, "dpsi": dpsi}))
    df = pd.concat(rows, ignore_index=True)

    # the groups must match the ones already stored with the errors
    kin = pd.read_parquet("results/kinematic_errors.parquet", columns=["scenario_id", "split", "motion"])
    kin = kin.drop_duplicates(["scenario_id", "split"])
    m = df.merge(kin, on=["scenario_id", "split"], how="outer", suffixes=("", "_kin"), indicator=True)
    if (m["_merge"] != "both").any() or (m["motion"] != m["motion_kin"]).any():
        raise SystemExit("motion groups differ from results/kinematic_errors.parquet")

    df.to_parquet(OUT, index=False)
    print(f"saved {len(df)} rows to {OUT}, groups match the error tables")

    # group sizes on test for every pair of thresholds, stationary and undefined do not depend on them
    test = df[df["split"] == "test"]
    moving = test["motion"].isin(["straight", "turn", "intermediate"])
    for straight, turn in THRESHOLDS:
        n_straight = (moving & (test["dpsi"] < straight)).sum()
        n_turn = (moving & (test["dpsi"] > turn)).sum()
        print(f"  {straight}/{turn} deg: straight {n_straight}, turn {n_turn}")


if __name__ == "__main__":
    main()