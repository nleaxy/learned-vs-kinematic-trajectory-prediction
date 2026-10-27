import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUT = DATA / "processed"
SEED = 42
N_VAL = 5000
N_TRAIN = 50000
LAST_OBS = 49
COLS = ["track_id", "focal_track_id", "timestep", "position_x", "position_y", "heading", "object_type"]
GROUPS = {
    "vehicle": "vehicle", "bus": "vehicle",
    "pedestrian": "pedestrian",
    "cyclist": "two_wheeler", "motorcyclist": "two_wheeler",
}


def read_focal(path, n_steps):
    df = pd.read_parquet(path, columns=COLS)
    df = df[df["track_id"] == df["focal_track_id"]].sort_values("timestep")
    steps = df["timestep"].to_numpy()
    if len(steps) != n_steps or not (steps == np.arange(n_steps)).all():
        return None
    xy = df[["position_x", "position_y"]].to_numpy()
    if not np.isfinite(xy).all():
        return None
    return {
        "track": df["track_id"].iloc[0],
        "xy": xy,
        "heading": df["heading"].iloc[LAST_OBS],
        "type": df["object_type"].iloc[0],
    }


def load_split(ids, folder, n_steps, future=None):
    rows, dropped = [], []
    for i, sid in enumerate(ids, 1):
        r = read_focal(DATA / folder / sid / f"scenario_{sid}.parquet", n_steps)
        # test history gets its 60 future points from the annotation file
        if r is not None and future is not None:
            track, fut = future.get(sid, (None, None))
            if track != r["track"] or fut is None or len(fut) != 60 or not np.isfinite(fut).all():
                r = None
            else:
                r["xy"] = np.vstack([r["xy"], fut])
        if r is None:
            dropped.append(sid)
        else:
            rows.append((sid, r["xy"], r["heading"], r["type"]))
        if i % 5000 == 0:
            print(f"  {folder}: {i}/{len(ids)}")
    return rows, dropped


def to_agent_frame(xy, heading):
    d = xy - xy[:, LAST_OBS:LAST_OBS + 1]
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)


def save(name, rows):
    ids = [r[0] for r in rows]
    np.savez_compressed(
        OUT / f"{name}.npz",
        ids=np.array(ids),
        xy=np.stack([r[1] for r in rows]),
        heading=np.array([r[2] for r in rows]),
        object_type=np.array([r[3] for r in rows]),
    )
    Path(f"configs/{name}_ids.txt").write_text("\n".join(ids) + "\n")
    groups = pd.Series([GROUPS.get(r[3], "other") for r in rows]).value_counts()
    print(f"{name}: {len(rows)} scenarios, {groups.to_dict()}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    subset = Path("configs/train_subset.txt").read_text().split()
    test_ids = sorted(p.name for p in (DATA / "test").iterdir() if p.is_dir())

    # stop before reading anything if the download is incomplete
    missing = [f"train/{s}" for s in subset if not (DATA / "train" / s / f"scenario_{s}.parquet").exists()]
    missing += [f"test/{s}" for s in test_ids if not (DATA / "test" / s / f"scenario_{s}.parquet").exists()]
    if missing:
        sys.exit(f"{len(missing)} scenario files are missing, e.g. {missing[:3]}. Finish the download first")

    rows, dropped_train = load_split(subset, "train", 110)
    print(f"train subset: {len(rows)} kept, {len(dropped_train)} dropped")
    if len(rows) < N_VAL + N_TRAIN:
        sys.exit("not enough scenarios after filtering")

    # shuffle a sorted list so the split does not depend on file order
    rows.sort(key=lambda r: r[0])
    random.Random(SEED).shuffle(rows)
    val, train = rows[:N_VAL], rows[N_VAL:N_VAL + N_TRAIN]

    ann = pd.read_parquet(DATA / "av2_mf_focal_test_annotations.parquet")
    future = {
        sid: (track, np.column_stack([x, y]))
        for sid, track, x, y in zip(ann["scenario_id"], ann["track_id"], ann["gt_trajectory_x"], ann["gt_trajectory_y"])
    }
    test, dropped_test = load_split(test_ids, "test", 50, future)
    print(f"test: {len(test)} kept, {len(dropped_test)} dropped")

    for name, part in [("train", train), ("val", val), ("test", test)]:
        save(name, part)

    (OUT / "dropped.txt").write_text(
        "".join(f"train {s}\n" for s in dropped_train) + "".join(f"test {s}\n" for s in dropped_test)
    )

    # input scaling from 10 Hz train histories in the agent frame
    xy = np.stack([r[1] for r in train])
    heading = np.array([r[2] for r in train])
    hist = to_agent_frame(xy, heading)[:, :LAST_OBS + 1].reshape(-1, 2)
    stats = {"mean": hist.mean(0).tolist(), "std": hist.std(0).tolist()}
    Path("configs/norm_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("norm stats", stats)


if __name__ == "__main__":
    main()