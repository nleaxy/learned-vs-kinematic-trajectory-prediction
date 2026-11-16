import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from groups import LAST_OBS, motion_type, road_user_group
from prepare_data import to_agent_frame
from train import RUNS, LOGS, Model, features, history_index

PROC = Path("data/processed")
OUT = Path("results/rnn_errors.parquet")
FULL = np.arange(LAST_OBS + 1)


def restore_grid(idx, variant):
    # weights that map the retained points back to the 50 steps of the 10 Hz grid
    eye = np.eye(len(idx))
    if variant == "linear":
        return np.stack([np.interp(FULL, idx, e) for e in eye], axis=1)
    # zero order hold, steps before the first retained point copy it
    last = np.maximum(np.searchsorted(idx, FULL, side="right") - 1, 0)
    return eye[last]


def sparse_input(hist10, freq, variant):
    idx = history_index(freq, "window")
    if variant == "naive":
        return hist10[:, idx]
    return np.einsum("tj,njc->ntc", restore_grid(idx, variant), hist10[:, idx])


def load_model(run, cfg):
    log = json.loads((LOGS / f"{run}.json").read_text())
    saved = torch.load(RUNS / f"{run}.pt")
    model = Model(cfg["model"]["input_size"], log["hidden"], cfg["model"]["output_points"])
    model.load_state_dict(saved["state"])
    model.eval()
    stats = {k: np.array(v) for k, v in saved["stats"].items()}
    return model, stats


@torch.no_grad()
def predict(model, stats, hist, batch=4096):
    x = torch.tensor((features(hist) - stats["mean"]) / stats["std"], dtype=torch.float32)
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).numpy()


def errors(pred, fut):
    d = np.linalg.norm(pred - fut, axis=2)
    return d.mean(1), d[:, -1]


def main():
    cfg = yaml.safe_load(open("configs/train.yaml"))
    runs = []
    for p in sorted(LOGS.glob("*.json")):
        r = json.loads(p.read_text())
        # grid runs with seed 0 and the timing and check runs are not evaluated
        if r.get("n_train") is None and r["seed"] != cfg["grid"]["seed"]:
            runs.append(r)
    missing = [r["run"] for r in runs if not (RUNS / f"{r['run']}.pt").exists()]
    if missing:
        raise SystemExit(f"weights not found for {len(missing)} runs, e.g. {missing[:3]}")

    splits = {}
    for split in ["val", "test"]:
        z = np.load(PROC / f"{split}.npz")
        frame = to_agent_frame(z["xy"], z["heading"])
        motion, _ = motion_type(z["xy"], z["heading"])
        splits[split] = dict(ids=z["ids"], hist=frame[:, :LAST_OBS + 1], fut=frame[:, LAST_OBS + 1:],
                             road_user=road_user_group(z["object_type"]), motion=motion)

    rows, check = [], []
    for r in runs:
        model, stats = load_model(r["run"], cfg)
        jobs = [("rnn", r["freq"], r["mode"], None)]
        # the 10 Hz models of the main mode are also applied to sparse inputs
        if r["freq"] == 10 and r["mode"] == "window":
            jobs += [(f"rnn10_{v}", f, "window", v) for f in [5, 2, 1] for v in cfg["sparse_input_for_10hz"]["variants"]]
        for name, freq, mode, variant in jobs:
            for split, s in splits.items():
                if variant is None:
                    hist = s["hist"][:, history_index(freq, mode)]
                else:
                    hist = sparse_input(s["hist"], freq, variant)
                ade, fde = errors(predict(model, stats, hist), s["fut"])
                rows.append(pd.DataFrame({
                    "scenario_id": s["ids"], "split": split, "model": name, "freq": freq, "mode": mode,
                    "seed": r["seed"], "road_user": s["road_user"], "motion": s["motion"],
                    "ade": ade.round(3).astype(np.float32), "fde": fde.round(3).astype(np.float32),
                    # miss uses the unrounded fde, the stored fde is rounded to millimeters
                    "miss": fde > 2.0,
                }))
                if name == "rnn" and split == "val":
                    check.append(abs(ade.mean() - r["best_val_ade"]))
        print(f"  {r['run']}: {len(jobs)} input conditions")

    df = pd.concat(rows, ignore_index=True)
    for c in ["split", "model", "mode", "road_user", "motion"]:
        df[c] = df[c].astype("category")
    # errors in millimeters are enough and keep the file small, val stays out of the file
    df[df["split"] == "test"].to_parquet(OUT, index=False)

    print(f"\nval ADE recomputed vs training logs, max difference {max(check):.5f} m")
    # only validation numbers are printed, test errors stay in the file until the main analysis
    val = df[df["split"] == "val"]
    print("\nvalidation ADE, mean over scenarios and seeds")
    print(val.pivot_table(index=["model", "mode"], columns="freq", values="ade", observed=True).round(3).to_string())
    print(f"\nsaved {(df['split'] == 'test').sum()} test rows to {OUT}, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()