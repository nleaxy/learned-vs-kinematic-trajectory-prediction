import json
from pathlib import Path

import pandas as pd
import yaml

LOGS = Path("results/runs")
OUT = Path("results/selected_hparams.json")
K = {10: 1, 5: 2, 2: 5, 1: 10}


def load_runs():
    rows = []
    for p in sorted(LOGS.glob("*.json")):
        r = json.loads(p.read_text())
        # timing and check runs used a part of train and are not part of the experiment
        if r.get("n_train") is not None:
            continue
        rows.append({k: r[k] for k in ["run", "freq", "mode", "hidden", "lr", "seed", "best_val_ade", "best_epoch", "epochs_run"]})
    return pd.DataFrame(rows)


def main():
    cfg = yaml.safe_load(open("configs/train.yaml"))
    df = load_runs()
    if df.empty:
        print("no runs found")
        return

    grid = df[(df["seed"] == cfg["grid"]["seed"]) & (df["mode"] == "window")]
    print("grid search, seed 0, val ADE")
    print(grid.pivot_table(index="freq", columns=["hidden", "lr"], values="best_val_ade").sort_index(ascending=False).round(3).to_string())

    n_cells = len(cfg["grid"]["hidden_size"]) * len(cfg["grid"]["lr"])
    best = {}
    for f, g in grid.groupby("freq"):
        if len(g) < n_cells:
            print(f"  {f} Hz: {len(g)} of {n_cells} grid runs, not selected yet")
            continue
        b = g.loc[g["best_val_ade"].idxmin()]
        best[int(f)] = {"hidden": int(b["hidden"]), "lr": float(b["lr"])}
    for f in sorted(best, reverse=True):
        print(f"  {f} Hz: hidden {best[f]['hidden']}, lr {best[f]['lr']:g}")
    if len(best) == 4:
        OUT.write_text(json.dumps({str(f): v for f, v in best.items()}, indent=1) + "\n")
        print(f"saved {OUT}")

    main_seeds = cfg["seeds"]["main"]
    runs = df[(df["mode"] == "window") & df["seed"].isin(main_seeds)]
    if not runs.empty:
        stats = runs.groupby("freq")["best_val_ade"].agg(["count", "mean", "std"]).sort_index(ascending=False)
        stats["std_share"] = stats["std"] / stats["mean"]
        print("\nmain runs, val ADE over seeds 1-5")
        print(stats.round(4).to_string())
        if len(stats) == 4 and (stats["count"] == len(main_seeds)).all():
            extra = (stats["std_share"] > cfg["seeds"]["extra_rule_cv"]).any()
            print("seeds 6-10 needed for all frequencies" if extra else "seeds 6-10 not needed")

    last5 = df[df["mode"] == "last5"]
    if not last5.empty:
        print("\nadditional mode, val ADE")
        print(last5.pivot_table(index="freq", columns="seed", values="best_val_ade").sort_index(ascending=False).round(3).to_string())

    if best:
        print("\ncommands with the selected settings")
        for f in sorted(best, reverse=True):
            h, lr = best[f]["hidden"], best[f]["lr"]
            print(f"foreach ($s in 1..5) {{ python src/train.py --freq {f} --hidden {h} --lr {lr:g} --seed $s }}")
        for f in sorted(best, reverse=True):
            # at 1 Hz the additional mode reuses the main mode models
            if K[f] not in cfg["seeds"]["last5_reuse_main_k"]:
                h, lr = best[f]["hidden"], best[f]["lr"]
                print(f"foreach ($s in 1..3) {{ python src/train.py --freq {f} --mode last5 --hidden {h} --lr {lr:g} --seed $s }}")


if __name__ == "__main__":
    main()