import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from stats import FREQS, PRIMARY

RESULTS = Path("results")
LOGS = RESULTS / "runs"

problems = []


def check(ok, message):
    if not ok:
        problems.append(message)


def check_cover(df, ids, label):
    # every scenario exactly once and finite errors
    check(len(df) == len(ids) and set(df["scenario_id"]) == ids, f"{label}: scenarios do not match the id list")
    check(np.isfinite(df[["ade", "fde"]].to_numpy()).all(), f"{label}: non finite errors")


def main():
    cfg = yaml.safe_load(open("configs/train.yaml"))
    ids = {s: set(Path(cfg["data"][f"{s}_ids"]).read_text().split()) for s in ["val", "test"]}

    kin = pd.read_parquet(RESULTS / "kinematic_errors.parquet")
    for split in ["val", "test"]:
        for model in ["cv", "ca"]:
            for f in FREQS:
                g = kin[(kin["split"] == split) & (kin["model"] == model) & (kin["freq"] == f)]
                check_cover(g, ids[split], f"{model} {f} Hz {split}")

    # extra seeds are expected only if the rule from the config fires on the main seeds
    logs = [json.loads(p.read_text()) for p in sorted(LOGS.glob("f*.json"))]
    runs = pd.DataFrame([{k: r[k] for k in ["freq", "mode", "seed", "best_val_ade"]} for r in logs])
    main_runs = runs[(runs["mode"] == "window") & runs["seed"].isin(cfg["seeds"]["main"])]
    spread = main_runs.groupby("freq")["best_val_ade"].agg(lambda x: x.std() / x.mean())
    seeds = cfg["seeds"]["main"] + (cfg["seeds"]["extra"] if (spread > cfg["seeds"]["extra_rule_cv"]).any() else [])
    print(f"seed spread on val by frequency: {spread.round(4).to_dict()}, expected seeds {seeds}")

    expected = [("rnn", f, "window", seeds) for f in FREQS]
    expected += [("rnn", f, "last5", cfg["seeds"]["last5_mode"]) for f in FREQS if f != 1]
    expected += [(f"rnn10_{v}", f, "window", seeds) for v in cfg["sparse_input_for_10hz"]["variants"] for f in FREQS[1:]]

    rnn = pd.read_parquet(RESULTS / "rnn_errors.parquet")
    found = set(rnn.groupby(["model", "freq", "mode"], observed=True).groups)
    check(found == {e[:3] for e in expected}, f"rnn conditions differ from the plan: {sorted(found ^ {e[:3] for e in expected})}")
    for model, f, mode, want in expected:
        g = rnn[(rnn["model"] == model) & (rnn["freq"] == f) & (rnn["mode"] == mode)]
        have = sorted(int(s) for s in g["seed"].unique())
        check(have == sorted(want), f"{model} {f} Hz {mode}: seeds {have}")
        for s, gs in g.groupby("seed"):
            check_cover(gs, ids["test"], f"{model} {f} Hz {mode} seed {s}")

    # input scaling of every run, a zero std would divide by zero
    for r in logs:
        std = np.array(r["input_stats"]["std"])
        check(np.isfinite(std).all() and (std > 0).all(), f"{r['run']}: input std {std}")

    # a non finite heading would give a nan angle and put the scenario into intermediate
    angles = pd.read_parquet(RESULTS / "motion_angles.parquet")
    check(np.isfinite(angles["dpsi"]).all(), "motion_angles: non finite heading change")
    for split in ["val", "test"]:
        check(set(angles.loc[angles["split"] == split, "scenario_id"]) == ids[split], f"motion_angles: {split} ids")

    table = pd.read_csv(RESULTS / "primary_comparisons.csv")
    keys = list(zip(table["hypothesis"], table["quantity"], table["freq"], table["group"]))
    check(keys == [p[:4] for p in PRIMARY], "primary_comparisons: the 8 parts differ from the plan")
    check(np.isfinite(table[["value", "lo", "hi"]].to_numpy()).all(), "primary_comparisons: non finite values")

    if problems:
        raise SystemExit("\n".join(["checks failed:"] + problems))
    print(f"all checks passed: {len(rnn)} rnn rows, {len(kin)} kinematic rows, {len(logs)} run logs")


if __name__ == "__main__":
    main()