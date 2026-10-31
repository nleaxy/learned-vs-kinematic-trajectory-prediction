import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
FREQS = [10, 5, 2, 1]
METRICS = ["ade", "fde", "miss"]
VARIANTS = ["linear", "zero_order_hold", "naive"]
LEVEL_MAIN = 1 - 0.05 / 8
LEVEL_DESC = 0.95
MARGIN = 0.05
MIN_GROUP = 300

# the 8 primary comparisons, side says where the interval has to lie for the part to be supported
PRIMARY = [
    ("H1", "delta", 5, "all", "below", 0.05),
    ("H1", "delta", 1, "all", "above", 0.20),
    ("H2", "adv", 2, "straight", "below", MARGIN),
    ("H2", "adv", 1, "straight", "below", MARGIN),
    ("H2", "adv", 2, "turn", "above", MARGIN),
    ("H2", "adv", 1, "turn", "above", MARGIN),
    ("H3", "extra", 2, "all", "above", MARGIN),
    ("H3", "extra", 1, "all", "above", MARGIN),
]


def load():
    kin = pd.read_parquet(RESULTS / "kinematic_errors.parquet")
    rnn = pd.read_parquet(RESULTS / "rnn_errors.parquet")
    for c in ["model", "mode", "road_user", "motion", "split"]:
        rnn[c] = rnn[c].astype(str)
    return kin, rnn


def build_conditions(kin, rnn):
    # every condition is a matrix seeds x scenarios plus the key of its seed draw
    test = kin[kin["split"] == "test"]
    ids = np.sort(test["scenario_id"].unique())
    scen = test.drop_duplicates("scenario_id").set_index("scenario_id").loc[ids, ["road_user", "motion"]]

    def matrix(df, metric):
        m = df.pivot(index="seed", columns="scenario_id", values=metric)
        if m.shape[1] != len(ids) or m.isna().any().any():
            raise SystemExit(f"condition does not cover all {len(ids)} test scenarios")
        return m[ids].to_numpy(np.float64)

    cond = {}
    for (name, f), g in test.groupby(["model", "freq"]):
        g = g.assign(seed=0)
        cond[(name, f, "window")] = ({m: matrix(g, m) for m in METRICS}, None)
    for (name, f, mode), g in rnn.groupby(["model", "freq", "mode"]):
        # the 10 Hz model on sparse inputs reuses the seed draw of the 10 Hz model
        key = ("rnn", 10, "window") if name.startswith("rnn10_") else (name, f, mode)
        cond[(name, f, mode)] = ({m: matrix(g, m) for m in METRICS}, key)
    # at 1 Hz both modes hold the same five points
    if ("rnn", 1, "last5") not in cond:
        cond[("rnn", 1, "last5")] = cond[("rnn", 1, "window")]
    return ids, scen, cond


def best_kinematic(kin, column, value):
    # better of cv and ca on the validation subset, per group and frequency
    val = kin[kin["split"] == "val"]
    if column is not None:
        val = val[val[column] == value]
    means = val.groupby(["freq", "model"])["ade"].mean().unstack("model")
    return means.idxmin(axis=1).to_dict(), means


def draw_seeds(cond, n_boot, rng):
    draws = {}
    for key, (mats, seed_key) in cond.items():
        if seed_key is None or seed_key in draws:
            continue
        n_seeds = cond[seed_key][0]["ade"].shape[0]
        # counts of each seed in every resample
        idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
        draws[seed_key] = np.stack([np.bincount(r, minlength=n_seeds) for r in idx]) / n_seeds
    return draws


def bootstrap(cond, cols, seed_draws, n_boot, rng, chunk=500):
    # resample means of every condition and metric, scenarios drawn once per resample for all models
    n = len(cols)
    out = {(key, m): np.empty(n_boot) for key in cond for m in METRICS}
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n)) + np.arange(b)[:, None] * n
        w = np.bincount(idx.ravel(), minlength=b * n).reshape(b, n) / n
        for key, (mats, seed_key) in cond.items():
            for m in METRICS:
                per_seed = mats[m][:, cols] @ w.T
                if seed_key is None:
                    out[(key, m)][start:start + b] = per_seed[0]
                else:
                    sw = seed_draws[seed_key][start:start + b]
                    out[(key, m)][start:start + b] = (sw * per_seed.T).sum(1)
    return out


def point(cond, cols):
    return {(key, m): mats[m][:, cols].mean() for key, (mats, _) in cond.items() for m in METRICS}


def interval(x, level):
    a = (1 - level) / 2
    return np.quantile(x, a), np.quantile(x, 1 - a)


def quantities(est, kin_choice, freq, mode="window"):
    # relative quantities from means, works for point estimates and for arrays of resample means
    rnn = lambda f: est[(("rnn", f, mode), "ade")]
    out = {}
    out["delta"] = (rnn(freq) - rnn(10)) / rnn(10)
    k = est[((kin_choice[freq], freq, "window"), "ade")]
    out["adv"] = (k - rnn(freq)) / k
    if freq != 10 and mode == "window":
        sparse = est[(("rnn10_linear", freq, "window"), "ade")]
        out["extra"] = (sparse - rnn(freq)) / rnn(freq)
    return out


def summarize_group(cond, cols, kin, column, value, seed_draws, n_boot, rng):
    kin_choice, _ = best_kinematic(kin, column, value)
    est = point(cond, cols)
    boot = bootstrap(cond, cols, seed_draws, n_boot, rng)
    rows = []
    for (key, m), x in boot.items():
        lo, hi = interval(x, LEVEL_DESC)
        seed_std = cond[key][0][m][:, cols].mean(1).std(ddof=1) if cond[key][1] is not None else np.nan
        rows.append(dict(quantity=m, model=key[0], freq=key[1], mode=key[2], value=est[(key, m)],
                         lo95=lo, hi95=hi, seed_std=seed_std))
    for mode in ["window", "last5"]:
        for f in FREQS:
            q_est = quantities(est, kin_choice, f, mode)
            q_boot = quantities(boot, kin_choice, f, mode)
            for q in q_est:
                lo, hi = interval(q_boot[q], LEVEL_DESC)
                lo_m, hi_m = interval(q_boot[q], LEVEL_MAIN)
                rows.append(dict(quantity=q, model="rnn", freq=f, mode=mode, value=q_est[q], lo95=lo, hi95=hi,
                                 lo_main=lo_m, hi_main=hi_m, kin=kin_choice[f]))
    df = pd.DataFrame(rows)
    df.insert(0, "group", value if column else "all")
    df.insert(0, "grouping", column or "all")
    df["n"] = len(cols)
    return df


def decide(lo, hi, side, thr):
    if side == "below":
        return "supported" if hi < thr else "contradicted" if lo > thr else "inconclusive"
    return "supported" if lo > thr else "contradicted" if hi < thr else "inconclusive"


def primary_table(summary):
    rows = []
    for hyp, q, f, group, side, thr in PRIMARY:
        grouping = "all" if group == "all" else "motion"
        r = summary[(summary["grouping"] == grouping) & (summary["group"] == group) & (summary["quantity"] == q)
                    & (summary["freq"] == f) & (summary["mode"] == "window")].iloc[0]
        result = decide(r["lo_main"], r["hi_main"], side, thr)
        if r["n"] < MIN_GROUP:
            result = "descriptive"
        rows.append(dict(hypothesis=hyp, quantity=q, freq=f, group=group, n=r["n"], side=side, threshold=thr,
                         value=r["value"], lo=r["lo_main"], hi=r["hi_main"], result=result))
    return pd.DataFrame(rows)


def verdicts(table):
    out = {}
    for hyp, g in table.groupby("hypothesis"):
        res = g["result"]
        if (res == "supported").all():
            out[hyp] = "accepted"
        elif (res == "supported").any():
            out[hyp] = "partly supported"
        else:
            out[hyp] = "not supported"
    return out


def min_sufficient(summary):
    # lowest frequency with the upper 95% bound of delta below 5% there and at every higher tested frequency
    rows = []
    d = summary[(summary["quantity"] == "delta") & (summary["mode"] == "window") & (summary["freq"] < 10)]
    for (grouping, group), g in d.groupby(["grouping", "group"]):
        if grouping not in ("all", "road_user") or g["n"].iloc[0] < MIN_GROUP:
            continue
        best = None
        for f in sorted(g["freq"], reverse=True):
            if g.loc[g["freq"] == f, "hi95"].iloc[0] >= MARGIN:
                break
            best = f
        rows.append(dict(grouping=grouping, group=group, min_sufficient_freq=best))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    kin, rnn = load()
    ids, scen, cond = build_conditions(kin, rnn)
    seed_draws = draw_seeds(cond, args.n_boot, rng)
    print(f"{len(ids)} test scenarios, {len(cond)} conditions, {args.n_boot} resamples")

    parts = [summarize_group(cond, np.arange(len(ids)), kin, None, None, seed_draws, args.n_boot, rng)]
    for column in ["road_user", "motion"]:
        for value in sorted(scen[column].unique()):
            cols = np.flatnonzero(scen[column].to_numpy() == value)
            parts.append(summarize_group(cond, cols, kin, column, value, seed_draws, args.n_boot, rng))
            print(f"  {column} {value}: {len(cols)} scenarios")
    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(RESULTS / "stats_summary.csv", index=False)

    table = primary_table(summary)
    table.to_csv(RESULTS / "primary_comparisons.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nprimary comparisons, {LEVEL_MAIN:.2%} intervals")
    print(table.round(4).to_string(index=False))
    for hyp, v in verdicts(table).items():
        print(f"  {hyp}: {v}")

    msf = min_sufficient(summary)
    msf.to_csv(RESULTS / "min_sufficient_freq.csv", index=False)
    print("\nminimum sufficient frequency")
    print(msf.to_string(index=False))
    print(f"\nsaved stats_summary.csv, primary_comparisons.csv, min_sufficient_freq.csv to {RESULTS}")


if __name__ == "__main__":
    main()