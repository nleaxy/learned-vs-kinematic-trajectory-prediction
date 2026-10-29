import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from groups import LAST_OBS
from prepare_data import to_agent_frame

PROC = Path("data/processed")
RUNS = Path("runs")
LOGS = Path("results/runs")
K = {10: 1, 5: 2, 2: 5, 1: 10}


def history_index(freq, mode):
    # every k-th point counting back from t 0, oldest first
    idx = np.arange(LAST_OBS, -1, -K[freq])[::-1]
    return idx[-5:] if mode == "last5" else idx


def features(hist):
    # positions and steps between retained points, the first step is zero
    steps = np.diff(hist, axis=1, prepend=hist[:, :1])
    return np.concatenate([hist, steps], axis=-1)


def load(split, freq, mode):
    z = np.load(PROC / f"{split}.npz")
    frame = to_agent_frame(z["xy"], z["heading"])
    return features(frame[:, history_index(freq, mode)]), frame[:, LAST_OBS + 1:]


def to_tensor(x, stats=None):
    return torch.tensor(x if stats is None else (x - stats["mean"]) / stats["std"], dtype=torch.float32)


class Model(nn.Module):
    def __init__(self, input_size, hidden, future_points):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.head = nn.Linear(hidden, future_points * 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).view(len(x), -1, 2)


def ade(pred, fut):
    return torch.linalg.norm(pred - fut, dim=-1).mean(1)


@torch.no_grad()
def evaluate(model, x, y, batch=4096):
    model.eval()
    return torch.cat([ade(model(x[i:i + batch]), y[i:i + batch]) for i in range(0, len(x), batch)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=int, choices=K, required=True)
    p.add_argument("--mode", choices=["window", "last5"], default="window")
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-train", type=int, default=None, help="only for the timing run")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open("configs/train.yaml"))
    tr = cfg["training"]
    name = args.name or f"f{args.freq}_{args.mode}_h{args.hidden}_lr{args.lr:g}_s{args.seed}"

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(tr["deterministic"])
    gen = torch.Generator().manual_seed(args.seed)

    x_tr, y_tr = load("train", args.freq, args.mode)
    x_va, y_va = load("val", args.freq, args.mode)
    # input scaling from the full train set of this condition
    stats = {"mean": x_tr.reshape(-1, 4).mean(0), "std": x_tr.reshape(-1, 4).std(0)}
    x_tr, y_tr = to_tensor(x_tr[:args.n_train], stats), to_tensor(y_tr[:args.n_train])
    x_va, y_va = to_tensor(x_va, stats), to_tensor(y_va)
    print(f"{name}: train {len(x_tr)}, val {len(x_va)}, history points {x_tr.shape[1]}, threads {torch.get_num_threads()}")

    model = Model(cfg["model"]["input_size"], args.hidden, cfg["model"]["output_points"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bs = tr["batch_size"]

    best, best_epoch, best_state, wait, log = float("inf"), 0, None, 0, []
    for epoch in range(1, tr["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(len(x_tr), generator=gen)
        train_loss = 0.0
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            loss = ade(model(x_tr[b]), y_tr[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(b)
        t1 = time.time()
        val = evaluate(model, x_va, y_va).mean().item()
        t2 = time.time()

        log.append(dict(epoch=epoch, train_ade=train_loss / len(x_tr), val_ade=val,
                        train_sec=t1 - t0, val_sec=t2 - t1))
        print(f"  epoch {epoch:2d}  train {train_loss / len(x_tr):.3f}  val {val:.3f}  "
              f"time {t1 - t0:.1f}s + {t2 - t1:.1f}s")

        if val < best:
            best, best_epoch, wait = val, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            # no early stop in the first epochs, the model may sit on a plateau there
            if wait >= tr["patience"] and epoch > tr["min_epochs"]:
                break

    RUNS.mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "stats": {k: v.tolist() for k, v in stats.items()}}, RUNS / f"{name}.pt")
    summary = dict(run=name, **vars(args), best_epoch=best_epoch, best_val_ade=best,
                   input_stats={k: v.tolist() for k, v in stats.items()},
                   epochs_run=len(log), total_sec=sum(r["train_sec"] + r["val_sec"] for r in log), log=log)
    (LOGS / f"{name}.json").write_text(json.dumps(summary, indent=1) + "\n")

    train_sec = np.mean([r["train_sec"] for r in log])
    val_sec = np.mean([r["val_sec"] for r in log])
    print(f"best epoch {best_epoch}, val ADE {best:.3f}, epochs run {len(log)}, total {summary['total_sec'] / 60:.1f} min")
    print(f"mean per epoch: train {train_sec:.2f}s on {len(x_tr)} scenarios, val {val_sec:.2f}s")


if __name__ == "__main__":
    main()