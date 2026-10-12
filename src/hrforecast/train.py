"""Trening i porównanie: persystencja, regresja liniowa, LSTM, ANFIS, ANFIS destylowany z LSTM.

Podział na zbiory odbywa się po całych aktywnościach (nie po oknach),
żeby sąsiednie, prawie identyczne okna nie trafiały jednocześnie do treningu i testu.
"""

import argparse
import copy
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from hrforecast.anfis import ANFIS
from hrforecast.features import ANFIS_FEATURES, SEQ_CHANNELS, SEQ_LEN, TARGET, usable_mask
from hrforecast.lstm import HrLSTM

Inputs = Callable[[torch.Tensor], torch.Tensor]


def split_activities(activities: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    acts = rng.permutation(np.unique(activities))
    if len(acts) < 3:
        raise SystemExit("Potrzeba co najmniej 3 aktywności (train/val/test).")
    n_test = max(1, round(0.15 * len(acts)))
    n_val = max(1, round(0.15 * len(acts)))
    return {
        "test": acts[:n_test],
        "val": acts[n_test : n_test + n_val],
        "train": acts[n_test + n_val :],
    }


def fit(
    model: nn.Module,
    inputs: Inputs,
    target: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    name: str,
) -> dict[str, list[float]]:
    """Adam + MSE, early stopping na zbiorze walidacyjnym; zostawia w modelu najlepsze wagi."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train": [], "val": []}
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=train_idx.device)]
        total = 0.0
        for start in range(0, len(perm), batch_size):
            b = perm[start : start + batch_size]
            loss = nn.functional.mse_loss(model(inputs(b)), target[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        val = nn.functional.mse_loss(predict(model, inputs, val_idx), target[val_idx]).item()
        history["train"].append(total / len(perm))
        history["val"].append(val)
        print(f"[{name}] epoka {epoch + 1:3d}  train MSE {history['train'][-1]:7.3f}  val MSE {val:7.3f}")
        if val < best:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(model: nn.Module, inputs: Inputs, idx: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    return torch.cat([model(inputs(idx[s : s + batch_size])) for s in range(0, len(idx), batch_size)])


def lse_batches(inputs: Inputs, target: torch.Tensor, idx: torch.Tensor, batch_size: int = 65536):
    for s in range(0, len(idx), batch_size):
        b = idx[s : s + batch_size]
        yield inputs(b), target[b]


def metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    err = pred - y
    return {"mae": err.abs().mean().item(), "rmse": err.pow(2).mean().sqrt().item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/samples.parquet"))
    parser.add_argument("--sport", default="running", help="'all' = wszystkie sporty")
    parser.add_argument("--n-mfs", type=int, default=3, help="liczba funkcji przynależności na wejście")
    parser.add_argument("--epochs-lstm", type=int, default=30)
    parser.add_argument("--epochs-anfis", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs") / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    df = pd.read_parquet(args.data)
    if args.sport != "all":
        df = df[df["sport"] == args.sport].reset_index(drop=True)
    usable = usable_mask(df).to_numpy()
    splits = split_activities(df.loc[usable, "activity"].to_numpy(), args.seed)
    idx_np = {k: np.flatnonzero(usable & df["activity"].isin(v).to_numpy()) for k, v in splits.items()}
    for k, v in idx_np.items():
        print(f"{k}: {len(splits[k])} aktywności, {len(v)} próbek")

    def tensor(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.nan_to_num(a), dtype=torch.float32, device=device)

    train_rows = idx_np["train"]
    feats_np = df[ANFIS_FEATURES].to_numpy(np.float64)
    seq_np = df[SEQ_CHANNELS].to_numpy(np.float64)
    seq_mean, seq_std = seq_np[train_rows].mean(0), seq_np[train_rows].std(0) + 1e-6
    feats = tensor(feats_np)
    seq = tensor((seq_np - seq_mean) / seq_std)
    y = tensor(df[TARGET].to_numpy())
    idx = {k: torch.tensor(v, device=device) for k, v in idx_np.items()}

    offsets = torch.arange(-SEQ_LEN + 1, 1, device=device)

    def seq_inputs(b: torch.Tensor) -> torch.Tensor:
        return seq[b[:, None] + offsets]

    def feat_inputs(b: torch.Tensor) -> torch.Tensor:
        return feats[b]

    common = dict(batch_size=args.batch_size, patience=5)
    results: dict[str, dict[str, float]] = {}
    histories: dict[str, dict[str, list[float]]] = {}

    # Punkty odniesienia
    results["persystencja (dhr=0)"] = metrics(torch.zeros_like(y[idx["test"]]), y[idx["test"]])
    a = np.c_[feats_np[train_rows], np.ones(len(train_rows))]
    coef, *_ = np.linalg.lstsq(a, df[TARGET].to_numpy()[train_rows], rcond=None)
    lin = feats[idx["test"]] @ tensor(coef[:-1]) + coef[-1]
    results["regresja liniowa"] = metrics(lin, y[idx["test"]])

    # LSTM (nauczyciel)
    lstm = HrLSTM(len(SEQ_CHANNELS)).to(device)
    histories["lstm"] = fit(lstm, seq_inputs, y, idx["train"], idx["val"],
                            epochs=args.epochs_lstm, lr=1e-3, name="LSTM", **common)
    results["LSTM"] = metrics(predict(lstm, seq_inputs, idx["test"]), y[idx["test"]])

    # ANFIS uczony na prawdziwym celu
    anfis = ANFIS(feats_np[train_rows], args.n_mfs).to(device)
    anfis.fit_consequents_lse(lse_batches(feat_inputs, y, idx["train"]))
    histories["anfis"] = fit(anfis, feat_inputs, y, idx["train"], idx["val"],
                             epochs=args.epochs_anfis, lr=3e-3, name="ANFIS", **common)
    results["ANFIS"] = metrics(predict(anfis, feat_inputs, idx["test"]), y[idx["test"]])

    # ANFIS destylowany: cel = predykcje LSTM (na zbiorach train/val)
    teacher = torch.zeros_like(y)
    for k in ("train", "val"):
        teacher[idx[k]] = predict(lstm, seq_inputs, idx[k])
    anfis_kd = ANFIS(feats_np[train_rows], args.n_mfs).to(device)
    anfis_kd.fit_consequents_lse(lse_batches(feat_inputs, teacher, idx["train"]))
    histories["anfis_distilled"] = fit(anfis_kd, feat_inputs, teacher, idx["train"], idx["val"],
                                       epochs=args.epochs_anfis, lr=3e-3, name="ANFIS-KD", **common)
    results["ANFIS destylowany z LSTM"] = metrics(predict(anfis_kd, feat_inputs, idx["test"]), y[idx["test"]])

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": lstm.state_dict(), "seq_mean": seq_mean, "seq_std": seq_std}, args.out / "lstm.pt")
    for name, model in (("anfis", anfis), ("anfis_distilled", anfis_kd)):
        torch.save({"n_mfs": args.n_mfs, "n_inputs": len(ANFIS_FEATURES), "state": model.state_dict()},
                   args.out / f"{name}.pt")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (args.out / "history.json").write_text(json.dumps(histories, indent=2))
    (args.out / "splits.json").write_text(json.dumps({k: v.tolist() for k, v in splits.items()}))

    print(f"\nTest ({len(idx_np['test'])} próbek), błąd prognozy tętna za 30 s [bpm]:")
    for name, m in results.items():
        print(f"  {name:28s} MAE {m['mae']:5.2f}   RMSE {m['rmse']:5.2f}")
    print(f"\nZapisano w {args.out}")


if __name__ == "__main__":
    main()
