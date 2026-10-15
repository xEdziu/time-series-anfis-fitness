"""Douczanie ANFIS w trakcie biegu — odpowiednik watch/source/Anfis.mc i HrForecastField.mc.

Przy ustalonych przesłankach wyjście ANFIS jest liniowe względem następników:
  y = theta . phi(x),  phi_r = wn_r * [z_1..z_n, 1]
więc zegarek doucza tylko następniki (theta, układ jak CONSEQ) znormalizowanym LMS:
  theta += eta * e * phi / (phi . phi + EPS),  e = clip(y - theta . phi, +-ERR_CLIP)
Odpowiedź y = hr(t + HORIZON_S) - hr(t) jest znana po HORIZON_S sekundach, więc krok dla próbki t
wykonujemy dopiero wtedy; prognoza w chwili t używa tylko odpowiedzi znanych do chwili t.

`hr-online` odtwarza to na zbiorze testowym zapisanego uruchomienia i porównuje kroki uczenia eta.
"""

import argparse
import itertools
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from hrforecast.anfis import EPS, flat_params, load_anfis
from hrforecast.features import ANFIS_FEATURES, HORIZON_S, TARGET

DEFAULT_ETA = 0.01
ERR_CLIP = 15.0  # bpm; pojedynczy artefakt z pasa HRM nie może mocno przestawić modelu
NLMS_EPS = 1e-6
HIT_BPM = 5.0  # prognoza „trafiona”, gdy błąd <= tyle bpm
ETAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]


def regressors(params: dict[str, np.ndarray], n_mfs: int, x: np.ndarray) -> np.ndarray:
    """phi dla porcji wejść x (B, n) -> (B, R * (n + 1))."""
    p = {k: v.astype(np.float64) for k, v in params.items()}
    n = x.shape[1]
    z = (x - p["MEAN"]) / p["STD"]
    centers, sigmas = p["CENTERS"].reshape(n, n_mfs), p["SIGMAS"].reshape(n, n_mfs)
    mu = np.exp(-0.5 * ((z[:, :, None] - centers) / sigmas) ** 2)  # (B, n, M)
    rules = np.array(list(itertools.product(range(n_mfs), repeat=n)))  # ostatnie wejście najszybciej
    w = mu[:, np.arange(n), rules].prod(axis=-1)  # (B, R)
    wn = w / (w.sum(axis=1, keepdims=True) + EPS)
    z1 = np.c_[z, np.ones(len(z))]
    return (wn[:, :, None] * z1[:, None, :]).reshape(len(x), -1)


def nlms_step(theta: np.ndarray, phi: np.ndarray, y: float, eta: float) -> None:
    e = np.clip(y - phi @ theta, -ERR_CLIP, ERR_CLIP)
    theta += eta * e * phi / (phi @ phi + NLMS_EPS)


def replay(phi: np.ndarray, y: np.ndarray, segment: np.ndarray, pos: np.ndarray,
           theta: np.ndarray, eta: float) -> np.ndarray:
    """Jedna aktywność w kolejności czasu: prognoza, a krok uczenia, gdy odpowiedź jest już znana.
    Zmienia theta w miejscu i zwraca prognozy. Po przerwie (nowy segment) oczekujące próbki
    przepadają — tak jak na zegarku, gdzie pauza czyści bufor."""
    pred = np.empty(len(y))
    pending: deque[int] = deque()
    for i in range(len(y)):
        while pending and (segment[pending[0]] != segment[i] or pos[pending[0]] + HORIZON_S <= pos[i]):
            j = pending.popleft()
            if segment[j] == segment[i]:
                nlms_step(theta, phi[j], y[j], eta)
        pred[i] = phi[i] @ theta
        pending.append(i)
    return pred


def evaluate(acts: list[tuple[np.ndarray, pd.DataFrame]], theta0: np.ndarray, eta: float, carry: bool) -> np.ndarray:
    """Błędy prognoz po kolei dla wszystkich aktywności. carry=True: theta przechodzi między
    aktywnościami (jak zapis w Application.Storage na zegarku), False: każda zaczyna od modelu bazowego."""
    theta = theta0.copy()
    errs = []
    for phi, g in acts:
        if not carry:
            theta = theta0.copy()
        y = g[TARGET].to_numpy()
        errs.append(replay(phi, y, g["segment"].to_numpy(), g["pos"].to_numpy(), theta, eta) - y)
    return np.concatenate(errs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="np. runs/<data>/anfis_distilled.pt (obok musi być predictions.parquet)")
    parser.add_argument("--etas", type=float, nargs="+", default=ETAS)
    args = parser.parse_args()

    params = flat_params(load_anfis(args.model))
    n_mfs = len(params["CENTERS"]) // len(ANFIS_FEATURES)
    theta0 = params["CONSEQ"].astype(np.float64)
    df = pd.read_parquet(args.model.parent / "predictions.parquet").sort_values(["activity", "segment", "pos"])
    acts = [(regressors(params, n_mfs, g[ANFIS_FEATURES].to_numpy()), g) for _, g in df.groupby("activity")]
    print(f"Zbiór testowy: {len(acts)} aktywności, {len(df)} próbek; prognoza tętna za {HORIZON_S} s.")

    naive = df[TARGET].abs()
    rows = []
    for eta in args.etas:
        carried = evaluate(acts, theta0, eta, carry=True)
        fresh = evaluate(acts, theta0, eta, carry=False)
        rows.append({
            "eta": eta,
            "MAE (zapis między biegami)": np.abs(carried).mean(),
            "MAE (każdy bieg od zera)": np.abs(fresh).mean(),
            f"trafienia ±{HIT_BPM:g} bpm": (np.abs(carried) <= HIT_BPM).mean(),
        })
    table = pd.DataFrame(rows)
    out = args.model.parent / f"online_{args.model.stem}.csv"
    table.to_csv(out, index=False)

    print(f"persystencja: MAE {naive.mean():.3f}, trafienia {(naive <= HIT_BPM).mean():.1%}")
    print("eta = 0 to model bazowy bez douczania.\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = table.loc[table["MAE (zapis między biegami)"].idxmin(), "eta"]
    print(f"\nZapisano {out}")
    print(f"Najlepsze eta = {best:g}. Eksport na zegarek: uv run hr-export {args.model} --eta {best:g}")


if __name__ == "__main__":
    main()
