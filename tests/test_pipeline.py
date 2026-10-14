import numpy as np
import pandas as pd
import torch

from hrforecast.anfis import ANFIS
from hrforecast.export_watch import flat_params, to_monkey_c
from hrforecast.features import ANFIS_FEATURES, HORIZON_S, TARGET, add_features, usable_mask
from hrforecast.train import fit, lse_batches, predict


def synthetic_run(seed: int, n: int = 1800) -> pd.DataFrame:
    """Tętno jako odpowiedź inercyjna (tau=60 s) na tempo zmieniane co 5 min i nachylenie."""
    rng = np.random.default_rng(seed)
    speed = np.repeat(rng.uniform(2.0, 5.0, n // 300 + 1), 300)[:n]
    grade = np.repeat(rng.uniform(-0.05, 0.08, n // 300 + 1), 300)[:n]
    hr = np.empty(n)
    hr[0] = 100.0
    for t in range(1, n):
        steady = 80 + 18 * speed[t] + 300 * grade[t]
        hr[t] = hr[t - 1] + (steady - hr[t - 1]) / 60
    dist = np.cumsum(speed)
    alt = np.cumsum(grade * speed)
    return pd.DataFrame({"hr": hr + rng.normal(0, 1, n), "speed": speed, "alt": alt, "dist": dist})


def watch_predict(params: dict[str, np.ndarray], n_mfs: int, x: np.ndarray) -> float:
    """Wierne odwzorowanie pętli z watch/source/Anfis.mc."""
    n = len(x)
    z = (x - params["MEAN"]) / params["STD"]
    mu = np.exp(-0.5 * ((np.repeat(z, n_mfs) - params["CENTERS"]) / params["SIGMAS"]) ** 2)
    num = den = 0.0
    for r in range(len(params["CONSEQ"]) // (n + 1)):
        rem, w = r, 1.0
        for i in range(n - 1, -1, -1):
            w *= mu[i * n_mfs + rem % n_mfs]
            rem //= n_mfs
        p = params["CONSEQ"][r * (n + 1) : (r + 1) * (n + 1)]
        num += w * (p[n] + p[:n] @ z)
        den += w
    return num / (den + 1e-12)


def test_features_target_alignment():
    df = add_features(synthetic_run(0))
    t = 500
    assert df[TARGET].iloc[t] == df["hr"].iloc[t + HORIZON_S] - df["hr"].iloc[t]
    assert usable_mask(df).sum() == len(df) - 119 - HORIZON_S


def test_anfis_learns_and_beats_persistence():
    torch.manual_seed(0)
    df = pd.concat([add_features(synthetic_run(s)) for s in range(8)], ignore_index=True)
    rows = np.flatnonzero(usable_mask(df).to_numpy())
    x = torch.tensor(df[ANFIS_FEATURES].to_numpy(), dtype=torch.float32).nan_to_num()
    y = torch.tensor(df[TARGET].to_numpy(), dtype=torch.float32).nan_to_num()
    train, val = torch.tensor(rows[: len(rows) * 3 // 4]), torch.tensor(rows[len(rows) * 3 // 4 :])

    model = ANFIS(x[train].numpy(), n_mfs=2)
    model.fit_consequents_lse(lse_batches(lambda b: x[b], y, train))
    fit(model, lambda b: x[b], y, train, val, epochs=15, lr=3e-3, batch_size=256, patience=5, name="test")
    mae = (predict(model, lambda b: x[b], val) - y[val]).abs().mean().item()
    persistence = y[val].abs().mean().item()
    assert mae < 0.75 * persistence


def test_watch_export_matches_torch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal([150, 0, 3, 3, 0], [15, 5, 0.6, 0.5, 0.03], size=(500, 5))
    model = ANFIS(x, n_mfs=3)
    with torch.no_grad():
        model.consequents.normal_()
        model.centers.add_(0.1 * torch.randn_like(model.centers))
    expected = model(torch.tensor(x[:20], dtype=torch.float32)).detach().numpy()
    params = flat_params(model)
    got = np.array([watch_predict(params, 3, row.astype(np.float32)) for row in x[:20]])
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert "const CONSEQ = [" in to_monkey_c(model)
