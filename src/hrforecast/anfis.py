"""ANFIS typu Takagi-Sugeno pierwszego rzędu, siatka reguł, gaussowskie funkcje przynależności.

Warstwy (Jang, 1993):
  1. rozmycie: mu_ij = exp(-0.5 * ((z_i - c_ij) / s_ij)^2)
  2. siła reguły: w_r = prod_i mu_{i, k_ri}   (iloczyn jako t-norma)
  3. normalizacja: wn_r = w_r / sum(w)
  4. następniki: f_r = p_r . z + q_r
  5. wyjście: y = sum_r wn_r * f_r
Uczenie: następniki inicjalizowane metodą najmniejszych kwadratów (krok LSE algorytmu
hybrydowego Janga), następnie wszystkie parametry dostrajane gradientowo (Adam).
"""

import itertools
from pathlib import Path

import numpy as np
import torch
from torch import nn

EPS = 1e-12


class ANFIS(nn.Module):
    def __init__(self, x_train: np.ndarray, n_mfs: int = 3):
        super().__init__()
        n = x_train.shape[1]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0) + 1e-6
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        # Środki w kwantylach danych, szerokość ~ połowa odstępu między środkami.
        z = (x_train - mean) / std
        centers = np.quantile(z, np.linspace(0.1, 0.9, n_mfs), axis=0).T  # (n, M)
        spacing = (centers[:, -1] - centers[:, 0]) / max(n_mfs - 1, 1)
        sigmas = np.repeat(np.maximum(spacing / 2, 0.1)[:, None], n_mfs, axis=1)
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.log_sigmas = nn.Parameter(torch.tensor(np.log(sigmas), dtype=torch.float32))

        # Reguła r = kombinacja indeksów MF; ostatnie wejście zmienia się najszybciej
        # (ta sama kolejność co w Anfis.mc na zegarku).
        rules = list(itertools.product(range(n_mfs), repeat=n))
        self.register_buffer("rules", torch.tensor(rules, dtype=torch.long))  # (R, n)
        self.consequents = nn.Parameter(torch.zeros(len(rules), n + 1))  # [p_1..p_n, q]

    @property
    def sigmas(self) -> torch.Tensor:
        return self.log_sigmas.exp()

    def firing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zwraca (z — wejścia znormalizowane, wn — znormalizowane siły reguł)."""
        z = (x - self.mean) / self.std
        mu = torch.exp(-0.5 * ((z[:, :, None] - self.centers) / self.sigmas) ** 2)  # (B, n, M)
        inputs = torch.arange(z.shape[1], device=x.device)[None, :]
        w = mu[:, inputs, self.rules].prod(dim=-1)  # (B, R)
        return z, w / (w.sum(dim=1, keepdim=True) + EPS)

    @torch.no_grad()
    def fit_consequents_lse(self, batches, ridge: float = 1.0) -> None:
        """Krok LSE z hybrydowego algorytmu Janga: przy ustalonych przesłankach wyjście
        jest liniowe względem następników, więc wyznaczamy je metodą najmniejszych kwadratów.
        `batches` to iterowalne pary (x, y); równania normalne sumujemy po porcjach."""
        k = self.consequents.numel()
        ata = torch.zeros(k, k, dtype=torch.float64, device=self.mean.device)
        aty = torch.zeros(k, dtype=torch.float64, device=self.mean.device)
        for x, y in batches:
            z, wn = self.firing(x)
            z1 = torch.cat([z, torch.ones_like(z[:, :1])], dim=1)  # (B, n+1)
            a = (wn[:, :, None] * z1[:, None, :]).reshape(len(x), k).double()
            ata += a.T @ a
            aty += a.T @ y.double()
        ata += ridge * torch.eye(k, dtype=torch.float64, device=ata.device)
        self.consequents.copy_(torch.linalg.solve(ata, aty).reshape(self.consequents.shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, wn = self.firing(x)
        f = z @ self.consequents[:, :-1].T + self.consequents[:, -1]  # (B, R)
        return (wn * f).sum(dim=1)


def load_anfis(path: Path) -> ANFIS:
    ckpt = torch.load(path, map_location="cpu")
    model = ANFIS(np.zeros((2, ckpt["n_inputs"])), ckpt["n_mfs"])
    model.load_state_dict(ckpt["state"])
    return model


def flat_params(model: ANFIS) -> dict[str, np.ndarray]:
    """Parametry spłaszczone dokładnie tak, jak czyta je Anfis.mc."""
    with torch.no_grad():
        return {
            "MEAN": model.mean.cpu().numpy(),
            "STD": model.std.cpu().numpy(),
            "CENTERS": model.centers.cpu().numpy().ravel(),  # [wejście][mf]
            "SIGMAS": model.sigmas.cpu().numpy().ravel(),
            "CONSEQ": model.consequents.cpu().numpy().ravel(),  # [reguła][p_1..p_n, q]
        }
