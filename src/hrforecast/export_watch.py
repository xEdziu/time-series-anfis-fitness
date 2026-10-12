"""Eksport wytrenowanego ANFIS do stałych Monkey C (watch/source/AnfisParams.mc)."""

import argparse
from pathlib import Path

import numpy as np
import torch

from hrforecast.anfis import ANFIS
from hrforecast.features import ANFIS_FEATURES, HORIZON_S

DEFAULT_OUT = Path("watch/source/AnfisParams.mc")


def load_anfis(path: Path) -> ANFIS:
    ckpt = torch.load(path, map_location="cpu")
    model = ANFIS(np.zeros((2, ckpt["n_inputs"])), ckpt["n_mfs"])
    model.load_state_dict(ckpt["state"])
    return model


def flat_params(model: ANFIS) -> dict[str, np.ndarray]:
    """Parametry spłaszczone dokładnie tak, jak czyta je Anfis.mc."""
    with torch.no_grad():
        return {
            "MEAN": model.mean.numpy(),
            "STD": model.std.numpy(),
            "CENTERS": model.centers.numpy().ravel(),  # [wejście][mf]
            "SIGMAS": model.sigmas.numpy().ravel(),
            "CONSEQ": model.consequents.numpy().ravel(),  # [reguła][p_1..p_n, q]
        }


def to_monkey_c(model: ANFIS) -> str:
    n_mfs = model.centers.shape[1]

    def arr(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.6f}" for v in values) + "] as Array<Float>"

    lines = [
        "// Wygenerowane przez `uv run hr-export` — nie edytuj ręcznie.",
        f"// Wejścia: {', '.join(ANFIS_FEATURES)}; wyjście: zmiana tętna za {HORIZON_S} s.",
        "import Toybox.Lang;",
        "",
        "module AnfisParams {",
        f"    const N_INPUTS = {len(ANFIS_FEATURES)};",
        f"    const N_MFS = {n_mfs};",
        f"    const HORIZON_S = {HORIZON_S};",
    ]
    lines += [f"    const {k} = {arr(v)};" for k, v in flat_params(model).items()]
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="np. runs/<data>/anfis_distilled.pt")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.write_text(to_monkey_c(load_anfis(args.model)), encoding="utf-8")
    print(f"Zapisano {args.out}")


if __name__ == "__main__":
    main()
