"""Eksport wytrenowanego ANFIS do stałych Monkey C (watch/source/AnfisParams.mc)."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from hrforecast.anfis import ANFIS, flat_params, load_anfis
from hrforecast.features import ANFIS_FEATURES, HORIZON_S
from hrforecast.online import DEFAULT_ETA, ERR_CLIP, HIT_BPM, NLMS_EPS

DEFAULT_OUT = Path("watch/source/AnfisParams.mc")


def to_monkey_c(model: ANFIS, eta: float = DEFAULT_ETA) -> str:
    n_mfs = model.centers.shape[1]

    def arr(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.6f}" for v in values) + "] as Array<Float>"

    params = [f"    const {k} = {arr(v)};" for k, v in flat_params(model).items()]
    # Zegarek trzyma douczone następniki tylko dla modelu o tym samym identyfikatorze,
    # więc eksport nowego modelu zaczyna douczanie od zera.
    model_id = hashlib.sha1("\n".join(params).encode()).hexdigest()[:12]
    lines = [
        "// Wygenerowane przez `uv run hr-export` — nie edytuj ręcznie.",
        f"// Wejścia: {', '.join(ANFIS_FEATURES)}; wyjście: zmiana tętna za {HORIZON_S} s.",
        "import Toybox.Lang;",
        "",
        "module AnfisParams {",
        f"    const N_INPUTS = {len(ANFIS_FEATURES)};",
        f"    const N_MFS = {n_mfs};",
        f"    const HORIZON_S = {HORIZON_S};",
        f'    const MODEL_ID = "{model_id}";',
        "    // Douczanie online (src/hrforecast/online.py)",
        f"    const ETA = {eta:.6f};",
        f"    const ERR_CLIP = {ERR_CLIP:.6f};",
        f"    const NLMS_EPS = {NLMS_EPS:.6f};",
        f"    const HIT_BPM = {HIT_BPM:.6f};",
        *params,
        "}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="np. runs/<data>/anfis_distilled.pt")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA,
                        help="krok douczania na zegarku (dobierz przez hr-online; 0 = bez douczania)")
    args = parser.parse_args()
    args.out.write_text(to_monkey_c(load_anfis(args.model), args.eta), encoding="utf-8")
    print(f"Zapisano {args.out}")


if __name__ == "__main__":
    main()
