"""data/raw/**/*.fit -> data/processed/samples.parquet"""

import argparse
from pathlib import Path

import pandas as pd

from hrforecast.features import add_features
from hrforecast.fit_io import load_activity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/samples.parquet"))
    args = parser.parse_args()

    files = sorted(p for p in args.raw.rglob("*") if p.suffix.lower() == ".fit")
    if not files:
        raise SystemExit(f"Brak plików .fit w {args.raw} — zobacz docs/GARMIN.md")

    frames = []
    for activity_id, path in enumerate(files):
        try:
            df, sport = load_activity(path)
        except Exception as exc:  # uszkodzone / nie-aktywnościowe pliki FIT pomijamy
            print(f"pomijam {path.name}: {exc}")
            continue
        segments = [add_features(seg) for _, seg in df.groupby("segment")]
        if not segments:
            print(f"pomijam {path.name}: brak tętna/prędkości/dystansu")
            continue
        act = pd.concat(segments)
        act["activity"] = activity_id
        act["file"] = path.name
        act["sport"] = sport
        frames.append(act)
        print(f"{path.name}: {sport}, {len(act)} s, {len(segments)} segm.")

    if not frames:
        raise SystemExit("Żaden plik nie zawierał użytecznych danych.")
    out = pd.concat(frames).reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nZapisano {len(out)} próbek z {len(frames)} aktywności -> {args.out}")
    print(out.groupby("sport")["activity"].nunique().rename("aktywności").to_string())


if __name__ == "__main__":
    main()
