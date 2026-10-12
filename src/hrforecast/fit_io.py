"""Wczytywanie plików FIT z zegarka do szeregu czasowego 1 Hz."""

from pathlib import Path

import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream

# Luki krótsze niż to (np. "smart recording", chwilowa utrata sygnału) interpolujemy;
# dłuższe (pauza, auto-pauza) dzielą trening na niezależne segmenty.
MAX_GAP_S = 10


def _first(record: dict, *keys: str) -> float:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return float(value)
    return np.nan


def _fill_short_gaps(s: pd.Series) -> pd.Series:
    """Interpoluje tylko luki NaN nie dłuższe niż MAX_GAP_S."""
    isna = s.isna()
    gap_len = isna.groupby((~isna).cumsum()).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return filled.where(~isna | (gap_len <= MAX_GAP_S))


def load_activity(path: Path) -> tuple[pd.DataFrame, str]:
    """Zwraca (DataFrame 1 Hz z kolumnami hr, speed, alt, dist, segment; sport)."""
    decoder = Decoder(Stream.from_file(str(path)))
    messages, errors = decoder.read()
    if errors:
        raise ValueError(f"{path.name}: {errors}")

    sessions = messages.get("session_mesgs", [])
    sport = str(sessions[0].get("sport", "unknown")) if sessions else "unknown"

    rows = [
        {
            "timestamp": r["timestamp"],
            "hr": _first(r, "heart_rate"),
            "speed": _first(r, "enhanced_speed", "speed"),
            "alt": _first(r, "enhanced_altitude", "altitude"),
            "dist": _first(r, "distance"),
        }
        for r in messages.get("record_mesgs", [])
        if r.get("timestamp") is not None
    ]
    if not rows:
        return pd.DataFrame(columns=["hr", "speed", "alt", "dist", "segment"]), sport

    df = pd.DataFrame(rows).groupby("timestamp").mean().sort_index()
    df = df.resample("1s").mean()

    # Brak wysokości (bieżnia) -> płasko; brak dystansu -> nie da się policzyć nachylenia.
    if df["alt"].isna().all():
        df["alt"] = 0.0

    df = df.apply(_fill_short_gaps)
    valid = df[["hr", "speed", "alt", "dist"]].notna().all(axis=1)
    df["segment"] = (~valid).cumsum()
    df = df[valid]
    return df, sport
