"""Cechy i cel predykcji.

Definicje muszą być identyczne z watch/source/HrForecastField.mc —
zegarek liczy te same cechy na żywo z bufora ostatnich WINDOW_S+1 sekund.
"""

import numpy as np
import pandas as pd

WINDOW_S = 30  # okno dla różnicy tętna i nachylenia
SPEED_WINDOW_S = 10  # krótka średnia prędkości — dłuższa zbyt późno widzi zmianę tempa
HORIZON_S = 30  # przewidujemy tętno za tyle sekund
SEQ_LEN = 120  # długość sekwencji wejściowej LSTM (s)
MIN_GRADE_DIST_M = 10.0  # poniżej tego dystansu nachylenie jest szumem
MAX_GRADE = 0.3

# Wejścia ANFIS (i modelu liniowego) — wszystkie policzalne na zegarku.
ANFIS_FEATURES = ["hr", "hr_d30", "speed10", "grade30"]
# Kanały sekwencji LSTM (per sekunda).
SEQ_CHANNELS = ["hr", "speed", "grade30"]
TARGET = "dhr"  # hr(t + HORIZON_S) - hr(t)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje cechy i cel dla jednego ciągłego segmentu 1 Hz."""
    out = df.copy()
    w = WINDOW_S
    out["hr_d30"] = out["hr"] - out["hr"].shift(w)
    out["speed10"] = out["speed"].rolling(SPEED_WINDOW_S).mean()
    d_dist = out["dist"] - out["dist"].shift(w)
    d_alt = out["alt"] - out["alt"].shift(w)
    out["grade30"] = (d_alt / np.maximum(d_dist, MIN_GRADE_DIST_M)).clip(-MAX_GRADE, MAX_GRADE)
    out[TARGET] = out["hr"].shift(-HORIZON_S) - out["hr"]
    out["pos"] = np.arange(len(out))
    return out


def usable_mask(df: pd.DataFrame) -> pd.Series:
    """Wiersze, dla których mamy pełne okno LSTM, wszystkie cechy ANFIS i cel."""
    cols = ANFIS_FEATURES + [TARGET]
    return (df["pos"] >= SEQ_LEN - 1) & df[cols].notna().all(axis=1)
