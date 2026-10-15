# time-series-anfis-fitness
Projekt na przedmiot Sieci neuronowe i systemy rozmyte na Politechnice Wrocławskiej.

**Prognoza tętna 30 s do przodu w trakcie biegu: głęboka sieć LSTM kontra neuro-rozmyty ANFIS, który działa na zegarku Garmin Fenix 8.**

Tętno reaguje na zmianę tempa czy nachylenia z opóźnieniem kilkudziesięciu sekund i z czasem dryfuje w górę.
Porównujemy dwa modele tej dynamiki:

| | LSTM (nauczyciel) | ANFIS (uczeń) |
|---|---|---|
| wejście | surowa sekwencja 120 s: tętno, prędkość, nachylenie | 5 cech: tętno, zmiana tętna w 30 s, średnia prędkość 10 s i 60 s, nachylenie 30 s |
| rozmiar | zespół 3 sieci, ~150 tys. parametrów | ~210 parametrów (32 reguły Sugeno) |
| interpretowalność | brak | reguły „JEŚLI … TO …” i funkcje przynależności |
| gdzie działa | GPU | na zegarku, w Monkey C |

ANFIS uczymy na dwa sposoby: bezpośrednio na danych oraz przez **destylację wiedzy** z LSTM.
Wyniki porównujemy z dwoma punktami odniesienia: persystencją (zakładamy, że tętno się nie zmieni) i regresją liniową.
Podział na zbiory treningowy, walidacyjny i testowy jest robiony po całych aktywnościach.

## Struktura
```
src/hrforecast/
  fit_io.py        FIT -> szereg 1 Hz (interpolacja krótkich luk, podział na segmenty przy pauzach)
  features.py      cechy i cel; te same definicje są w watch/source/HrForecastField.mc
  anfis.py         ANFIS Takagi-Sugeno w PyTorch
  lstm.py          LSTM
  prepare.py       hr-prepare: data/raw -> data/processed/samples.parquet
  train.py         hr-train: trening wszystkich modeli i metryki -> runs/<data>/
  report.py        wykresy i tabele do sprawozdania -> runs/<data>/report/ (hr-report odtwarza je bez treningu)
  online.py        douczanie ANFIS na zegarku (LMS); hr-online dobiera krok uczenia na zbiorze testowym
  export_watch.py  hr-export: ANFIS -> watch/source/AnfisParams.mc
watch/             pole danych Connect IQ (Monkey C) dla Fenixa 8
docs/GARMIN.md     podłączenie zegarka, pobieranie danych, budowanie i wgrywanie
docs/OPIS_PROJEKTU.md  szczegółowy opis kodu, treningu, wyników i każdego wykresu (także dla asystentów AI)
```

## Szybki start
```powershell
uv sync
uv run pytest                 # testy na danych syntetycznych, w tym zgodność eksportu z wnioskowaniem na zegarku
# skopiuj pliki .fit do data/raw/  (docs/GARMIN.md)
uv run hr-prepare
uv run hr-train
uv run hr-online runs\<data>\anfis_distilled.pt
uv run hr-export runs\<data>\anfis_distilled.pt --eta <eta z hr-online>
.\watch\build.ps1 -Device fenix847mm
```
