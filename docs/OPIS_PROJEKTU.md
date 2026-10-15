# Opis projektu: kod, dane, modele i wykresy

Ten dokument jest pisany tak, żeby mógł go przeczytać człowiek albo asystent AI bez dostępu do autora.
Opisuje, co robi każdy fragment kodu, jak przebiega trening, co zawiera katalog wyników
i jak czytać każdy wykres oraz każdą tabelę. Na końcu są wyniki referencyjne i wnioski z eksperymentów.

## 1. Cel projektu w jednym akapicie

Przewidujemy, **o ile zmieni się tętno biegacza za 30 sekund**, na podstawie tego, co zegarek wie teraz
(tętno, prędkość, nachylenie). Porównujemy głęboką sieć **LSTM** (dokładniejszą, ale za dużą na zegarek)
z neuro-rozmytym modelem **ANFIS** (mały, interpretowalny, działa na zegarku Garmin Fenix 8).
ANFIS uczymy na dwa sposoby: bezpośrednio na danych oraz przez **destylację wiedzy** z LSTM
(ANFIS uczy się naśladować prognozy LSTM). Punktami odniesienia są persystencja i regresja liniowa.

## 2. Przepływ danych (pipeline)

```
pliki .fit z zegarka (data/raw/)
   │  hr-prepare   (prepare.py + fit_io.py + features.py)
   ▼
data/processed/samples.parquet   ← szereg 1 Hz z cechami i celem
   │  hr-train     (train.py + lstm.py + anfis.py + report.py)
   ▼
runs/<data-godzina>/             ← modele, metryki, predykcje, report/ z wykresami
   │  hr-online    (online.py)        ← dobór kroku douczania eta na zbiorze testowym
   │  hr-export    (export_watch.py)
   ▼
watch/source/AnfisParams.mc      ← parametry ANFIS i stałe douczania jako stałe Monkey C
   │  watch/build.ps1
   ▼
watch/bin/HrForecast.prg         ← pole danych na zegarek
```

Komendy (Windows / PowerShell, w katalogu repozytorium):

```powershell
uv sync                     # instalacja zależności (PyTorch z CUDA 12.8)
uv run pytest               # testy na danych syntetycznych
uv run hr-prepare           # data/raw -> data/processed/samples.parquet
uv run hr-train             # trening wszystkich modeli + raport
uv run hr-report runs\<data>   # ponowne narysowanie wykresów bez treningu
uv run hr-online runs\<data>\anfis_distilled.pt   # dobór kroku douczania na zegarku
uv run hr-export runs\<data>\anfis_distilled.pt --eta <eta>   # model -> zegarek
.\watch\build.ps1 -Device fenix847mm
```

Szczegóły pobierania danych z Garmina, budowania i wgrywania na zegarek: `docs/GARMIN.md`.

## 3. Pliki z kodem

| plik | rola |
|---|---|
| `src/hrforecast/fit_io.py` | Czyta plik FIT i zwraca szereg czasowy 1 Hz: `hr`, `speed`, `alt`, `dist`. Luki ≤ 10 s interpoluje, dłuższe (pauza, auto-pauza) dzielą aktywność na niezależne **segmenty**. |
| `src/hrforecast/features.py` | Definicje cech i celu. **Muszą być identyczne z kodem na zegarku** (`watch/source/HrForecastField.mc`). |
| `src/hrforecast/prepare.py` | `hr-prepare`: przetwarza wszystkie pliki FIT, liczy cechy osobno dla każdego segmentu, zapisuje `samples.parquet`. Pliki bez tętna, prędkości lub dystansu pomija. |
| `src/hrforecast/lstm.py` | `HrLSTM` (2 warstwy LSTM, 64 jednostki, dropout 0,1, warstwa liniowa na wyjściu) oraz `LSTMEnsemble` (średnia z kilku LSTM). |
| `src/hrforecast/anfis.py` | ANFIS Takagi–Sugeno 1. rzędu w PyTorch: gaussowskie funkcje przynależności, siatka reguł, krok LSE z algorytmu hybrydowego Janga. |
| `src/hrforecast/train.py` | `hr-train`: podział danych, trening wszystkich modeli, ewaluacja na zbiorze testowym, zapis wyników, wywołanie raportu. |
| `src/hrforecast/report.py` | Wszystkie wykresy i tabele do sprawozdania. `hr-report` odtwarza je z zapisanych plików, bez treningu. |
| `src/hrforecast/online.py` | Douczanie następników ANFIS znormalizowanym LMS, identyczne jak na zegarku. `hr-online` odtwarza je na zbiorze testowym dla kilku kroków `eta`. |
| `src/hrforecast/export_watch.py` | `hr-export`: zapisuje parametry ANFIS i stałe douczania jako stałe Monkey C. |
| `watch/source/HrForecastField.mc` | Pole danych na zegarku: bufor ostatnich 61 s, liczenie cech na żywo, prognoza, douczanie, skuteczność, ekran ze strefami i tempem. |
| `watch/source/Anfis.mc` | ANFIS w Monkey C: regresory `phi`, prognoza `theta . phi` (odpowiednik `ANFIS.forward`) i krok douczania. |
| `tests/test_pipeline.py` | Testy: wyrównanie celu w czasie, ANFIS uczy się lepiej niż persystencja, eksport na zegarek daje te same wyniki co PyTorch, krok LMS i kolejność douczania (tylko znane odpowiedzi). |

## 4. Dane, cechy i cel

- Dane pochodzą z jednego zegarka (eksport z Garmin Connect). Po filtrowaniu jest **78 aktywności**
  (54 biegi, 8 rower, 8 spacer, 7 wędrówki, 1 pływanie), razem ok. 448 tys. sekund.
  Pozostałe ~3700 plików FIT z eksportu to nie są treningi (monitoring całodobowy itp.).
- Domyślnie trenujemy i testujemy **tylko na biegach** (`--sport running`).
- **Cel** (`dhr`): `hr(t + 30 s) − hr(t)`, czyli zmiana tętna za 30 s w bpm.
  Prognozowane tętno = aktualne tętno + prognoza `dhr`.
- **Wejścia ANFIS i regresji liniowej** (5 cech, wszystkie policzalne na zegarku na żywo):

| cecha | znaczenie |
|---|---|
| `hr` | aktualne tętno [bpm] |
| `hr_d30` | zmiana tętna w ostatnich 30 s [bpm] |
| `speed10` | średnia prędkość z ostatnich 10 s [m/s] |
| `speed60` | średnia prędkość z ostatnich 60 s [m/s]; w parze ze `speed10` mówi, czy tempo właśnie wzrosło czy spadło |
| `grade30` | nachylenie z ostatnich 30 s (Δwysokość / Δdystans, przycięte do ±0,3) |

- **Wejście LSTM**: surowa sekwencja ostatnich 120 s, kanały `hr`, `speed`, `grade30`,
  znormalizowane średnią i odchyleniem ze zbioru treningowego.
- **Próbki użyteczne**: od 120. sekundy segmentu (pełne okno LSTM), ze wszystkimi cechami i znanym celem.
  Dzięki temu wszystkie modele są oceniane na **tych samych** próbkach.

## 5. Jak przebiega trening (`hr-train`)

1. **Podział po całych aktywnościach** (nie po oknach): 70% train, 15% val, 15% test, losowo z `--seed`.
   Sąsiednie okna są prawie identyczne, więc podział po oknach zawyżałby wyniki.
   Przy seedzie 0: 38 / 8 / 8 biegów.
2. **Persystencja**: prognoza `dhr = 0` („tętno się nie zmieni”). Najprostszy punkt odniesienia.
3. **Regresja liniowa** na 5 cechach (metoda najmniejszych kwadratów).
4. **LSTM (nauczyciel)**: zespół `--n-lstm` (domyślnie 3) niezależnie uczonych sieci; prognoza = średnia.
   Każda sieć: Adam, lr 1e-3, MSE, batch 1024, maks. 30 epok, **early stopping** z cierpliwością 5 epok
   na zbiorze walidacyjnym; na koniec wczytywane są wagi z najlepszej epoki.
5. **ANFIS** (uczony na prawdziwym celu), domyślnie 2 funkcje przynależności na wejście,
   czyli 2⁵ = 32 reguły i 212 parametrów:
   - inicjalizacja: środki funkcji przynależności w kwantylach danych, szerokości z odstępów między nimi;
   - **krok LSE**: przy ustalonych przesłankach wyjście jest liniowe względem następników,
     więc wyznaczamy je metodą najmniejszych kwadratów (z regularyzacją ridge = 1);
   - **dostrajanie gradientowe** wszystkich parametrów: Adam, lr 3e-3, maks. 60 epok, early stopping.
   - Wynik zapisujemy dwa razy: „ANFIS (tylko LSE)” i „ANFIS” (po dostrajaniu).
6. **ANFIS destylowany z LSTM (ANFIS-KD)**: to samo co w punkcie 5, ale celem są **prognozy zespołu LSTM**
   na zbiorach train i val. Na zbiorze testowym zawsze porównujemy z **prawdziwym** tętnem.
7. Zapis wyników i generowanie raportu.

Ważne opcje: `--sport` (`all` = wszystkie sporty), `--n-mfs`, `--n-lstm`, `--epochs-lstm`, `--epochs-anfis`,
`--seed`, `--out`.

## 6. Co jest w katalogu `runs/<data-godzina>/`

| plik | zawartość |
|---|---|
| `metrics.json` | Dla każdego modelu na zbiorze testowym: `mae`, `rmse` [bpm], `r2`, `parametry`, `czas_uczenia_s`. Zawiera też wyniki pojedynczych sieci z zespołu (`LSTM #1 (pojedynczy)` itd.). |
| `history.json` | MSE na train i val w każdej epoce, osobno dla każdego LSTM i obu ANFIS. |
| `splits.json` | Identyfikatory aktywności w zbiorach train / val / test. |
| `config.json` | Wszystkie argumenty uruchomienia i urządzenie (cuda/cpu). |
| `predictions.parquet` | Wiersz = próbka testowa: `activity`, `segment`, `pos` (sekunda w segmencie), `hr`, cechy, `dhr` (prawda) i kolumna z prognozą `dhr` każdego modelu. Z tego pliku da się odtworzyć wszystkie wykresy błędów. |
| `lstm.pt`, `anfis.pt`, `anfis_distilled.pt` | Wagi modeli. Na zegarek eksportuje się jeden z plików ANFIS. |
| `report/` | Wykresy (każdy jako `.png` do podglądu i `.pdf` wektorowo do LaTeX-a) oraz tabele `.csv`. |

**Metryki**: MAE = średni błąd bezwzględny, RMSE = pierwiastek z błędu średniokwadratowego
(mocniej karze duże pomyłki), R² = jaka część zmienności `dhr` jest wyjaśniona
(0 = tyle co przewidywanie średniej, 1 = idealnie). Wszystkie liczone na zbiorze testowym.

## 7. Wykresy w `report/` i jak je czytać

Każdy model ma stały kolor na wszystkich wykresach. Warianty „(tylko LSE)” pojawiają się tylko
na wykresie metryk i w tabelach, żeby nie zaśmiecać pozostałych wykresów.

### `krzywe_uczenia`
- **Co pokazuje**: dla każdej uczonej sieci (każdy LSTM z zespołu, ANFIS, ANFIS-KD) MSE na zbiorze
  treningowym (niebieska linia) i walidacyjnym (pomarańczowa) w kolejnych epokach.
  Kropka oznacza epokę z najniższym błędem walidacyjnym, czyli wagi, które zostały zachowane.
- **Jak czytać**: jeśli błąd treningowy spada, a walidacyjny rośnie, model się przeucza
  (tak jest w LSTM po kilku epokach). Płaska krzywa ANFIS oznacza, że krok LSE dał już prawie optimum.
- **Uwaga**: krzywe ANFIS-KD mają inną skalę, bo cel to prognozy LSTM, a nie prawdziwe tętno
  (prognozy LSTM są gładsze, więc łatwiej je naśladować).

### `porownanie_metryk`
- **Co pokazuje**: słupki MAE i RMSE na zbiorze testowym dla wszystkich modeli, z wartościami liczbowymi.
- **Jak czytać**: niżej = lepiej. Kluczowe porównania do sprawozdania:
  ANFIS vs persystencja (czy model w ogóle coś daje), ANFIS-KD vs ANFIS (czy destylacja pomaga),
  ANFIS-KD vs LSTM (ile tracimy, schodząc z dużej sieci na model mieszczący się na zegarku),
  „tylko LSE” vs pełny ANFIS (ile daje dostrajanie gradientowe).

### `prognoza_vs_rzeczywistosc`
- **Co pokazuje**: wykres rozrzutu, osobny panel dla każdego modelu. Oś X: prawdziwa zmiana tętna za 30 s,
  oś Y: prognoza. Przerywana linia y = x to prognoza idealna. Pokazane jest losowe 20 tys. punktów,
  osie obcięte na 99,5. percentylu.
- **Jak czytać**: im bardziej chmura układa się wzdłuż przekątnej, tym lepiej. Chmura bardziej płaska
  niż przekątna oznacza, że model jest „ostrożny” i niedoszacowuje dużych zmian
  (typowe przy niskim R²). Persystencja to pozioma linia w zerze.

### `rozklad_bledow`
- **Co pokazuje**: histogramy błędu (prognoza − prawda) każdego modelu, znormalizowane do gęstości,
  bez 1% najbardziej skrajnych wartości.
- **Jak czytać**: węższy i wyższy pik wokół zera = lepszy model. Przesunięcie piku od zera
  oznacza systematyczne zawyżanie lub zaniżanie.

### `blad_wg_zmiany_tetna` (+ tabela `.csv`)
- **Co pokazuje**: MAE każdego modelu w przedziałach prawdziwej zmiany tętna
  (od „< −20 bpm” do „> 20 bpm”). Tabela zawiera też liczbę próbek w każdym przedziale.
- **Jak czytać**: tutaj widać, **gdzie** modele zawodzą. Dla małych zmian (−2…2 bpm) wszystkie modele
  są dobre, a duże zmiany (początek interwału, zatrzymanie) są trudne dla każdego.
  Przedziały skrajne mają mało próbek, więc ich wartości są mniej pewne (sprawdź kolumnę z liczbą próbek).

### `blad_na_aktywnosc` (+ tabela `.csv`)
- **Co pokazuje**: wykres pudełkowy MAE liczonego osobno dla każdej aktywności testowej;
  kropki to pojedyncze biegi.
- **Jak czytać**: pokazuje, jak bardzo wynik zależy od konkretnego biegu. Duży rozrzut
  (np. 2–6 bpm) oznacza, że różnice między modelami rzędu 0,1 bpm są mniejsze niż różnice
  między biegami. Warto o tym napisać przy interpretacji wyników.

### `przyklady_przebiegow`
- **Co pokazuje**: 3 najdłuższe ciągłe fragmenty biegów testowych w czasie.
  Szara linia: tętno teraz (to jest też prognoza persystencji). Czarna: prawdziwe tętno za 30 s.
  Kolorowe: prognozy tętna za 30 s z LSTM, ANFIS i ANFIS-KD.
- **Jak czytać**: dobra prognoza powinna leżeć na czarnej linii, czyli wyprzedzać szarą o 30 s.
  Gdy kolorowe linie pokrywają się z szarą, model w praktyce powtarza aktualne tętno.
  Przy gwałtownych zmianach tempa widać, które modele reagują wcześniej.

### `funkcje_przynaleznosci_ANFIS` i `funkcje_przynaleznosci_ANFIS-KD`
- **Co pokazuje**: dla każdej z 5 cech funkcje przynależności ANFIS w jednostkach fizycznych
  (bpm, m/s itd.). Linia kropkowana: stan początkowy, ciągła: po uczeniu.
  Kolory to zbiory rozmyte („niskie”, „wysokie”; przy 3 MF także „średnie”).
  Szare tło: histogram wartości cechy w danych treningowych.
- **Jak czytać**: pokazuje interpretowalność ANFIS, np. od ilu bpm tętno jest „wysokie”.
  Małe przesunięcia linii ciągłych względem kropkowanych oznaczają, że dostrajanie gradientowe
  niewiele zmieniło (zgodnie z krzywymi uczenia).

### `aktywnosc_regul` (+ tabele `reguly_ANFIS.csv`, `reguly_ANFIS-KD.csv`)
- **Co pokazuje**: 15 reguł o największej średniej znormalizowanej sile odpalenia na zbiorze testowym,
  osobno dla ANFIS i ANFIS-KD.
- **Tabele `reguly_*.csv`**: wszystkie reguły w czytelnej postaci, np.
  „hr jest wysokie I hr_d30 jest niskie I …”, ze średnią aktywacją i współczynnikami następnika
  (`p_<cecha>` dla każdej cechy oraz wyraz wolny `q`). Uwaga: następnik działa na cechach
  **znormalizowanych** (z = (x − średnia) / odchylenie), więc `p` mówi o kierunku i względnej sile wpływu,
  a nie o bpm na jednostkę cechy.
- **Jak czytać**: kilka reguł zwykle dominuje (opisują typowy bieg), a reszta obsługuje rzadkie sytuacje.
  Do sprawozdania warto opisać słownie 2–3 najczęstsze reguły.

### `rozklady_danych` (+ tabela `statystyki_danych.csv`)
- **Co pokazuje**: rozkłady 5 cech i celu `dhr` w zbiorach train, val i test (zakres 0,5–99,5 percentyla).
  Tabela zawiera statystyki opisowe (liczność, średnia, odchylenie, kwantyle) dla każdego zbioru.
- **Jak czytać**: sprawdza, czy zbiory są do siebie podobne. Jeśli test wyraźnie różni się od train
  (np. inne tempo), wyniki testowe mogą być zaskakująco dobre lub złe.

### Pozostałe tabele
- `metryki.csv`: to samo co `metrics.json`, w formie tabeli (jeden wiersz = model).

## 8. Część na zegarku

- `HrForecastField.mc` co sekundę dopisuje tętno, prędkość, wysokość i dystans do bufora 61 s,
  liczy 5 cech **dokładnie tak jak `features.py`** i prognozuje `hr + ANFIS(cechy)`.
  Przez pierwsze 60 s po starcie (lub po pauzie) prognoza to `--`, bo bufor się zapełnia.
- Ekran (`DataField.onUpdate`, układ z 1 polem): łuk stref tętna z `UserProfile.getHeartRateZones`
  dla bieżącego sportu (znacznik pełny = tętno teraz, pusty = prognoza), prognoza, tętno teraz ze strefą,
  tempo ze średniej prędkości z 10 s oraz skuteczność od startu aktywności. W mniejszym polu tylko prognoza.
  Opis dla użytkownika: `docs/GARMIN.md`, kroki 6–7.
- `Anfis.mc` jest ogólny: liczba wejść i funkcji przynależności pochodzi z `AnfisParams.mc`.
- **Douczanie online.** Przy ustalonych funkcjach przynależności wyjście ANFIS jest liniowe względem następników:
  `y = theta . phi(x)`, `phi_r = wn_r · [z, 1]`. Zegarek doucza więc tylko następniki (192 liczby przy 5 wejściach
  i 2 MF) znormalizowanym LMS: `theta += eta · e · phi / (phi·phi + eps)`, błąd `e` obcięty do ±15 bpm.
  Krok dla prognozy z chwili `t` jest robiony w chwili `t + 30 s`, gdy prawdziwe tętno jest już znane, a prognoza
  w chwili `t` używa tylko odpowiedzi znanych do `t` (ocena prequential, bez przecieku przyszłości).
  RLS odpada, bo potrzebuje macierzy 192 × 192 i tyle samo operacji co sekundę.
  Douczone `theta` trafia do `Application.Storage` razem z `MODEL_ID` (skrót parametrów modelu bazowego),
  więc nowy eksport zaczyna od zera.
- **Skuteczność na zegarku**: równolegle liczona jest prognoza modelu bazowego (`AnfisParams.CONSEQ`)
  i naiwna (tętno się nie zmieni). Ekran pokazuje MAE wszystkich trzech od początku aktywności.
- **`hr-online`** odtwarza ten sam algorytm na zbiorze testowym (`predictions.parquet` obok modelu) w dwóch wariantach:
  z zapisem `theta` między biegami (jak na zegarku) i z każdym biegiem od modelu bazowego. Wynik zapisuje do
  `runs/<data>/online_<model>.csv` i podpowiada `eta`. `eta = 0` to model bazowy, czyli ten sam wynik co w `metrics.json`.
  Zbiór testowy to tylko kilka biegów, więc korzyść z douczania między biegami jest tu niedoszacowana.
- **Niezmiennik**: każda zmiana cech w `features.py` wymaga tej samej zmiany w `HrForecastField.mc`,
  ponownego `hr-prepare`, `hr-train` i `hr-export`. Test `test_watch_export_matches_torch` sprawdza
  zgodność wnioskowania, ale nie liczenia cech na zegarku. Tak samo każda zmiana douczania w `online.py`
  (krok, obcięcie błędu, moment aktualizacji) wymaga tej samej zmiany w `Anfis.mc` i `HrForecastField.mc`.

## 9. Wyniki referencyjne

Uruchomienie `runs/20261013-223529`: tylko biegi, seed 0, 2 MF, zespół 3 LSTM, test = 8 biegów (22 356 próbek).

| model | MAE [bpm] | RMSE [bpm] | R² | parametry |
|---|---|---|---|---|
| persystencja (dhr = 0) | 3,37 | 5,13 | 0,00 | 0 |
| regresja liniowa | 3,28 | 4,87 | 0,10 | 6 |
| LSTM (zespół 3 sieci) | **3,04** | **4,53** | **0,22** | 153 027 |
| LSTM, pojedyncze sieci | 3,06–3,13 | 4,54–4,66 | 0,17–0,22 | 51 009 |
| ANFIS (tylko LSE) | 3,23 | 4,73 | 0,15 | 212 |
| ANFIS | 3,22 | 4,74 | 0,15 | 212 |
| ANFIS-KD (tylko LSE) | 3,14 | 4,71 | 0,16 | 212 |
| ANFIS-KD | **3,13** | 4,70 | 0,16 | 212 |

Najważniejsze wnioski:
- ANFIS-KD jest ok. 7% lepszy od persystencji, a od LSTM gorszy tylko o 0,1 bpm, mając ~720 razy mniej parametrów.
- Destylacja pomaga: ANFIS-KD jest lepszy od ANFIS uczonego bezpośrednio (3,13 vs 3,22).
- Dostrajanie gradientowe po kroku LSE prawie nic nie daje (różnice ≤ 0,01 bpm).
- Zespół LSTM jest lepszy i stabilniejszy niż pojedyncza sieć.
- R² ≈ 0,2 nawet dla LSTM: zmiany tętna za 30 s w dużej mierze nie da się przewidzieć z przeszłości
  (model nie wie, że biegacz zaraz przyspieszy).

## 10. Wnioski z eksperymentów projektowych

Te eksperymenty wykonano w trakcie projektowania (3-krotna walidacja krzyżowa po biegach).
Skryptów nie ma w repozytorium. Liczby to średnie MAE z 3 foldów, więc różnią się od tabeli w punkcie 9.

- **Liczba funkcji przynależności**: 2 MF (16 reguł przy 4 cechach) było lepsze niż 3 (81 reguł) i 4 (256 reguł).
  Więcej reguł = przeuczenie. Stąd domyślne `--n-mfs 2`.
- **Nowe cechy dla ANFIS**: spośród sprawdzonych kandydatów (zmiany tętna w 10, 60, 120 s,
  średnie prędkości 30, 60, 120 s, różnice prędkości, tętno względem średniej z 120 s, czas od startu)
  najwięcej dała średnia prędkość z 60 s: MAE 3,38 → 3,32. Kolejne cechy dawały < 0,005 bpm.
  Czas od startu (dryf tętna) pogarszał wynik.
- **Dane z innych sportów** (rower, spacer, wędrówki): dla LSTM poprawa o ok. 0,02 bpm (w granicach szumu,
  odchylenie między foldami ~0,1), dla ANFIS-KD lekkie pogorszenie. Inne sporty mają inną zależność
  tętna od prędkości. Dlatego domyślnie trenujemy tylko na biegach.
- **Zespół 3 LSTM vs pojedynczy**: LSTM lepszy o ok. 0,07 bpm; uczeń ANFIS-KD zyskuje na tym tylko ok. 0,01 bpm.
- **Wąskie gardło**: wszystkie warianty ANFIS-KD mieściły się w wąskim przedziale niezależnie od nauczyciela.
  Ograniczeniem jest informacja zawarta w cechach wejściowych, a nie sposób uczenia.

## 11. Ograniczenia, o których warto napisać

- Dane z jednej osoby i jednego zegarka; 54 biegi. Model nie musi działać dla innych biegaczy.
- Zbiór testowy to tylko 8 biegów. Wyniki zależą od podziału (`--seed`); do sprawozdania
  najlepiej uruchomić trening dla kilku seedów i podać średnią ± odchylenie:
  `0..4 | % { uv run hr-train --seed $_ --out runs\seed$_ }`.
- Tętno z czujnika nadgarstkowego ma opóźnienia i artefakty (widoczne jako schodki w przebiegach).
- Kod Monkey C nie ma automatycznych testów liczenia cech; należy go sprawdzić w symulatorze
  (`docs/GARMIN.md`, krok 6).
