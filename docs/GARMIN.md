# Podłączenie Garmina: dane, budowanie i wgrywanie pola danych

Instrukcja dla Fenixa 8 z pasem HRM (Windows). Nazwy menu na zegarku mogą się trochę różnić
w zależności od wersji oprogramowania i języka.

## 1. Ustawienia zegarka (raz, przed zbieraniem danych)

1. **Zapis co sekundę.** Domyślny tryb „Inteligentny” zapisuje punkty nieregularnie, co psuje szereg czasowy.
   _Ustawienia → System → Zapis danych → Interwał zapisu → Co sekundę._
2. **Sparuj pas HRM.** Załóż pas i zwilż elektrody, a potem:
   _Ustawienia → Czujniki i akcesoria → Dodaj nowy → Tętno._
   Przy następnej aktywności zegarek sam użyje pasa zamiast czujnika optycznego.
   Pas jest tu ważny, bo czujnik optyczny ma opóźnienie i artefakty, a my modelujemy właśnie dynamikę tętna.
3. **Zbieraj zróżnicowane biegi.** Model uczy się, jak tętno reaguje na zmiany obciążenia,
   więc najwięcej dają interwały, podbiegi, fartlek i biegi progresywne.
   Same spokojne biegi w strefie 2 niewiele go nauczą.
   Na start wystarczy około 20–30 aktywności, a im więcej, tym lepiej.

## 2. Pobranie plików FIT do `data/raw/`

Wystarczy dowolny z trzech sposobów. Pliki możesz wrzucać do `data/raw/`, także do podfolderów.

### A) Kabel USB, czyli najszybciej dla ostatnich treningów

1. Podłącz zegarek kablem do komputera.
2. W Eksploratorze Windows zegarek pojawi się w „Ten komputer” jako urządzenie **fēnix 8**.
   To urządzenie MTP, a nie dysk z literą, więc pliki trzeba skopiować ręcznie (przeciągnij i upuść).
3. Otwórz `Internal Storage\GARMIN\Activity\` i skopiuj pliki `*.fit` do `data/raw/`.

Zegarek trzyma tylko ograniczoną historię, więc starsze treningi weź sposobem B lub C.

### B) Pojedyncze treningi z Garmin Connect (www)

Otwórz aktywność na connect.garmin.com, kliknij ikonę zębatki i wybierz **Eksportuj plik oryginalny**.
Pobierzesz `.zip` z plikiem `.fit` w środku. Rozpakuj go do `data/raw/`.

### C) Pełna historia (eksport RODO)

Na garmin.com wejdź w _Konto Garmin → Zarządzanie danymi → Eksportuj swoje dane_.
Link przychodzi mailem, zwykle po kilku dniach. W archiwum, w katalogu `DI_CONNECT/DI-Connect-Uploaded-Files/`,
są paczki zip z oryginalnymi plikami FIT. Rozpakuj je do `data/raw/`.
Pliki, które nie są aktywnościami, `hr-prepare` sam pominie.

### Przetworzenie

```powershell
uv run hr-prepare        # data/raw/**/*.fit -> data/processed/samples.parquet
```

Skrypt wypisuje, ile sekund danych i ile aktywności każdego sportu znalazł.

## 3. Trening

```powershell
uv sync                                   # PyTorch z CUDA 12.8 (wymagane dla RTX 50xx)
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
uv run hr-train                           # domyślnie tylko bieganie; --sport all = wszystko
uv run hr-export runs\<data>\anfis_distilled.pt   # albo anfis.pt, jeśli wypadł lepiej
```

`hr-export` nadpisuje `watch/source/AnfisParams.mc` parametrami wytrenowanego modelu.

## 4. Klucz dewelopera (raz)

Każdy plik `.prg` musi być podpisany. Klucz jest prywatny i nie trafia do gita (`watch/developer_key*` jest w `.gitignore`).

W Git Bash:

```bash
openssl genrsa -out watch/developer_key.pem 4096
openssl pkcs8 -topk8 -inform PEM -outform DER -in watch/developer_key.pem -out watch/developer_key -nocrypt
rm watch/developer_key.pem
```

Możesz też zainstalować w VS Code rozszerzenie **Monkey C** (Garmin) i uruchomić
`Monkey C: Generate a Developer Key`. Klucz zapisz wtedy jako `watch/developer_key` (tak domyślnie nazywa go rozszerzenie).

## 5. Budowanie

```powershell
.\watch\build.ps1 -Device fenix847mm
```

Identyfikatory modeli Fenixa 8 to: `fenix843mm`, `fenix847mm`, `fenix8pro47mm`, `fenix8solar47mm`, `fenix8solar51mm`.
Wybierz swój, bo od tego zależą rozdzielczość i czcionki.

## 6. Test w symulatorze z odtworzeniem prawdziwego treningu

```powershell
$sdk = (Get-Content "$env:APPDATA\Garmin\ConnectIQ\current-sdk.cfg" -Raw).Trim()
& "$sdk\bin\connectiq.bat"                                          # uruchamia symulator
& "$sdk\bin\monkeydo.bat" watch\bin\HrForecast.prg fenix847mm       # ładuje pole danych
```

W symulatorze:

1. _Simulation → FIT Data → Playback File…_ i wybierz plik `.fit` z `data/raw/`.
   Symulator zacznie podawać tętno, prędkość, wysokość i dystans z tego treningu.
2. Uruchom timer aktywności (_Simulation → Activity Data → Timer: Start_ lub odpowiednik w Twojej wersji SDK).
   Pole pokazuje `--`, dopóki timer nie jest uruchomiony, a po starcie przez pierwsze 60 s, czyli do zapełnienia bufora (średnia prędkość z 60 s).

To jest dobre demo na obronę: obok widać aktualne tętno, a pole „HR +30s” wyprzedza je o pół minuty.

## 7. Wgranie na zegarek (sideload)

1. Zbuduj aplikację dla swojego modelu (krok 5).
2. Podłącz zegarek przez USB i skopiuj `watch\bin\HrForecast.prg` do `Internal Storage\GARMIN\Apps\`.
3. Odłącz zegarek. Pole zostanie zainstalowane po chwili.
4. Dodaj pole do ekranu aktywności: na ekranie startu biegu wejdź w _Menu (przytrzymaj środkowy lewy przycisk) →
   Ustawienia biegu → Ekrany danych → wybierz ekran → Edytuj pole → Connect IQ → HR Forecast_.
   Najlepiej ustawić je obok zwykłego pola „Tętno”.

Aby odinstalować, usuń plik z `GARMIN\Apps\`.
Pola danych wgrane w ten sposób nie przechodzą przez sklep Connect IQ i to wystarczy na potrzeby projektu.
