"""Wykresy i tabele do sprawozdania, zapisywane przez hr-train do runs/<data>/report/.

Wszystko jest liczone z predykcji na zbiorze testowym zapisanych w predictions.parquet,
więc wykresy można odtworzyć bez ponownego treningu: `regenerate(runs/<data>)`.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from hrforecast.anfis import ANFIS  # noqa: E402
from hrforecast.features import ANFIS_FEATURES, HORIZON_S, TARGET  # noqa: E402

# Stała kolejność kolorów: model zawsze ma ten sam kolor na każdym wykresie.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FEATURE_LABELS = {
    "hr": "tętno [bpm]",
    "hr_d30": "zmiana tętna w 30 s [bpm]",
    "speed10": "średnia prędkość 10 s [m/s]",
    "speed60": "średnia prędkość 60 s [m/s]",
    "grade30": "nachylenie 30 s [-]",
}
MF_LABELS = {2: ["niskie", "wysokie"], 3: ["niskie", "średnie", "wysokie"]}

plt.rcParams.update({
    "figure.dpi": 110,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984",
    "axes.titleweight": "bold",
    "lines.linewidth": 2,
    "font.size": 10,
})


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):  # PNG do podglądu, PDF (wektorowo) do LaTeX-a
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def _color(models: list[str], name: str) -> str:
    return COLORS[models.index(name) % len(COLORS)]


def _main(models: list[str]) -> list[str]:
    """Modele końcowe — warianty „tylko LSE” są w tabelach i na wykresie metryk, ale zaśmiecałyby resztę."""
    return [m for m in models if "tylko LSE" not in m]


def learning_curves(histories: dict[str, dict[str, list[float]]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(histories), figsize=(4.2 * len(histories), 3.4), squeeze=False)
    for ax, (name, h) in zip(axes[0], histories.items()):
        epochs = np.arange(1, len(h["train"]) + 1)
        ax.plot(epochs, h["train"], color=COLORS[0], label="trening")
        ax.plot(epochs, h["val"], color=COLORS[1], label="walidacja")
        best = int(np.argmin(h["val"]))
        ax.scatter([best + 1], [h["val"][best]], s=40, color=COLORS[1], edgecolor="white", zorder=3)
        ax.annotate(f"najlepsza: ep. {best + 1}", (best + 1, h["val"][best]),
                    textcoords="offset points", xytext=(4, 8), fontsize=8, color="#52514e")
        ax.set(title=name, xlabel="epoka", ylabel="MSE [bpm²]")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(frameon=False)
    fig.suptitle("Krzywe uczenia")
    _save(fig, out, "krzywe_uczenia")


def metrics_bars(results: dict[str, dict[str, float]], out: Path) -> None:
    models = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(models) + 1.5), sharey=True)
    for ax, key in zip(axes, ("mae", "rmse")):
        vals = [results[m][key] for m in models]
        bars = ax.barh(models, vals, color=[_color(models, m) for m in models], height=0.6)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set(xlabel=f"{key.upper()} [bpm]", title=key.upper())
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()  # osie są współdzielone, więc odwracamy raz
    fig.suptitle(f"Błąd prognozy tętna za {HORIZON_S} s (zbiór testowy)")
    _save(fig, out, "porownanie_metryk")


def pred_vs_true(pred: pd.DataFrame, models: list[str], out: Path) -> None:
    shown = _main(models)
    n = len(shown)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False, sharex=True, sharey=True)
    y = pred[TARGET].to_numpy()
    lim = np.quantile(np.abs(y), 0.995)
    sample = np.random.default_rng(0).choice(len(y), min(len(y), 20000), replace=False)
    for ax, m in zip(axes.flat, shown):
        ax.scatter(y[sample], pred[m].to_numpy()[sample], s=3, alpha=0.15, color=_color(models, m), linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], color="#52514e", linewidth=1, linestyle="--")
        ax.set(title=m, xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"rzeczywista zmiana tętna za {HORIZON_S} s [bpm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("prognoza [bpm]")
    fig.suptitle("Prognoza a wartość rzeczywista (zbiór testowy)")
    _save(fig, out, "prognoza_vs_rzeczywistosc")


def error_distribution(pred: pd.DataFrame, models: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    errs = {m: pred[m].to_numpy() - pred[TARGET].to_numpy() for m in _main(models)}
    lim = np.quantile(np.abs(np.concatenate(list(errs.values()))), 0.99)
    bins = np.linspace(-lim, lim, 81)
    for m, e in errs.items():
        ax.hist(e, bins=bins, histtype="step", linewidth=1.6, density=True, color=_color(models, m), label=m)
    ax.set(xlabel="błąd prognozy [bpm]", ylabel="gęstość", title="Rozkład błędów (zbiór testowy)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "rozklad_bledow")


def error_by_bins(pred: pd.DataFrame, models: list[str], out: Path) -> pd.DataFrame:
    """MAE w zależności od wielkości rzeczywistej zmiany tętna — gdzie modele zawodzą."""
    edges = [-np.inf, -20, -10, -5, -2, 2, 5, 10, 20, np.inf]
    labels = ["<-20", "-20..-10", "-10..-5", "-5..-2", "-2..2", "2..5", "5..10", "10..20", ">20"]
    bucket = pd.cut(pred[TARGET], edges, labels=labels)
    table = pd.DataFrame({m: (pred[m] - pred[TARGET]).abs().groupby(bucket, observed=False).mean() for m in models})
    table["liczba próbek"] = bucket.value_counts().reindex(labels)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    for m in _main(models):
        ax.plot(x, table[m], marker="o", markersize=5, color=_color(models, m), label=m)
    ax.set_xticks(x, labels)
    ax.set(xlabel=f"rzeczywista zmiana tętna za {HORIZON_S} s [bpm]", ylabel="MAE [bpm]",
           title="Błąd w zależności od wielkości zmiany tętna")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "blad_wg_zmiany_tetna")
    return table


def per_activity(pred: pd.DataFrame, models: list[str], out: Path) -> pd.DataFrame:
    table = pred.groupby("activity").apply(
        lambda g: pd.Series({m: (g[m] - g[TARGET]).abs().mean() for m in models} | {"próbek": len(g)}),
        include_groups=False,
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    shown = _main(models)
    ax.boxplot([table[m] for m in shown], tick_labels=shown, widths=0.5,
               medianprops={"color": "#0b0b0b"}, patch_artist=True,
               boxprops={"facecolor": "#f4f3f0", "edgecolor": "#8a8984"})
    for i, m in enumerate(shown):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(table))
        ax.scatter(i + 1 + jitter, table[m], s=14, color=_color(models, m), zorder=3)
    ax.set(ylabel="MAE na aktywność [bpm]", title="Zmienność błędu między aktywnościami testowymi")
    ax.tick_params(axis="x", labelrotation=20)
    _save(fig, out, "blad_na_aktywnosc")
    return table


def example_timelines(pred: pd.DataFrame, models: list[str], out: Path, n: int = 3) -> None:
    """Najdłuższe ciągłe segmenty testowe: rzeczywiste hr(t+30) vs prognozy modeli uczonych.
    Persystencja to po prostu hr(t), więc linia „tętno teraz” pokazuje też ją."""
    shown = [m for m in _main(models) if m not in ("persystencja (dhr=0)", "regresja liniowa")]
    longest = pred.groupby(["activity", "segment"]).size().nlargest(n).index
    fig, axes = plt.subplots(len(longest), 1, figsize=(11, 3 * len(longest)), squeeze=False)
    for ax, (act, seg) in zip(axes[:, 0], longest):
        g = pred[(pred["activity"] == act) & (pred["segment"] == seg)].sort_values("pos")
        t = g["pos"].to_numpy() / 60
        ax.plot(t, g["hr"], color="#a3a29d", linewidth=1, label="tętno teraz = persystencja")
        ax.plot(t, g["hr"] + g[TARGET], color="#0b0b0b", linewidth=2, label=f"rzeczywiste tętno za {HORIZON_S} s")
        for m in shown:
            ax.plot(t, g["hr"] + g[m], color=_color(models, m), linewidth=1.2, alpha=0.9, label=m)
        ax.set(title=f"aktywność {act}, segment {seg}", ylabel="tętno [bpm]")
    axes[-1, 0].set_xlabel("czas od początku segmentu [min]")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    _save(fig, out, "przyklady_przebiegow")


def membership_functions(anfis_models: dict[str, tuple[ANFIS, ANFIS]], x_train: np.ndarray, out: Path) -> None:
    """Funkcje przynależności przed i po uczeniu, w jednostkach oryginalnych."""
    for name, (before, after) in anfis_models.items():
        n_in = len(ANFIS_FEATURES)
        fig, axes = plt.subplots(1, n_in, figsize=(4 * n_in, 3.4))
        for i, ax in enumerate(axes):
            lo, hi = np.quantile(x_train[:, i], [0.005, 0.995])
            xs = np.linspace(lo, hi, 400)
            ax.hist(x_train[:, i], bins=60, range=(lo, hi), density=True, color="#e4e3df")
            twin = ax.twinx()
            for model, style in ((before, ":"), (after, "-")):
                mean, std = model.mean[i].item(), model.std[i].item()
                c = model.centers[i].detach().cpu().numpy()
                s = model.sigmas[i].detach().cpu().numpy()
                z = (xs - mean) / std
                for k in range(len(c)):
                    twin.plot(xs, np.exp(-0.5 * ((z - c[k]) / s[k]) ** 2), linestyle=style, color=COLORS[k],
                              label=MF_LABELS.get(len(c), [str(j) for j in range(len(c))])[k] if style == "-" else None)
            twin.set_ylim(0, 1.05)
            twin.grid(False)
            ax.set_yticks([])
            ax.set(xlabel=FEATURE_LABELS.get(ANFIS_FEATURES[i], ANFIS_FEATURES[i]))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            if i < n_in - 1:
                twin.set_yticklabels([])
        handles, labels = twin.get_legend_handles_labels()
        handles += [plt.Line2D([], [], color="#52514e", linestyle=ls) for ls in ("-", ":")]
        labels += ["po uczeniu", "początkowe"]
        fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
        fig.suptitle(f"Funkcje przynależności — {name} (tło: rozkład danych treningowych)")
        _save(fig, out, f"funkcje_przynaleznosci_{name}")


def rules_table(model: ANFIS, x_eval: torch.Tensor) -> pd.DataFrame:
    """Reguły Sugeno w czytelnej postaci + średnia aktywacja na danych testowych."""
    with torch.no_grad():
        _, wn = model.firing(x_eval)
    n_mfs = model.centers.shape[1]
    names = MF_LABELS.get(n_mfs, [str(j) for j in range(n_mfs)])
    conseq = model.consequents.detach().cpu().numpy()
    rows = []
    for r, combo in enumerate(model.rules.cpu().numpy()):
        rows.append({
            "reguła": r,
            "JEŚLI": " I ".join(f"{f} jest {names[k]}" for f, k in zip(ANFIS_FEATURES, combo)),
            "średnia aktywacja": wn[:, r].mean().item(),
            **{f"p_{f}": conseq[r, i] for i, f in enumerate(ANFIS_FEATURES)},
            "q": conseq[r, -1],
        })
    return pd.DataFrame(rows).sort_values("średnia aktywacja", ascending=False)


def rule_activation(tables: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 5), squeeze=False)
    for ax, (name, t) in zip(axes[0], tables.items()):
        top = t.head(15).iloc[::-1]
        ax.barh(top["reguła"].astype(str), top["średnia aktywacja"], color=COLORS[0], height=0.6)
        ax.set(title=f"{name}: 15 najczęściej aktywnych reguł", xlabel="średnia znormalizowana siła", ylabel="reguła")
        ax.grid(axis="y", visible=False)
    _save(fig, out, "aktywnosc_regul")


def data_overview(df: pd.DataFrame, rows: dict[str, np.ndarray], out: Path) -> pd.DataFrame:
    cols = ANFIS_FEATURES + [TARGET]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.2))
    for ax, c in zip(axes, cols):
        all_vals = df.loc[np.concatenate(list(rows.values())), c]
        lo, hi = np.quantile(all_vals, [0.005, 0.995])
        for i, (split, r) in enumerate(rows.items()):
            ax.hist(df.loc[r, c], bins=50, range=(lo, hi), density=True, histtype="step",
                    linewidth=1.5, color=COLORS[i], label=split)
        ax.set(xlabel=FEATURE_LABELS.get(c, f"{c} = zmiana tętna za {HORIZON_S} s [bpm]"))
        ax.set_yticks([])
    axes[-1].legend(frameon=False)
    fig.suptitle("Rozkłady cech i celu w zbiorach train / val / test")
    _save(fig, out, "rozklady_danych")
    return pd.concat({s: df.loc[r, cols].describe().T for s, r in rows.items()})


def write_report(
    out: Path,
    results: dict[str, dict[str, float]],
    histories: dict[str, dict[str, list[float]]],
    pred: pd.DataFrame,
    models: list[str],
) -> Path:
    """Wykresy i tabele zależne tylko od zapisanych plików (metrics/history/predictions)."""
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    learning_curves(histories, rep)
    metrics_bars({m: results[m] for m in models}, rep)
    pred_vs_true(pred, models, rep)
    error_distribution(pred, models, rep)
    error_by_bins(pred, models, rep).to_csv(rep / "blad_wg_zmiany_tetna.csv")
    per_activity(pred, models, rep).to_csv(rep / "blad_na_aktywnosc.csv")
    example_timelines(pred, models, rep)
    pd.DataFrame(results).T.to_csv(rep / "metryki.csv")
    return rep


def regenerate(run_dir: Path) -> Path:
    """Odtwarza wykresy z gotowego katalogu runs/<data> bez treningu."""
    results = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    histories = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    models = [m for m in results if m in pred.columns]
    return write_report(run_dir, results, histories, pred, models)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Odtwarza wykresy i tabele z katalogu runs/<data> bez treningu.")
    parser.add_argument("run_dir", type=Path)
    print(f"Zapisano w {regenerate(parser.parse_args().run_dir)}")
