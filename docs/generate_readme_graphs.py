"""Generate README figures from saved experiments: python docs/generate_readme_graphs.py.

Requires matplotlib. No training assets or network access are needed.
"""

import csv
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/readme-assets"
COLORS = {"CLIP-GRU-32": "#2878B5", "CLIP-GRU-64": "#168C91",
          "PE-GRU-32": "#8054A6", "baseline": "#808891",
          "heads": "#32935B", "ensemble": "#DB842B"}
TEST_MODELS = ["Released GenD-CLIP (per-frame)", "Released GenD-PE (per-frame)",
               "Ensemble of released heads (CLIP+PE)", "CLIP-GRU-32", "PE-GRU-32",
               "Ensemble of GRUs (CLIP-GRU+PE-GRU)"]


def read_csv(relative, columns):
    path = ROOT / relative
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{relative}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{relative}: empty CSV")
    return rows


def number(row, key, probability=False):
    value = float(row[key])
    if not math.isfinite(value) or value < 0 or (probability and value > 1):
        raise ValueError(f"Invalid {key}: {row[key]}")
    return value


def load_curves():
    curves = {}
    for label, folder, prefix, length, model in [
        ("CLIP-GRU-32", "results", "gru32", 32, "GRU"),
        ("CLIP-GRU-64", "results", "gru64", 64, "GRU"),
        ("PE-GRU-32", "results_pe", "gru32_pe", 32, "GRU-PE"),
    ]:
        seeds = []
        for seed in range(5):
            relative = f"temporal/{folder}/{prefix}_seed{seed}_epochs.csv"
            rows = read_csv(relative, ["model", "sequence_length", "training_seed",
                                       "split", "epoch", "bce"])
            keyed = {}
            for row in rows:
                if (int(row["training_seed"]) != seed or int(row["sequence_length"]) != length
                        or row["model"] != model):
                    raise ValueError(f"{relative}: unexpected seed or configuration")
                key = (row["split"], int(row["epoch"]))
                if key in keyed:
                    raise ValueError(f"{relative}: duplicate {key}")
                keyed[key] = number(row, "bce")
            expected = {(split, epoch) for split in ("train", "validation")
                        for epoch in range(1, 21)}
            if set(keyed) != expected:
                raise ValueError(f"{relative}: expected exactly train/validation epochs 1–20")
            seeds.append(keyed)
        curves[label] = {split: [mean(seed[split, epoch] for seed in seeds)
                                 for epoch in range(1, 21)]
                         for split in ("train", "validation")}
    return curves


def validation_values():
    clip = read_csv("temporal/results/validation_summary.csv",
                    ["model", "sequence_length", "split", "runs", "auroc_mean"])
    pe = read_csv("temporal/results_pe/gru32_pe_validation_summary.csv",
                  ["model", "sequence_length", "split", "runs", "auroc_mean"])
    values = []
    for rows, model, length, runs in [
        (clip, "GenD", 32, 1), (clip, "GRU", 32, 5),
        (clip, "GenD", 64, 1), (clip, "GRU", 64, 5),
        (pe, "GenD-PE per-frame head", 32, 1),
        (pe, "GRU-PE (mean of 5 seeds)", 32, 5),
    ]:
        matches = [r for r in rows if r["model"] == model
                   and int(r["sequence_length"]) == length and r["split"] == "validation"]
        if len(matches) != 1 or int(matches[0]["runs"]) != runs:
            raise ValueError(f"Expected one validation summary for {model}, length {length}, runs {runs}")
        values.append(number(matches[0], "auroc_mean", True))
    return values


def test_values(dataset, videos, fakes):
    rows = read_csv(f"temporal/results_pe/final_comparison_{dataset}.csv",
                    ["model", "videos", "fakes", "auroc"])
    if len(rows) != 6 or {r["model"] for r in rows} != set(TEST_MODELS):
        raise ValueError(f"{dataset}: expected the six comparison configurations")
    by_model = {r["model"]: r for r in rows}
    for row in rows:
        if int(row["videos"]) != videos or int(row["fakes"]) != fakes:
            raise ValueError(f"{dataset}: unexpected dataset counts")
    return [number(by_model[model], "auroc", True) for model in TEST_MODELS]


def save(fig, filename, note):
    fig.text(0.03, 0.025, note, fontsize=9, color="#404852", va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    path = OUTPUT / filename
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Generated {path.relative_to(ROOT)}")


def bar_chart(filename, title, labels, values, colors, digits, note, highlight=False):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    bars = ax.barh(labels, values, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.set(title=title, xlabel="AUROC (full scale: 0–1)", xlim=(0, 1))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    best = max(range(len(values)), key=values.__getitem__)
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value - 0.012, bar.get_y() + bar.get_height() / 2,
                f"{value:.{digits}f}", ha="right", va="center", color="white",
                fontsize=11, fontweight="bold")
        if highlight and i == best:
            bar.set_edgecolor("#163E28")
            bar.set_linewidth(2.5)
            ax.get_yticklabels()[i].set_fontweight("bold")
    save(fig, filename, note)


def main():
    # Validate all inputs before writing any figures.
    curves = load_curves()
    validation = validation_values()
    ffpp = test_values("ffpp", 100, 50)
    celebdf = test_values("celebdf", 518, 340)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 15, "axes.titlepad": 16,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#202832", "axes.labelcolor": "#202832"})
    for split, filename, title, ylabel in [
        ("train", "train-bce.png", "Training BCE Loss Across 20 Epochs",
         "Binary Cross-Entropy (BCE) Loss"),
        ("validation", "validation-bce.png", "Validation BCE Loss Across 20 Epochs",
         "Validation BCE Loss"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for label, data in curves.items():
            ax.plot(range(1, 21), data[split], label=label, color=COLORS[label],
                    linewidth=2.2, marker="o", markersize=3)
        ax.set(title=title, xlabel="Epochs", ylabel=ylabel, xlim=(1, 20), ylim=(0, None))
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        save(fig, filename, "Lines represent the mean of five training seeds (0–4) at each epoch.")
    bar_chart("validation-auroc.png", "FaceForensics++ Validation AUROC",
              ["GenD-CLIP-32", "CLIP-GRU-32", "GenD-CLIP-64", "CLIP-GRU-64",
               "GenD-PE-32", "PE-GRU-32"], validation,
              [COLORS[k] for k in ["baseline", "CLIP-GRU-32", "baseline", "CLIP-GRU-64",
                                   "baseline", "PE-GRU-32"]], 5,
              "GRU bars: mean AUROC across five separately selected seed checkpoints.\n"
              "Source: best-checkpoint validation summaries; not prediction-ensemble AUROC.")
    colors = [COLORS[k] for k in ["baseline", "baseline", "heads", "CLIP-GRU-32",
                                  "PE-GRU-32", "ensemble"]]
    bar_chart("ffpp-test-auroc.png", "FaceForensics++ Held-Out Test AUROC",
              TEST_MODELS, ffpp, colors, 4,
              "100 videos: 50 real / 50 fake. GRU AUROC uses five-seed averaged probabilities,\n"
              "not the mean of individual seed AUROCs.")
    bar_chart("celebdf-test-auroc.png", "Celeb-DF v2 Cross-Dataset Test AUROC",
              TEST_MODELS, celebdf, colors, 4,
              "518 videos: 178 real / 340 fake. Bold outline marks the highest observed AUROC.\n"
              "GRU AUROC uses five-seed averaged probabilities; external test without retraining.", True)


if __name__ == "__main__":
    main()
