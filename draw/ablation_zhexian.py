import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 900,
        "savefig.dpi": 900,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

SAVE_DIR = Path(__file__).resolve().parent / "outputs" / "ablation"

METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]
METRIC_COLORS = {
    "Accuracy": "#4E79A7",
    "Precision": "#F28E2B",
    "Recall": "#59A14F",
    "F1-score": "#E15759",
}

def draw_dumbbell_delta_plot(data, filename, title):
    ionfusion = data.loc[data["Model"] == "IonFusion", METRICS].iloc[0]
    ablations = data[data["Model"] != "IonFusion"].copy()

    records = []
    for _, row in ablations.iterrows():
        for metric in METRICS:
            records.append(
                {
                    "Model": row["Model"],
                    "Metric": metric,
                    "Value": row[metric],
                    "IonFusion": ionfusion[metric],
                    "Delta": row[metric] - ionfusion[metric],
                }
            )

    plot_data = pd.DataFrame(records)
    model_order = list(ablations["Model"])
    row_gap = len(METRICS) + 1
    y_positions = {}
    y_labels = {}

    for model_index, model in enumerate(model_order):
        base = (len(model_order) - 1 - model_index) * row_gap
        y_labels[base + 1.5] = model
        for metric_index, metric in enumerate(METRICS):
            y_positions[(model, metric)] = base + (len(METRICS) - 1 - metric_index)

    fig_height = max(4.2, 1.1 * len(model_order) + 1.4)
    fig, ax = plt.subplots(figsize=(8.4, fig_height))

    ax.axvline(0, color="#222222", linewidth=1.2, linestyle="-", zorder=1)

    for _, row in plot_data.iterrows():
        y = y_positions[(row["Model"], row["Metric"])]
        delta = row["Delta"]
        color = METRIC_COLORS[row["Metric"]]

        ax.hlines(y, 0, delta, color=color, linewidth=2.4, alpha=0.7, zorder=2)
        ax.scatter(0, y, s=46, color="#2F2F2F", edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(delta, y, s=64, color=color, edgecolor="white", linewidth=0.8, zorder=4)

        sign = "+" if delta >= 0 else ""
        text_x = delta + (0.004 if delta >= 0 else -0.004)
        ha = "left" if delta >= 0 else "right"
        ax.text(
            text_x,
            y,
            f"{sign}{delta:.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            color=color,
        )

    max_abs_delta = max(0.01, float(plot_data["Delta"].abs().max()))
    x_padding = max(0.012, max_abs_delta * 0.28)
    ax.set_xlim(-max_abs_delta - x_padding, max_abs_delta + x_padding)

    ax.set_yticks(list(y_labels.keys()))
    ax.set_yticklabels(list(y_labels.values()), fontsize=11)
    ax.set_xlabel("Metric change compared with IonFusion")
    ax.set_title(title, pad=10)

    ax.grid(axis="x", linestyle=":", alpha=0.45, color="#8A8A8A")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=metric,
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
        )
        for metric, color in METRIC_COLORS.items()
    ]
    handles.insert(
        0,
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label="IonFusion baseline",
            markerfacecolor="#2F2F2F",
            markeredgecolor="white",
            markersize=7,
        ),
    )
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        output_path = SAVE_DIR / f"{filename}.{ext}"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"[Saved] {output_path}")
    plt.close(fig)

def main():
    global SAVE_DIR
    parser = argparse.ArgumentParser(description="Plot ablation metric changes from CSV files.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=SAVE_DIR)
    args = parser.parse_args()
    SAVE_DIR = args.output_dir
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    required = {"Model", *METRICS}
    for filename, output_name, title in (
        ("ablation_evolution.csv", "figure_ablation_dumbbell_evolution", "Ablation metric changes vs IonFusion"),
        ("ablation_mechanisms.csv", "figure_ablation_dumbbell_mechanisms", "Mechanism ablation metric changes vs IonFusion"),
    ):
        data = pd.read_csv(args.data_dir / filename)
        missing = required - set(data.columns)
        if missing:
            parser.error(f"{filename}: missing columns {sorted(missing)}")
        if (data["Model"] == "IonFusion").sum() != 1:
            parser.error(f"{filename}: exactly one IonFusion row is required")
        draw_dumbbell_delta_plot(data, output_name, title)

if __name__ == "__main__":
    main()
