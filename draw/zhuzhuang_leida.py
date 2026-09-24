import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = os.environ.get("DRAW_FONT_PATH")
if font_path and Path(font_path).is_file():
    font_manager.fontManager.addfont(font_path)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial","Liberation Sans" , "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 900,
    "savefig.dpi": 900,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})

SAVE_DIR = Path(__file__).resolve().parent / "outputs" / "comparison"

method_map = {
    "seq-ipr-stru": "IonFusion",
    "CPSS": "IonFusion",
    "CPFSS": "IonFusion",
    "VGIchan_acid_based_SVM": "AAC-SVM",
    "VGIchan_acid_based_MLP": "AAC-MLP",
    "VGIchan_Dipeptide_based_SVM": "DPC-SVM",
    "VGIchan_Dipeptide_based_MLP": "DPC-MLP",
    "VGIchan_PSI_BLAST_based_SVM": "PSI-SVM",
    "VGIchan_PSI_BLAST_based_MLP": "PSI-MLP",
    "PSI_BLAST_based_SVM": "PSI-SVM",
    "PSI_BLAST_based_MLP": "PSI-MLP",
    "DeepPLM": "DeepPLM",
    "DeepIon": "DeepIon",
    "ipr-stru": "IPR+Str",
    "seq-stru": "Seq+Str",
    "seq-ipr": "Seq+IPR",
    "seq": "Seq",
    "no_layered": "NoLayer",
    "no_autogressive": "NoAR"
}

def move_ionfusion_last(df):
    return pd.concat(
        [df[df["Method"] != "IonFusion"], df[df["Method"] == "IonFusion"]],
        ignore_index=True,
    )

def load_table(data_dir, filename, expected_columns):
    df = pd.read_csv(data_dir / filename)
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{filename}: missing columns {sorted(missing)}")
    df["Method"] = df["Method"].replace(method_map)
    return move_ionfusion_last(df)

method_colors_9 = [
    "#E89DA0", "#88CEE6", "#F6C8A8", "#B2D3A4", "#9FBA95",
    "#E6CECF", "#B696B6", "#80C1C4", "#FCBB44"
]

method_colors_7 = [
    "#E89DA0", "#88CEE6", "#F6C8A8", "#B2D3A4",
    "#9FBA95", "#E6CECF", "#B696B6"
]

def plot_metric_or_class_grouped(
    df,
    title,
    ylabel,
    save_name,
    colors,
    ylim=(0.5, 1.02),
    figsize=(7.6, 3.6),
    group_gap=1.12,
    total_span=0.82
):
    """
    Keep Figure 1 and Figure 4 as grouped bar plots.
    """
    group_names = df.columns[1:].tolist()
    methods = df["Method"].tolist()
    data = df.iloc[:, 1:].to_numpy()

    n_groups = len(group_names)
    n_methods = len(methods)

    group_centers = np.arange(n_groups) * group_gap
    step = total_span / (n_methods - 1) if n_methods > 1 else total_span
    width = step * 0.90

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_methods):
        x_positions = group_centers + (i - (n_methods - 1) / 2) * step

        ax.bar(
            x_positions,
            data[i, :],
            width=width,
            color=colors[i % len(colors)],
            edgecolor="none",
            linewidth=0,
            label=methods[i],
            zorder=3
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_names)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
        zorder=0
    )

    ax.tick_params(axis="both", direction="out", length=3, width=0.8)

    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.png"), bbox_inches="tight")
    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.svg"), format="svg", bbox_inches="tight")
    plt.close()

def plot_radar_x_class_legend_method(
    df,
    ylabel,
    save_name,
    colors,
    ylim=(0.0, 1.0),
    figsize=(5.8, 5.8),
    annotate_cpss=True
):
    """
    Figure 2 / Figure 3:
    radar chart
    axes = classes
    each polygon = one method
    values are formatted to 3 decimals
    legend is not drawn in the figure
    """
    classes = df.columns[1:].tolist()
    methods = df["Method"].tolist()
    data = df.iloc[:, 1:].to_numpy()  # [n_methods, n_classes]

    n_classes = len(classes)

    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False).tolist()

    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw=dict(polar=True)
    )

    cpss_idx = methods.index("CPSS") if "CPSS" in methods else None

    for i, method in enumerate(methods):
        values = data[i, :].tolist()
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=1.5,
            color=colors[i % len(colors)],
            label=method,
            zorder=3
        )

        ax.fill(
            angles,
            values,
            color=colors[i % len(colors)],
            alpha=0.08,
            zorder=2
        )

        if annotate_cpss and cpss_idx is not None and i == cpss_idx:
            for angle, value in zip(angles[:-1], values[:-1]):
                ax.text(
                    angle,
                    min(value + 0.045, ylim[1] + 0.02),
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10
                )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # 先隐藏默认标签

    label_radius = ylim[1] + 0.12  # 控制类别文字往外移动的距离，可改成 0.10 / 0.15

    for angle, label in zip(angles[:-1], classes):
        if np.cos(angle) > 0.15:
            ha = "left"
        elif np.cos(angle) < -0.15:
            ha = "right"
        else:
            ha = "center"

        if np.sin(angle) > 0.15:
            va = "bottom"
        elif np.sin(angle) < -0.15:
            va = "top"
        else:
            va = "center"

        ax.text(
            angle,
            label_radius,
            label,
            ha=ha,
            va=va,
            fontsize=16,          # 类别文字大小，想更大可改 17
            clip_on=False
        )

    ax.set_ylim(*ylim)

    yticks = np.linspace(ylim[0], ylim[1], 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.3f}" for y in yticks], fontsize=11)

    ax.set_ylabel("")

    ax.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.45
    )

    plt.subplots_adjust(left=0.16, right=0.84, top=0.86, bottom=0.12)

    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.png"), bbox_inches="tight")
    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.svg"), format="svg", bbox_inches="tight")
    plt.close()

def save_method_legend(methods, colors, save_name="legend_model"):
    """
    Save separate legend image as a two-column table:
    left column = color
    right column = method name
    """
    import matplotlib.patches as patches

    n_methods = len(methods)

    fig_height = max(2.2, 0.32 * n_methods + 0.6)
    fig, ax = plt.subplots(figsize=(3.2, fig_height))

    ax.axis("off")

    ax.text(
        0.18, 1.02,
        "Color",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        transform=ax.transAxes
    )

    ax.text(
        0.48, 1.02,
        "Model",
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        transform=ax.transAxes
    )

    y_positions = np.linspace(0.92, 0.08, n_methods)

    for i, (method, y) in enumerate(zip(methods, y_positions)):
        color = colors[i % len(colors)]

        ax.plot(
            [0.12, 0.24],
            [y, y],
            color=color,
            linewidth=3.5,
            transform=ax.transAxes,
            solid_capstyle="round"
        )

        ax.plot(
            0.18, y,
            marker="o",
            markersize=8,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="None",
            transform=ax.transAxes
        )

        ax.text(
            0.34,
            y,
            method,
            ha="left",
            va="center",
            fontsize=13,
            transform=ax.transAxes
        )

    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.png"), bbox_inches="tight")
    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.svg"), format="svg", bbox_inches="tight")
    plt.close()

def main():
    global SAVE_DIR
    parser = argparse.ArgumentParser(description="Plot classification comparison from CSV files.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=SAVE_DIR)
    args = parser.parse_args()
    SAVE_DIR = args.output_dir
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    macro = ["Method", "Accuracy", "Precision", "Recall", "F1-score"]
    classes = ["Method"] + [f"Class {i}" for i in range(9)]
    df1 = load_table(args.data_dir, "table1_macro_comparison.csv", macro)
    df2 = load_table(args.data_dir, "table2_per_class_precision.csv", classes)
    df3 = load_table(args.data_dir, "table3_per_class_recall.csv", classes)
    df4 = load_table(args.data_dir, "table4_ablation_macro.csv", macro)
    df5 = load_table(args.data_dir, "table5_per_class_f1score.csv", classes)

    plot_metric_or_class_grouped(
        df=df1,
        title="Performance Comparison of Different Ion Channel Protein Classification Methods",
        ylabel="Score",
        save_name="figure1_performance_comparison",
        colors=method_colors_9,
        ylim=(0.55, 1.02),
        figsize=(7.6, 3.6),
        group_gap=1.12,
        total_span=0.82
    )

    plot_radar_x_class_legend_method(
        df=df2,
        ylabel="Per-class Precision",
        save_name="figure2_per_class_precision_radar",
        colors=method_colors_9,
        ylim=(0.0, 1.0),
        figsize=(5.8, 5.8),
        annotate_cpss=False
    )
    plot_radar_x_class_legend_method(
        df=df3,
        ylabel="Per-class Recall",
        save_name="figure3_per_class_recall_radar",
        colors=method_colors_9,
        ylim=(0.0, 1.0),
        figsize=(5.8, 5.8),
        annotate_cpss=False
    )

    plot_metric_or_class_grouped(
        df=df4,
        title="Ablation Study Results",
        ylabel="Score",
        save_name="figure4_ablation_study",
        colors=method_colors_7,
        ylim=(0.70, 1.02),
        figsize=(7.0, 3.6),
        group_gap=1.12,
        total_span=0.72
    )
    plot_radar_x_class_legend_method(
        df=df5,
        ylabel="Per-class F1-score",
        save_name="figure4_per_class_f1score_radar",
        colors=method_colors_9,
        ylim=(0.0, 1.0),
        figsize=(5.8, 5.8),
        annotate_cpss=False
    )

    save_method_legend(
        methods=df1["Method"].tolist(),
        colors=method_colors_9,
        save_name="legend_comparison_9_methods"
    )

    save_method_legend(
        methods=df4["Method"].tolist(),
        colors=method_colors_7,
        save_name="legend_ablation_7_methods"
    )

if __name__ == "__main__":
    main()
