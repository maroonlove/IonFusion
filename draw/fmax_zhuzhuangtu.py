import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "fmax"
font_path = os.environ.get("DRAW_FONT_PATH")
if font_path and Path(font_path).is_file():
    font_manager.fontManager.addfont(font_path)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 22,
    "axes.labelsize": 26,
    "axes.titlesize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "figure.dpi": 900,
    "savefig.dpi": 900,
    "axes.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})

method_order = ["esm2", "esm3", "interpro", "deepgozero", "deepgose", "CPFSS_plus"]

method_display_map = {
    "esm2": "ESM2",
    "esm3": "ESM3",
    "interpro": "InterPro",
    "deepgozero": "DeepGOZero",
    "deepgose": "DeepGOSE",
    "CPFSS_plus": "IonFusion"
}

legend_methods = [method_display_map[m] for m in method_order]

seq_order = [10, 30, 50]
metrics = ["Fmax", "AUPR", "ICAUPR", "DPAUPR"]
ontologies = ["bp", "cc", "mf"]

thresholds = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

method_colors = ["#9FD3E8", "#F3C9A9", "#B7D7A8", "#E3D3D8", "#B59ABD", "#F4B544"]
method_color_map = dict(zip(method_order, method_colors))

def get_bar_ylim_max(global_max):
    """查找最接近且大于等于最大值的阈值点"""
    for t in thresholds:
        if t >= global_max:
            return float(t)
    return 1.0

def get_line_ylim_range(global_min, global_max):
    """查找包裹最小和最大值的精细阈值区间"""
    padded_min = global_min - 0.005
    padded_max = global_max + 0.005

    y_min = 0.0
    for t in reversed(thresholds):
        if t <= padded_min:
            y_min = float(t)
            break

    y_max = 1.0
    for t in thresholds:
        if t >= padded_max:
            y_max = float(t)
            break

    return y_min, y_max

def set_threshold_yticks(ax, ylim):
    """
    纵坐标只显示 thresholds 中的阈值刻度，
    不显示 matplotlib 自动生成的中间值，例如 0.45、0.55。
    """
    y_min, y_max = ylim
    eps = 1e-9

    yticks = [
        float(t) for t in thresholds
        if t >= y_min - eps and t <= y_max + eps
    ]

    if len(yticks) == 0:
        yticks = [y_min, y_max]

    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.1f}" for t in yticks], fontweight="bold")

def generate_all_plots(df, figsize=(8.4, 4.0), line_figsize=(5.2, 3.8)):

    for seq_id in seq_order:

        group_df = df[df["seq_identity"] == seq_id]

        all_vals = group_df[metrics].values.flatten()
        global_min = np.nanmin(all_vals)
        global_max = np.nanmax(all_vals)

        bar_max = get_bar_ylim_max(global_max)
        bar_ylim = (0.0, bar_max)

        line_min, line_max = get_line_ylim_range(global_min, global_max)
        line_ylim = (line_min, line_max)

        for m in metrics:
            for onto in ontologies:
                sub_df = df[(df["class"] == onto) & (df["seq_identity"] == seq_id)].copy()
                sub_df["method"] = pd.Categorical(sub_df["method"], categories=method_order, ordered=True)
                sub_df = sub_df.sort_values("method")

                fig_bar, ax_bar = plt.subplots(figsize=figsize)
                ax_bar.set_facecolor("#F6F6F6")
                fig_bar.patch.set_facecolor("white")

                x_indices = np.arange(len(method_order))

                for idx, method in enumerate(method_order):
                    row = sub_df[sub_df["method"] == method]
                    if len(row) == 0: continue
                    val = row[m].values[0]

                    ax_bar.bar(
                        idx, val, width=0.55,
                        color=method_color_map[method], edgecolor="none", alpha=0.95, zorder=3
                    )

                ax_bar.set_xticks(x_indices)
                ax_bar.set_xticklabels([])
                ax_bar.tick_params(axis="x", bottom=False)

                ax_bar.set_xlabel("")
                ax_bar.set_ylabel(m, fontweight="bold")

                ax_bar.set_ylim(bar_ylim)
                set_threshold_yticks(ax_bar, bar_ylim)
                ax_bar.set_xlim(-0.6, len(method_order) - 0.4)

                finalize_axes_style(ax_bar)
                save_plot(fig_bar, OUT_DIR, f"{m}_predict{seq_id}_{onto}_bar")

    x_line = np.array([0.0, 0.72, 1.44])

    for m in metrics:
        for onto in ontologies:
            slice_df = df[df["class"] == onto].copy()
            slice_df["seq_identity"] = pd.Categorical(slice_df["seq_identity"], categories=seq_order, ordered=True)
            slice_df["method"] = pd.Categorical(slice_df["method"], categories=method_order, ordered=True)
            slice_df = slice_df.sort_values(["seq_identity", "method"])

            pivot_df = slice_df.pivot(index="seq_identity", columns="method", values=m)
            pivot_df = pivot_df.loc[seq_order, method_order]

            l_min, l_max = get_line_ylim_range(pivot_df.values.min(), pivot_df.values.max())

            fig_line, ax_line = plt.subplots(figsize=line_figsize)
            ax_line.set_facecolor("#F6F6F6")
            fig_line.patch.set_facecolor("white")

            for method in method_order:
                values = pivot_df[method].values

                ax_line.plot(
                    x_line, values,
                    color=method_color_map[method], linestyle="-", linewidth=2.0, zorder=4
                )
                ax_line.scatter(
                    x_line, values,
                    color=method_color_map[method], edgecolor="white", linewidth=0.8, s=35, zorder=5
                )

            ax_line.set_xticks(x_line)
            ax_line.set_xticklabels([f"\u2264{sid}%" for sid in seq_order], fontweight="bold", fontsize=22)
            ax_line.set_xlabel("")
            ax_line.set_ylabel(m, fontweight="bold")

            line_ylim = (l_min, l_max)
            ax_line.set_ylim(line_ylim)
            set_threshold_yticks(ax_line, line_ylim)
            ax_line.set_xlim(x_line[0] - 0.06, x_line[-1] + 0.06)

            finalize_axes_style(ax_line)
            save_plot(fig_line, OUT_DIR, f"{m}_{onto}_line")

def finalize_axes_style(ax):
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#222222")
    ax.tick_params(axis="both", width=1.3, length=5)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.subplots_adjust(bottom=0.20)

def save_plot(fig, out_dir, filename):
    pdf_path = os.path.join(out_dir, f"{filename}.pdf")
    png_path = os.path.join(out_dir, f"{filename}.png")
    svg_path = os.path.join(out_dir, f"{filename}.svg")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=900, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

def save_method_legend(methods, colors, save_name="legend_model"):
    """
    Save separate legend image as a two-column table:
    left column = color
    right column = method name
    """
    n_methods = len(methods)

    fig_height = max(2.2, 0.52 * n_methods + 0.9)
    fig, ax = plt.subplots(figsize=(4.4, fig_height + 1.0))

    ax.axis("off")

    ax.text(
        0.18, 1.02,
        "Color",
        ha="center",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        transform=ax.transAxes
    )

    ax.text(
        0.48, 1.02,
        "Model",
        ha="left",
        va="bottom",
        fontsize=22,
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
            linewidth=5.0,
            transform=ax.transAxes,
            solid_capstyle="round"
        )

        ax.plot(
            0.18, y,
            marker="o",
            markersize=11,
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
            fontsize=22,
            transform=ax.transAxes
        )

    plt.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_name}.png")
    svg_path = os.path.join(OUT_DIR, f"{save_name}.svg")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=900, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Plot GO prediction metrics from a CSV file.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    required = {"method", "class", "seq_identity", *metrics}
    missing = required - set(df.columns)
    if missing:
        parser.error(f"Missing input columns: {sorted(missing)}")
    generate_all_plots(df)
    save_method_legend(legend_methods, method_colors)

if __name__ == "__main__":
    main()
