#!/usr/bin/env python3

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DRAW_DATA_DIR", BASE_DIR / "data"))
AUPR_DIR = DATA_DIR / "go_aupr"
STAT_DIR = DATA_DIR / "go_label_stats"
OUT_DIR = Path(os.environ.get("DRAW_OUTPUT_DIR", BASE_DIR / "outputs")) / "go_aupr_by_depth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQIDS = [10, 30, 50]
NAMESPACES = ["bp", "cc", "mf"]
ONLY_VALID_AUPR = True
SAVE_PNG = True
SAVE_PDF = True
FIG_DPI = 900

font_path = os.environ.get("DRAW_FONT_PATH")
if font_path and Path(font_path).is_file():
    font_manager.fontManager.addfont(font_path)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 22,
        "axes.labelsize": 26,
        "axes.titlesize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "axes.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

method_order = [
    "ESM2-MLP",
    "ESM3-MLP",
    "IPR-MLP",
    "DeepGOZero",
    "DeepGO-SE",
    "IonFusion",
]
method_colors_9 = [
    "#9FD3E8",  # esm2
    "#F3C9A9",  # esm3
    "#B7D7A8",  # interpro
    "#E3D3D8",  # deepgozero
    "#B59ABD",  # deepgose
    "#F4B544",  # CPFSS_plus
]

method_color_map = dict(zip(method_order, method_colors_9))

DEPTH_BINS = [6.5, np.inf]
DEPTH_LABELS = [">6"]

def normalize_namespace(x):
    return str(x).lower().strip()

def normalize_method_name(x):
    x = str(x).strip()

    rename = {
        "ESM2_MLP": "ESM2-MLP",
        "ESM3_MLP": "ESM3-MLP",
        "IPR_MLP": "IPR-MLP",
        "DeepGOZero": "DeepGOZero",
        "DeepGoSE": "DeepGO-SE",
        "CPFSS_PLUS": "IonFusion",
    }

    return rename.get(x, x)

def finalize_axes_style(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#222222")
    ax.tick_params(axis="both", direction="out", width=1.0, length=4)
    ax.set_axisbelow(True)

def normalize_valid_bool(x):
    if pd.isna(x):
        return False

    if isinstance(x, bool):
        return x

    x = str(x).strip().lower()
    return x in {"true", "1", "yes", "y", "t"}

def find_column_case_insensitive(df, candidates):
    col_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]

    return None

def infer_method_from_filename(path):
    name = Path(path).name
    name = name.replace("_all_go_aupr.csv", "")
    return normalize_method_name(name)

def load_all_go_aupr():
    files = sorted(glob.glob(str(AUPR_DIR / "*_all_go_aupr.csv")))

    if not files:
        raise FileNotFoundError(f"No *_all_go_aupr.csv found in {AUPR_DIR}")

    dfs = []

    for f in files:
        df = pd.read_csv(f)

        required = {"go_id", "aupr", "seq_identity", "namespace"}
        missing = required - set(df.columns)

        if missing:
            warnings.warn(f"[Skip] {f}, missing columns: {missing}")
            continue

        if "method" not in df.columns:
            df["method"] = infer_method_from_filename(f)

        df["method"] = df["method"].map(normalize_method_name)
        df["namespace"] = df["namespace"].map(normalize_namespace)
        df["seq_identity"] = df["seq_identity"].astype(int)
        df["aupr"] = pd.to_numeric(df["aupr"], errors="coerce")

        if ONLY_VALID_AUPR and "valid_aupr" in df.columns:
            df["_valid_aupr_bool"] = df["valid_aupr"].map(normalize_valid_bool)
            df = df[df["_valid_aupr_bool"]].copy()

        df = df[np.isfinite(df["aupr"])].copy()
        df = df[df["aupr"] < 1.0].copy()

        keep_cols = [
            "method",
            "seq_identity",
            "namespace",
            "go_id",
            "aupr",
            "valid_aupr",
            "n_test_proteins",
            "n_positive",
            "n_negative",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]

        dfs.append(df[keep_cols])

    if not dfs:
        raise RuntimeError("No valid GO-level AUPR files loaded.")

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    return all_df

def load_depth_stats(seqid, namespace):
    stat_file = STAT_DIR / f"seqid_{seqid}_{namespace}_frequency_ic_depth.csv"

    if not stat_file.exists():
        raise FileNotFoundError(f"Missing stat file: {stat_file}")

    df = pd.read_csv(stat_file)

    go_col = find_column_case_insensitive(df, ["go_id", "go", "term", "term_id"])
    depth_col = find_column_case_insensitive(
        df,
        ["depth", "go_depth", "level"],
    )

    if go_col is None or depth_col is None:
        raise ValueError(
            f"Cannot detect go_id/depth columns in {stat_file}. "
            f"Current columns = {list(df.columns)}"
        )

    df = df.rename(
        columns={
            go_col: "go_id",
            depth_col: "depth",
        }
    )

    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")

    df = df[["go_id", "depth"]].drop_duplicates("go_id")
    df = df.dropna(subset=["go_id", "depth"])

    return df

def add_depth_bin(df):
    df = df.copy()

    df["depth_bin"] = pd.cut(
        df["depth"],
        bins=DEPTH_BINS,
        labels=DEPTH_LABELS,
        include_lowest=True,
        right=True,
    )

    df = df.dropna(subset=["depth_bin"]).copy()
    df["depth_bin"] = df["depth_bin"].astype(str)

    return df

def summarize_by_depth(df):
    summary = (
        df.groupby(["depth_bin", "method"], observed=True)
        .agg(
            mean_aupr=("aupr", "mean"),
            std_aupr=("aupr", "std"),
            n_go=("go_id", "nunique"),
        )
        .reset_index()
    )

    summary["sem_aupr"] = summary["std_aupr"] / np.sqrt(summary["n_go"].clip(lower=1))

    return summary

def plot_ic_violin(df, seqid, namespace):
    """
    Raincloud 风格：
    左侧半小提琴图 + 原始散点 + 右侧箱线图。
    df 必须是已经 add_ic_bin() 之后的 merged 数据。
    """

    bins = [b for b in DEPTH_LABELS if b in set(df["depth_bin"].astype(str))]

    existing_methods = set(df["method"].drop_duplicates())
    methods = [m for m in method_order if m in existing_methods]

    if len(bins) == 0 or len(methods) == 0:
        print(f"[Skip Plot] Empty violin data: seqid={seqid}, namespace={namespace}")
        return

    violin_width = 0.26     # 半小提琴最大宽度
    method_step = 0.38      # 同一个 IC bin 内，不同 method 之间的距离
    group_gap = 0.78        # 不同 IC bin 之间的距离

    point_size = 10
    point_alpha = 0.82

    box_width = 0.08 #控制箱线图矩形宽度
    box_gap = 0.000 #小提琴右边界和箱型图左边界之间的间隙

    rng = np.random.default_rng(123)

    group_span = (len(methods) - 1) * method_step
    group_step = group_span + group_gap

    data_list = []
    center_pos_list = []
    box_pos_list = []
    color_list = []

    group_centers = []

    for i, depth_bin in enumerate(bins):
        group_center = i * group_step
        group_centers.append(group_center)

        start_pos = group_center - group_span / 2

        for j, method in enumerate(methods):
            sub = df[
                (df["depth_bin"].astype(str) == depth_bin)
                & (df["method"] == method)
            ]

            values = sub["aupr"].dropna().values

            if len(values) == 0:
                continue

            center_pos = start_pos + j * method_step
            box_pos = center_pos + box_gap + box_width / 2

            data_list.append(values)
            center_pos_list.append(center_pos)
            box_pos_list.append(box_pos)
            color_list.append(method_color_map.get(method, "#CCCCCC"))

    fig_width = max(9, 0.95 * len(bins) + 3)
    fig_height = 6.4

    plt.figure(figsize=(fig_width, fig_height))

    vp = plt.violinplot(
        data_list,
        positions=center_pos_list,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, center_pos, color in zip(vp["bodies"], center_pos_list, color_list):
        body.set_facecolor("none")
        body.set_edgecolor(color)
        body.set_linewidth(1.05)
        body.set_alpha(0.95)

        path = body.get_paths()[0]
        vertices = path.vertices
        vertices[:, 0] = np.minimum(vertices[:, 0], center_pos)

    for values, center_pos, color in zip(data_list, center_pos_list, color_list):
        jitter_x = rng.uniform(
            low=center_pos - violin_width * 0.45,
            high=center_pos - violin_width * 0.08,
            size=len(values),
        )

        plt.scatter(
            jitter_x,
            values,
            s=point_size,
            color=color,
            alpha=point_alpha,
            edgecolor="white",
            linewidth=0.25,
            zorder=4,
        )

    bp = plt.boxplot(
        data_list,
        positions=box_pos_list,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(
            linewidth=0,
            edgecolor="none",
        ),
        medianprops=dict(
            color="red",
            linewidth=1.4,
        ),
        whiskerprops=dict(
            color="black",
            linewidth=0.9,
        ),
        capprops=dict(
            color="black",
            linewidth=0.9,
        ),
    )

    for patch, color in zip(bp["boxes"], color_list):
        patch.set_facecolor(color)
        patch.set_edgecolor("none")
        patch.set_linewidth(0)
        patch.set_alpha(0.65)
        patch.set_zorder(5)

    for key in ["medians", "whiskers", "caps"]:
        for artist in bp[key]:
            artist.set_zorder(6)

    plt.xticks(group_centers, [""] * len(group_centers))
    plt.xlabel("")
    plt.ylabel("M-AUPR")
    plt.ylim(0, 1.0)
    plt.title("")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    finalize_axes_style(plt.gca())

    plt.tight_layout()

    out_prefix = OUT_DIR / f"seqid_{seqid}_{namespace}_Depth_raincloud_aupr"

    if SAVE_PNG:
        plt.savefig(f"{out_prefix}.png", dpi=FIG_DPI)
        print(f"[Saved] {out_prefix}.png")

    if SAVE_PDF:
        plt.savefig(f"{out_prefix}.pdf")
        print(f"[Saved] {out_prefix}.pdf")

    plt.savefig(f"{out_prefix}.svg", format="svg")
    print(f"[Saved] {out_prefix}.svg")

    legend_labels = {
        "esm2": "ESM2-MLP",
        "esm3": "ESM3-MLP",
        "interpro": "IPR-MLP",
        "deepgozero": "DeepGOZero",
        "deepgose": "DeepGO-SE",
        "CPFSS_plus": "IonFusion",
        "ESM2-MLP": "ESM2-MLP",
        "ESM3-MLP": "ESM3-MLP",
        "IPR-MLP": "IPR-MLP",
        "DeepGOZero": "DeepGOZero",
        "DeepGO-SE": "DeepGO-SE",
        "CPFSS": "IonFusion",
    }

    legend_handles = []
    legend_texts = []

    for method in methods:
        handle = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=method_color_map.get(method, "#CCCCCC"),
            edgecolor="none",
            alpha=0.90,
        )
        legend_handles.append(handle)
        legend_texts.append(legend_labels.get(method, method))

    fig_leg = plt.figure(figsize=(3.4, 4.4))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")

    ax_leg.legend(
        legend_handles,
        legend_texts,
        loc="center",
        frameon=False,
        fontsize=22,
        ncol=1,
        handlelength=1.3,
        handletextpad=0.8,
        labelspacing=0.9,
    )

    legend_prefix = OUT_DIR / "method_legend_vertical"

    if SAVE_PNG:
        fig_leg.savefig(
            f"{legend_prefix}.png",
            dpi=FIG_DPI,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print(f"[Saved] {legend_prefix}.png")

    if SAVE_PDF:
        fig_leg.savefig(
            f"{legend_prefix}.pdf",
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print(f"[Saved] {legend_prefix}.pdf")

    fig_leg.savefig(
        f"{legend_prefix}.svg",
        format="svg",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    print(f"[Saved] {legend_prefix}.svg")

    plt.close(fig_leg)
    plt.close()

def main():
    all_aupr = load_all_go_aupr()

    for seqid in SEQIDS:
        for namespace in NAMESPACES:
            print(f"[Process] seqid={seqid}, namespace={namespace}, metric=depth")

            aupr_sub = all_aupr[
                (all_aupr["seq_identity"] == seqid)
                & (all_aupr["namespace"] == namespace)
            ].copy()

            if aupr_sub.empty:
                print(f"[Skip] No AUPR records for seqid={seqid}, namespace={namespace}")
                continue

            stat_df = load_depth_stats(seqid, namespace)

            merged = aupr_sub.merge(stat_df, on="go_id", how="inner")

            if merged.empty:
                continue

            merged_path = OUT_DIR / f"seqid_{seqid}_{namespace}_depth_merged.csv"
            merged.to_csv(merged_path, index=False)
            print(f"[Saved] merged -> {merged_path}")

            merged = add_depth_bin(merged)
            summary = summarize_by_depth(merged)

            summary_path = OUT_DIR / f"seqid_{seqid}_{namespace}_depth_bin_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"[Saved] summary -> {summary_path}")

            plot_ic_violin(merged, seqid, namespace)

    print(f"\n[Done] Output dir: {OUT_DIR}")

if __name__ == "__main__":
    main()
