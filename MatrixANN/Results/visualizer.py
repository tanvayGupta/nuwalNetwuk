#!/usr/bin/env python3
"""
visualizer.py

Visualizes XOR neural-network training results stored in a CSV file with
columns: error_00, error_01, error_10, error_11, mean_absolute_error,
iterations, learning_rate. Each row is one training run.

Usage:
    python visualizer.py [path/to/results.csv]

If no path is given, the script picks the first *.csv file it finds in
the current directory. Plots are saved as PNGs into a ./plots/ subfolder
(created if it doesn't exist) and are not shown interactively, so this
also works headless / over SSH.

Designed to keep working as more `iterations` values get added to the
CSV over time: every plot facets or groups by `iterations` automatically,
and the iterations x learning_rate heatmap only activates once there's
more than one iterations value to compare.
"""

import sys
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False

ERROR_COLS = ["error_00", "error_01", "error_10", "error_11"]
REQUIRED_COLS = set(ERROR_COLS + ["mean_absolute_error", "iterations", "learning_rate"])


def find_csv():
    if len(sys.argv) > 1:
        return sys.argv[1]
    csvs = sorted(glob.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            "No CSV file found in the current directory. "
            "Pass a path explicitly: python visualizer.py path/to/file.csv"
        )
    return csvs[0]


def load_data(path):
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    return df


def out_path(name):
    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, name)


def plot_mae_vs_lr(df):
    """
    Scatter of every run's MAE against learning_rate, plus the mean MAE
    (with std error bars) per learning_rate. One subplot per distinct
    `iterations` value so runs trained for different lengths aren't mixed.
    This is the main "which learning rate converges best" plot.
    """
    iters_values = sorted(df["iterations"].unique())
    fig, axes = plt.subplots(1, len(iters_values),
                              figsize=(6 * len(iters_values), 5), squeeze=False)
    axes = axes[0]

    for ax, it in zip(axes, iters_values):
        sub = df[df["iterations"] == it]
        ax.scatter(sub["learning_rate"], sub["mean_absolute_error"],
                    alpha=0.4, s=25, label="individual runs")

        stats = sub.groupby("learning_rate")["mean_absolute_error"].agg(["mean", "std"])
        ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                     color="crimson", marker="o", linewidth=2,
                     capsize=3, label="mean ± std")

        ax.set_title(f"iterations = {it}")
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("mean_absolute_error")
        ax.legend()

    fig.suptitle("MAE vs Learning Rate")
    fig.tight_layout()
    fig.savefig(out_path("mae_vs_learning_rate.png"), dpi=150)
    plt.close(fig)


def plot_error_components_vs_lr(df):
    """
    Small multiples: one subplot per XOR input (00, 01, 10, 11) showing
    that input's final signed error against learning_rate. Useful for
    spotting which specific input(s) the network fails to converge on,
    and for spotting saturation (errors clustering near +/-0.5) instead
    of shrinking toward 0.
    """
    for it in sorted(df["iterations"].unique()):
        sub = df[df["iterations"] == it]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        for ax, col in zip(axes.flat, ERROR_COLS):
            ax.scatter(sub["learning_rate"], sub[col], alpha=0.5, s=20)
            means = sub.groupby("learning_rate")[col].mean()
            ax.plot(means.index, means.values, color="crimson", marker="o")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(col)
            ax.set_xlabel("learning_rate")
            ax.set_ylabel("error")

        fig.suptitle(f"Per-input error vs Learning Rate (iterations={it})")
        fig.tight_layout()
        fig.savefig(out_path(f"error_components_iterations_{it}.png"), dpi=150)
        plt.close(fig)


def plot_mae_boxplot(df):
    """
    Boxplot of MAE grouped by learning_rate, faceted by iterations. Shows
    spread across repeated runs at each learning rate -- i.e. which rates
    are reliably good vs. which ones sometimes get stuck in a bad local
    minimum (wide box / high outliers).
    """
    iters_values = sorted(df["iterations"].unique())
    fig, axes = plt.subplots(1, len(iters_values),
                              figsize=(6 * len(iters_values), 5), squeeze=False)
    axes = axes[0]

    for ax, it in zip(axes, iters_values):
        sub = df[df["iterations"] == it]
        if HAVE_SEABORN:
            sns.boxplot(data=sub, x="learning_rate", y="mean_absolute_error", ax=ax)
        else:
            groups = [g["mean_absolute_error"].values for _, g in sub.groupby("learning_rate")]
            labels = sorted(sub["learning_rate"].unique())
            ax.boxplot(groups, labels=labels)
        ax.set_title(f"iterations = {it}")
        ax.tick_params(axis="x", rotation=90)

    fig.suptitle("Spread of MAE per Learning Rate")
    fig.tight_layout()
    fig.savefig(out_path("mae_boxplot.png"), dpi=150)
    plt.close(fig)


def plot_heatmap_iterations_lr(df):
    """
    Heatmap of mean MAE across iterations (rows) x learning_rate (cols).
    Only meaningful -- and only generated -- once the CSV has more than
    one distinct `iterations` value.
    """
    if df["iterations"].nunique() < 2:
        return

    pivot = df.pivot_table(index="iterations", columns="learning_rate",
                            values="mean_absolute_error", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(0.6 * len(pivot.columns) + 2,
                                     0.6 * len(pivot.index) + 2))
    if HAVE_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r", ax=ax)
    else:
        im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax)

    ax.set_title("Mean MAE: iterations vs learning_rate")
    fig.tight_layout()
    fig.savefig(out_path("mae_heatmap.png"), dpi=150)
    plt.close(fig)


def main():
    path = find_csv()
    print(f"Loading: {path}")
    df = load_data(path)
    print(f"{len(df)} runs | "
          f"{df['iterations'].nunique()} distinct iteration count(s) | "
          f"{df['learning_rate'].nunique()} distinct learning rate(s)")

    plot_mae_vs_lr(df)
    plot_error_components_vs_lr(df)
    plot_mae_boxplot(df)
    plot_heatmap_iterations_lr(df)

    print("Saved plots to ./plots/")


if __name__ == "__main__":
    main()