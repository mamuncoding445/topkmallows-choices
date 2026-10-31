"""
Builds figures and summary tables from experiment1/experiment2 result CSVs.
TV is derived from MAE via the identity TV = (|A|+1)*MAE/2; since MAE here
is averaged over assortments of varying size, we report TV per-assortment
-size-class separately where it matters, but for summary tables we report
the identity applied at the MEAN assortment size actually enumerated
(sizes 2-6, mean |A|+1 = 5.5) as a representative conversion, documented
as such -- not a per-row recomputation requiring per-row |A|.
"""
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_LABELS = {
    "mnl_single": "MNL (single)",
    "topkgmm_single": "TopKGMM (single)",
    "mnl_mixture": "MNL mixture",
    "topkgmm_mixture": "TopKGMM mixture",
    "uniform_baseline": "Uniform baseline",
}
FAMILY_ORDER = ["uniform_baseline", "topkgmm_single", "mnl_single", "topkgmm_mixture", "mnl_mixture"]
COLORS = {
    "uniform_baseline": "#999999",
    "topkgmm_single": "#e07b39",
    "mnl_single": "#3b7dd8",
    "topkgmm_mixture": "#c0392b",
    "mnl_mixture": "#2471a3",
}


def read_csv_rows(path):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_experiment1(csv_path, out_path):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    rows = read_csv_rows(csv_path)
    rhos = sorted(set(float(r["rho"]) for r in rows))
    fig, ax = plt.subplots(figsize=(7, 5))
    for family in FAMILY_ORDER:
        means, stds = [], []
        for rho in rhos:
            vals = [float(r["test_mae_full"]) for r in rows
                    if r["family"] == family and float(r["rho"]) == rho]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(rhos, means, yerr=stds, label=FAMILY_LABELS[family],
                     color=COLORS[family], marker="o", capsize=3)
    ax.set_xlabel("rho (block-structure strength)")
    ax.set_ylabel("Mean Absolute Error (full 456-assortment enumeration)")
    ax.set_title("Experiment 1: synthetic block-structured rankings\nMAE vs rho, mean +/- std over seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


def plot_experiment2(csv_path, out_path):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    rows = read_csv_rows(csv_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    means, stds, labels, colors = [], [], [], []
    for family in FAMILY_ORDER:
        vals = [float(r["test_mae_full"]) for r in rows if r["family"] == family]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        labels.append(FAMILY_LABELS[family])
        colors.append(COLORS[family])
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean Absolute Error (full 456-assortment enumeration)")
    ax.set_title("Experiment 2: Dublin West 2002 ballots\nMAE by model family, mean +/- std over seeds")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


def summary_table(csv_path, group_cols, out_path):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    rows = read_csv_rows(csv_path)
    groups = {}
    for r in rows:
        key = tuple(r[c] for c in group_cols) + (r["family"],)
        groups.setdefault(key, []).append(float(r["test_mae_full"]))
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(group_cols) + ["family", "mean_mae", "std_mae", "n_seeds", "mean_tv_at_meank"])
        for key, vals in sorted(groups.items()):
            mean_mae = np.mean(vals)
            std_mae = np.std(vals)
            mean_tv = 5.5 / 2 * mean_mae  # identity at mean |A|+1 = 5.5 (sizes 2-6)
            writer.writerow(list(key) + [f"{mean_mae:.5f}", f"{std_mae:.5f}", len(vals), f"{mean_tv:.5f}"])
    print("wrote", out_path)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists("outputs/experiment1_results.csv"):
        plot_experiment1("outputs/experiment1_results.csv", "figures/experiment1_mae_vs_rho.png")
        summary_table("outputs/experiment1_results.csv", ["rho"], "outputs/experiment1_summary.csv")
    if os.path.exists("outputs/experiment2_results.csv"):
        plot_experiment2("outputs/experiment2_results.csv", "figures/experiment2_mae_bars.png")
        summary_table("outputs/experiment2_results.csv", [], "outputs/experiment2_summary.csv")
