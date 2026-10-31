"""
Regenerates the TopKGMM-vs-paper's-MNL comparison figure as a vector PDF
(matches REPORT.md Section 3.3: left = synthetic vs rho, right = Dublin West
bars; only topkgmm_single/mixture and mnl_single_paper/mixture_paper shown).
"""
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = ["topkgmm_single", "topkgmm_mixture", "mnl_single_paper", "mnl_mixture_paper"]
LABELS = {
    "topkgmm_single": "TopKGMM (single)",
    "topkgmm_mixture": "TopKGMM (mixture)",
    "mnl_single_paper": "Paper's MNL (single)",
    "mnl_mixture_paper": "Paper's MNL (mixture)",
}
STYLES = {
    "topkgmm_single": dict(color="#1f77b4", linestyle="-", marker="o"),
    "topkgmm_mixture": dict(color="#1f77b4", linestyle="-", marker="s", alpha=0.6),
    "mnl_single_paper": dict(color="#d62728", linestyle="--", marker="o"),
    "mnl_mixture_paper": dict(color="#d62728", linestyle="--", marker="s", alpha=0.6),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: synthetic experiment 1, MAE vs rho
rows = list(csv.DictReader(open("outputs/experiment1_results.csv")))
byrho = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["family"] in FAMILIES:
        byrho[float(r["rho"])][r["family"]].append(float(r["test_mae_full"]))
rhos = sorted(byrho.keys())

ax = axes[0]
for fam in FAMILIES:
    means = [np.mean(byrho[rho][fam]) for rho in rhos]
    ax.plot(rhos, means, label=LABELS[fam], **STYLES[fam])
ax.set_xlabel(r"$\rho$ (block-structure correlation)")
ax.set_ylabel("Test MAE (full enumeration)")
ax.set_title("Synthetic correlated-block data")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

# Right: Dublin West, bar chart of mean MAE per family
rows2 = list(csv.DictReader(open("outputs/experiment2_results.csv")))
byfam = defaultdict(list)
for r in rows2:
    if r["family"] in FAMILIES:
        byfam[r["family"]].append(float(r["test_mae_full"]))

ax = axes[1]
means = [np.mean(byfam[f]) for f in FAMILIES]
stds = [np.std(byfam[f]) for f in FAMILIES]
colors = [STYLES[f]["color"] for f in FAMILIES]
alphas = [STYLES[f].get("alpha", 1.0) for f in FAMILIES]
bars = ax.bar(range(len(FAMILIES)), means, yerr=stds, capsize=4,
              color=colors, tick_label=[LABELS[f] for f in FAMILIES])
for b, a in zip(bars, alphas):
    b.set_alpha(a)
ax.set_xticklabels([LABELS[f] for f in FAMILIES], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Test MAE (full enumeration)")
ax.set_title("Dublin West 2002 STV ballots")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("TopKGMM vs. the paper's own MNL baseline", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("figures/topkgmm_vs_paper_mnl.pdf")
print("wrote figures/topkgmm_vs_paper_mnl.pdf")
