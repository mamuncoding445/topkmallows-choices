"""
Experiment 1 (synthetic) rerun, FIXING the validation-set leakage flagged
in review: hyperparameter selection now scores candidates against
EMPIRICAL probabilities built from the actually-sampled validation
rankings (previously sampled and discarded), instead of exact ground
truth restricted to a 40-assortment subsample. Final test MAE still uses
exact ground truth over the full 456-assortment enumeration (unchanged --
that part was never the issue).

Families: mnl_single_paper, mnl_mixture_paper, topkgmm_single,
topkgmm_mixture only (paper-style MNL + TopKGMM, per project scope).
BETA_GRID is the already-corrected grid from model_selection.py
([0.01, 0.02, 0.05, 0.1, 0.3, 0.6]) -- fixed, not re-widened.

Additional controls (per review instructions):
  - Same choice rule for validation and final evaluation: both
    val_true_fn (empirical_probs_from_rankings -> choice_from_full_rank)
    and true_fn (exact_choice_probs, whose block branch also calls
    choice_from_full_rank and whose uniform branch implements the
    identical "first visible item of A else outside" rule in closed
    form) apply the SAME deterministic choice rule; only the SOURCE
    (empirical over sampled val rankings vs. exact enumeration) and the
    ASSORTMENT SET (40 selection vs. 456 full) differ, as intended.
  - Same train/val samples for all four families within a run: `train`
    and `val` are drawn once per (rho, seed) and passed identically into
    every select_* call below.
  - All randomness seeded: `rng` (train/val sampling) is seeded by
    (seed, rho); row-subsampling inside fit_center is seeded via the
    `seed=seed` argument threaded through every select_* call; BuCchoi's
    previously-unseeded Choice() fallback is now also seeded (see the
    `random.seed(seed)` call added in topkgmm_fit.fit_center).
  - Previous (leaky) results are untouched: this script writes to a NEW
    file (outputs/experiment1_results_valfix.csv by default), never
    overwriting outputs/experiment1_results.csv.
"""
import argparse
import csv
import json
import random
import time

from block_generator import sample_topk, exact_choice_probs, choice_from_full_rank, N as SYN_N, K as SYN_K
from mixture import LinkageCache
from model_selection import (make_selection_assortments, val_mae_for_prob_fn,
                              select_single_topkgmm, select_topkgmm_mixture,
                              select_single_paper_mnl, select_paper_mnl_mixture)


def empirical_probs_from_rankings(A, rankings, k=SYN_K):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    Anull = A + [0]
    counts = {x: 0 for x in Anull}
    for r in rankings:
        c = choice_from_full_rank(A, r, k)
        counts[c] += 1
    m = len(rankings)
    return {x: counts[x] / m for x in Anull}


def run_one(rho, seed, n_train=8000, n_val=2000):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    t_start = time.time()
    rng = random.Random(1000 * seed + int(rho * 100))
    train = [sample_topk(rho, rng) for _ in range(n_train)]
    val = [sample_topk(rho, rng) for _ in range(n_val)]  # NOW ACTUALLY USED

    full_assortments, selection_assortments = make_selection_assortments(SYN_N)

    def val_true_fn(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return empirical_probs_from_rankings(A, val)

    def true_fn(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return exact_choice_probs(A, rho)

    linkage_cache = LinkageCache(train, SYN_N)

    results = {}
    results["mnl_single_paper"] = select_single_paper_mnl(
        train, SYN_N, selection_assortments, val_true_fn, seed=seed)
    results["mnl_mixture_paper"] = select_paper_mnl_mixture(
        train, SYN_N, selection_assortments, val_true_fn, linkage_cache, seed=seed)
    results["topkgmm_single"] = select_single_topkgmm(
        train, SYN_N, selection_assortments, val_true_fn, seed=seed)
    results["topkgmm_mixture"] = select_topkgmm_mixture(
        train, SYN_N, selection_assortments, val_true_fn, linkage_cache, seed=seed)

    rows = []
    for family, res in results.items():
        test_mae = val_mae_for_prob_fn(res["prob_fn"], true_fn, full_assortments)
        row = {"rho": rho, "seed": seed, "family": family,
               "test_mae_full": test_mae, "val_mae_selection": res["val_mae"],
               "hyperparams": json.dumps(res.get("hyperparams", {})),
               "runtime_sec": res.get("runtime")}
        rows.append(row)

    total_runtime = time.time() - t_start
    print(f"[rho={rho} seed={seed}] done in {total_runtime:.1f}s; "
          + "; ".join(f"{r['family']}={r['test_mae_full']:.4f}" for r in rows))
    return rows


def main():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=str, default="outputs/experiment1_results_valfix.csv")
    args = parser.parse_args()

    all_rows = []
    fieldnames = ["rho", "seed", "family", "test_mae_full", "val_mae_selection",
                  "hyperparams", "runtime_sec"]
    for rho in args.rhos:
        for seed in args.seeds:
            rows = run_one(rho, seed)
            all_rows.extend(rows)
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
