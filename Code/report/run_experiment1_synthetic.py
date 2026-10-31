"""
Experiment 1: synthetic block-structured ranking experiment.

n=9 items, 3 blocks of 3, rho in {0,0.25,0.5,0.75,1}, k=6 top-k visible
rankings, 8000 train / 2000 val rankings per (rho,seed), >=5 seeds.
For each (rho,seed): select hyperparameters by validation MAE on a fixed
75-assortment subsample, then evaluate FINAL metrics on the FULL
enumeration of all 456 assortments (sizes 2-6) against EXACT reference
probabilities, for all four model families: single MNL, single TopKGMM,
MNL mixture, TopKGMM mixture, plus the uniform baseline.
"""
import argparse
import csv
import json
import os
import random
import time

from block_generator import sample_topk, exact_choice_probs, N as SYN_N
from metrics import enumerate_assortments, subsample_assortments_per_size, mae, uniform_baseline
from mnl_fit import empirical_outside_rate
from mixture import LinkageCache
from LearningAlgos import Choice
from model_selection import (make_selection_assortments, val_mae_for_prob_fn,
                              select_single_mnl, select_single_topkgmm,
                              select_mnl_mixture, select_topkgmm_mixture,
                              SELECTION_SUBSAMPLE_PER_SIZE, SELECTION_SEED)


def run_one(rho, seed, n_train=8000, n_val=2000, out_dir="outputs"):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    t_start = time.time()
    rng = random.Random(1000 * seed + int(rho * 100))
    train = [sample_topk(rho, rng) for _ in range(n_train)]
    val = [sample_topk(rho, rng) for _ in range(n_val)]

    full_assortments, selection_assortments = make_selection_assortments(SYN_N)

    def true_fn(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return exact_choice_probs(A, rho)

    def choice_fn(A, tau):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return Choice(A, tau, SYN_N)

    linkage_cache = LinkageCache(train, SYN_N)

    results = {}
    results["mnl_single"] = select_single_mnl(
        train, full_assortments, SYN_N, choice_fn, selection_assortments, true_fn)
    results["topkgmm_single"] = select_single_topkgmm(
        train, SYN_N, selection_assortments, true_fn)
    results["mnl_mixture"] = select_mnl_mixture(
        train, full_assortments, SYN_N, choice_fn, selection_assortments, true_fn, linkage_cache)
    results["topkgmm_mixture"] = select_topkgmm_mixture(
        train, SYN_N, selection_assortments, true_fn, linkage_cache)

    final_rows = []
    for family, res in results.items():
        test_mae = val_mae_for_prob_fn(res["prob_fn"], true_fn, full_assortments)
        row = {
            "rho": rho, "seed": seed, "family": family,
            "test_mae_full": test_mae,
            "val_mae_selection": res["val_mae"],
            "hyperparams": json.dumps(res.get("hyperparams", {})),
            "runtime_sec": res.get("runtime"),
        }
        if "subsample" in res:
            row["subsample"] = json.dumps(res["subsample"])
        if "subsample_report" in res:
            row["subsample"] = json.dumps(res["subsample_report"])
        final_rows.append(row)

    uniform_mae = val_mae_for_prob_fn(lambda A: uniform_baseline(A + [0]), true_fn, full_assortments)
    final_rows.append({"rho": rho, "seed": seed, "family": "uniform_baseline",
                        "test_mae_full": uniform_mae, "val_mae_selection": None,
                        "hyperparams": "{}", "runtime_sec": None})

    total_runtime = time.time() - t_start
    print(f"[rho={rho} seed={seed}] done in {total_runtime:.1f}s; "
          + "; ".join(f"{r['family']}={r['test_mae_full']:.4f}" for r in final_rows))
    return final_rows


def main():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n_train", type=int, default=8000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--out", type=str, default="outputs/experiment1_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_rows = []
    for rho in args.rhos:
        for seed in args.seeds:
            rows = run_one(rho, seed, n_train=args.n_train, n_val=args.n_val)
            all_rows.extend(rows)
            # write incrementally so partial progress is never lost
            fieldnames = ["rho", "seed", "family", "test_mae_full", "val_mae_selection",
                          "hyperparams", "runtime_sec", "subsample"]
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
