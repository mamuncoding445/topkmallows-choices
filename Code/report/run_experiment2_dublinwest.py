"""
Experiment 2: Dublin West 2002 STV ballots.

Common top-k length k chosen from TRAINING data only (largest k such that
>=50% of training ballots rank >=k candidates); ballots shorter than k are
dropped from the split they appear in (train/val/test each independently
filtered to length>=k, then truncated to their own top-k prefix).
Voter-level 60/20/20 split (ballot multiplicities expanded), reproducible
per seed. Reference probabilities for evaluation are EMPIRICAL (no latent
generator exists for real data): deterministic rule applied to the FULLY
OBSERVED visible top-k ballot -- first item of A appearing in the ballot's
top-k, else outside -- aggregated over the TEST split.
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict

from dublin_west_data import parse_blt, expand_ballots, choose_common_k, truncate_to_k, split_train_val_test
from metrics import enumerate_assortments, mae, uniform_baseline
from mnl_fit import empirical_outside_rate
from mixture import LinkageCache
from LearningAlgos import Choice
from model_selection import (make_selection_assortments, val_mae_for_prob_fn,
                              select_single_mnl, select_single_topkgmm,
                              select_mnl_mixture, select_topkgmm_mixture)


def choice_from_full_topk(A, topk_ranking):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Deterministic rule for a FULLY OBSERVED top-k ballot: first item of
    A appearing in the ballot, else outside (0). No randomness -- the
    full visible top-k is already known with certainty.
    """
    for item in topk_ranking:
        if item in A:
            return item
    return 0


def empirical_probs_from_rankings(A, rankings):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    Anull = A + [0]
    counts = {x: 0 for x in Anull}
    for r in rankings:
        c = choice_from_full_topk(A, r)
        counts[c] += 1
    m = len(rankings)
    return {x: counts[x] / m for x in Anull}


def run_one(seed, data_path="data/dublin_west/Dublin_West.dat", out_dir="outputs"):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    t_start = time.time()
    data = parse_blt(data_path)
    n = data["num_candidates"]
    expanded = expand_ballots(data["ballots"])

    train_full, val_full, test_full = split_train_val_test(expanded, seed=seed)
    k = choose_common_k(train_full, min_frac=0.5)

    train = [truncate_to_k(r, k) for r in train_full if len(r) >= k]
    val = [truncate_to_k(r, k) for r in val_full if len(r) >= k]
    test = [truncate_to_k(r, k) for r in test_full if len(r) >= k]

    full_assortments, selection_assortments = make_selection_assortments(n)

    def true_fn(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return empirical_probs_from_rankings(A, test)

    def val_true_fn(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return empirical_probs_from_rankings(A, val)

    def choice_fn(A, tau):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return Choice(A, tau, n)

    linkage_cache = LinkageCache(train, n)

    results = {}
    results["mnl_single"] = select_single_mnl(
        train, full_assortments, n, choice_fn, selection_assortments, val_true_fn)
    results["topkgmm_single"] = select_single_topkgmm(
        train, n, selection_assortments, val_true_fn)
    results["mnl_mixture"] = select_mnl_mixture(
        train, full_assortments, n, choice_fn, selection_assortments, val_true_fn, linkage_cache)
    results["topkgmm_mixture"] = select_topkgmm_mixture(
        train, n, selection_assortments, val_true_fn, linkage_cache)

    final_rows = []
    for family, res in results.items():
        test_mae = val_mae_for_prob_fn(res["prob_fn"], true_fn, full_assortments)
        row = {
            "seed": seed, "k": k, "family": family,
            "test_mae_full": test_mae,
            "val_mae_selection": res["val_mae"],
            "hyperparams": json.dumps(res.get("hyperparams", {})),
            "runtime_sec": res.get("runtime"),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
        }
        if "subsample" in res:
            row["subsample"] = json.dumps(res["subsample"])
        if "subsample_report" in res:
            row["subsample"] = json.dumps(res["subsample_report"])
        final_rows.append(row)

    uniform_mae = val_mae_for_prob_fn(lambda A: uniform_baseline(A + [0]), true_fn, full_assortments)
    final_rows.append({"seed": seed, "k": k, "family": "uniform_baseline",
                        "test_mae_full": uniform_mae, "val_mae_selection": None,
                        "hyperparams": "{}", "runtime_sec": None,
                        "n_train": len(train), "n_val": len(val), "n_test": len(test)})

    total_runtime = time.time() - t_start
    print(f"[seed={seed}] k={k} done in {total_runtime:.1f}s; "
          + "; ".join(f"{r['family']}={r['test_mae_full']:.4f}" for r in final_rows))
    return final_rows


def main():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=str, default="outputs/experiment2_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_rows = []
    for seed in args.seeds:
        rows = run_one(seed)
        all_rows.extend(rows)
        fieldnames = ["seed", "k", "family", "test_mae_full", "val_mae_selection",
                      "hyperparams", "runtime_sec", "n_train", "n_val", "n_test", "subsample"]
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({k2: row.get(k2, "") for k2 in fieldnames})
    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
