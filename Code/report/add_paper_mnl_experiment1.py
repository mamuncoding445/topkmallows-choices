"""
Adds mnl_single_paper / mnl_mixture_paper rows to the Experiment 1
results CSV -- the paper's ORIGINAL moment-based MNL estimator, fit on
the exact same train data (same seeding) as every other model, evaluated
against the same full-456-assortment exact reference probabilities.
Does not touch any existing rows; appends only.
"""
import argparse
import csv
import json
import random
import time

from block_generator import sample_topk, exact_choice_probs, N as SYN_N
from mixture import LinkageCache
from model_selection import (make_selection_assortments, val_mae_for_prob_fn,
                              select_single_paper_mnl, select_paper_mnl_mixture)


def run_one(rho, seed, n_train=8000, n_val=2000):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    t_start = time.time()
    rng = random.Random(1000 * seed + int(rho * 100))
    train = [sample_topk(rho, rng) for _ in range(n_train)]
    _ = [sample_topk(rho, rng) for _ in range(n_val)]

    full_assortments, selection_assortments = make_selection_assortments(SYN_N)

    def true_fn(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return exact_choice_probs(A, rho)

    linkage_cache = LinkageCache(train, SYN_N)

    res_single = select_single_paper_mnl(train, SYN_N, selection_assortments, true_fn, seed=seed)
    res_mixture = select_paper_mnl_mixture(train, SYN_N, selection_assortments, true_fn, linkage_cache, seed=seed)

    rows = []
    for family, res in [("mnl_single_paper", res_single), ("mnl_mixture_paper", res_mixture)]:
        test_mae = val_mae_for_prob_fn(res["prob_fn"], true_fn, full_assortments)
        rows.append({"rho": rho, "seed": seed, "family": family,
                      "test_mae_full": test_mae, "val_mae_selection": res["val_mae"],
                      "hyperparams": json.dumps(res.get("hyperparams", {})),
                      "runtime_sec": res.get("runtime"), "subsample": ""})

    print(f"[rho={rho} seed={seed}] paper-MNL done in {time.time()-t_start:.1f}s; "
          + "; ".join(f"{r['family']}={r['test_mae_full']:.4f}" for r in rows))
    return rows


def main():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--existing", type=str, default="outputs/experiment1_results.csv")
    parser.add_argument("--out", type=str, default="outputs/experiment1_results.csv")
    args = parser.parse_args()

    existing_rows = []
    with open(args.existing, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["family"] not in ("mnl_single_paper", "mnl_mixture_paper"):
                existing_rows.append(row)

    new_rows = []
    for rho in args.rhos:
        for seed in args.seeds:
            new_rows.extend(run_one(rho, seed))

    all_rows = existing_rows + new_rows
    family_order = {"mnl_single": 0, "mnl_single_paper": 1, "topkgmm_single": 2,
                     "mnl_mixture": 3, "mnl_mixture_paper": 4, "topkgmm_mixture": 5,
                     "uniform_baseline": 6}
    all_rows.sort(key=lambda r: (float(r["rho"]), int(r["seed"]), family_order[r["family"]]))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
