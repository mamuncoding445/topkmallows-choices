"""
Adds mnl_single_paper / mnl_mixture_paper rows to the Experiment 2
(Dublin West) results CSV -- the paper's original moment-based MNL
estimator, fit on the same train split (same seeding) as every other
model. Does not touch any existing rows; appends only.
"""
import argparse
import csv
import json
import time

from dublin_west_data import parse_blt, expand_ballots, choose_common_k, truncate_to_k, split_train_val_test
from mixture import LinkageCache
from model_selection import make_selection_assortments, val_mae_for_prob_fn, select_single_paper_mnl, select_paper_mnl_mixture
from run_experiment2_dublinwest import empirical_probs_from_rankings


def run_one(seed, data_path="data/dublin_west/Dublin_West.dat"):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    t_start = time.time()
    data = parse_blt(data_path)
    n = data["num_candidates"]
    expanded = expand_ballots(data["ballots"])

    train_full, val_full, test_full = split_train_val_test(expanded, seed=seed)
    k = choose_common_k(train_full, min_frac=0.5)

    train = [truncate_to_k(r, k) for r in train_full if len(r) >= k]
    test = [truncate_to_k(r, k) for r in test_full if len(r) >= k]

    full_assortments, selection_assortments = make_selection_assortments(n)

    def true_fn(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return empirical_probs_from_rankings(A, test)

    def val_true_fn(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        val = [truncate_to_k(r, k) for r in val_full if len(r) >= k]
        return empirical_probs_from_rankings(A, val)

    linkage_cache = LinkageCache(train, n)

    res_single = select_single_paper_mnl(train, n, selection_assortments, val_true_fn, seed=seed)
    res_mixture = select_paper_mnl_mixture(train, n, selection_assortments, val_true_fn, linkage_cache, seed=seed)

    rows = []
    for family, res in [("mnl_single_paper", res_single), ("mnl_mixture_paper", res_mixture)]:
        test_mae = val_mae_for_prob_fn(res["prob_fn"], true_fn, full_assortments)
        rows.append({"seed": seed, "k": k, "family": family,
                      "test_mae_full": test_mae, "val_mae_selection": res["val_mae"],
                      "hyperparams": json.dumps(res.get("hyperparams", {})),
                      "runtime_sec": res.get("runtime"),
                      "n_train": len(train), "n_val": sum(1 for r in val_full if len(r) >= k),
                      "n_test": len(test), "subsample": ""})

    print(f"[seed={seed}] k={k} paper-MNL done in {time.time()-t_start:.1f}s; "
          + "; ".join(f"{r['family']}={r['test_mae_full']:.4f}" for r in rows))
    return rows


def main():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--existing", type=str, default="outputs/experiment2_results.csv")
    parser.add_argument("--out", type=str, default="outputs/experiment2_results.csv")
    args = parser.parse_args()

    existing_rows = []
    with open(args.existing, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["family"] not in ("mnl_single_paper", "mnl_mixture_paper"):
                existing_rows.append(row)

    new_rows = []
    for seed in args.seeds:
        new_rows.extend(run_one(seed))

    all_rows = existing_rows + new_rows
    family_order = {"mnl_single": 0, "mnl_single_paper": 1, "topkgmm_single": 2,
                     "mnl_mixture": 3, "mnl_mixture_paper": 4, "topkgmm_mixture": 5,
                     "uniform_baseline": 6}
    all_rows.sort(key=lambda r: (int(r["seed"]), family_order[r["family"]]))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k2: row.get(k2, "") for k2 in fieldnames})
    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
