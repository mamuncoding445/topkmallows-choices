"""
Targeted rerun: recompute ONLY topkgmm_single and topkgmm_mixture rows
for Experiment 2 (Dublin West) with the corrected BETA_GRID. See
rerun_topkgmm_experiment1.py's docstring for rationale -- MNL rows are
unaffected and left untouched.
"""
import argparse
import csv
import json
import time

from dublin_west_data import parse_blt, expand_ballots, choose_common_k, truncate_to_k, split_train_val_test
from mixture import LinkageCache
from model_selection import make_selection_assortments, val_mae_for_prob_fn, select_single_topkgmm, select_topkgmm_mixture
from run_experiment2_dublinwest import empirical_probs_from_rankings


def rerun_one(seed, data_path="data/dublin_west/Dublin_West.dat"):
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

    res_single = select_single_topkgmm(train, n, selection_assortments, val_true_fn)
    res_mixture = select_topkgmm_mixture(train, n, selection_assortments, val_true_fn, linkage_cache)

    rows = []
    for family, res in [("topkgmm_single", res_single), ("topkgmm_mixture", res_mixture)]:
        test_mae = val_mae_for_prob_fn(res["prob_fn"], true_fn, full_assortments)
        row = {"seed": seed, "k": k, "family": family,
               "test_mae_full": test_mae, "val_mae_selection": res["val_mae"],
               "hyperparams": json.dumps(res.get("hyperparams", {})),
               "runtime_sec": res.get("runtime"),
               "n_train": len(train), "n_val": sum(1 for r in val_full if len(r) >= k),
               "n_test": len(test)}
        if "subsample" in res:
            row["subsample"] = json.dumps(res["subsample"])
        if "subsample_report" in res:
            row["subsample"] = json.dumps(res["subsample_report"])
        rows.append(row)

    print(f"[seed={seed}] k={k} rerun done in {time.time()-t_start:.1f}s; "
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
            if row["family"] not in ("topkgmm_single", "topkgmm_mixture"):
                existing_rows.append(row)

    new_rows = []
    for seed in args.seeds:
        new_rows.extend(rerun_one(seed))

    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda r: (int(r["seed"]),
                                  {"mnl_single": 0, "topkgmm_single": 1, "mnl_mixture": 2,
                                   "topkgmm_mixture": 3, "uniform_baseline": 4}[r["family"]]))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k2: row.get(k2, "") for k2 in fieldnames})
    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
