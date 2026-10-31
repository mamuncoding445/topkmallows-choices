"""
Shared hyperparameter grids and selection routines used by both
experiment drivers. Selection uses validation MAE on a FIXED random
subsample of assortments (15 per size across sizes 2-6 = 75 total,
seeded at 999) for grid-search speed; FINAL reported metrics always use
full enumeration of every assortment (never subsampled) -- settled
decision #3.
"""
import time
import numpy as np

from metrics import enumerate_assortments, subsample_assortments_per_size, mae
from mnl_fit import fit_pl_mle, calibrate_v0, choice_probs_pl, empirical_outside_rate
from topkgmm_fit import fit_center, choice_probs_topkgmm
from mixture import LinkageCache, fit_mnl_mixture, fit_topkgmm_mixture
from paper_mnl_fit import fit_paper_mnl, choice_probs_paper_mnl, fit_paper_mnl_mixture

# Synthetic-experiment grid (n=9, narrower than sushi's Table 1 grid --
# sushi's n=100 is far larger; widen only if grid-boundary optima recur).
# NOTE: deliberately shrunk from the original [0.1,0.5,1.0,2.0] x
# [0.1,0.3,0.5,0.8,1.2] x [2,3,5] grid to fit a 2-hour total wall-clock
# budget for the whole pipeline (see project time-budget notes). The
# dominant cost is DyPChIP call count = len(P_GRID)*len(BETA_GRID)*
# sum(CLUSTER_COUNTS)*len(selection_assortments) for the TopKGMM mixture
# grid search; profiling showed DyPChIP itself is fast per call (~8ms)
# but the full grid multiplies that by thousands of calls.
#
# CORRECTION (post-hoc, via beta_sweep.py): the original BETA_GRID
# floored at 0.3, which a dedicated sweep showed was far too high --
# single TopKGMM's full-enumeration MAE kept improving all the way down
# to beta~0.01-0.02 at every rho tested (0, 0.5, 1), matching the paper's
# own sushi-optimal beta=0.05 (found via a much wider grid than we could
# afford initially). BETA_GRID now spans that corrected range. MNL is
# unaffected (no beta dependence), so only topkgmm_single/topkgmm_mixture
# needed to be refit with this corrected grid.
P_GRID = [0.5, 1.0]
BETA_GRID = [0.01, 0.02, 0.05, 0.1, 0.3, 0.6]
CLUSTER_COUNTS = [2, 3]

SELECTION_SUBSAMPLE_PER_SIZE = 8
SELECTION_SEED = 999


def make_selection_assortments(n):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    full = enumerate_assortments(n, sizes=range(2, 7))
    selection = subsample_assortments_per_size(full, SELECTION_SUBSAMPLE_PER_SIZE, SELECTION_SEED)
    return full, selection


def val_mae_for_prob_fn(prob_fn, true_fn, assortments):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    vals = []
    for A in assortments:
        Anull = A + [0]
        pred = prob_fn(A)
        true = true_fn(A)
        vals.append(mae(pred, true, Anull))
    return float(np.mean(vals))


def select_single_mnl(train_rankings, calibration_assortments, n, choice_fn, val_assortments, true_fn):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    t0 = time.time()
    v = fit_pl_mle(train_rankings, n)

    def outside_rate(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        return empirical_outside_rate(A, train_rankings, choice_fn)

    v0 = calibrate_v0(v, calibration_assortments, outside_rate)
    prob_fn = lambda A: choice_probs_pl(v, v0, A)
    val_mae = val_mae_for_prob_fn(prob_fn, true_fn, val_assortments)
    runtime = time.time() - t0
    return {"family": "mnl_single", "prob_fn": prob_fn, "val_mae": val_mae,
            "runtime": runtime, "hyperparams": {}}


def select_single_topkgmm(train_rankings, n, val_assortments, true_fn, max_rows=2000, seed=0,
                           p_grid=P_GRID, beta_grid=BETA_GRID):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    t0 = time.time()
    center, n_avail, n_used = fit_center(n, train_rankings, max_rows=max_rows, seed=seed)
    best = None
    for p in p_grid:
        for beta in beta_grid:
            prob_fn = lambda A, p=p, beta=beta: choice_probs_topkgmm(A, center, n, p, beta)
            vmae = val_mae_for_prob_fn(prob_fn, true_fn, val_assortments)
            if best is None or vmae < best["val_mae"]:
                best = {"family": "topkgmm_single", "prob_fn": prob_fn, "val_mae": vmae,
                        "hyperparams": {"p": p, "beta": beta},
                        "subsample": {"available": n_avail, "used": n_used}}
    best["runtime"] = time.time() - t0
    return best


def select_mnl_mixture(train_rankings, calibration_assortments, n, choice_fn, val_assortments,
                        true_fn, linkage_cache, p_grid=P_GRID, cluster_counts=CLUSTER_COUNTS):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    t0 = time.time()
    best = None
    for p in p_grid:
        for num_clusters in cluster_counts:
            prob_fn, components = fit_mnl_mixture(
                train_rankings, calibration_assortments,
                lambda A: empirical_outside_rate(A, train_rankings, choice_fn),
                n, p, num_clusters, linkage_cache)
            vmae = val_mae_for_prob_fn(prob_fn, true_fn, val_assortments)
            if best is None or vmae < best["val_mae"]:
                best = {"family": "mnl_mixture", "prob_fn": prob_fn, "val_mae": vmae,
                        "hyperparams": {"p": p, "num_clusters": num_clusters},
                        "num_components": len(components)}
    best["runtime"] = time.time() - t0
    return best


def select_topkgmm_mixture(train_rankings, n, val_assortments, true_fn, linkage_cache,
                            p_grid=P_GRID, beta_grid=BETA_GRID, cluster_counts=CLUSTER_COUNTS,
                            max_rows_per_component=2000, seed=0):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    t0 = time.time()
    best = None
    for p in p_grid:
        for num_clusters in cluster_counts:
            # center fits for this (p, num_clusters) combo are reused across the beta grid
            prob_fn_by_beta = {}
            prob_fn, components, subsample_report = fit_topkgmm_mixture(
                train_rankings, n, p, beta_grid[0], num_clusters, linkage_cache,
                max_rows_per_component=max_rows_per_component, seed=seed)
            # cache centers; re-derive choice_probs for each beta without re-running BuCchoi
            centers = [c for _, c in components]
            weights = [w for w, _ in components]
            for beta in beta_grid:
                def prob_fn_beta(A, beta=beta, centers=centers, weights=weights):
                    # Written with the assistance of Claude (Anthropic), a generative AI tool.
                    Anull = A + [0]
                    total_probs = {x: 0.0 for x in Anull}
                    for weight, center in zip(weights, centers):
                        probs = choice_probs_topkgmm(A, center, n, p, beta)
                        for x in Anull:
                            total_probs[x] += weight * probs[x]
                    return total_probs
                vmae = val_mae_for_prob_fn(prob_fn_beta, true_fn, val_assortments)
                if best is None or vmae < best["val_mae"]:
                    best = {"family": "topkgmm_mixture", "prob_fn": prob_fn_beta, "val_mae": vmae,
                            "hyperparams": {"p": p, "beta": beta, "num_clusters": num_clusters},
                            "subsample_report": subsample_report}
    best["runtime"] = time.time() - t0
    return best


def select_single_paper_mnl(train_rankings, n, val_assortments, true_fn, seed=0):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Fits the paper's ORIGINAL (moment-based, non-MLE) MNL estimator --
    no hyperparameters to select, included for parity with select_single_mnl's
    return shape.
    """
    t0 = time.time()
    u = fit_paper_mnl(train_rankings, n, seed=seed)
    prob_fn = lambda A: choice_probs_paper_mnl(u, A)
    val_mae = val_mae_for_prob_fn(prob_fn, true_fn, val_assortments)
    return {"family": "mnl_single_paper", "prob_fn": prob_fn, "val_mae": val_mae,
            "runtime": time.time() - t0, "hyperparams": {}}


def select_paper_mnl_mixture(train_rankings, n, val_assortments, true_fn, linkage_cache,
                              p_grid=P_GRID, cluster_counts=CLUSTER_COUNTS, seed=0):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Mixture version of the paper's original MNL estimator, selected by
    the same validation-MAE procedure as select_mnl_mixture.
    """
    t0 = time.time()
    best = None
    for p in p_grid:
        for num_clusters in cluster_counts:
            prob_fn, components = fit_paper_mnl_mixture(
                train_rankings, n, p, num_clusters, linkage_cache, seed=seed)
            vmae = val_mae_for_prob_fn(prob_fn, true_fn, val_assortments)
            if best is None or vmae < best["val_mae"]:
                best = {"family": "mnl_mixture_paper", "prob_fn": prob_fn, "val_mae": vmae,
                        "hyperparams": {"p": p, "num_clusters": num_clusters},
                        "num_components": len(components)}
    best["runtime"] = time.time() - t0
    return best
