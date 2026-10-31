"""
BuCchoi + DyPChIP wrapper for fitting a single TopKGMM component and
computing its choice probabilities.

Includes two fixes to the underlying repo code (see project notes):
  - Center-length cap at min(tractability_cap, (n+1)//2): DyPChIP's DP
    divides by (n+1-k-j) while building the empty-profile base case, for
    j up to k-ell-1 in the worst case; this requires k <= (n+1)//2 or a
    ZeroDivisionError results. At n=9 this binds (cap = 5), unlike at
    sushi's n=100 scale where the repo's own ~12-15 tractability cap
    binds first.
  - BuCchoi row-subsampling: BuCchoi cost scales with row count and is
    the dominant runtime cost. We subsample to a reproducible random
    subset of up to `max_rows` training rankings (default 2000) before
    calling BuCchoi, using all rows when fewer are available.
  - BuCchoi fallback seeding: LearningAlgos.Choice() falls back to
    `random.choice` (the GLOBAL random module) whenever none of a
    singleton/pairwise query's candidates appear in a given training
    ranking's visible top-k. BuCchoi itself is called unmodified (not
    edited), but we now seed the global random module immediately before
    each call so that fallback is reproducible given `seed`, instead of
    depending on whatever global RNG state happened to exist beforehand.
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from LearningAlgos import BuCchoi
from DyPChiP import DyPChIP_final

TRACTABILITY_CAP = 12


def effective_center_cap(n):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    return min(TRACTABILITY_CAP, (n + 1) // 2)


def subsample_rows(rankings, max_rows, seed):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Reproducible subsample of up to max_rows rankings. Returns
    (subsampled_rows, n_available, n_used).
    """
    n_available = len(rankings)
    if n_available <= max_rows:
        return list(rankings), n_available, n_available
    rng = random.Random(seed)
    idx = list(range(n_available))
    rng.shuffle(idx)
    chosen = idx[:max_rows]
    return [rankings[i] for i in chosen], n_available, max_rows


def fit_center(n, rankings, max_rows=2000, seed=0):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Fit a TopKGMM center via BuCchoi on a (possibly subsampled) set of
    training top-k rankings. Returns (center_list, n_available, n_used).
    Center length is capped at effective_center_cap(n) before being
    returned, so downstream DyPChIP calls never hit the (n+1)//2 bound.
    """
    sub_rankings, n_available, n_used = subsample_rows(rankings, max_rows, seed)
    random.seed(seed)  # seeds Choice()'s global-random fallback inside BuCchoi
    _, center_list = BuCchoi(list(range(1, n + 1)), sub_rankings)
    cap = effective_center_cap(n)
    if len(center_list) > cap:
        center_list = center_list[:cap]
    return center_list, n_available, n_used


def choice_probs_topkgmm(A, center, n, p, beta, w=None):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Choice probabilities for assortment A under a single TopKGMM with
    the given center, hyperparameters (p=incomparable penalty, beta), and
    weights w (defaults to all-ones per project scope: learning nonuniform
    weights is out of scope).
    """
    k = len(center)
    if w is None:
        w = [1 for _ in range(k + 1)]
    return DyPChIP_final(A, center, n, k, p, beta, w)
