"""
Faithful reimplementation of the repo's ORIGINAL MNL estimator
(Learn_utilities_MNL / choice_probs_MNL, Sushi_analysis.py lines
240-259) -- the moment-based, non-MLE estimator this report otherwise
replaced with a proper Plackett-Luce MLE (mnl_fit.py), kept here
specifically to answer: does TopKGMM's deficit to MNL narrow as block
correlation (rho) increases, under the SAME MNL baseline the paper
itself used for its sushi comparison -- not under our strengthened
PL-MLE, which could itself be confounding that comparison?

Original code (for reference):
    def Learn_utilities_MNL(train):
        fixed_val = 1
        u = np.zeros(n+1)
        u[0] = fixed_val
        for i in range(1, n+1):
            A = [i]
            count = np.sum([1 for tau in train if Choice(A,tau,n) != 0])
            u[i] = count / (fixed_val * len(preferences))   # <-- BUG: `preferences`
                                                              #     is an undefined/
                                                              #     global name here,
                                                              #     not `train`
        u_total = np.sum(u)
        return u / u_total

    def choice_probs_MNL(utilities, A):
        ANull = A + [0]
        u_A = [utilities[i] for i in ANull]
        total_u = sum(u_A)
        return {k: utilities[k]/total_u for k in ANull}

One bug fixed here (nothing else changed): `len(preferences)` is an
undefined-variable / wrong-scope bug in the original script (relies on
a global `preferences` list from Sushi_analysis.py's module-level code
that happens to equal the full dataset there) -- clearly meant to be
`len(train)`, the normalizing sample count. This is an interpreter-level
bug, not a modeling choice, so we correct it; everything else about the
estimator (fixed_val=1 outside anchor, per-item single-assortment count,
linear not softmax normalization, Choice()'s random tie-break) is kept
exactly as written, since THAT is the thing we are deliberately testing
against TopKGMM here.

u[i] for i>=1 is proportional to (# training rankings where item i
appears) + random noise from Choice()'s 50/50 fallback when i is absent
from a truncated ranking -- i.e. an ad hoc popularity count, not a
likelihood-based estimate. u[0] (outside) is a FIXED constant (1),
never learned from data at all, then renormalized alongside the rest.
"""
import random
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from LearningAlgos import Choice


def fit_paper_mnl(train, n, seed=0):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    rng = random.Random(seed)
    fixed_val = 1
    u = np.zeros(n + 1)
    u[0] = fixed_val
    n_train = len(train)
    for i in range(1, n + 1):
        A = [i]
        count = 0
        for tau in train:
            c = _choice_seeded(A, tau, n, rng)
            if c != 0:
                count += 1
        u[i] = count / (fixed_val * n_train)
    u_total = np.sum(u)
    return u / u_total


def _choice_seeded(A, tau, n, rng):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Choice(A, tau, n) with its random fallback driven by a local,
    seeded RNG instead of the global `random` module, so the paper-MNL
    fit is reproducible -- the original Choice() uses random.choice
    directly, which this preserves in effect while making the result
    deterministic given a seed.
    """
    Anull = A + [0]
    for item in tau:
        if item in Anull:
            return item
    return rng.choice(Anull)


def choice_probs_paper_mnl(u, A):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    Anull = A + [0]
    u_A = {k: u[k] for k in Anull}
    total = sum(u_A.values())
    return {k: u_A[k] / total for k in Anull}


def fit_paper_mnl_mixture(train_rankings, n, p, num_clusters, linkage_cache, seed=0):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Mixture version of the paper's original MNL estimator, via the
    same shared K_p clustering used for mnl_mixture/topkgmm_mixture
    (mixture.py) -- each cluster gets its own fit_paper_mnl fit, combined
    by cluster-size weight. Mirrors mixture.fit_mnl_mixture's structure
    exactly, with fit_paper_mnl/choice_probs_paper_mnl substituted for
    the PL-MLE + v0-calibration pipeline.
    """
    from mixture import cluster_rankings

    labels, D = linkage_cache.cluster_labels(p, num_clusters)
    clusters = cluster_rankings(train_rankings, labels)
    total_rows = len(train_rankings)

    components = []
    for lab, rows in clusters.items():
        if len(rows) < 2:
            continue
        weight = len(rows) / total_rows
        u = fit_paper_mnl(rows, n, seed=seed)
        components.append((weight, u))

    def choice_probs(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        Anull = A + [0]
        total_probs = {x: 0.0 for x in Anull}
        for weight, u in components:
            probs = choice_probs_paper_mnl(u, A)
            for x in Anull:
                total_probs[x] += weight * probs[x]
        return total_probs

    return choice_probs, components
