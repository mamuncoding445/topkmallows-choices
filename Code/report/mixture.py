"""
Mixture-of-MNL and mixture-of-TopKGMM models via shared clustering.

Clustering-then-fit procedure (paper Appendix B.1, with one intentional
deviation per report instructions): cluster training rankings via the K_p
distance for a grid of p, into {2,3,5} clusters; the paper keeps only
clusters with positive silhouette score and combines by cluster-size
weight. We DEVIATE: we select component count and (beta,p) by HELD-OUT
VALIDATION MAE instead of silhouette score ("clustering-then-fit, selected
by held-out validation error rather than silhouette"). Clusters are still
combined via cluster-size weights.

Shared hyperparameters (project settled decision #5):
  - p is the SAME quantity in the K_p clustering distance and in each
    TopKGMM component's own incomparable-pair penalty -- shared, not
    independently tuned per role.
  - beta is shared across all components of a given TopKGMM-mixture
    candidate (not fit per component). This keeps hyperparameter COUNT
    comparable to the single-model baseline; it does NOT make the mixture
    and single-model fits equally complex overall (the mixture still has
    one independently learned center per component).

Linkage caching: the clustering distance matrix (and its linkage) depends
only on p, not on num_clusters or which family (MNL-mixture vs
TopKGMM-mixture) -- cached once per p and reused across num_clusters in
{2,3,5} and across both mixture families within one (rho,seed) run.
"""
import sys
import os
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from LearningAlgos import Choice

from distances import kp_distance_matrix
from mnl_fit import fit_pl_mle, calibrate_v0, choice_probs_pl, empirical_outside_rate
from topkgmm_fit import fit_center, choice_probs_topkgmm


class LinkageCache:
    """Caches the K_p distance matrix and its hierarchical linkage, keyed
    by p only (not num_clusters, not family) -- see module docstring.
    """
    def __init__(self, rankings, n):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        self.rankings = rankings
        self.n = n
        self._cache = {}

    def get(self, p):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        if p not in self._cache:
            D = kp_distance_matrix(self.rankings, self.n, p)
            condensed = D[np.triu_indices(len(D), k=1)]
            Z = linkage(condensed, method="average")
            self._cache[p] = (D, Z)
        return self._cache[p]

    def cluster_labels(self, p, num_clusters):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        D, Z = self.get(p)
        labels = fcluster(Z, t=num_clusters, criterion="maxclust")
        return labels, D


def cluster_rankings(rankings, labels):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    clusters = defaultdict(list)
    for idx, lab in enumerate(labels):
        clusters[lab].append(rankings[idx])
    return clusters


def fit_mnl_mixture(train_rankings, train_assortments_for_calibration,
                     empirical_outside_rate_fn, n, p, num_clusters, linkage_cache):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Fit a mixture of MNL models via shared clustering at the given p
    and num_clusters. Each component gets its own PL-MLE fit and its own
    v0 calibration (v0 calibration uses the SAME training assortments for
    every component, per settled design -- weights combine at the
    mixture level, not within a single component's calibration).
    Returns a callable choice_probs(A) -> dict.
    """
    labels, D = linkage_cache.cluster_labels(p, num_clusters)
    clusters = cluster_rankings(train_rankings, labels)
    total = len(train_rankings)

    components = []
    for lab, rows in clusters.items():
        if len(rows) < 2:
            continue
        weight = len(rows) / total
        v = fit_pl_mle(rows, n)

        def outside_rate_for_cluster(A, rows=rows):
            # Written with the assistance of Claude (Anthropic), a generative AI tool.
            return empirical_outside_rate(A, rows, lambda A_, tau: Choice(A_, tau, n))

        v0 = calibrate_v0(v, train_assortments_for_calibration, outside_rate_for_cluster)
        components.append((weight, v, v0))

    def choice_probs(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        Anull = A + [0]
        total_probs = {x: 0.0 for x in Anull}
        for weight, v, v0 in components:
            probs = choice_probs_pl(v, v0, A)
            for x in Anull:
                total_probs[x] += weight * probs[x]
        return total_probs

    return choice_probs, components


def fit_topkgmm_mixture(train_rankings, n, p, beta, num_clusters, linkage_cache,
                         max_rows_per_component=2000, seed=0):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Fit a mixture of TopKGMM models via shared clustering at the given
    p and num_clusters; beta is SHARED across all components (not fit
    per-component). Each component's center is learned independently via
    BuCchoi with row subsampling. Returns (choice_probs callable,
    subsample_report list of dicts with available/used per component).
    """
    labels, D = linkage_cache.cluster_labels(p, num_clusters)
    clusters = cluster_rankings(train_rankings, labels)
    total = len(train_rankings)

    components = []
    subsample_report = []
    for lab, rows in clusters.items():
        if len(rows) < 2:
            continue
        weight = len(rows) / total
        center, n_avail, n_used = fit_center(n, rows, max_rows=max_rows_per_component, seed=seed)
        components.append((weight, center))
        subsample_report.append({"cluster": int(lab), "available": n_avail, "used": n_used,
                                  "weight": weight, "center_len": len(center)})

    def choice_probs(A):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        Anull = A + [0]
        total_probs = {x: 0.0 for x in Anull}
        for weight, center in components:
            probs = choice_probs_topkgmm(A, center, n, p, beta)
            for x in Anull:
                total_probs[x] += weight * probs[x]
        return total_probs

    return choice_probs, components, subsample_report
