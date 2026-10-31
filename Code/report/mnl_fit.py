"""
Proper Plackett-Luce MLE for MNL, replacing the repo's moment-based
Learn_utilities_MNL/choice_probs_MNL (linear, not softmax, normalization --
not a real MLE; see project settled decision #2).

Two-stage, explicitly documented:
  Stage 1 (MLE): fit item utilities theta_1..theta_n (v_j = exp(theta_j))
    by maximizing the exploded-logit log-likelihood of the OBSERVED TOP-K
    RANKINGS directly (standard Plackett-Luce partial-ranking MLE). The
    outside option plays no role in how a ranking is generated, so it
    cannot be estimated from this same likelihood.
  Stage 2 (method-of-moments calibration): calibrate v0 (outside-option
    value) separately, by least-squares matching the model's predicted
    outside-share v0/(v0+sum_{j in A} v_j) against the empirical no-choice
    rate, on a sample of TRAINING assortments only.

This is an intentional, documented two-stage convention -- not a claim
about latent "declining" behavior.
"""
import numpy as np
from scipy.optimize import minimize


def fit_pl_mle(top_k_rankings, n, l2=1e-4):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Fit Plackett-Luce utilities theta (item 1..n) via MLE on a list of
    observed top-k rankings (each a list of distinct items, length k,
    items in 1..n, best-first order). Returns v = exp(theta), normalized
    so that sum(v) = n (identifiability -- only relative scale matters for
    choice probabilities; v0 is calibrated separately and compared on the
    same normalized scale downstream).
    """
    def neg_log_lik(theta):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        v = np.exp(theta)
        ll = 0.0
        grad = np.zeros(n)
        for ranking in top_k_rankings:
            remaining = set(range(1, n + 1))
            for item in ranking:
                idx_remaining = np.array(sorted(remaining)) - 1
                v_rem = v[idx_remaining]
                denom = v_rem.sum()
                ll += theta[item - 1] - np.log(denom)
                remaining.discard(item)
        return -ll + l2 * np.sum(theta ** 2)

    def neg_log_lik_grad(theta):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        v = np.exp(theta)
        grad = np.zeros(n)
        for ranking in top_k_rankings:
            remaining = set(range(1, n + 1))
            for item in ranking:
                idx_remaining = np.array(sorted(remaining)) - 1
                v_rem = v[idx_remaining]
                denom = v_rem.sum()
                grad[item - 1] += 1.0
                grad[idx_remaining] -= v_rem / denom
                remaining.discard(item)
        return -grad + 2 * l2 * theta

    theta0 = np.zeros(n)
    res = minimize(neg_log_lik, theta0, jac=neg_log_lik_grad, method="L-BFGS-B")
    theta_hat = res.x
    v = np.exp(theta_hat)
    v = v / v.sum() * n
    return v  # v[i-1] is utility of item i


def calibrate_v0(v, train_assortments, empirical_outside_rate_fn):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Least-squares calibration of v0 against empirical outside-share on
    TRAINING assortments only. empirical_outside_rate_fn(A) -> float in
    [0,1], the empirical fraction of training choices in which the
    customer chose "outside" for assortment A.

    empirical_outside_rate_fn does not depend on v0, so it is evaluated
    exactly once per assortment up front (not inside the optimizer loop) --
    a 456-assortment x ~50-100-NM-iteration x O(n_train) recomputation
    was previously the dominant cost of the whole MNL fit.
    """
    emp_rates = np.array([empirical_outside_rate_fn(A) for A in train_assortments])
    vA_sums = np.array([sum(v[a - 1] for a in A) for A in train_assortments])

    def sq_err(log_v0):
        # Written with the assistance of Claude (Anthropic), a generative AI tool.
        v0 = np.exp(log_v0[0])
        pred = v0 / (v0 + vA_sums)
        return float(np.sum((pred - emp_rates) ** 2))

    res = minimize(sq_err, x0=[0.0], method="Nelder-Mead")
    v0 = float(np.exp(res.x[0]))
    return v0


def choice_probs_pl(v, v0, A):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """MNL choice probabilities for assortment A given fitted utilities v
    (1-indexed by item, i.e. v[i-1]) and calibrated outside value v0.
    """
    vA = {a: v[a - 1] for a in A}
    total = v0 + sum(vA.values())
    probs = {a: vA[a] / total for a in A}
    probs[0] = v0 / total
    return probs


def empirical_outside_rate(A, training_topk_rankings, choice_fn):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Empirical no-choice rate for assortment A on training data, using
    the repo's Choice(A, tau, n) (random fallback) applied to each
    training top-k ranking -- this is the "generating/simulating choices"
    context (settled decision #1), appropriate here since we need
    simulated choice OUTCOMES from truncated rankings for calibration.
    """
    n_total = len(training_topk_rankings)
    n_outside = sum(1 for tau in training_topk_rankings if choice_fn(A, tau) == 0)
    return n_outside / n_total if n_total > 0 else 0.0
