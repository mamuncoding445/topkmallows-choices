"""Sanity check: fit_pl_mle recovers known item utilities when training
data is generated from a KNOWN Plackett-Luce model (not a TopKGMM,
not the block generator) -- i.e. the MLE is correctly implemented."""
import random
import numpy as np


def pl_sample_topk(v, k, rng):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Sample a top-k Plackett-Luce ranking: repeatedly draw without
    replacement with probability proportional to v among remaining
    items."""
    n = len(v)
    remaining = list(range(1, n + 1))
    weights = list(v)
    ranking = []
    for _ in range(k):
        total = sum(weights)
        probs = [w / total for w in weights]
        choice_idx = rng.choices(range(len(remaining)), probs)[0]
        ranking.append(remaining[choice_idx])
        del remaining[choice_idx]
        del weights[choice_idx]
    return ranking


def main():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    from mnl_fit import fit_pl_mle
    rng = random.Random(0)
    n = 9
    true_v = np.array([3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4])
    true_v_normalized = true_v / true_v.sum() * n
    k = 6
    train = [pl_sample_topk(list(true_v), k, rng) for _ in range(20000)]

    v_hat = fit_pl_mle(train, n)
    print("true v (normalized): ", np.round(true_v_normalized, 3))
    print("fitted v:            ", np.round(v_hat, 3))
    rel_err = np.abs(v_hat - true_v_normalized) / true_v_normalized
    print("relative error per item:", np.round(rel_err, 3))
    print("max relative error:", rel_err.max())
    assert rel_err.max() < 0.15, "PL MLE did not recover known utilities within tolerance"
    print("PASS: PL MLE recovers known utilities within 15% relative error at n_train=20000")


if __name__ == "__main__":
    main()
