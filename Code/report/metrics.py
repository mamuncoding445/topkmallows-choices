"""
Metrics (MAE, TV), uniform baseline, and assortment enumeration/subsampling.
"""
import random
import itertools
import numpy as np


def mae(pred, true, Anull):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Mean absolute error over items in Anull (assortment + outside option)."""
    return float(np.mean([abs(pred.get(x, 0.0) - true.get(x, 0.0)) for x in Anull]))


def total_variation(pred, true, Anull):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """TV distance = 0.5 * sum |p-q|. Identity: TV = (|A|+1)*MAE/2."""
    return 0.5 * float(np.sum([abs(pred.get(x, 0.0) - true.get(x, 0.0)) for x in Anull]))


def uniform_baseline(Anull):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    m = len(Anull)
    return {x: 1.0 / m for x in Anull}


def enumerate_assortments(n, sizes=range(2, 7)):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    items = list(range(1, n + 1))
    out = []
    for s in sizes:
        out.extend(list(c) for c in itertools.combinations(items, s))
    return out


def subsample_assortments_per_size(all_assortments, per_size, seed):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Fixed-seed subsample of `per_size` assortments per distinct size,
    used ONLY for hyperparameter selection (grid search speed); final
    reported metrics always use the full `all_assortments` list.
    """
    rng = random.Random(seed)
    by_size = {}
    for a in all_assortments:
        by_size.setdefault(len(a), []).append(a)
    out = []
    for size, lst in by_size.items():
        lst = list(lst)
        rng.shuffle(lst)
        out.extend(lst[:per_size])
    return out


def avg_mae_over_assortments(prob_fn, true_fn, assortments):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """prob_fn(A)->dict, true_fn(A)->dict. Returns mean MAE across assortments."""
    vals = []
    for A in assortments:
        Anull = A + [0]
        pred = prob_fn(A)
        true = true_fn(A)
        vals.append(mae(pred, true, Anull))
    return float(np.mean(vals))
