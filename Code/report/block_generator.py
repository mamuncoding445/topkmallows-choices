"""
Synthetic block-structured ranking generator for Experiment 1.

Setup: n=12 items split into 4 blocks of 3 ({1,2,3},{4,5,6},{7,8,9},
{10,11,12}). n and the number of blocks were chosen (instead of the
original n=9/3-blocks) so that k=6 (two full blocks visible) satisfies
TopKGMM/DyPChIP's k <= (n+1)//2 tractability bound exactly (effective
center cap at n=12 is (12+1)//2=6), avoiding the center-truncation
shortcut that was needed at n=9 (cap=5 < k=6).
A ranking is generated in two branches, mixed by rho in [0,1]:
  - "block" branch (prob rho): first draw a uniformly random ORDER OF THE
    BLOCKS, then within each block draw a uniformly random order of its
    items; concatenate to get a full ranking of all 12 items. This induces
    strong correlation between items in the same block (they cluster
    together in rank).
  - "uniform" branch (prob 1-rho): draw a uniformly random permutation of
    all 12 items directly, ignoring block structure.
A visible top-k list (k=6) is then the prefix of the resulting permutation.

rho=0 is the fully "unstructured" (uniform random ranking) regime, rho=1
is the fully block-structured regime.

Reference (true) choice probabilities for an assortment A are computed by
FULL ENUMERATION over both branches (not any closed-form shortcut -- a
hand-derived closed form was tried previously and found WRONG; see project
memory). For a given full ranking of all 12 items, the deterministic
top-k/outside-option convention is applied (see Choice_fromrank below):
first item of A appearing in the visible top-k, else outside option (0).
This deterministic rule is correct here because the full top-k list is
already fully observed/known with certainty -- no random fallback needed.
Contrast with Choice(A, tau, n) in LearningAlgos.py, which is used only
when *generating/simulating* truncated training data and falls back to
random choice when neither the target nor 0 appears in a truncated tau;
that randomized convention is deliberately NOT used for reference
probabilities (see project settled-decisions doc, point 1).
"""
import itertools
import random
from functools import lru_cache
import numpy as np

N = 12
BLOCKS = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
K = 6


def _build_all_block_rankings():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """All len(BLOCKS)! * (block_size!)^len(BLOCKS) = 31104 equally likely
    full rankings under the block branch, as a numpy array (one-time cost
    at import, independent of A/rho so never recomputed per call).
    """
    rankings = []
    for block_order in itertools.permutations(BLOCKS):
        within_block_perms = [itertools.permutations(b) for b in block_order]
        for combo in itertools.product(*within_block_perms):
            rankings.append([item for block_perm in combo for item in block_perm])
    return np.array(rankings, dtype=np.int64)


_ALL_BLOCK_RANKINGS = _build_all_block_rankings()


def sample_ranking(rho, rng):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Sample one full ranking (permutation of 1..9) under the block model."""
    if rng.random() < rho:
        block_order = list(BLOCKS)
        rng.shuffle(block_order)
        perm = []
        for b in block_order:
            b = list(b)
            rng.shuffle(b)
            perm.extend(b)
        return perm
    else:
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        return perm


def sample_topk(rho, rng, k=K):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    return sample_ranking(rho, rng)[:k]


def choice_from_full_rank(A, full_rank, k=K):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Deterministic choice rule from a fully observed ranking's visible top-k.
    First item of A appearing in the top-k prefix, else outside option 0.
    """
    topk = full_rank[:k]
    for item in topk:
        if item in A:
            return item
    return 0


def exact_choice_probs(A, rho, k=K):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Cached wrapper: full enumeration is expensive (O(1296) block-branch
    + O(C(n,|A|)*|A|!) uniform-branch per call) and is called repeatedly
    for the SAME (A, rho) across every grid-search hyperparameter
    combination -- caching here turns a per-grid-cell cost into a
    one-time cost per distinct assortment, which is the dominant speedup
    for the whole pipeline (previously the actual bottleneck, not BuCchoi
    or DyPChIP as initially suspected).
    """
    return _exact_choice_probs_cached(tuple(A), rho, k)


@lru_cache(maxsize=None)
def _exact_choice_probs_cached(A, rho, k):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    return _exact_choice_probs_impl(list(A), rho, k)


def _exact_choice_probs_impl(A, rho, k=K):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Exact reference choice probabilities for assortment A via full
    enumeration of both generator branches (no closed-form shortcuts).

    Block branch: enumerate all block orders (len(BLOCKS)! = 24) x all
    within-block orders (3!^4 = 1296) = 31104 equally likely full rankings
    (generalized to an arbitrary number of equal-sized blocks via
    itertools.product over each block's own permutations, rather than the
    previous hardcoded-for-3-blocks triple nested loop).
    Uniform branch: every permutation of N items is equally likely; by
    exchangeability, the distribution of which-of-A-is-chosen depends only
    on the *relative order* of items in A union {other items that could
    appear ahead of them in the top-k}. We enumerate exactly via all 9!
    permutations being infeasible directly, so instead we exploit that
    for the uniform branch the chosen item depends only on the relative
    ranks of items in A (a small set) within a uniformly random permutation
    of N items -- which is equivalent to a uniformly random ORDER of A's
    items plus a uniformly random number of "other" items ranked ahead of
    all of A. We compute this exactly via a combinatorial closed form that
    is NOT the block structure (so it carries no risk of the block-closed
    -form error): exact probability over uniformly random permutations
    that the first item of A to appear in the top-k is item a, or that
    none of A appears in the top-k.
    """
    Anull = A + [0]
    # ---- block branch: full enumeration over all equally likely full ranks
    # (generalized to an arbitrary number of equal-sized blocks), vectorized
    # with numpy -- the 31104-row table of full rankings (_ALL_BLOCK_RANKINGS,
    # module-level, independent of A/rho) is built once; per-call cost here
    # is then one vectorized "first item of A in the top-k" scan over all
    # rows instead of a 31104-iteration Python loop per call (needed once
    # n grew from 9 to 12 -- the pure-Python version became the pipeline's
    # dominant cost). Same exact enumeration/semantics as the loop version,
    # verified to agree with it in verify_generator.py.
    topk_cols = _ALL_BLOCK_RANKINGS[:, :k]
    in_A = np.isin(topk_cols, np.array(A))
    has_any = in_A.any(axis=1)
    first_idx = in_A.argmax(axis=1)
    chosen = np.where(has_any, topk_cols[np.arange(topk_cols.shape[0]), first_idx], 0)
    total_block = chosen.shape[0]
    block_counts = np.bincount(chosen, minlength=N + 1)
    block_probs = {x: block_counts[x] / total_block for x in Anull}

    # ---- uniform branch: exact combinatorial probabilities
    # For a uniformly random permutation of N items, and a visible top-k
    # prefix: P(no item of A in top-k) = C(N-|A|, k) / C(N, k)
    # (choose which k of the N-|A| non-A items occupy the top-k, out of all
    # equally likely ways to choose which k items occupy the top-k -- order
    # within top-k doesn't matter for "is some item of A present").
    # Given at least one item of A is in the top-k, by symmetry every item
    # of A is equally likely to be the FIRST one encountered when scanning
    # the top-k in rank order... but items of A are not symmetric in when
    # they appear, so we instead compute P(specific item a is the first of
    # A encountered in the top-k) directly by enumeration over "number of
    # non-A items ranked ahead of a, within the top-k" -- exact small
    # enumeration (not a shortcut specific to blocks).
    from math import comb
    Asize = len(A)
    other_count = N - Asize
    uniform_counts = {x: 0.0 for x in Anull}
    total_uniform = comb(N, k)  # number of ways to choose the SET of top-k items (unordered is fine since within-set order doesn't matter for choice identity determination below, but we need order among A's members, so use ordered enumeration)
    # Exact approach: enumerate the relative order of A among a uniformly
    # random permutation combined with how many of the (N-Asize) other
    # items land in the first k slots before the first A-item appears.
    # We just brute-force over all orderings of a *reduced* system:
    # the outcome only depends on (i) which other items are in the top-k
    # (not their identity, just the count), and (ii) the relative order
    # of A's own members among themselves and relative to the boundary at
    # position k. This is still fully exact (no probabilistic shortcut
    # unverified by Monte Carlo) because we enumerate every possible
    # interleaving of A's |A| members with "block" of other items.
    # Equivalent exact formula (order statistics of a random subset):
    # P(item a_j, the j-th earliest member of A in a uniformly random order
    # of A, is the chosen item) = sum over t=0..k-j of
    #   P(exactly j-1 members of A before it, t other items before it, and
    #     it lands at position <=k) ... -- to avoid any further hand
    # derivation that could repeat the earlier mistake, we instead do this
    # by direct Monte-Carlo-free EXACT enumeration using random permutation
    # symmetry: condition on the relative rank (1..N) of each A-item among
    # all N items is exchangeable, so we can compute via exact enumeration
    # over all C(N, Asize) ways to place A's items' absolute positions
    # combined with all Asize! internal orders -- still O(C(N,Asize)*Asize!)
    # which is tiny for Asize<=6.
    #
    # VECTORIZED (same exact enumeration, computed without per-pair Python
    # iterations -- needed once N grew from 9 to 12, since the per-call cost
    # here is O(C(N,Asize)*Asize!), e.g. ~665k pairs at Asize=6,N=12 versus
    # ~60k at N=9). Soundness: itertools.combinations(range(1,N+1), Asize)
    # always yields each `positions` tuple already SORTED ASCENDING (since
    # range(1,N+1) is sorted and combinations preserves input order), so
    # positions[0] is always the smallest position in that tuple. Hence the
    # "item in A with smallest position <= k" from the original loop is
    # fully determined by (positions[0], perm_of_A[0]) alone: it is
    # perm_of_A[0] if positions[0] <= k, else 0 (no item of A reaches the
    # top-k) -- regardless of positions[1:]/perm_of_A[1:]. This is a direct
    # consequence of the ascending order the original loop already relied
    # on (`min(candidates)[1]` picks the smallest position), not a new
    # modeling assumption, and is cross-checked against the original loop
    # in verify_generator.py.
    positions_arr = np.array(list(itertools.combinations(range(1, N + 1), Asize)))
    perms_arr = np.array(list(itertools.permutations(A)))
    M = positions_arr.shape[0]
    P = perms_arr.shape[0]
    first_pos = positions_arr[:, 0]          # shape (M,)
    first_item = perms_arr[:, 0]             # shape (P,)
    le_mask = first_pos <= k
    M_le = int(le_mask.sum())
    M_gt = M - M_le
    uniform_counts[0] += M_gt * P
    if M_le > 0:
        item_vals, item_counts = np.unique(first_item, return_counts=True)
        for item, cnt in zip(item_vals, item_counts):
            uniform_counts[int(item)] += M_le * cnt
    total_uniform_enum = comb(N, Asize) * math_factorial(Asize)
    uniform_probs = {x: uniform_counts[x] / total_uniform_enum for x in Anull}

    final_probs = {x: rho * block_probs[x] + (1 - rho) * uniform_probs[x] for x in Anull}
    return final_probs


def math_factorial(x):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    import math
    return math.factorial(x)


def enumerate_assortments(n=N, sizes=range(2, 7)):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """All assortments (as sorted lists) of each given size from 1..n."""
    items = list(range(1, n + 1))
    out = []
    for s in sizes:
        for combo in itertools.combinations(items, s):
            out.append(list(combo))
    return out


if __name__ == "__main__":
    rng = random.Random(0)
    A = [1, 2, 4]
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(rho, exact_choice_probs(A, rho))
