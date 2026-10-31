"""Sanity check: exact_choice_probs (full enumeration) matches Monte Carlo
simulation of the block generator, and reproduces the two verified
findings from the project notes:
  (1) compositions with unequal per-block representation carry strong
      rho-signal; the only EXACTLY degenerate case is the assortment
      being precisely two full blocks (size 6).
  (2) a single TopKGMM structurally cannot match the block generator's
      prediction pattern for an unequal-composition assortment -- checked
      separately in verify_topkgmm_recovery.py / the experiment driver
      itself (model mismatch, not a fitting bug).
"""
import random
from collections import Counter
from block_generator import sample_ranking, choice_from_full_rank, exact_choice_probs


def mc_probs(A, rho, n_trials=2_000_000, seed=42):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    rng = random.Random(seed)
    counts = Counter()
    for _ in range(n_trials):
        fr = sample_ranking(rho, rng)
        c = choice_from_full_rank(A, fr)
        counts[c] += 1
    return {k: v / n_trials for k, v in counts.items()}


def main():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    cases = [[1, 2, 4], [1, 4, 7], [1, 2, 3, 4, 5, 6], [2, 3, 5, 6, 8, 9], [1, 1][:1] + [2, 3]]
    cases = [[1, 2, 4], [1, 4, 7], [1, 2, 3, 4, 5, 6], [2, 3, 5, 6, 8, 9]]
    max_overall_diff = 0.0
    for A in cases:
        for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
            exact = exact_choice_probs(A, rho)
            mc = mc_probs(A, rho)
            diff = max(abs(exact.get(x, 0) - mc.get(x, 0)) for x in set(exact) | set(mc))
            max_overall_diff = max(max_overall_diff, diff)
            print(f"A={A} rho={rho}: exact={ {k: round(v,4) for k,v in exact.items()} } "
                  f"mc={ {k: round(v,4) for k,v in mc.items()} } maxdiff={diff:.4f}")
    print(f"\nmax diff across all cases: {max_overall_diff:.4f} (expect < ~0.002 at 2M MC trials)")

    print("\nDegenerate check: two full blocks [1,2,3,4,5,6] -- should be IDENTICAL across all rho")
    A = [1, 2, 3, 4, 5, 6]
    vals = [exact_choice_probs(A, rho) for rho in [0.0, 0.5, 1.0]]
    identical = all(v == vals[0] for v in vals)
    print("Identical across rho:", identical, vals[0])
    assert identical
    assert max_overall_diff < 0.01


if __name__ == "__main__":
    main()
