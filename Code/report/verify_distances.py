"""Sanity check: vectorized kp_distance_matrix exactly matches the
pure-Python reference (num_misorders/num_incomparable_diff logic from
Sushi_analysis.py, reproduced here since that file has a syntax error and
cannot be imported as-is -- see project notes, bug #1)."""
import random
import numpy as np
from distances import kp_distance_matrix


def position(top):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    return {t: i for i, t in enumerate(top, start=1)}


def num_misorders(top_1, top_2, n):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    count = 0
    if len(top_1) != len(top_2):
        return count
    k = len(top_1)
    pos_1 = position(top_1)
    pos_2 = position(top_2)
    s = pos_1.keys() & pos_2.keys()
    l = len(s)
    for i in s:
        for j in s:
            if (pos_1[i] - pos_1[j]) * (pos_2[i] - pos_2[j]) < 0:
                count += 1
    count /= 2
    for i in top_1:
        for j in s:
            if i in s:
                continue
            if pos_1[i] < pos_1[j]:
                count += 1
    for i in top_2:
        for j in s:
            if i in s:
                continue
            if pos_2[i] < pos_2[j]:
                count += 1
    count += (k - l) * (k - l)
    return count


def num_incomparable_diff(top_1, top_2, n):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    count = 0
    if len(top_1) != len(top_2):
        return count
    k = len(top_1)
    pos_1 = position(top_1)
    pos_2 = position(top_2)
    s = pos_1.keys() & pos_2.keys()
    l = len(s)
    count += (k - l) * (k - l - 1)
    count += (k - l) * (n - 2 * k + l) * 2
    return count


def main():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    random.seed(1)
    n = 9
    all_ok = True
    for trial in range(10):
        k = random.randint(2, 6)
        rows = 15
        rankings = []
        for _ in range(rows):
            perm = list(range(1, n + 1))
            random.shuffle(perm)
            rankings.append(perm[:k])
        p = random.choice([0.01, 0.1, 0.5, 1, 2, 5])
        D = kp_distance_matrix(rankings, n, p)
        ref = np.zeros((rows, rows))
        for i in range(rows):
            for j in range(rows):
                if i == j:
                    continue
                mis = num_misorders(rankings[i], rankings[j], n)
                inc = num_incomparable_diff(rankings[i], rankings[j], n)
                ref[i, j] = mis + p * inc
        ok = np.allclose(D, ref)
        all_ok = all_ok and ok
        print(f"trial {trial}: k={k} p={p} match={ok} max_abs_diff={np.max(np.abs(D-ref))}")
    print("ALL OK" if all_ok else "MISMATCH DETECTED")
    assert all_ok


if __name__ == "__main__":
    main()
