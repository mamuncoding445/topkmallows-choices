"""
Vectorized K_p distance matrix for clustering top-k rankings (misorders +
p * incomparables), matching Sushi_analysis.py's num_misorders /
num_incomparable_diff exactly, computed via numpy matrix algebra instead
of O(rows^2 * k^2) pure Python loops. Verified exact match against the
pure-Python reference in verify_distances.py.

Benchmarked (per project notes): ~1.5s for 8000x8000, ~8s for ~18000x18000.

Derivation summary (n = number of items, k = top-k length, both small: 9):
  Let s_ij = common items between rankings i,j, l_ij = |s_ij|.
  misorders(i,j) = misorders_common(i,j) + cross1(i,j) + cross2(i,j) + (k-l_ij)^2
  where:
    misorders_common(i,j) = (l_ij*(l_ij-1) - S_ij) / 4
      with S_ij = Sign_i_flat . Sign_j_flat (dot product of flattened
      n x n sign matrices), Sign_i[a,b] = +1 if a,b both present in
      ranking i and pos_i[a] < pos_i[b]; -1 if pos_i[a] > pos_i[b]; 0 if
      either item absent from ranking i (a==b always 0).
    cross1(i,j) = sum_{a in top_i\\s, b in s} [pos_i[a] < pos_i[b]]
      = Term1[i,j] - Term2[i,j], where
      Term1[i,j] = sum_b c_i[b] * presence_j[b],  c_i[b] = presence_i[b]*(pos_i[b]-1)
      Term2[i,j] = presence_j^T M_i presence_j,   M_i[a,b] = [a,b present in i, pos_i[a]<pos_i[b]]
    cross2(i,j) = cross1(j,i)  (symmetric swap of roles)
  incomparables(i,j) = (k-l_ij)*(k-l_ij-1) + (k-l_ij)*(n-2k+l_ij)*2
"""
import numpy as np


def _presence_and_pos(rankings, n):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    rows = len(rankings)
    pos = np.zeros((rows, n), dtype=np.float64)
    for i, ranking in enumerate(rankings):
        for rank_idx, item in enumerate(ranking):
            pos[i, item - 1] = rank_idx + 1
    presence = (pos > 0)
    return pos, presence


def kp_distance_matrix(rankings, n, p):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    rows = len(rankings)
    k = len(rankings[0])
    pos, presence = _presence_and_pos(rankings, n)
    presence_f = presence.astype(np.float64)

    l_mat = presence_f @ presence_f.T  # common-support size per pair

    # Sign matrices and flattened dot products -> S_ij
    Sign = np.zeros((rows, n, n), dtype=np.float64)
    M = np.zeros((rows, n, n), dtype=np.float64)
    c = np.zeros((rows, n), dtype=np.float64)
    for i in range(rows):
        p_i = pos[i]
        pres_i = presence[i]
        diff = p_i[:, None] - p_i[None, :]
        both_present = pres_i[:, None] & pres_i[None, :]
        sign_i = np.sign(diff) * both_present
        np.fill_diagonal(sign_i, 0.0)
        Sign[i] = sign_i
        lt = (diff < 0) & both_present
        np.fill_diagonal(lt, False)
        M[i] = lt.astype(np.float64)
        c[i] = presence[i] * (pos[i] - 1)

    Sign_flat = Sign.reshape(rows, -1)
    S = Sign_flat @ Sign_flat.T  # rows x rows

    misorders_common = (l_mat * (l_mat - 1) - S) / 4.0

    Term1 = c @ presence_f.T  # Term1[i,j] = sum_b c_i[b]*presence_j[b]

    Term2 = np.zeros((rows, rows), dtype=np.float64)
    for i in range(rows):
        vec = presence_f @ M[i]          # rows x n : vec[j,b] = sum_a presence_j[a]*M_i[a,b]
        Term2[i, :] = (vec * presence_f).sum(axis=1)  # presence_j^T M_i presence_j

    cross1 = Term1 - Term2
    cross2 = cross1.T

    misorders = misorders_common + cross1 + cross2 + (k - l_mat) ** 2
    np.fill_diagonal(misorders, 0.0)

    kl = k - l_mat
    incomparables = kl * (kl - 1) + kl * (n - 2 * k + l_mat) * 2
    np.fill_diagonal(incomparables, 0.0)

    D = misorders + p * incomparables
    np.fill_diagonal(D, 0.0)
    return D
