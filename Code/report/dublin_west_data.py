"""
BLT-format parser for the Dublin West 2002 STV election ballots.

File format (verified against the downloaded file):
  line 1: title
  line 2: header "<seats> <candidates> <num_ballot_lines> <total_votes>"
          e.g. "3 9 10335 29988" -- 3 seats, 9 candidates, 10335 distinct
          weighted ballot lines, 29988 total weighted votes.
  next `candidates` lines: "Name,Party,ElectedFlag"
  then weighted ballot lines: "weight c1 c2 ... cm 0" (0-terminated ranking)
  ending in a bare "0".

Cross-checks (both must hold exactly, matching the header):
  sum of ballot weights == total_votes (29988)
  number of distinct ballot lines == num_ballot_lines (10335)
"""
import random


def parse_blt(path):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    title = lines[0]
    header = lines[1].split()
    num_seats, num_candidates, num_ballot_lines, total_votes = map(int, header)

    candidates = []
    idx = 2
    for i in range(num_candidates):
        parts = lines[idx].split(",")
        candidates.append({"name": parts[0], "party": parts[1], "elected": int(parts[2])})
        idx += 1

    ballots = []  # list of (weight, ranking) where ranking is list of candidate ids (1-indexed)
    while True:
        parts = lines[idx].split()
        idx += 1
        if parts == ["0"]:
            break
        weight = int(parts[0])
        ranking_with_zero = list(map(int, parts[1:]))
        assert ranking_with_zero[-1] == 0
        ranking = ranking_with_zero[:-1]
        ballots.append((weight, ranking))

    sum_weights = sum(w for w, _ in ballots)
    n_lines = len(ballots)
    assert sum_weights == total_votes, f"weight sum {sum_weights} != {total_votes}"
    assert n_lines == num_ballot_lines, f"line count {n_lines} != {num_ballot_lines}"

    return {
        "title": title,
        "num_seats": num_seats,
        "num_candidates": num_candidates,
        "num_ballot_lines": num_ballot_lines,
        "total_votes": total_votes,
        "candidates": candidates,
        "ballots": ballots,  # (weight, ranking) pairs
    }


def expand_ballots(ballots):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Expand weighted ballot lines into individual-voter rankings (one
    entry per voter, multiplicities expanded), for voter-level splitting.
    """
    expanded = []
    for weight, ranking in ballots:
        expanded.extend([ranking] * weight)
    return expanded


def choose_common_k(expanded_train_rankings, min_frac=0.5):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Largest k such that >= min_frac of TRAINING ballots rank >= k
    candidates (decided from training data only, per report instructions).
    """
    lengths = [len(r) for r in expanded_train_rankings]
    n_train = len(lengths)
    lengths_sorted = sorted(lengths, reverse=True)
    best_k = 1
    for k in range(1, max(lengths) + 1):
        frac_at_least_k = sum(1 for l in lengths if l >= k) / n_train
        if frac_at_least_k >= min_frac:
            best_k = k
    return best_k


def truncate_to_k(ranking, k):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    """Full ballot ranking truncated to its own top-k prefix (if it has
    fewer than k ranked candidates, keep as-is -- these are filtered out
    by the caller when retaining only ballots with length >= k).
    """
    return ranking[:k]


def split_train_val_test(rankings, seed, train_frac=0.6, val_frac=0.2):
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    rng = random.Random(seed)
    shuffled = list(rankings)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


if __name__ == "__main__":
    data = parse_blt("data/dublin_west/Dublin_West.dat")
    print("parsed:", data["title"], data["num_seats"], data["num_candidates"],
          data["num_ballot_lines"], data["total_votes"])
    expanded = expand_ballots(data["ballots"])
    print("total expanded voters:", len(expanded))
    from collections import Counter
    length_counts = Counter(len(r) for r in expanded)
    print("ballot length distribution:", dict(sorted(length_counts.items())))
