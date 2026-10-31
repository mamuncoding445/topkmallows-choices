"""
Diagnostic: does single TopKGMM's near-rho-invariant error (0.1210 at
rho=0 vs 0.1283 at rho=1, full Experiment 1 results) reflect a fixed
inability to flatten toward uniform (a beta-grid-floor artifact, since
our shrunk BETA_GRID floors at 0.3 while the paper's sushi-optimal beta
was 0.05), a genuine structural mismatch (single center can't represent
block-dependent reversals), or both?

Method: refit TopKGMM's center ONCE per rho (same train data/seed as the
real Experiment 1 run, rho in {0.0, 0.5, 1.0}, seed=0), then sweep beta
over a MUCH wider/lower range than the main run's BETA_GRID=[0.3,0.6,1.0],
holding p fixed at 0.5 (our own selected value, and also the paper's
sushi-optimal p), evaluating full 456-assortment MAE at every beta.

Prediction under the "two distinct mechanisms" hypothesis: a lower beta
should substantially help at rho=0 (true distribution is close to
uniform, lower beta = more indifference = reachable) but do little or
nothing at rho=1 (the problem there is the WRONG SHAPE of the
prediction -- a single order cannot represent composition-dependent
reversals -- not insufficient flattening).
"""
import csv
import random
import time

from block_generator import sample_topk, exact_choice_probs, N as SYN_N
from metrics import enumerate_assortments, mae, uniform_baseline
from topkgmm_fit import fit_center, choice_probs_topkgmm

BETA_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.6, 1.0]
P_FIXED = 0.5
RHOS = [0.0, 0.5, 1.0]
N_TRAIN = 8000
SEED = 0


def run():
    # Written with the assistance of Claude (Anthropic), a generative AI tool.
    full_assortments = enumerate_assortments(SYN_N, sizes=range(2, 7))
    rows = []
    for rho in RHOS:
        t0 = time.time()
        rng = random.Random(1000 * SEED + int(rho * 100))
        train = [sample_topk(rho, rng) for _ in range(N_TRAIN)]
        center, n_avail, n_used = fit_center(SYN_N, train, max_rows=2000, seed=SEED)

        def true_fn(A):
            # Written with the assistance of Claude (Anthropic), a generative AI tool.
            return exact_choice_probs(A, rho)

        uniform_mae = sum(mae(uniform_baseline(A + [0]), true_fn(A), A + [0])
                           for A in full_assortments) / len(full_assortments)

        for beta in BETA_SWEEP:
            vals = []
            for A in full_assortments:
                Anull = A + [0]
                pred = choice_probs_topkgmm(A, center, SYN_N, P_FIXED, beta)
                true = true_fn(A)
                vals.append(mae(pred, true, Anull))
            m = sum(vals) / len(vals)
            rows.append({"rho": rho, "beta": beta, "p": P_FIXED,
                         "test_mae_full": m, "uniform_mae": uniform_mae,
                         "center": str(center)})
            print(f"rho={rho} beta={beta}: MAE={m:.4f} (uniform={uniform_mae:.4f})")
        print(f"rho={rho} done in {time.time()-t0:.1f}s, center={center}")

    with open("outputs/beta_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rho", "beta", "p", "test_mae_full",
                                                "uniform_mae", "center"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("wrote outputs/beta_sweep.csv")


if __name__ == "__main__":
    run()
