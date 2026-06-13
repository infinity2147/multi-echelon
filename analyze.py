"""Aggregate per-seed CSV logs into training-curve plots and a results table.

Produces, under figures/ and results/:
  - <case>_curve.png : mean cost across seeds vs iteration, 95% CI band,
                       benchmark line (paper Fig. 2/3/5 style).
  - summary.csv / summary table printed to stdout: per-case mean/best/worst
    final cost across seeds vs benchmark, plus converged-only stats.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, results-subdir, benchmark-cost, paper-PPO-cost, eval-every-iters)
CASES = [
    ("Linear", "linear", 3259, 2726, 100),
    ("Divergent", "divergent", 4059, 3600, 100),
    ("General (a) no limit", "general_nolimit", 4636, 314923, 500),
    ("General (b) transition limit", "general_limit", 4636, 4481, 500),
]
# benchmark costs measured in *our* simulator (fairer than the paper's printed
# value, since PPO is evaluated in the same env). Used for the % comparison.
BENCH_SIM = {"linear": None, "divergent": 3934, "general_nolimit": 4636,
             "general_limit": 4636}


MIN_EVALS = 50  # ignore barely-started seeds (need >=25k iters to count)


def load_series(case):
    """Return {seed: (iterations, costs)} for seeds with >= MIN_EVALS points."""
    series = {}
    for f in sorted(glob.glob(f"results/{case}/seed*.csv")):
        rows = [l.strip().split(",") for l in open(f)][1:]
        if len(rows) < MIN_EVALS:
            continue
        seed = int(f.split("seed")[-1].split(".")[0])
        series[seed] = (np.array([int(r[0]) for r in rows]),
                        np.array([float(r[2]) for r in rows]))
    return series


def load(case):
    """(iterations, matrix[seed, eval]) truncated to the common-length prefix."""
    series = load_series(case)
    if not series:
        return None, None, None
    n = min(len(v[1]) for v in series.values())
    its = next(iter(series.values()))[0][:n]
    mat = np.vstack([v[1][:n] for v in series.values()])
    return its, mat, sorted(series)


def plot_case(label, case, bench, ylim_cap=None):
    its, mat, seeds = load(case)
    if its is None:
        return
    mean, std = mat.mean(0), mat.std(0)
    ci = 1.96 * std / np.sqrt(mat.shape[0])
    median = np.median(mat, 0)

    # Linear plot, y capped so the benchmark + converged region are legible.
    # Mean is shown but dominated by any blown-up seeds, so the median (the
    # typical run) is the headline line; individual seeds are faint.
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(mat.shape[0]):
        ax.plot(its, mat[i], color="0.75", lw=0.6, alpha=0.7,
                label="individual seeds" if i == 0 else None)
    ax.plot(its, median, color="C0", lw=2.0, label="PPO (median over seeds)")
    ax.plot(its, mean, color="C2", lw=1.2, ls="--", label="PPO (mean over seeds)")
    ax.fill_between(its, mean - ci, mean + ci, color="C2", alpha=0.15)
    ax.axhline(bench, color="C1", lw=1.5, label=f"Benchmark ({bench})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title(f"{label}  (n={mat.shape[0]} seeds)")
    ax.set_ylim(0, ylim_cap or max(bench * 4, np.percentile(mat[:, -1], 60) * 1.5))
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{case}_curve.png", dpi=130)
    plt.close(fig)

    # Log-scale version showing the full range including blow-ups.
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(mat.shape[0]):
        ax.plot(its, np.maximum(mat[i], 1), color="0.75", lw=0.6, alpha=0.7,
                label="individual seeds" if i == 0 else None)
    ax.plot(its, median, color="C0", lw=2.0, label="PPO (median)")
    ax.plot(its, mean, color="C2", lw=1.2, ls="--", label="PPO (mean)")
    ax.axhline(bench, color="C1", lw=1.5, label=f"Benchmark ({bench})")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost (log)")
    ax.set_title(f"{label}  (n={mat.shape[0]} seeds, log scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/{case}_curve_log.png", dpi=130)
    plt.close(fig)


def main():
    print(f"{'case':30s} {'n':>2s} {'mean':>10s} {'best':>9s} {'worst':>11s} "
          f"{'conv':>5s} {'conv_mean':>9s} {'bench':>7s}")
    rows = []
    for label, case, bench, paper, _ in CASES:
        series = load_series(case)
        if not series:
            print(f"{label:30s}  (no data yet)")
            continue
        # per-seed FINAL cost = each seed's last available eval (seeds stopped
        # early at ~29k are already converged; finished seeds report 50k).
        final = np.array([series[s][1][-1] for s in sorted(series)])
        conv = final[final < bench * 3]
        cmean = conv.mean() if len(conv) else float("nan")
        print(f"{label:30s} {len(final):2d} {final.mean():10.0f} {final.min():9.0f} "
              f"{final.max():11.0f} {len(conv):2d}/{len(final):<2d} {cmean:9.0f} {bench:7d}")
        rows.append((label, case, len(final), final.mean(), final.min(),
                     final.max(), len(conv), cmean, bench, paper))
        cap = bench * 4 if case.startswith("general") else None
        plot_case(label, case, bench, ylim_cap=cap)

    with open("results/summary.csv", "w") as f:
        f.write("case,key,n_seeds,mean_final,best_final,worst_final,"
                "n_converged,converged_mean,benchmark,paper_ppo\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("\nwrote results/summary.csv and figures/*.png")


if __name__ == "__main__":
    main()
