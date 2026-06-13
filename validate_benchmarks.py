"""Validate env implementations by reproducing the paper's benchmark costs.

Targets:
  divergent: DA base-stock [124,30,30,30]  -> cost over periods 26-75 ~ 4,059
  general  : base-stock [37,47,33,63,30x5] -> cost over periods 51-100 ~ 4,797
             with plant fill rates ~ 98%
"""
import numpy as np
from envs import DivergentEnv, GeneralEnv

N_RUNS = 500


def run_divergent(seed0=10_000):
    totals = []
    for i in range(N_RUNS):
        env = DivergentEnv(episode_len=10**9)
        env.reset(seed=seed0 + i)
        costs = []
        for _ in range(env.EVAL_HORIZON):
            _, _, _, _, info = env.step_raw(env.benchmark_action_raw())
            costs.append(info["cost"])
        totals.append(sum(costs[env.WARMUP:]))
    return np.array(totals)


def run_general(seed0=20_000):
    totals, fills = [], []
    for i in range(N_RUNS):
        env = GeneralEnv(episode_len=10**9, transition_limit=False)
        _, info = env.reset(seed=seed0 + i)
        costs, served, demand = [info["cost"]], [0.0], [1e-9]
        for _ in range(env.EVAL_HORIZON - 1):
            _, _, _, _, info = env.step_raw(env.benchmark_action_raw())
            costs.append(info["cost"])
            served.append(info["served"])
            demand.append(info["demand"])
        totals.append(sum(costs[env.WARMUP:]))
        fills.append(sum(served[env.WARMUP:]) / sum(demand[env.WARMUP:]))
    return np.array(totals), np.array(fills)


if __name__ == "__main__":
    d = run_divergent()
    print(f"DIVERGENT benchmark : mean {d.mean():8.1f}  (paper: 4059)  "
          f"ci95 +/-{1.96 * d.std() / np.sqrt(len(d)):.1f}")
    g, f = run_general()
    print(f"GENERAL   benchmark : mean {g.mean():8.1f}  (paper: 4797)  "
          f"ci95 +/-{1.96 * g.std() / np.sqrt(len(g)):.1f}   "
          f"plant fill {f.mean() * 100:.2f}% (~98%)")
