"""Reproduce the paper's Fig. 6: per-stockpoint fill rates of a trained general
policy, next to the benchmark base-stock policy.

Warehouse (mill) fill rate = shipped / requested across its edges.
Retailer (plant) fill rate  = customer demand served / customer demand.
Averaged over post-warm-up periods, 100 simulation runs, for the best
general_limit seed (lowest final eval cost).
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs import GeneralEnv

N_RUNS = 100


def best_seed(case="general_limit"):
    """Lowest final-cost seed that has a saved model (finished a full run)."""
    best, bestc = None, 1e18
    for f in sorted(glob.glob(f"results/{case}/seed*.csv")):
        seed = int(f.split("seed")[-1].split(".")[0])
        if not os.path.exists(f"results/{case}/seed{seed}_model.zip"):
            continue
        rows = [l.strip().split(",") for l in open(f)][1:]
        if not rows:
            continue
        c = float(rows[-1][2])
        if c < bestc:
            bestc, best = c, seed
    return best, bestc


def fill_rates(predict, transition_limit, seed0):
    mill_req = np.zeros(4); mill_ship = np.zeros(4)
    plant_dem = np.zeros(5); plant_srv = np.zeros(5)
    for i in range(N_RUNS):
        env = GeneralEnv(episode_len=10 ** 9, transition_limit=transition_limit)
        obs, _ = env.reset(seed=seed0 + i)
        for t in range(env.EVAL_HORIZON - 1):
            obs, _, _, _, info = env.step(predict(obs))
            if t >= env.WARMUP:
                mill_req += info["mill_req"]; mill_ship += info["mill_ship"]
                plant_dem += info["demand_p"]; plant_srv += info["served_p"]
    wh = np.divide(mill_ship, mill_req, out=np.ones(4), where=mill_req > 0)
    rt = np.divide(plant_srv, plant_dem, out=np.ones(5), where=plant_dem > 0)
    return wh * 100, rt * 100


def main():
    seed, cost = best_seed("general_limit")
    print(f"best general_limit seed = {seed} (final eval cost {cost:.0f})")
    model = PPO.load(f"results/general_limit/seed{seed}_model")
    ppo_wh, ppo_rt = fill_rates(
        lambda o: model.predict(o, deterministic=True)[0], True, 700_000)

    bench_wh, bench_rt = fill_rates(
        lambda o: None, False, 700_000) if False else (None, None)
    # benchmark via raw base-stock action
    mill_req = np.zeros(4); mill_ship = np.zeros(4)
    plant_dem = np.zeros(5); plant_srv = np.zeros(5)
    for i in range(N_RUNS):
        env = GeneralEnv(episode_len=10 ** 9)
        env.reset(seed=700_000 + i)
        for t in range(env.EVAL_HORIZON - 1):
            _, _, _, _, info = env.step_raw(env.benchmark_action_raw())
            if t >= env.WARMUP:
                mill_req += info["mill_req"]; mill_ship += info["mill_ship"]
                plant_dem += info["demand_p"]; plant_srv += info["served_p"]
    bench_wh = np.divide(mill_ship, mill_req, out=np.ones(4), where=mill_req > 0) * 100
    bench_rt = np.divide(plant_srv, plant_dem, out=np.ones(5), where=plant_dem > 0) * 100

    labels = [f"w{i+1}" for i in range(4)] + [f"r{i+1}" for i in range(5)]
    ppo = np.concatenate((ppo_wh, ppo_rt))
    bench = np.concatenate((bench_wh, bench_rt))
    x = np.arange(9)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, bench, 0.4, label="Benchmark", color="C1")
    ax.bar(x + 0.2, ppo, 0.4, label=f"PPO (limit, seed {seed})", color="C0")
    for xi, v in zip(x - 0.2, bench):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    for xi, v in zip(x + 0.2, ppo):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.axhline(98, color="0.5", ls=":", lw=1, label="98% target")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Fill rate (%)"); ax.set_ylim(0, 110)
    ax.set_title("Average fill rate per stock point (CardBoard / general case)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/general_fillrate.png", dpi=130)
    print("PPO   fill:", dict(zip(labels, ppo.round(0))))
    print("Bench fill:", dict(zip(labels, bench.round(0))))
    print("wrote figures/general_fillrate.png")


if __name__ == "__main__":
    main()
