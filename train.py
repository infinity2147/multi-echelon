"""PPO training for the Geevers et al. (2024) reproduction.

Hyperparameters from the paper's Table 4 (Schulman et al. 2017 defaults):
2x64 tanh, Glorot-uniform init, buffer 256, batch 64, 10 update epochs,
gamma 0.99, GAE lambda 0.95, clip 0.2, Adam lr 1e-4.
10,000 iterations (linear, divergent), 50,000 (general); 1 iteration = one
256-step buffer. Evaluation every 100 iterations (500 for general):
100 deterministic simulation runs, mean total cost over the case's horizon
(warm-up discarded). Transition-limit envs are evaluated WITHOUT the limit,
as prescribed in Sect. 4.3.

Usage: train.py --case {linear,divergent,general_nolimit,general_limit} --seed N
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs import make_env

ITERS = {"linear": 10_000, "divergent": 10_000,
         "general_nolimit": 50_000, "general_limit": 50_000}
EVAL_EVERY_ITERS = {"linear": 100, "divergent": 100,
                    "general_nolimit": 500, "general_limit": 500}
N_EVAL_RUNS = 100
N_STEPS = 256


def eval_case(case):
    """Eval uses the same env as training. The transition limit only clips the
    NN's observation (no physical/cost difference), so the limit-trained policy
    must be evaluated with the same clipped observation it learned on."""
    return case


def evaluate(model, case, seed):
    """Mean/std of total post-warm-up cost over N_EVAL_RUNS simulations."""
    envs = [make_env(eval_case(case), episode_len=10 ** 9) for _ in range(N_EVAL_RUNS)]
    horizon = envs[0].EVAL_HORIZON
    warmup = getattr(envs[0], "WARMUP", 0)
    costs = np.zeros((N_EVAL_RUNS, horizon))
    obs_list, n_steps = [], horizon
    for i, env in enumerate(envs):
        ob, info = env.reset(seed=seed + i)
        obs_list.append(ob)
        if "cost" in info:  # linear/general: period-0 cost precedes 1st action
            costs[i, 0] = info["cost"]
    if "cost" in info:
        n_steps = horizon - 1
        col0 = 1
    else:
        col0 = 0
    obs = np.array(obs_list)
    for t in range(n_steps):
        actions, _ = model.predict(obs, deterministic=True)
        for i, env in enumerate(envs):
            ob, _, _, _, info = env.step(actions[i])
            obs[i] = ob
            costs[i, col0 + t] = info["cost"]
    totals = costs[:, warmup:].sum(axis=1)
    return totals.mean(), totals.std()


class EvalCallback(BaseCallback):
    def __init__(self, case, seed, csv_path):
        super().__init__()
        self.case, self.seed, self.csv_path = case, seed, csv_path
        self.every_steps = EVAL_EVERY_ITERS[case] * N_STEPS
        self.eval_idx = 0
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["iteration", "timesteps", "mean_cost", "std_cost"])

    def _on_step(self):
        if self.num_timesteps % self.every_steps == 0:
            self.eval_idx += 1
            mean, std = evaluate(self.model, self.case,
                                 seed=1_000_000 * (self.seed + 1) + 1000 * self.eval_idx)
            it = self.num_timesteps // N_STEPS
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([it, self.num_timesteps,
                                        f"{mean:.2f}", f"{std:.2f}"])
        return True


def glorot_init(model):
    """Paper: Glorot-uniform weight init (SB3 default differs)."""
    for module in (model.policy.mlp_extractor, model.policy.action_net,
                   model.policy.value_net):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(ITERS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--iters", type=int, default=None, help="override #iterations")
    args = ap.parse_args()

    torch.set_num_threads(1)
    outdir = os.path.join(args.outdir, args.case)
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"seed{args.seed}.csv")

    env = make_env(args.case)  # 256-step training episodes
    env.reset(seed=10_000 + args.seed)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=1e-4, n_steps=N_STEPS, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        vf_coef=0.5, max_grad_norm=1e9,  # Spinning Up has no grad clipping
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                           activation_fn=nn.Tanh, ortho_init=False,
                           log_std_init=0.0),
        seed=args.seed, device="cpu", verbose=0,
    )
    glorot_init(model)

    iters = args.iters or ITERS[args.case]
    t0 = time.time()
    model.learn(total_timesteps=iters * N_STEPS,
                callback=EvalCallback(args.case, args.seed, csv_path))
    model.save(os.path.join(outdir, f"seed{args.seed}_model"))
    print(f"done {args.case} seed {args.seed} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
