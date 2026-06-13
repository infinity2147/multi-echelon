# Multi-echelon inventory optimization with PPO — a reproduction

A reproduction of the PPO baseline from **Geevers, van Hezewijk & Mes (2024),
"Multi-echelon inventory optimization using deep reinforcement learning,"**
*Central European Journal of Operations Research* 32:653–683
([DOI 10.1007/s10100-023-00872-2](https://doi.org/10.1007/s10100-023-00872-2),
open access). Implementation details follow the authors' MSc thesis
([essay.utwente.nl/85432](http://essay.utwente.nl/85432/1/Geevers_MA_BMS.pdf)).

The goal was to reproduce their PPO results on three network structures (linear,
divergent, general) and, in particular, the training **instability** they report
for the general "CardBoard Company" network.

**Read [`REPORT.md`](REPORT.md) for the full write-up:** results vs the paper,
every assumption, and an honest account of what did and did not reproduce.

## TL;DR results

| Case | my PPO (converged) | benchmark | paper PPO | outcome |
|---|---|---|---|---|
| Linear | mean 1,242 / best 1,047 | 3,259 | 2,726 | beats benchmark (by more than the paper — see §5) |
| Divergent | mean 3,872 / best 3,255 | 4,059 | ~3,600 | quantitative match |
| General (a) no limit | mean 3,856 / best 2,645 | 4,636 | 314,923 / 4,175 | beats benchmark; **did not blow up** |
| General (b) transition limit | mean 5,082 / best 2,790 | 4,636 | 4,481 / 3,935 | converges |

Two narrative claims did **not** reproduce and are documented rather than
smoothed over: (i) the paper's central general-case finding that the
order-per-edge action space is unstable without a transition limit — my no-limit
case converged in all seeds; (ii) linear/divergent show occasional seed-level
blow-ups the paper's tight confidence bands do not.

## Layout

| file | purpose |
|---|---|
| `envs.py` | the three Gymnasium envs, built from the paper's spec (not the OR-Gym repo) |
| `validate_benchmarks.py` | reproduces the paper's benchmark policy costs (env fidelity check) |
| `train.py` | SB3 PPO with the paper's Table-4 hyperparameters |
| `analyze.py` | training-curve plots + results table |
| `eval_fillrate.py` | per-stock-point fill-rate figure (reproduces paper Fig. 6) |
| `run_all.sh`, `run_general.sh` | parallel experiment runners (thread-pinned) |
| `figures/`, `results/` | generated plots and per-seed eval logs |

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh          # linear + divergent (10 seeds each)
./run_general.sh      # general case (10 seeds x 2 sub-cases)
.venv/bin/python analyze.py
.venv/bin/python eval_fillrate.py
```

> **Note:** the runners export `OMP_NUM_THREADS=1` etc. Without single-thread
> pinning, parallel PyTorch processes oversubscribe the cores and a general run
> takes ~16 h instead of ~1 h.

Source PDFs (paper, thesis) and the third-party `SupplyChainv0_gym` clone are
git-ignored; trained-model `.zip`s are regenerable via `train.py`.
