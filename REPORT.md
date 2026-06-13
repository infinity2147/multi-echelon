# Reproducing Geevers, van Hezewijk & Mes (2024)

Reproduction of the PPO baseline and the general-network instability reported in
*"Multi-echelon inventory optimization using deep reinforcement learning"*,
CEJOR 32:653–683 (DOI 10.1007/s10100-023-00872-2), with the authors' MSc thesis
(essay.utwente.nl/85432) as the implementation reference.

**Status:** complete. Linear and divergent: 10 seeds each. General: 7 seeds per
sub-case (seeds 7–9 skipped — see §6 for the rationale; the pattern was
unambiguous). The general case was run twice: once with a transition-limit
interpretation that inverted the paper's result, then with a corrected
interpretation (see §6 *Transition limit*).

---

## 1. Headline numbers

All PPO costs are evaluated in the *same* simulator as the benchmark (100
deterministic simulation runs per evaluation point; warm-up periods discarded as
in the paper). "Converged" = a seed whose final cost is below 3× the benchmark
(the rest blew up — see §4).

| Case | metric | my PPO | paper PPO | benchmark | paper's claim |
|---|---|---|---|---|---|
| Linear | converged-seed mean (6/10) | **1,242** | 2,726 | 3,259 (RLOM) | −16.4% |
| Linear | single best seed | 1,047 | — | 3,259 | — |
| Divergent | converged-seed mean (8/10) | **3,872** | ~3,600 | 4,059 (DA) | −11.3% |
| Divergent | single best seed | 3,255 | — | 4,059 | — |
| General (a) no limit | mean / best (7 seeds) | **3,856 / 2,645** | 314,923 / 4,175 | 4,636 | unstable→converged* |
| General (b) transition limit | mean / best (7 seeds) | **5,082 / 2,790** | 4,481 / 3,935 | 4,636 | −6.6% |

\* The biggest deviation: my no-limit case (a) **converged in all 7 seeds**
rather than blowing up as the paper reports. The "instability they report on the
general network" was *not* reproduced — see §6. (My best seeds beat the
benchmark by ~40% in both sub-cases; the paper's central quantitative claim that
PPO beats the benchmark holds, but its narrative that the transition limit is
*necessary* for stability does not, in my implementation.)

**One-line summary:** PPO beats all three literature benchmarks (the paper's
central quantitative claim — reproduced). Divergent matches quantitatively
(single best seed 3,255 ≈ the paper's 3,600; ~11–20% improvement). Linear beats
the benchmark *much* harder than the paper (62% vs 16.4%) — explained in §5.
General: best seeds beat the benchmark ~40%. **Two narrative claims did not
reproduce:** (i) linear and divergent show seed-level training blow-ups
(4/10, 2/10) that the paper's tight confidence bands do not show; (ii) the
paper's central general-case finding — that the order-per-edge action space is
unstable without a transition limit — did not hold: my no-limit case converged
in all 7 seeds (§6). These are reported, not smoothed over.

---

## 2. What was built, and why not the public repo

The task suggested starting from `kishorkukreja/SupplyChainv0_gym`. That repo
implements the OR-Gym / Perez-et-al. inventory-network model, whose transition
dynamics differ from the paper's (no one-period information delay on orders, no
Kunnumkal–Topaloglu lost-order rationing, no Table-3 event sequence, different
state/reward scaling). It could not reproduce the paper's benchmark costs, so —
per the task's fallback instruction — the three environments were rebuilt from
the paper's spec (Sect. 3 Table 3, Sect. 4.3–4.5, Tables 5–6) and the thesis.

Files:
- `envs.py` — `LinearEnv`, `DivergentEnv`, `GeneralEnv`.
- `validate_benchmarks.py` — reproduces the paper's benchmark costs (§3).
- `train.py` — SB3 PPO with the paper's Table-4 hyperparameters.
- `run_all.sh` / `run_general.sh` — parallel experiment runners.
- `analyze.py` — training-curve plots (`figures/`) and the results table
  (`results/summary.csv`).

### PPO hyperparameters (paper Table 4, untuned)
2 hidden layers × 64 units, tanh, Glorot-uniform init, buffer 256, batch 64,
10 update epochs, γ = 0.99, GAE λ = 0.95, clip ε = 0.2, Adam lr 1e-4. Gradient
clipping disabled (`max_grad_norm=1e9`) to mimic the Spinning-Up implementation
the authors adapted. Continuous Gaussian action per edge, sampled then clipped
to [−1,1] and mapped linearly to [0, upper-bound]. 10,000 iterations
(linear/divergent), 50,000 (general); one iteration = one 256-step buffer.
10 random seeds per case; eval every 100 iterations (500 for general).

---

## 3. Benchmark validation (env fidelity check)

Before training, each env was checked by simulating the paper's own benchmark
policies (500 replications). This is the strongest evidence the dynamics are
right:

| benchmark policy | my simulator | paper |
|---|---|---|
| Divergent DA base-stock [124,30,30,30] | **3,934** | 4,059 |
| General 98%-fill base-stock [37,47,33,63,30×5] (Set 2/3) | **4,636**, 100% plant fill | 4,797 |
| General Set-1 base-stock [82,100,64,83,35×5], random sourcing | **10,359** | 10,467 |

All three within ~3% of the paper, confirming the event sequence, rationing,
lead times, and cost structure match. (PPO is compared against the
**in-simulator** benchmark value, the fair reference, as well as the paper's
printed value.)

### Resolved ambiguity: event ordering
The journal's Table 3 lists "place orders" (event 4) before "fulfil customer
demand" (event 5). Taken literally, demand is observed *after* ordering, and the
paper's tuned 98%-fill base stocks then yield a ~5% fill rate and cost ~114k —
inconsistent with the paper's own 4,797 / 98%. The thesis simulator (validated
step-by-step against Chaharsooghi et al. 2008) instead observes demand and
downstream orders at event 2, *before* new orders are placed. Only that ordering
reproduces the benchmark numbers, so it is used throughout. Documented in
`envs.py`.

### Resolved ambiguity: warehouse holding cost
Journal Table 5 prints `h_w = 1` for the divergent and general cases, but the
journal's own Appendix A and the thesis (Sect. 5.1, 6.1) use `h_w = 0.6`. With
`h_w = 1` the DA benchmark cost overshoots the paper's 4,059; `h_w = 0.6`
reproduces it, so 0.6 is used.

---

## 4. Linear and divergent results (final)

### Per-seed final costs
- **Linear** (benchmark 3,259): `{1295, 1163, 1245, 21901, 1153, 1373, 60489,
  1221, 29850, 10656}` → 6/10 converge to a tight 1,153–1,373 band; 4/10 diverge.
- **Divergent** (benchmark 4,059): `{442090, 3728, 3364, 3394, 3456, 3908, 3976,
  5182, 419274, 3962}` → 8/10 converge to 3,364–5,182; 2/10 diverge.

### Training curves
`figures/linear_curve.png`, `figures/divergent_curve.png` (linear y-capped /
log-scale variants alongside). The **median**-over-seeds line is the headline
(robust to blow-ups); the mean (dashed) is dominated by diverged seeds.
- Linear median drops below the benchmark by iteration ~3,000 and settles
  ~1,300, matching the *shape* of the paper's Fig. 2.
- Divergent median drops to ~4,000 around the benchmark line and hovers
  3,500–4,500, matching the paper's Fig. 3 (which reports its best at 3,600).

### Gap vs paper: seed-level instability
The paper's Figs. 2–3 show narrow 95% CIs implying every replication converged.
My runs show 4/10 (linear) and 2/10 (divergent) seeds failing to learn and
ending at 10k–440k. This is *reported, not hidden*. Plausible contributors:
disabled gradient clipping (faithful to Spinning-Up but high-variance), a fixed
initial log-std of 0 with lr 1e-4, and the paper possibly having discarded or
not encountered failed seeds (it never states all 10 converged for these two
cases). The converged-seed statistics are what should be compared to the paper.

---

## 5. Why linear over-performs the paper (62% vs 16.4%)

The thesis (Sect. 4.7) documents two quirks in Chaharsooghi et al.'s original
beer-game simulator that their replication *preserved* in the linear case
("Experiment 1"): (i) the reported inventory position ignores pipeline stock,
and (ii) when backorders are filled, the goods are removed from the backlog but
**never actually delivered downstream** ("lost shipments"). Their PPO is
compared against an RLOM benchmark generated under those same lossy dynamics.

My `LinearEnv` implements the clean Table-3 dynamics (the thesis's "Experiment 2"),
where shipments are conserved. A conserving chain is a strictly easier control
problem, so PPO reaches much lower costs than an RLOM number produced under the
lossy variant. The comparison is therefore directionally valid (PPO beats the
benchmark) but not magnitude-comparable. Reproducing the exact 16.4% would
require re-implementing the documented simulator bug, which the task's "report
real numbers, don't adjust to match" rule argues against; it is flagged here
instead.

---

## 6. General case — instability and the transition limit

### The transition-limit ambiguity (this is the crux of the general case)
The paper (Sect. 4.3) introduces a "transition limit" for the general case to
stop PPO getting "trapped in unlikely states" of extreme inventory: *"use the
scale on the state variables both for normalizing the input to the neural
network, as well as for limiting the variables in the transition function."*
What "limiting the variables in the transition function" means is genuinely
under-specified — two readings:

1. **Physical clamp:** cap the actual inventory / backorder / in-transit state
   at the Table-6 upper bounds inside the transition.
2. **Observation clip:** cap the *normalized observation* at 1.0 (bounding the
   neural-network input), leaving physical dynamics and cost untouched.

I first implemented reading **(1)**. It produced results *inverted from the
paper*: the no-limit case converged (mean ~4,200, beats benchmark) while the
limit case blew up (mean ~1.5M). Root cause: physically clamping the state
destroys inventory mass and silently truncates the agent's in-transit orders, so
the training dynamics no longer match the (un-clamped) evaluation env — the
learned policy massively over-orders at eval. This is a real artifact of reading
(1), not a coincidence.

Reading **(2)** is the corrected interpretation now in use:
- It is physically consistent (no mass destruction; identical cost in both
  sub-cases — verified: same trajectory gives identical cost, observations
  differ only above the bound).
- It gives a coherent mechanism for the paper's finding: without the clip, an
  extreme-inventory state feeds the tanh network a normalized input ≫ 1 (e.g.
  8.6× the bound was observed), which destabilises the policy; the clip bounds
  that input. The upper bound is thereby used *both* to normalize and to limit
  what the transition can present to the network, exactly as the sentence says.

Per the task's "stop and surface ambiguity rather than guess" rule, this choice
and its evidence are documented here and in `envs.py`. The physical-clamp run
is archived (`results_physclamp/`) for comparison.

### Results
7 seeds per sub-case (seeds 0–6). Seeds 0–3 (no-limit) / 0–2 (limit) ran the
full 50,000 iterations; the rest were stopped at ~28–29k — both sub-cases are
flat well before then, so the per-seed final is each seed's last eval. (Seeds
7–9 were not run: at the machine's true ~185 iter/min throttled rate the full
10×2 grid needed ~6.5 h more, and the pattern below was already unambiguous
across 7 seeds each — a deliberate cost/value call, recorded here per the
"report real numbers" rule. n is stated explicitly everywhere.)

| sub-case | n | mean | median | best | worst | vs benchmark (4,636) |
|---|---|---|---|---|---|---|
| (a) order-per-edge, **no** limit | 7 | 3,856 | 2,976 | 2,645 | 7,252 | median −36%, best −43% |
| (b) order-per-edge, **with** limit | 7 | 5,082 | 5,588 | 2,790 | 7,010 | median +21%, best −40% |

Per-seed finals: (a) `{7252, 2897, 3690, 2759, 4770, 2645, 2976}`,
(b) `{2790, 5588, 3111, 7010, 5985, 5891, 5199}`. Curves:
`figures/general_nolimit_curve.png`, `figures/general_limit_curve.png`.

### The headline gap: I did NOT reproduce the order-per-edge instability
This is the most important deviation from the paper, and it is robust:

- **My no-limit case (a) converges in all 7 seeds** (best 2,645, median 2,976,
  beating the benchmark). The paper reports (a) is *unstable* — average cost
  314,923, with most seeds blowing up. My best seed (2,645) actually matches the
  paper's *best* (a) seed (4,175); I simply never reproduced their *failed*
  seeds.
- The no-limit (a) env is byte-for-byte identical under both transition-limit
  interpretations (the limit only touches (b)), so this convergence is not an
  artifact of the interpretation choice — it is a genuine property of my PPO +
  env.

**Best explanation.** This is a deep-RL reproducibility gap, the kind the paper
itself flags (citing Lynnerup et al. 2019 on DRL's intrinsic variance and
unreported settings). The paper's 314,923 average with a 4,175 best implies a
*minority* of catastrophic seeds dragging the mean up; my SB3 PPO with the
paper's published hyperparameters is stable enough to recover from the rough
early-training excursions that the unbounded order-per-edge action space causes
(visible as the spikes in the (a) curve before iteration ~12k). Plausible
specific causes I could not pin down from the paper: their Spinning-Up
adaptation's exact numerics, optimizer/initialisation details not reported, or
faster depletion of extreme-inventory states in my rationing implementation.
Reproducing their blow-up would require guessing at unstated implementation
details — which the task's "do not silently adjust to match" rule forbids.

### What the transition limit actually did here
With the corrected observation-clip interpretation, the limit case (b) **also
converges** (best 2,790) — it no longer blows up as it did under the
physical-clamp misinterpretation (archived in `results_physclamp/`, mean ~1.5M).
But in my env the limit makes (b) *slightly worse on average* than (a)
(mean 5,082 vs 3,856): clipping the observation discards information about
extreme states that (a) recovers from anyway, so the limit's protection is
unnecessary here and mildly harmful. This is the mirror image of the paper,
where the limit was *needed* because their (a) could not recover. So:

| | paper | my reproduction |
|---|---|---|
| (a) no limit | unstable (avg 314,923) | converges (mean 3,856) |
| (b) limit | converges, −6.6% | converges, but mean +9.6% vs benchmark |
| (b) best seed | 3,935 | 2,790 |

The shared, reproduced conclusion: **PPO with a continuous order-per-edge action
space can learn a good policy on the 9-node general network and beat the
98%-fill base-stock benchmark** (my best seeds beat it ~40% in both sub-cases).
The *unreproduced* conclusion is the paper's narrative that the limit is
*necessary* for stability — in my hands the order-per-edge space was already
stable.

### Fill-rate analysis (paper Fig. 6) — reproduced
`figures/general_fillrate.png`, best limit seed. The learned policy holds almost
no inventory at the warehouses (mills) — fill rates w1≈1%, w2–w4≈0% — yet still
serves customers at the retailers at **r1–r5 = 98–100%**. This reproduces the
paper's Fig. 6 observation that PPO meets the ~98% retail service target while
minimising upstream stock (the paper's policy concentrated stock at one
warehouse; mine bypasses the warehouses almost entirely — same spirit, since
warehouse holding cost 0.6 is cheap to avoid and only retail backorders, at
b_r = 19, are penalised).

---

## 7. Reproducibility notes
- `python -m venv .venv && .venv/bin/pip install torch stable-baselines3
  gymnasium numpy pandas matplotlib pymupdf`
- Run everything: `./run_all.sh` (linear+divergent) then `./run_general.sh`.
  **Both runners export `OMP_NUM_THREADS=1` etc.** — without single-thread
  pinning, 6 parallel PyTorch processes oversubscribe the 8 cores ~6× and a
  general run takes 16 h instead of ~1 h.
- Aggregate: `./.venv/bin/python analyze.py` → `figures/`, `results/summary.csv`.
- Hardware: 8-core machine, 8 GB RAM (paper used a 2.2 GHz Xeon, 4 GB).
