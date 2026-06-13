"""
Environments reproducing Geevers, van Hezewijk & Mes (2024),
"Multi-echelon inventory optimization using deep reinforcement learning",
CEJOR 32:653-683 (DOI 10.1007/s10100-023-00872-2).

Built from the paper's spec (Sect. 3, Sect. 4.3-4.5, Tables 5/6) plus
implementation details from Geevers' MSc thesis (essay.utwente.nl/85432).
The public SupplyChainv0_gym repo follows the OR-Gym/Perez model (different
dynamics), so the envs are built from the paper instead.

Event order note: the journal's Table 3 lists "place orders" (event 4) before
"fulfil customer demand" (event 5), but the thesis simulator the paper is built
on (validated step-by-step against Chaharsooghi et al. 2008) observes demand
and downstream orders at event 2, BEFORE new orders are placed at event 4.
Only the thesis order is consistent with the paper's own benchmarks: with
demand-after-ordering the general-case base stocks [37,47,33,63,30x5] yield a
~5% fill rate and costs ~114k (vs the paper's 98% / 4,797). We therefore use
the thesis event order: the decision epoch is event 4, observations are the
post-demand (pre-ordering) state.

Cases:
  LinearEnv    - Chaharsooghi beer game. One stochastic lead time U{0,4} per
                 period shared by all shipments, demand U{0,15}, h=1, b=2,
                 init: I=12, pipeline 4@t0+4@t1, initial outstanding orders 4.
  DivergentEnv - Kunnumkal & Topaloglu. Orders placed simultaneously (event 1),
                 warehouse ships same period (ascending-IP rationing,
                 unfulfilled orders LOST), arrivals next period, demand last.
                 h_w=0.6 (journal Table 5 says 1, but the journal's Appendix A
                 and the thesis use 0.6), h_r=1, b_r=19, d~Pois(U[5,15]).
  GeneralEnv   - CardBoard Company, 4 mills + 5 plants, adjacency from Fig. 1c
                 vector data: w1-{r1,r2,r3}, w2-{r1,r2,r3,r4}, w3-{r4,r5},
                 w4-{r1..r5}. d~Pois(15), h_w=0.6, h_r=1, b_w=0, b_r=19.
                 Optional transition limit: state capped at Table 6 bounds
                 inside the transition (training only).

All envs: action in [-1,1]^n mapped to [0, UB] per edge (clipped, rounded);
reward = -(holding+backorder cost)/1000; training episodes truncated at 256
steps (Spinning-Up epoch-reset semantics).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


TRAIN_EPISODE_LEN = 256


def _scale_action(a, ub):
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    return np.rint((a + 1.0) / 2.0 * ub)


class LinearEnv(gym.Env):
    """Beer game, 4 stock points in series (0 = most upstream, 3 = retailer)."""

    N = 4
    H, B = 1.0, 2.0
    ACT_UB = np.full(4, 30.0)
    MAX_LT = 4
    EVAL_HORIZON = 35  # = initial period (cost in reset info) + 34 steps

    UB_TI, UB_TB, UB_I, UB_BO, UB_IT = 4000.0, 4000.0, 1000.0, 1000.0, 150.0

    def __init__(self, episode_len=TRAIN_EPISODE_LEN):
        self.episode_len = episode_len
        n_obs = 2 + 4 + 4 + self.N * self.MAX_LT
        self.observation_space = spaces.Box(-1.0, 1.0, (n_obs,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.I = np.full(self.N, 12.0)
        self.BO_edge = np.zeros(self.N)  # owed to i by its upstream (i=1..3)
        self.BO_cust = 0.0
        self.pipeline = [dict() for _ in range(self.N)]
        for i in range(self.N):
            self.pipeline[i][0] = 4.0
            self.pipeline[i][1] = 4.0
        self.steps = 0
        cost = self._run_period(np.full(self.N, 4.0))  # initial outstanding orders
        return self._obs(), {"cost": cost}

    def _run_period(self, orders):
        """Events 1-2 of period self.t: receive, observe orders+demand, ship."""
        t = self.t
        # Event 1: receive everything due at or before t
        for i in range(self.N):
            for a in [a for a in self.pipeline[i] if a <= t]:
                self.I[i] += self.pipeline[i].pop(a)

        lt = int(self.np_random.integers(0, self.MAX_LT + 1))  # shared lead time
        d = float(self.np_random.integers(0, 16))

        # Event 2: infinite supplier ships stock point 0's order
        if orders[0] > 0:
            self.pipeline[0][t + lt] = self.pipeline[0].get(t + lt, 0.0) + orders[0]
        # each stock point ships its downstream neighbour's order + backorders
        for i in range(1, self.N):
            u = i - 1
            req = orders[i] + self.BO_edge[i]
            q = min(self.I[u], req)
            self.I[u] -= q
            self.BO_edge[i] = req - q
            if q > 0:
                self.pipeline[i][t + lt] = self.pipeline[i].get(t + lt, 0.0) + q
        # retailer serves customer backorders + new demand
        avail = self.I[3]
        self.I[3] = max(avail - self.BO_cust - d, 0.0)
        self.BO_cust = max(self.BO_cust + d - avail, 0.0)

        self.t += 1
        return self.H * self.I.sum() + self.B * (self.BO_edge[1:].sum() + self.BO_cust)

    def _obs(self):
        it_slots = np.zeros((self.N, self.MAX_LT))
        for i in range(self.N):
            for arr, q in self.pipeline[i].items():
                s = min(max(arr - self.t, 0), self.MAX_LT - 1)
                it_slots[i, s] += q
        raw = np.concatenate((
            [self.I.sum(), self.BO_edge[1:].sum() + self.BO_cust], self.I,
            self.BO_edge[1:], [self.BO_cust], it_slots.flatten(),
        ))
        ub = np.concatenate((
            [self.UB_TI, self.UB_TB], np.full(4, self.UB_I), np.full(4, self.UB_BO),
            np.full(self.N * self.MAX_LT, self.UB_IT),
        ))
        return np.clip(raw / ub, 0, None) * 2.0 - 1.0

    def step(self, action):
        cost = self._run_period(_scale_action(action, self.ACT_UB))
        self.steps += 1
        truncated = self.steps >= self.episode_len
        return self._obs(), -cost / 1000.0, False, truncated, {"cost": cost}


class DivergentEnv(gym.Env):
    """Kunnumkal & Topaloglu one-warehouse three-retailer system."""

    H_W, H_R, B_R = 0.6, 1.0, 19.0
    ACT_UB = np.array([300.0, 75.0, 75.0, 75.0])
    EVAL_HORIZON, WARMUP = 75, 25

    UB_TI, UB_TB, UB_I, UB_BOR, UB_ITW, UB_ITR = 1000.0, 450.0, 250.0, 150.0, 300.0, 75.0

    def __init__(self, episode_len=TRAIN_EPISODE_LEN):
        self.episode_len = episode_len
        self.observation_space = spaces.Box(-1.0, 1.0, (13,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.I_w = 0.0
        self.I_r = np.zeros(3)
        self.BO_r = np.zeros(3)
        self.in_w = 0.0
        self.in_r = np.zeros(3)
        self.steps = 0
        return self._obs(), {}

    def _obs(self):
        raw = np.concatenate((
            [self.I_w + self.I_r.sum(), self.BO_r.sum(), self.I_w],
            self.I_r, self.BO_r, [self.in_w], self.in_r,
        ))
        ub = np.concatenate((
            [self.UB_TI, self.UB_TB, self.UB_I], np.full(3, self.UB_I),
            np.full(3, self.UB_BOR), [self.UB_ITW], np.full(3, self.UB_ITR),
        ))
        return np.clip(raw / ub, 0, None) * 2.0 - 1.0

    def step(self, action):
        orders = _scale_action(action, self.ACT_UB)
        O_w, O_r = orders[0], orders[1:]

        # Event 2: supplier ships O_w; warehouse ships this period's retailer
        # orders by ascending inventory position; unfulfilled portions lost.
        ship_r = np.zeros(3)
        if O_r.sum() <= self.I_w:
            ship_r = O_r.copy()
        else:
            ip = self.I_r + self.in_r - self.BO_r
            for r in np.argsort(ip):
                ship_r[r] = max(min(self.I_w - ship_r.sum(), O_r[r]), 0.0)
        self.I_w -= ship_r.sum()

        # Event 3: receive shipments sent in the previous period
        self.I_w += self.in_w
        self.I_r += self.in_r
        self.in_w, self.in_r = O_w, ship_r

        # Event 4: demand Pois(U[5,15]) per retailer
        lam = self.np_random.uniform(5.0, 15.0, 3)
        d = self.np_random.poisson(lam).astype(np.float64)
        avail = self.I_r.copy()
        self.I_r = np.maximum(avail - self.BO_r - d, 0.0)
        self.BO_r = np.maximum(self.BO_r + d - avail, 0.0)

        cost = self.H_W * self.I_w + self.H_R * self.I_r.sum() + self.B_R * self.BO_r.sum()
        self.steps += 1
        truncated = self.steps >= self.episode_len
        return self._obs(), -cost / 1000.0, False, truncated, {"cost": cost}

    def benchmark_action_raw(self, S=(124.0, 30.0, 30.0, 30.0)):
        o_w = max(S[0] - (self.I_w + self.in_w), 0.0)
        o_r = np.maximum(np.asarray(S[1:]) - (self.I_r + self.in_r - self.BO_r), 0.0)
        return np.concatenate(([o_w], o_r))

    def step_raw(self, raw_orders):
        a = np.asarray(raw_orders, dtype=np.float64) / self.ACT_UB * 2.0 - 1.0
        return self.step(a)


# CardBoard network, extracted from the paper's Fig. 1c vector graphics
CONN = {0: [0, 1, 2], 1: [0, 1, 2, 3], 2: [3, 4], 3: [0, 1, 2, 3, 4]}
EDGES = [(m, p) for m in range(4) for p in CONN[m]]  # 14 mill->plant edges
PLANT_EDGES = {p: [k for k, (m, pp) in enumerate(EDGES) if pp == p] for p in range(5)}
MILL_EDGES = {m: [k for k, (mm, p) in enumerate(EDGES) if mm == m] for m in range(4)}


class GeneralEnv(gym.Env):
    """CardBoard Company general network, order-per-edge action space."""

    H_W, H_R, B_R = 0.6, 1.0, 19.0
    DEMAND_MEAN = 15.0
    EVAL_HORIZON, WARMUP = 100, 50  # period 0 cost in reset info + 99 steps
    N_M, N_P, N_E = 4, 5, len(EDGES)

    UB_TI, UB_TB = 4500.0, 8250.0
    UB_I, UB_BOE, UB_BOP = 500.0, 500.0, 250.0
    UB_ITM, UB_ITE = 150.0, 75.0
    ACT_UB = np.concatenate((np.full(4, 150.0), np.full(len(EDGES), 75.0)))

    def __init__(self, episode_len=TRAIN_EPISODE_LEN, transition_limit=False):
        # transition_limit: the paper's Sect. 4.3 option for the general case.
        # Interpretation (see REPORT.md "Transition limit"): the Table-6 upper
        # bounds are used to CLIP THE NORMALIZED OBSERVATION to [0,1], bounding
        # the neural-network input. The physical transition and cost are
        # IDENTICAL in both sub-cases (no inventory is destroyed); only the
        # network's view of extreme states differs. Without the limit the
        # normalized state can exceed 1 when inventory/backorders blow past the
        # bounds, feeding the tanh network unbounded inputs.
        self.episode_len = episode_len
        self.transition_limit = transition_limit
        n_obs = 2 + self.N_M + self.N_P + self.N_E + self.N_P + self.N_M + self.N_E
        hi = 1.0 if transition_limit else np.inf
        self.observation_space = spaces.Box(0.0, hi, (n_obs,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (self.N_M + self.N_E,), np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.I_m = np.zeros(self.N_M)
        self.I_p = np.zeros(self.N_P)
        self.BO_e = np.zeros(self.N_E)
        self.BO_p = np.zeros(self.N_P)
        self.in_m = np.zeros(self.N_M)   # arriving at next period's event 1
        self.in_e = np.zeros(self.N_E)
        self.steps = 0
        cost = self._run_period(np.zeros(self.N_M + self.N_E))
        return self._obs(), {"cost": cost}

    def _run_period(self, orders):
        O_m, O_e = orders[:4], orders[4:]

        # Event 1: receive shipments sent in the previous period
        self.I_m += self.in_m
        for k, (m, p) in enumerate(EDGES):
            self.I_p[p] += self.in_e[k]

        # Event 2: suppliers ship mill orders, arriving next period's event 1
        # (orders placed end of period t are on the mill's shelf at t+2,
        # journal Table 3 event 2: IT_{w,t,t+1} = O_{w,t-1}).
        self.in_m = O_m.copy()

        # Mills ship this period's plant orders + edge backorders from
        # post-receipt stock and the goods are DELIVERED within the same
        # period, before demand (order placed end of period t is on the
        # plant's shelf during period t+1). This receipt lag is the only
        # timing consistent with the paper's tuned 98%-fill base stocks
        # [37,47,33,63,30x5] -> cost 4,797 (see REPORT.md). Rationing by
        # ascending (plant inventory - total backorders owed to the plant).
        owed_p = np.array([sum(self.BO_e[k] for k in PLANT_EDGES[p]) for p in range(5)])
        prio = np.argsort(self.I_p - owed_p)
        mill_req = np.zeros(self.N_M)   # units requested of each mill (Fig-6 fill rate)
        mill_ship = np.zeros(self.N_M)  # units the mill actually shipped
        for m in range(self.N_M):
            req = {k: O_e[k] + self.BO_e[k] for k in MILL_EDGES[m]}
            mill_req[m] = sum(req.values())
            if sum(req.values()) <= self.I_m[m]:
                for k in MILL_EDGES[m]:
                    self.I_p[EDGES[k][1]] += req[k]
                    self.I_m[m] -= req[k]
                    self.BO_e[k] = 0.0
                mill_ship[m] = mill_req[m]
            else:
                for p in prio:
                    for k in MILL_EDGES[m]:
                        if EDGES[k][1] != p:
                            continue
                        q = max(min(self.I_m[m], req[k]), 0.0)
                        self.I_p[p] += q
                        self.I_m[m] -= q
                        self.BO_e[k] = req[k] - q
                        mill_ship[m] += q
        self._mill_req, self._mill_ship = mill_req, mill_ship

        # plants serve customer backorders + new demand
        d = self.np_random.poisson(self.DEMAND_MEAN, self.N_P).astype(np.float64)
        avail = self.I_p.copy()
        served_p = np.minimum(np.maximum(avail - self.BO_p, 0.0), d)
        self._served_p, self._demand_p = served_p, d
        self._served, self._demand = served_p.sum(), d.sum()
        self.I_p = np.maximum(avail - self.BO_p - d, 0.0)
        self.BO_p = np.maximum(self.BO_p + d - avail, 0.0)

        cost = (self.H_W * self.I_m.sum() + self.H_R * self.I_p.sum()
                + self.B_R * self.BO_p.sum())
        return cost

    def _obs(self):
        raw = np.concatenate((
            [self.I_m.sum() + self.I_p.sum(), self.BO_e.sum() + self.BO_p.sum()],
            self.I_m, self.I_p, self.BO_e, self.BO_p, self.in_m, self.in_e,
        ))
        ub = np.concatenate((
            [self.UB_TI, self.UB_TB],
            np.full(self.N_M + self.N_P, self.UB_I),
            np.full(self.N_E, self.UB_BOE), np.full(self.N_P, self.UB_BOP),
            np.full(self.N_M, self.UB_ITM), np.full(self.N_E, self.UB_ITE),
        ))
        # transition limit = cap the normalized observation at 1.0
        hi = 1.0 if self.transition_limit else None
        return np.clip(raw / ub, 0.0, hi)

    def step(self, action):
        cost = self._run_period(_scale_action(action, self.ACT_UB))
        self.steps += 1
        truncated = self.steps >= self.episode_len
        info = {"cost": cost, "served": self._served, "demand": self._demand,
                "served_p": self._served_p, "demand_p": self._demand_p,
                "mill_req": self._mill_req, "mill_ship": self._mill_ship}
        return self._obs(), -cost / 1000.0, False, truncated, info

    # 98% fill-rate base-stock benchmark; plant orders split evenly per edge
    def benchmark_action_raw(self, S=(37.0, 47.0, 33.0, 63.0, 30.0, 30.0, 30.0, 30.0, 30.0)):
        S_m, S_p = np.asarray(S[:4]), np.asarray(S[4:])
        owed_m = np.array([sum(self.BO_e[k] for k in MILL_EDGES[m]) for m in range(4)])
        o_m = np.maximum(S_m - (self.I_m + self.in_m - owed_m), 0.0)
        o_e = np.zeros(self.N_E)
        for p in range(self.N_P):
            ks = PLANT_EDGES[p]
            ip = (self.I_p[p] + sum(self.in_e[k] + self.BO_e[k] for k in ks)
                  - self.BO_p[p])
            need = max(S_p[p] - ip, 0.0)
            for k in ks:
                o_e[k] = need / len(ks)
        return np.concatenate((o_m, o_e))

    def step_raw(self, raw_orders):
        a = np.asarray(raw_orders, dtype=np.float64) / self.ACT_UB * 2.0 - 1.0
        return self.step(a)


def make_env(case, episode_len=TRAIN_EPISODE_LEN):
    if case == "linear":
        return LinearEnv(episode_len)
    if case == "divergent":
        return DivergentEnv(episode_len)
    if case == "general_nolimit":
        return GeneralEnv(episode_len, transition_limit=False)
    if case == "general_limit":
        return GeneralEnv(episode_len, transition_limit=True)
    raise ValueError(case)
