
import json, time, math, os, sys, argparse, csv
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

def _get_xp():
    try:
        import cupy as cp  # type: ignore
        _ = cp.asarray([0.0]) + 0.0
        return cp, True
    except Exception:
        import numpy as cp  # type: ignore
        return cp, False

xp, _GPU = _get_xp()

@dataclass
class G3Params:
    v0_kms: float = 250.0
    rc0_kpc: float = 15.0
    gamma: float = 0.5
    beta: float = 0.10
    sigma_star: float = 150.0
    alpha: float = 1.1
    kappa: float = 2.0
    eta: float = 1.0
    delta_kpc: float = 0.8
    p_in: float = 1.2
    p_out: float = 1.0
    g_sat: float = 3000.0
    rc_ref_kpc: float = 30.0
    sigma0: float = 150.0
    gating_type: str = "rational"
    screen_type: str = "sigmoid"
    exponent_type: str = "logistic_r"

def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def _softplus(x):
    return xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(x, 0.0)

class G3Model:
    def __init__(self, params: G3Params):
        self.p = params

    def _rc_eff(self, r_half_kpc: float, sigma_bar: float) -> float:
        p = self.p
        rc = p.rc0_kpc * (r_half_kpc / p.rc_ref_kpc) ** p.gamma * (sigma_bar / p.sigma0) ** (-p.beta)
        return float(rc)

    def _p_of_R(self, R_kpc, r_half_kpc, Sigma_loc):
        p = self.p
        if p.exponent_type == "logistic_r":
            z = (R_kpc - p.eta * r_half_kpc) / max(p.delta_kpc, 1e-6)
            w = 1.0 / (1.0 + xp.exp(-z))
            return p.p_in + (p.p_out - p.p_in) * w
        elif p.exponent_type == "sigma_tuned":
            s = _safe_div(self.p.sigma_star, Sigma_loc)
            w = 1.0 / (1.0 + s**self.p.alpha)
            return p.p_out + (p.p_in - p.p_out) * w
        else:
            return xp.full_like(R_kpc, p.p_out)

    def _screen(self, Sigma_loc):
        p = self.p
        if p.screen_type == "none":
            return xp.ones_like(Sigma_loc)
        if p.screen_type == "exp":
            s = _safe_div(p.sigma_star, Sigma_loc)
            return xp.exp(-p.alpha * s**p.kappa)
        if p.screen_type == "sigmoid":
            s = _safe_div(p.sigma_star, Sigma_loc)
            return 1.0 / (1.0 + s**p.alpha) ** p.kappa
        if p.screen_type == "powerlaw":
            s = _safe_div(Sigma_loc, Sigma_loc + p.sigma_star)
            return s ** p.alpha
        return xp.ones_like(Sigma_loc)

    def _gate(self, R_kpc, rc_eff, power):
        p = self.p
        if p.gating_type == "rational":
            Rp = xp.power(R_kpc, power)
            return _safe_div(Rp, Rp + rc_eff**power)
        if p.gating_type == "rational2":
            Rp = xp.power(R_kpc, 2.0 * power)
            return _safe_div(Rp, Rp + (rc_eff ** (2.0 * power)))
        if p.gating_type == "softplus":
            s = _softplus((R_kpc - rc_eff))
            return _safe_div(s, s + _softplus(rc_eff))
        if p.gating_type == "tanh":
            w = xp.maximum(1e-3, rc_eff / (xp.mean(power) + 1e-6))
            return 0.5 * (1.0 + xp.tanh((R_kpc - rc_eff) / w))
        return xp.power(R_kpc, power) / (xp.power(R_kpc, power) + rc_eff**power)

    def predict_vel_kms(self, R_kpc, gN_kms2_per_kpc, Sigma_loc, r_half_kpc, Sigma_bar):
        p = self.p
        rc_eff = self._rc_eff(r_half_kpc, Sigma_bar)
        power = self._p_of_R(R_kpc, r_half_kpc, Sigma_loc)
        gate = self._gate(R_kpc, rc_eff, power)
        screen = self._screen(Sigma_loc)
        g_tail = _safe_div(p.v0_kms ** 2, R_kpc) * gate * screen
        g_tail = p.g_sat * xp.tanh(_safe_div(g_tail, p.g_sat))
        g_tot = gN_kms2_per_kpc + g_tail
        v = xp.sqrt(xp.clip(R_kpc * g_tot, 0.0, None))
        return v

def median_abs_pct_err(v_pred, v_obs, eps=1e-9):
    return float(xp.median(xp.abs(v_pred - v_obs) / (xp.maximum(xp.abs(v_obs), eps))) * 100.0)

def mean_abs_pct_err(v_pred, v_obs, eps=1e-9):
    return float(xp.mean(xp.abs(v_pred - v_obs) / (xp.maximum(xp.abs(v_obs), eps))) * 100.0)

TEMPLATE_BOUNDS = {
    "v0_kms": (120.0, 380.0),
    "rc0_kpc": (5.0, 40.0),
    "gamma": (0.0, 2.0),
    "beta": (0.0, 0.8),
    "sigma_star": (20.0, 400.0),
    "alpha": (0.5, 3.0),
    "kappa": (0.5, 3.5),
    "eta": (0.4, 1.6),
    "delta_kpc": (0.1, 2.5),
    "p_in": (0.2, 2.5),
    "p_out": (0.2, 2.5),
    "g_sat": (800.0, 8000.0),
}

def objective_mw(params_vec, R_kpc, v_obs_kms, gN_kms2_per_kpc, Sigma_loc, r_half_kpc, Sigma_bar, param_space, variant, metric="median"):
    keys = list(param_space.keys())
    lo = xp.asarray([param_space[k][0] for k in keys])
    hi = xp.asarray([param_space[k][1] for k in keys])
    x = xp.clip(params_vec, lo, hi)
    p = G3Params(
        v0_kms=float(x[0]), rc0_kpc=float(x[1]), gamma=float(x[2]), beta=float(x[3]),
        sigma_star=float(x[4]), alpha=float(x[5]), kappa=float(x[6]), eta=float(x[7]),
        delta_kpc=float(x[8]), p_in=float(x[9]), p_out=float(x[10]), g_sat=float(x[11]),
        gating_type=variant.get("gating_type","rational"),
        screen_type=variant.get("screen_type","sigmoid"),
        exponent_type=variant.get("exponent_type","logistic_r"),
    )
    model = G3Model(p)
    v_pred = model.predict_vel_kms(R_kpc, gN_kms2_per_kpc, Sigma_loc, r_half_kpc, Sigma_bar)
    if metric == "median":
        return median_abs_pct_err(v_pred, v_obs_kms)
    return mean_abs_pct_err(v_pred, v_obs_kms)

class PSO:
    def __init__(self, param_space, n_particles=64, w=0.6, c1=1.6, c2=1.0, seed=0):
        self.space = param_space; self.dim = len(param_space); self.n = n_particles
        self.w, self.c1, self.c2 = w, c1, c2
        rs = xp.random.RandomState(seed)
        self.lo = xp.asarray([param_space[k][0] for k in param_space])
        self.hi = xp.asarray([param_space[k][1] for k in param_space])
        self.pos = self.lo + (self.hi - self.lo) * rs.rand(self.n, self.dim)
        self.vel = 0.1 * (rs.rand(self.n, self.dim) - 0.5) * (self.hi - self.lo)
        self.pbest = self.pos.copy(); self.pbest_val = xp.full((self.n,), xp.inf)
        self.gbest = None; self.gbest_val = xp.inf; self.rng = rs

    def step(self, f):
        vals = xp.asarray([f(self.pos[i]) for i in range(self.n)])
        improved = vals < self.pbest_val
        self.pbest[improved] = self.pos[improved]; self.pbest_val = xp.minimum(self.pbest_val, vals)
        i_best = int(xp.argmin(self.pbest_val))
        if float(self.pbest_val[i_best]) < float(self.gbest_val):
            self.gbest = self.pbest[i_best].copy(); self.gbest_val = float(self.pbest_val[i_best])
        r1 = self.rng.rand(self.n, self.dim); r2 = self.rng.rand(self.n, self.dim)
        cognitive = self.c1 * r1 * (self.pbest - self.pos); social = self.c2 * r2 * (self.gbest - self.pos)
        self.vel = self.w * self.vel + cognitive + social; self.pos = xp.clip(self.pos + self.vel, self.lo, self.hi)
        return float(self.gbest_val), xp.asarray(self.gbest)

class DifferentialEvolution:
    def __init__(self, param_space, pop=96, F=0.5, CR=0.9, seed=0):
        self.space = param_space; self.dim = len(param_space); self.pop = pop; self.F = F; self.CR = CR
        rs = xp.random.RandomState(seed)
        self.lo = xp.asarray([param_space[k][0] for k in param_space])
        self.hi = xp.asarray([param_space[k][1] for k in param_space])
        self.X = self.lo + (self.hi - self.lo) * rs.rand(self.pop, self.dim); self.rng = rs

    def step(self, f):
        newX = self.X.copy(); scores = xp.asarray([f(self.X[i]) for i in range(self.pop)])
        for i in range(self.pop):
            idx = list(range(self.pop)); idx.remove(i)
            a, b, c = self.rng.choice(idx, 3, replace=False)
            mutant = self.X[a] + self.F * (self.X[b] - self.X[c])
            cross_mask = self.rng.rand(self.dim) < self.CR
            trial = xp.where(cross_mask, mutant, self.X[i]); trial = xp.clip(trial, self.lo, self.hi)
            f_trial = f(trial)
            if f_trial < scores[i]: newX[i] = trial; scores[i] = f_trial
        self.X = newX; i_best = int(xp.argmin(scores))
        return float(scores[i_best]), self.X[i_best].copy()

def _now(): return time.time()
def _as_list(d): return list(d) if hasattr(d, '__len__') else [d]

class SolverOrchestrator:
    def __init__(self, param_space, eval_fn, variants, patience_iters=50, min_delta=0.05, global_patience=200, time_limit_s=None, seed=0):
        self.space = param_space; self.eval_fn = eval_fn; self.variants = variants
        self.patience_iters = patience_iters; self.min_delta = min_delta; self.global_patience = global_patience
        self.time_limit_s = time_limit_s; self.seed = seed; self.history = []; self.best = {"score": float('inf'), "x": None, "variant": None}
        self._start = _now()

    def _init_solver(self, variant_id, solver_name):
        if solver_name == "pso": return PSO(self.space, n_particles=64, seed=self.seed + 13 * variant_id)
        if solver_name == "de": return DifferentialEvolution(self.space, pop=96, seed=self.seed + 29 * variant_id)
        raise ValueError("Unknown solver")

    def run(self, branches=3, iters_per_branch=200, outdir="out_g3_gpu_suite"):
        os.makedirs(outdir, exist_ok=True)
        branch_states = []
        for b in range(branches):
            v_id = b % len(self.variants); solver_name = "pso" if (b % 2 == 0) else "de"
            solver = self._init_solver(v_id, solver_name)
            fv = lambda x: self.eval_fn(x, self.variants[v_id])
            score_init = fv(solver.pos[0]) if hasattr(solver, "pos") else fv(solver.X[0])
            branch_states.append([v_id, solver_name, solver, score_init, 0])
        no_global_improve = 0
        def maybe_time_up(): return (self.time_limit_s is not None) and ((_now() - self._start) > self.time_limit_s)
        for t in range(iters_per_branch):
            if maybe_time_up(): break
            for b in range(branches):
                v_id, sname, solver, bbest, since = branch_states[b]
                fv = lambda x: self.eval_fn(x, self.variants[v_id])
                score, xbest = solver.step(fv)
                if score + 1e-9 < bbest - self.min_delta: bbest = score; since = 0
                else: since += 1
                if score + 1e-9 < self.best["score"]:
                    self.best = {"score": float(score), "x": [float(a) for a in _as_list(xbest)], "variant": self.variants[v_id]}
                    no_global_improve = 0
                else: no_global_improve += 1
                if since >= self.patience_iters or no_global_improve >= self.global_patience:
                    v_id = (v_id + 1) % len(self.variants); sname = "de" if sname == "pso" else "pso"; solver = self._init_solver(v_id, sname)
                    bbest = float("inf"); since = 0; no_global_improve = 0
                branch_states[b] = [v_id, sname, solver, bbest, since]
                self.history.append({"iter": t, "branch": b, "variant_id": v_id, "solver": sname, "score": float(score), "best_score": float(self.best["score"]), "time_s": _now() - self._start})
            if t % 10 == 0 or t == iters_per_branch - 1:
                with open(os.path.join(outdir, "best.json"), "w") as f: json.dump(self.best, f, indent=2)
                with open(os.path.join(outdir, "history.json"), "w") as f: json.dump(self.history, f, indent=2)
        return self.best

def _load_mw_csv(path: str):
    R_kpc, v_obs, gN, Sigma = [], [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            R_kpc.append(float(row["R_kpc"]))
            v_obs.append(float(row["v_obs_kms"]))
            gN.append(float(row["gN_kms2_per_kpc"]))
            Sigma.append(float(row["Sigma_loc_Msun_pc2"]))
    return xp.asarray(R_kpc), xp.asarray(v_obs), xp.asarray(gN), xp.asarray(Sigma)

def objective_mw_factory(R, v_obs, gN, Sigma, r_half_kpc, Sigma_bar, metric):
    def fn(xvec, variant):
        return objective_mw(xvec, R, v_obs, gN, Sigma, r_half_kpc, Sigma_bar, TEMPLATE_BOUNDS, variant, metric=metric)
    return fn

def _variants_default():
    return [
        {"gating_type": "rational", "screen_type": "sigmoid", "exponent_type": "logistic_r"},
        {"gating_type": "rational2", "screen_type": "sigmoid", "exponent_type": "logistic_r"},
        {"gating_type": "rational", "screen_type": "exp", "exponent_type": "logistic_r"},
        {"gating_type": "tanh", "screen_type": "sigmoid", "exponent_type": "sigma_tuned"},
        {"gating_type": "softplus", "screen_type": "powerlaw", "exponent_type": "logistic_r"},
    ]

def cli_mw_runner():
    ap = argparse.ArgumentParser(description="G3 GPU solver suite: Milky Way search")
    ap.add_argument("--mw_csv", type=str, required=True, help="CSV with columns: R_kpc,v_obs_kms,gN_kms2_per_kpc,Sigma_loc_Msun_pc2")
    ap.add_argument("--r_half_kpc", type=float, required=True, help="Half-mass radius proxy")
    ap.add_argument("--sigma_bar", type=float, required=True, help="Mean baryonic surface density (Msun/pc^2)")
    ap.add_argument("--metric", type=str, default="median", choices=["median","mean"])
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--global_patience", type=int, default=160)
    ap.add_argument("--min_delta", type=float, default=0.02)
    ap.add_argument("--time_limit_s", type=float, default=None)
    ap.add_argument("--outdir", type=str, default="out_g3_gpu_suite")
    args = ap.parse_args()

    R, v_obs, gN, Sigma = _load_mw_csv(args.mw_csv)
    eval_fn = objective_mw_factory(R, v_obs, gN, Sigma, args.r_half_kpc, args.sigma_bar, args.metric)
    orch = SolverOrchestrator(TEMPLATE_BOUNDS, eval_fn, _variants_default(),
                              patience_iters=args.patience, min_delta=args.min_delta,
                              global_patience=args.global_patience, time_limit_s=args.time_limit_s, seed=0)
    best = orch.run(branches=args.branches, iters_per_branch=args.iters, outdir=args.outdir)
    summary = {"gpu": _GPU, "best_score_percent": best["score"], "best_params_vector": best["x"], "best_variant": best["variant"], "bounds": TEMPLATE_BOUNDS}
    with open(os.path.join(args.outdir, "mw_best_summary.json"), "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

def cli_sparc_zero_shot():
    ap = argparse.ArgumentParser(description="Zero-shot SPARC evaluation wrapper.")
    ap.add_argument("--gal_list", type=str, required=True)
    ap.add_argument("--params_json", type=str, required=True)
    ap.add_argument("--rotmod_parquet", type=str, required=True)
    ap.add_argument("--all_tables_parquet", type=str, required=True)
    ap.add_argument("--NR", type=int, default=128)
    ap.add_argument("--NZ", type=int, default=128)
    ap.add_argument("--outdir", type=str, default="out_zero_shot_sparc")
    args = ap.parse_args()

    with open(args.params_json, "r") as f: best = json.load(f)
    x = best["best_params_vector"]; var = best["best_variant"]
    keys = list(TEMPLATE_BOUNDS.keys()); friendly = dict(zip(keys, x)); friendly.update(var)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "params_manifest.json"), "w") as f: json.dump(friendly, f, indent=2)
    cmd_text = []
    gals = [g.strip() for g in args.gal_list.split(",")]
    for g in gals:
        cmd = (f'py -u root-m\\pde\\run_sparc_pde.py --axisym_maps '
               f'--galaxy "{g}" --rotmod_parquet {args.rotmod_parquet} '
               f'--all_tables_parquet {args.all_tables_parquet} '
               f'--NR {args.NR} --NZ {args.NZ} '
               f'--S0 {friendly["v0_kms"]} --rc_kpc {friendly["rc0_kpc"]} '
               f'--rc_gamma {friendly["gamma"]} --sigma_beta {friendly["beta"]} '
               f'--sigma0_Msun_pc2 {friendly["sigma_star"]} --g0_kms2_per_kpc 1200 '
               f'--outdir root-m\\out\\sparc_zero_shot\\{g}')
        cmd_text.append(cmd)
    with open(os.path.join(args.outdir, "commands_to_run.txt"), "w") as f: f.write("\\n".join(cmd_text))
    print(f"Wrote suggested commands to: {os.path.join(args.outdir, 'commands_to_run.txt')}")
