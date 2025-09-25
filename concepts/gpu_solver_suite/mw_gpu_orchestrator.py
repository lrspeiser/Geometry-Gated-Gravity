#!/usr/bin/env python3
"""
GPU-Accelerated Multi-Solver Orchestrator for Milky Way G³ Optimization
Runs multiple optimizers and model variants in parallel to break through plateaus
"""

import numpy as np
import argparse
import json
import time
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
    XP = cp
    print("[✓] GPU acceleration enabled (CuPy)")
except ImportError:
    HAS_CUPY = False
    XP = np
    print("[!] GPU not available, using CPU (NumPy)")

# ============================================================================
# Model Variants (Geometry-Only Gates)
# ============================================================================

class ModelVariant:
    """Base class for G³ model variants"""
    
    def __init__(self, name: str, n_params: int, bounds: Tuple[np.ndarray, np.ndarray]):
        self.name = name
        self.n_params = n_params
        self.bounds = bounds
        self.bounds_lo, self.bounds_hi = bounds
    
    def compute_g_tail(self, R, z, gN, Sigma_loc, theta):
        raise NotImplementedError

class RationalLogisticSigma(ModelVariant):
    """Rational × Logistic × Sigma-screen variant"""
    
    def __init__(self):
        bounds_lo = np.array([50., 5.0, 0.5, 1.0, 0.5, 50., 0.7])
        bounds_hi = np.array([250., 40.0, 8.0, 6.0, 2.0, 200., 2.0])
        super().__init__("rational_logistic_sigma", 7, (bounds_lo, bounds_hi))
    
    def compute_g_tail(self, R, z, gN, Sigma_loc, theta):
        v0, rc, r0, delta, alpha, Sigma0, z_scale = theta
        
        # Rational gate
        rational = R / (R + rc)
        
        # Logistic gate
        x = (R - r0) / delta
        logistic = 1.0 / (1.0 + XP.exp(-x))
        
        # Sigma screen
        sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
        sigma_screen = 1.0 / (1.0 + XP.power(sigma_ratio, alpha))
        
        # Z-dependence (exponential suppression)
        z_suppress = XP.exp(-(XP.abs(z) / z_scale))
        
        # Combined tail acceleration
        g_tail = (v0**2 / R) * rational * logistic * sigma_screen * z_suppress
        
        return g_tail

class RationalSigma(ModelVariant):
    """Rational × Sigma-screen (no logistic)"""
    
    def __init__(self):
        bounds_lo = np.array([50., 5.0, 0.5, 50., 0.7])
        bounds_hi = np.array([250., 40.0, 2.0, 200., 2.0])
        super().__init__("rational_sigma", 5, (bounds_lo, bounds_hi))
    
    def compute_g_tail(self, R, z, gN, Sigma_loc, theta):
        v0, rc, alpha, Sigma0, z_scale = theta
        
        rational = R / (R + rc)
        sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
        sigma_screen = 1.0 / (1.0 + XP.power(sigma_ratio, alpha))
        z_suppress = XP.exp(-(XP.abs(z) / z_scale))
        
        g_tail = (v0**2 / R) * rational * sigma_screen * z_suppress
        return g_tail

class CurvatureGate(ModelVariant):
    """Rational × Curvature-based suppression"""
    
    def __init__(self):
        bounds_lo = np.array([50., 5.0, 0.5, 1.0, 1.0, 0.5, 0.7])
        bounds_hi = np.array([250., 40.0, 8.0, 6.0, 10.0, 3.0, 2.0])
        super().__init__("curvature_gate", 7, (bounds_lo, bounds_hi))
    
    def compute_g_tail(self, R, z, gN, Sigma_loc, theta):
        v0, rc, r0, delta, chi0, p, z_scale = theta
        
        # Rational gate
        rational = R / (R + rc)
        
        # Logistic gate
        x = (R - r0) / delta
        logistic = 1.0 / (1.0 + XP.exp(-x))
        
        # Curvature gate (dimensionless shear)
        # Approximate d(ln gN)/d(ln R) with finite differences
        dR = 0.01  # Small perturbation
        gN_plus = gN  # We'd compute gN(R+dR) in full implementation
        chi = XP.abs(XP.log(gN + 1e-10))  # Simplified proxy
        curvature_suppress = 1.0 / (1.0 + XP.power(chi / chi0, p))
        
        # Z-suppression
        z_suppress = XP.exp(-(XP.abs(z) / z_scale))
        
        g_tail = (v0**2 / R) * rational * logistic * curvature_suppress * z_suppress
        return g_tail

class ThicknessGate(ModelVariant):
    """Variant with enhanced z-dependence for thick disk"""
    
    def __init__(self):
        bounds_lo = np.array([50., 5.0, 0.5, 1.0, 50., 0.5, 0.3, 1.0])
        bounds_hi = np.array([250., 40.0, 8.0, 6.0, 200., 2.0, 1.5, 3.0])
        super().__init__("thickness_gate", 8, (bounds_lo, bounds_hi))
    
    def compute_g_tail(self, R, z, gN, Sigma_loc, theta):
        v0, rc, r0, delta, Sigma0, alpha, hz, m = theta
        
        # Modified rc with z-dependence
        rc_eff = rc * (1.0 + XP.power(XP.abs(z) / hz, m))
        
        rational = R / (R + rc_eff)
        x = (R - r0) / delta
        logistic = 1.0 / (1.0 + XP.exp(-x))
        
        sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
        sigma_screen = 1.0 / (1.0 + XP.power(sigma_ratio, alpha))
        
        g_tail = (v0**2 / R) * rational * logistic * sigma_screen
        return g_tail

# ============================================================================
# Objective Function
# ============================================================================

def robust_loss(v_model, v_obs, v_err=None, loss_type='huber', delta=0.1):
    """Robust loss function with optional error weighting"""
    
    # Relative error
    rel_error = XP.abs((v_model - v_obs) / (v_obs + 1e-6))
    
    if v_err is not None:
        # Weight by inverse variance
        weights = 1.0 / (v_err**2 + (0.05 * v_obs)**2)  # Add noise floor
        rel_error *= XP.sqrt(weights / XP.mean(weights))
    
    if loss_type == 'huber':
        # Huber loss (quadratic near 0, linear in tails)
        mask = rel_error <= delta
        loss = XP.where(mask, 
                       0.5 * rel_error**2,
                       delta * (rel_error - 0.5 * delta))
    elif loss_type == 'tukey':
        # Tukey biweight (completely ignores outliers)
        c = 4.685 * delta
        mask = rel_error <= c
        loss = XP.where(mask,
                       (c**2 / 6) * (1 - (1 - (rel_error/c)**2)**3),
                       c**2 / 6)
    else:  # L1
        loss = rel_error
    
    return XP.median(loss)

def evaluate_variant(theta, data, variant: ModelVariant):
    """Evaluate a model variant on the data"""
    
    R = data['R']
    z = data.get('z', XP.zeros_like(R))
    v_obs = data['v_obs']
    v_err = data.get('v_err', None)
    gN = data['gN']
    Sigma_loc = data.get('Sigma_loc', XP.ones_like(R) * 100.0)
    
    # Compute tail acceleration
    g_tail = variant.compute_g_tail(R, z, gN, Sigma_loc, theta)
    
    # Total circular velocity
    g_tot = gN + g_tail
    v_model = XP.sqrt(XP.maximum(R * g_tot, 0.0))
    
    # Robust loss
    return robust_loss(v_model, v_obs, v_err, loss_type='huber')

# ============================================================================
# Differential Evolution (Global)
# ============================================================================

class DifferentialEvolution:
    """GPU-accelerated Differential Evolution"""
    
    def __init__(self, func, bounds, popsize=15, F=0.8, CR=0.9, seed=42):
        self.func = func
        self.bounds = bounds
        self.dim = len(bounds[0])
        self.popsize = popsize * self.dim  # DE rule of thumb
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        
        XP.random.seed(seed)
        
        # Initialize population
        self.pop = self._init_population()
        self.fitness = None
        self.best_idx = None
        self.best_x = None
        self.best_f = np.inf
    
    def _init_population(self):
        """Latin hypercube sampling for diverse initialization"""
        lo, hi = XP.asarray(self.bounds[0]), XP.asarray(self.bounds[1])
        pop = XP.random.uniform(0, 1, (self.popsize, self.dim))
        
        # Latin hypercube
        for j in range(self.dim):
            idx = XP.random.permutation(self.popsize)
            pop[:, j] = (idx + pop[:, j]) / self.popsize
        
        # Scale to bounds
        pop = lo + pop * (hi - lo)
        return pop
    
    def step(self):
        """One generation of DE"""
        lo, hi = XP.asarray(self.bounds[0]), XP.asarray(self.bounds[1])
        
        # Evaluate current population if needed
        if self.fitness is None:
            self.fitness = XP.array([self.func(x) for x in self.pop])
            self.best_idx = XP.argmin(self.fitness)
            self.best_x = self.pop[self.best_idx].copy()
            self.best_f = float(self.fitness[self.best_idx])
        
        # Generate trial vectors
        trials = XP.empty_like(self.pop)
        
        for i in range(self.popsize):
            # Select 3 distinct individuals
            idxs = [j for j in range(self.popsize) if j != i]
            a, b, c = XP.random.choice(idxs, 3, replace=False)
            
            # Mutation: best/1/bin
            mutant = self.best_x + self.F * (self.pop[b] - self.pop[c])
            
            # Crossover
            cross_points = XP.random.rand(self.dim) < self.CR
            if not XP.any(cross_points):
                cross_points[XP.random.randint(self.dim)] = True
            
            trial = XP.where(cross_points, mutant, self.pop[i])
            
            # Bound constraints
            trial = XP.clip(trial, lo, hi)
            trials[i] = trial
        
        # Evaluate trials
        trial_fitness = XP.array([self.func(x) for x in trials])
        
        # Selection
        mask = trial_fitness < self.fitness
        self.pop[mask] = trials[mask]
        self.fitness[mask] = trial_fitness[mask]
        
        # Update best
        min_idx = XP.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_f:
            self.best_idx = min_idx
            self.best_x = self.pop[min_idx].copy()
            self.best_f = float(self.fitness[min_idx])
        
        return self.best_x, self.best_f

# ============================================================================
# SPSA-Adam (Local Refinement)
# ============================================================================

class SPSAAdam:
    """Simultaneous Perturbation Stochastic Approximation with Adam"""
    
    def __init__(self, func, x0, bounds, lr=0.01, c=0.01, seed=42):
        self.func = func
        self.x = XP.asarray(x0, dtype=XP.float64)
        self.bounds = bounds
        self.lr = lr
        self.c = c  # Perturbation size
        
        # Adam parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.m = XP.zeros_like(self.x)
        self.v = XP.zeros_like(self.x)
        self.t = 0
        
        XP.random.seed(seed)
    
    def step(self):
        """One SPSA-Adam step"""
        self.t += 1
        lo, hi = XP.asarray(self.bounds[0]), XP.asarray(self.bounds[1])
        
        # Generate random perturbation
        delta = 2 * XP.random.randint(0, 2, self.x.shape) - 1
        delta = delta.astype(XP.float64)
        
        # Perturbed evaluations
        x_plus = XP.clip(self.x + self.c * delta, lo, hi)
        x_minus = XP.clip(self.x - self.c * delta, lo, hi)
        
        f_plus = self.func(x_plus)
        f_minus = self.func(x_minus)
        
        # Gradient estimate
        g = (f_plus - f_minus) / (2 * self.c * delta + 1e-10)
        
        # Adam update
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update with adaptive LR
        lr_t = self.lr / (1 + 0.001 * self.t)  # Decay
        self.x -= lr_t * m_hat / (XP.sqrt(v_hat) + self.eps)
        
        # Project to bounds
        self.x = XP.clip(self.x, lo, hi)
        
        f_current = self.func(self.x)
        return self.x.copy(), float(f_current)

# ============================================================================
# Orchestrator with Plateau Detection
# ============================================================================

class SolverOrchestrator:
    """Multi-solver orchestrator with automatic switching"""
    
    def __init__(self, data: Dict, variants: List[ModelVariant], 
                 out_dir: str, patience: int = 10, min_rel_improve: float = 1e-3):
        self.data = data
        self.variants = variants
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.patience = patience
        self.min_rel_improve = min_rel_improve
        
        self.incumbent_x = None
        self.incumbent_f = np.inf
        self.incumbent_variant = None
        
        self.history = []
        self.plateau_counter = 0
        self.last_best = np.inf
    
    def run_solver_chain(self, variant: ModelVariant, max_iters: int = 1000):
        """Run DE -> SPSA-Adam chain on a variant"""
        
        print(f"\n[Variant: {variant.name}]")
        func = lambda x: evaluate_variant(x, self.data, variant)
        
        # Phase 1: Differential Evolution (global)
        print("  Phase 1: Differential Evolution...")
        de = DifferentialEvolution(func, variant.bounds, popsize=15)
        
        de_iters = min(150, max_iters // 3)
        for i in range(de_iters):
            x, f = de.step()
            
            if f < self.incumbent_f:
                self.incumbent_x = x.copy()
                self.incumbent_f = f
                self.incumbent_variant = variant.name
                self._save_best()
                print(f"    [DE iter {i+1}] NEW BEST: {f:.4f}")
            
            if i % 20 == 0:
                print(f"    [DE iter {i+1}] Current: {f:.4f}, Best: {self.incumbent_f:.4f}")
        
        # Phase 2: SPSA-Adam (local refinement)
        print("  Phase 2: SPSA-Adam refinement...")
        spsa = SPSAAdam(func, de.best_x, variant.bounds, lr=0.01)
        
        spsa_iters = max_iters - de_iters
        for i in range(spsa_iters):
            x, f = spsa.step()
            
            if f < self.incumbent_f:
                self.incumbent_x = x.copy()
                self.incumbent_f = f
                self.incumbent_variant = variant.name
                self._save_best()
                print(f"    [SPSA iter {i+1}] NEW BEST: {f:.4f}")
            
            if i % 50 == 0:
                print(f"    [SPSA iter {i+1}] Current: {f:.4f}, Best: {self.incumbent_f:.4f}")
            
            # Check for plateau
            if i % self.patience == 0:
                rel_improve = (self.last_best - self.incumbent_f) / (abs(self.last_best) + 1e-10)
                if rel_improve < self.min_rel_improve:
                    print(f"    Plateau detected (rel_improve={rel_improve:.2e}), switching...")
                    self.last_best = self.incumbent_f
                    return True  # Signal to switch
                self.last_best = self.incumbent_f
        
        return False
    
    def orchestrate(self, max_cycles: int = 10):
        """Main orchestration loop"""
        
        print("\n" + "="*70)
        print("SOLVER ORCHESTRATOR")
        print("="*70)
        print(f"Variants: {[v.name for v in self.variants]}")
        print(f"Patience: {self.patience}, Min improvement: {self.min_rel_improve}")
        print("="*70)
        
        # Cycle through variants
        for cycle in range(max_cycles):
            print(f"\n[CYCLE {cycle+1}/{max_cycles}]")
            
            for variant in self.variants:
                plateau = self.run_solver_chain(variant, max_iters=300)
                self._log_progress(cycle, variant.name)
                
                if plateau:
                    print(f"  Switching from {variant.name} due to plateau")
            
            # Check global plateau
            if cycle > 0:
                rel_improve = (self.history[-2]['best_f'] - self.incumbent_f) / (abs(self.history[-2]['best_f']) + 1e-10)
                if rel_improve < self.min_rel_improve:
                    print(f"\nGlobal plateau reached after {cycle+1} cycles")
                    print(f"Final best: {self.incumbent_f:.4f} (variant: {self.incumbent_variant})")
                    break
        
        self._final_report()
    
    def _save_best(self):
        """Save current best to JSON"""
        best_dict = {
            'variant': self.incumbent_variant,
            'loss': float(self.incumbent_f),
            'accuracy': float(1.0 - self.incumbent_f),
            'params': self.incumbent_x.tolist() if HAS_CUPY else self.incumbent_x.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.out_dir / 'best.json', 'w') as f:
            json.dump(best_dict, f, indent=2)
    
    def _log_progress(self, cycle: int, variant: str):
        """Log progress to CSV"""
        entry = {
            'cycle': cycle,
            'variant': variant,
            'best_f': self.incumbent_f,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(entry)
        
        # Write CSV
        import csv
        csv_path = self.out_dir / 'search_log.csv'
        with open(csv_path, 'w', newline='') as f:
            if self.history:
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                writer.writeheader()
                writer.writerows(self.history)
    
    def _final_report(self):
        """Generate final report"""
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        print(f"Best variant: {self.incumbent_variant}")
        print(f"Best loss: {self.incumbent_f:.4f}")
        print(f"Best accuracy: {(1.0 - self.incumbent_f)*100:.1f}%")
        print(f"Total evaluations: {len(self.history)} variant-cycles")
        print(f"Output directory: {self.out_dir}")
        print("="*70)

# ============================================================================
# Data Loading
# ============================================================================

def load_mw_data(npz_path: str) -> Dict:
    """Load MW data from NPZ file"""
    
    data = np.load(npz_path)
    
    # Transfer to GPU
    mw_data = {
        'R': XP.asarray(data['R_kpc'], dtype=XP.float64),
        'v_obs': XP.asarray(data['v_obs_kms'], dtype=XP.float64)
    }
    
    # Optional fields
    if 'z_kpc' in data:
        mw_data['z'] = XP.asarray(data['z_kpc'], dtype=XP.float64)
    if 'v_err_kms' in data:
        mw_data['v_err'] = XP.asarray(data['v_err_kms'], dtype=XP.float64)
    if 'gN_kms2_per_kpc' in data:
        mw_data['gN'] = XP.asarray(data['gN_kms2_per_kpc'], dtype=XP.float64)
    else:
        # Simple Newtonian estimate if not provided
        mw_data['gN'] = 200.0**2 / mw_data['R']  # Flat rotation curve baseline
    
    if 'Sigma_loc_Msun_pc2' in data:
        mw_data['Sigma_loc'] = XP.asarray(data['Sigma_loc_Msun_pc2'], dtype=XP.float64)
    
    print(f"Loaded {len(mw_data['R'])} stars from {npz_path}")
    return mw_data

def create_demo_data(n_stars: int = 10000) -> Dict:
    """Create synthetic demo data"""
    
    print("Creating synthetic demo data...")
    
    # Radial distribution
    R = XP.random.uniform(3.0, 20.0, n_stars)
    z = XP.random.normal(0, 0.3, n_stars)
    
    # Simple rotation curve with scatter
    v_flat = 220.0
    v_obs = v_flat * XP.sqrt(1.0 - XP.exp(-R / 5.0))
    v_obs += XP.random.normal(0, 15, n_stars)
    v_err = XP.ones_like(v_obs) * 10.0
    
    # Newtonian baseline
    gN = (180.0**2) / R * XP.exp(-R / 8.0)
    
    # Surface density
    Sigma_loc = 100.0 * XP.exp(-R / 4.0)
    
    return {
        'R': R,
        'z': z,
        'v_obs': v_obs,
        'v_err': v_err,
        'gN': gN,
        'Sigma_loc': Sigma_loc
    }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='GPU Multi-Solver Orchestrator')
    parser.add_argument('--data', type=str, help='Path to NPZ data file')
    parser.add_argument('--demo', action='store_true', help='Run with demo data')
    parser.add_argument('--out', type=str, default='out/orchestrator', help='Output directory')
    parser.add_argument('--variants', type=str, 
                       default='rational_logistic_sigma,rational_sigma,curvature_gate,thickness_gate',
                       help='Comma-separated list of variants')
    parser.add_argument('--patience', type=int, default=10, help='Plateau patience')
    parser.add_argument('--min_rel', type=float, default=5e-4, help='Min relative improvement')
    parser.add_argument('--max_cycles', type=int, default=10, help='Max orchestrator cycles')
    
    args = parser.parse_args()
    
    # Load or create data
    if args.demo:
        data = create_demo_data(n_stars=10000)
    elif args.data:
        data = load_mw_data(args.data)
    else:
        print("Error: Specify --data or --demo")
        return
    
    # Initialize variants
    variant_map = {
        'rational_logistic_sigma': RationalLogisticSigma(),
        'rational_sigma': RationalSigma(),
        'curvature_gate': CurvatureGate(),
        'thickness_gate': ThicknessGate()
    }
    
    variant_names = args.variants.split(',')
    variants = [variant_map[name] for name in variant_names if name in variant_map]
    
    if not variants:
        print(f"Error: No valid variants in {args.variants}")
        return
    
    # Run orchestrator
    orchestrator = SolverOrchestrator(
        data=data,
        variants=variants,
        out_dir=args.out,
        patience=args.patience,
        min_rel_improve=args.min_rel
    )
    
    orchestrator.orchestrate(max_cycles=args.max_cycles)

if __name__ == '__main__':
    main()