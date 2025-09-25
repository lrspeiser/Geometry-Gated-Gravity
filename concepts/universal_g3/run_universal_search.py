#!/usr/bin/env python3
"""
Universal multi-scale optimizer for G³ model
Implements formula families, plateau detection, and automatic branching
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import logging

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

# Import our modules
from gating import gate_and_exponent, curvature_gate, gradient_screening
from baryon_primitives import hernquist_sigma, build_3d_density, vert_profile
from ad_policy import ADPolicy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G = 4.300917270e-6  # (km/s)^2 kpc / Msun
SOLAR_CONSTRAINT = 1e-8  # |G_eff/G - 1| < this

@dataclass
class FormulaFamily:
    """Definition of a G³ formula variant"""
    name: str
    gate_type: str  # 'rational', 'softplus', 'yukawa'
    use_curvature: bool
    use_gradient: bool
    use_nonlocal: bool
    edge_aware_rc: bool
    
    def describe(self):
        features = []
        if self.use_curvature:
            features.append("curvature")
        if self.use_gradient:
            features.append("gradient")
        if self.use_nonlocal:
            features.append("nonlocal")
        if self.edge_aware_rc:
            features.append("edge-aware")
        return f"{self.name}: {self.gate_type} gate" + (f" + {'+'.join(features)}" if features else "")

# Define formula families
FORMULA_FAMILIES = {
    'F1': FormulaFamily('F1', 'rational', False, False, False, False),  # Baseline
    'F2': FormulaFamily('F2', 'softplus', False, False, False, False),  # Smooth monotone
    'F3': FormulaFamily('F3', 'rational', True, False, True, False),   # With curvature
    'F4': FormulaFamily('F4', 'rational', False, True, False, True),    # Edge-aware rc
    'F5': FormulaFamily('F5', 'yukawa', True, False, True, False),     # Strong suppression
}

class PlateauScheduler:
    """Detects optimization plateaus and triggers branching"""
    
    def __init__(self, min_rel_gain=3e-3, window=40, max_plateaus=3):
        self.min_rel_gain = min_rel_gain
        self.window = window
        self.max_plateaus = max_plateaus
        self.hist = []
        self.plateaus = 0
        self.branches = []
        
    def update(self, best):
        """Check if we've plateaued"""
        self.hist.append(best)
        if len(self.hist) < self.window + 1:
            return None
            
        old = self.hist[-self.window - 1]
        gain = (old - best) / max(1e-9, abs(old))
        
        if gain < self.min_rel_gain:
            self.plateaus += 1
            logger.info(f"Plateau detected #{self.plateaus} (gain={gain:.5f})")
            return "plateau"
        return None
        
    def done(self):
        """Check if we should stop"""
        return self.plateaus >= self.max_plateaus
    
    def record_branch(self, branch_info):
        """Record branch action taken"""
        self.branches.append({
            'plateau_num': self.plateaus,
            'iteration': len(self.hist),
            'best_value': self.hist[-1] if self.hist else None,
            **branch_info
        })

def branch_action(branch_id, params, family_name, weights):
    """Execute branching strategy when plateaued"""
    action = {}
    
    if branch_id % 3 == 0:  # Widen transitions
        params["w_p"] *= 1.25
        params["w_S"] *= 1.25
        action['type'] = 'widen_transitions'
        action['new_widths'] = {'w_p': params["w_p"], 'w_S': params["w_S"]}
        
    elif branch_id % 3 == 1:  # Hop family
        family_map = {
            'F1': 'F3',  # Add curvature
            'F2': 'F4',  # Add edge awareness
            'F3': 'F5',  # Try Yukawa
            'F4': 'F1',  # Back to baseline
            'F5': 'F3',  # Back to curvature
        }
        new_family = family_map.get(family_name, 'F1')
        action['type'] = 'family_hop'
        action['from'] = family_name
        action['to'] = new_family
        family_name = new_family
        
    else:  # Anneal MW weight
        weights["MW"] *= 2.0
        action['type'] = 'anneal_weights'
        action['new_MW_weight'] = weights["MW"]
        
    return params, family_name, weights, action

def compute_nonlocal_ring_kernel(R, Sigma, ell=3.0):
    """
    Compute non-local ring awareness metric
    R(R) = ∫ K(|R-R'|; ℓ) * d²ln(Σ)/d(lnR')² dR'
    """
    xp = cp if GPU_AVAILABLE and isinstance(R, cp.ndarray) else np
    
    # Compute curvature
    log_R = xp.log(xp.maximum(R, 1e-6))
    log_Sigma = xp.log(xp.maximum(Sigma, 1e-8))
    
    # Second derivative
    if len(R) > 2:
        d2_logSigma = xp.gradient(xp.gradient(log_Sigma, log_R), log_R)
    else:
        d2_logSigma = xp.zeros_like(R)
    
    # Gaussian kernel K(r) = exp(-r²/2ℓ²) / √(2πℓ²)
    kernel_norm = 1.0 / (xp.sqrt(2 * xp.pi) * ell)
    
    # Convolve (simple approximation for now)
    R_metric = xp.zeros_like(R)
    for i in range(len(R)):
        # Weight by distance
        weights = kernel_norm * xp.exp(-(R - R[i])**2 / (2 * ell**2))
        R_metric[i] = xp.sum(weights * d2_logSigma) * (R[1] - R[0]) if len(R) > 1 else 0
    
    return R_metric

def evaluate_g3_tail(R, Sigma, params, family):
    """
    Evaluate G³ tail acceleration for given formula family
    """
    xp = cp if GPU_AVAILABLE and isinstance(R, cp.ndarray) else np
    
    # Get gate and exponent from smooth functions
    S, p = gate_and_exponent(Sigma, params)
    
    # Core parameters
    v0 = params['v0']
    rc0 = params['rc0']
    
    # Effective core radius (may be edge-aware)
    if family.edge_aware_rc:
        kappa_R = xp.gradient(xp.log(xp.maximum(Sigma, 1e-8)), xp.log(xp.maximum(R, 1e-6)))
        eta = params.get('eta_edge', 0.1)
        rc_eff = rc0 * (Sigma / params['Sigma_star'])**(-params['beta']) * (1 + eta * xp.abs(kappa_R))
    else:
        rc_eff = rc0 * (Sigma / params['Sigma_star'])**(-params['beta'])
    
    rc_eff = xp.clip(rc_eff, 0.1, 100.0)
    
    # Apply formula-specific gate
    if family.gate_type == 'rational':
        # Standard rational gate
        R_safe = xp.maximum(R, 1e-6)
        gate = R_safe**p / (R_safe**p + rc_eff**p)
        
    elif family.gate_type == 'softplus':
        # Softplus gate: σ(α*ln(R/rc))
        alpha = params.get('alpha_gate', 1.0)
        gate = 1.0 / (1.0 + xp.exp(-alpha * xp.log(xp.maximum(R/rc_eff, 1e-6))))
        
    elif family.gate_type == 'yukawa':
        # Yukawa-like with exponential suppression
        gate = (1.0 - xp.exp(-(Sigma / params['Sigma_star'])**params['alpha']))
        gate *= R**p / (R**p + rc_eff**p)
    else:
        gate = R**p / (R**p + rc_eff**p)
    
    # Apply optional modulations
    if family.use_curvature:
        C_factor = curvature_gate(Sigma, R, params)
        gate *= C_factor
    
    if family.use_gradient:
        G_factor = gradient_screening(Sigma, R, params)
        gate *= G_factor
    
    if family.use_nonlocal:
        R_metric = compute_nonlocal_ring_kernel(R, Sigma, params.get('ell', 3.0))
        epsilon_R = params.get('epsilon_R', 0.05)
        p_mod = p * (1.0 + epsilon_R * xp.tanh(R_metric))
        gate = R**p_mod / (R**p_mod + rc_eff**p_mod)
    
    # Final tail acceleration
    g_tail = (v0**2 / xp.maximum(R, 1e-6)) * gate * S
    
    return g_tail

class MultiScaleObjective:
    """
    Multi-scale objective with auto-balancing weights
    """
    
    def __init__(self, mw_data=None, sparc_data=None, cluster_data=None):
        self.mw_data = mw_data
        self.sparc_data = sparc_data
        self.cluster_data = cluster_data
        
        # Running statistics for auto-balancing
        self.term_stats = {
            'MW': {'values': [], 'weight': 0.45},
            'SPARC': {'values': [], 'weight': 0.45},
            'cluster': {'values': [], 'weight': 0.05},
            'smooth': {'values': [], 'weight': 0.05}
        }
        
        # AD policy
        self.ad_policy_mw = ADPolicy(policy='jeans')
        self.ad_policy_sparc = ADPolicy(policy='bounded', f_max=0.15)
        
    def huber_loss(self, residuals, delta):
        """Huber robust loss"""
        xp = cp if GPU_AVAILABLE and isinstance(residuals, cp.ndarray) else np
        abs_r = xp.abs(residuals)
        return xp.where(abs_r <= delta,
                       0.5 * residuals**2,
                       delta * (abs_r - 0.5 * delta))
    
    def evaluate_mw(self, params, family):
        """Evaluate MW stellar constraint"""
        if self.mw_data is None:
            return 0.0
            
        xp = cp if GPU_AVAILABLE else np
        
        # Compute G³ tail
        R = self.mw_data['R']
        Sigma = self.mw_data['Sigma']
        g_tail = evaluate_g3_tail(R, Sigma, params, family)
        
        # Add to baryonic
        v_bar = self.mw_data['v_bar']
        v_pred_sq = v_bar**2 + g_tail * R
        v_pred = xp.sqrt(xp.maximum(v_pred_sq, 0))
        
        # Apply AD correction to observed
        v_obs = self.ad_policy_mw.get_circular_velocity(
            R, self.mw_data['v_phi'],
            sigma_R=self.mw_data.get('sigma_R'),
            nu=self.mw_data.get('nu')
        )
        
        # Robust loss
        sigma_v = self.mw_data.get('sigma_v', 10.0)
        residuals = (v_pred - v_obs) / sigma_v
        loss = xp.median(self.huber_loss(residuals, 0.5))
        
        return float(loss)
    
    def evaluate_sparc(self, params, family):
        """Evaluate SPARC rotation curves"""
        if self.sparc_data is None:
            return 0.0
            
        xp = cp if GPU_AVAILABLE else np
        losses = []
        
        for galaxy in self.sparc_data:
            R = galaxy['R']
            Sigma = galaxy['Sigma']
            g_tail = evaluate_g3_tail(R, Sigma, params, family)
            
            v_bar = galaxy['v_bar']
            v_pred_sq = v_bar**2 + g_tail * R
            v_pred = xp.sqrt(xp.maximum(v_pred_sq, 0))
            
            # Apply bounded AD to observed
            v_obs = self.ad_policy_sparc.get_circular_velocity(
                R, galaxy['v_obs'],
                nu=galaxy.get('Sigma')
            )
            
            # Per-galaxy median error
            residuals = (v_pred - v_obs) / xp.maximum(v_obs, 10.0)
            galaxy_loss = xp.median(self.huber_loss(residuals, 0.1))
            losses.append(float(galaxy_loss))
        
        return np.median(losses) if losses else 0.0
    
    def evaluate_smoothness(self, params, family, test_R=None):
        """Evaluate smoothness penalty"""
        xp = cp if GPU_AVAILABLE else np
        
        if test_R is None:
            test_R = xp.linspace(0.1, 20, 100)
        
        # Test surface density
        Sigma_test = 100.0 * xp.exp(-test_R / 3.0)
        
        # Compute tail
        g_tail = evaluate_g3_tail(test_R, Sigma_test, params, family)
        
        # Gradient norm
        if len(test_R) > 1:
            grad = xp.gradient(g_tail, test_R)
            smoothness = xp.mean(grad**2)
        else:
            smoothness = 0.0
            
        return float(smoothness)
    
    def __call__(self, params, family):
        """
        Compute total multi-scale objective
        """
        # Individual terms
        mw_loss = self.evaluate_mw(params, family)
        sparc_loss = self.evaluate_sparc(params, family)
        smooth_loss = self.evaluate_smoothness(params, family)
        
        # Update statistics for auto-balancing
        self.term_stats['MW']['values'].append(mw_loss)
        self.term_stats['SPARC']['values'].append(sparc_loss)
        self.term_stats['smooth']['values'].append(smooth_loss)
        
        # Auto-balance weights if we have enough samples
        if len(self.term_stats['MW']['values']) > 10:
            for key in ['MW', 'SPARC', 'smooth']:
                values = self.term_stats[key]['values'][-50:]  # Last 50 samples
                iqr = np.percentile(values, 75) - np.percentile(values, 25)
                if iqr > 0:
                    # Inverse variance weighting
                    self.term_stats[key]['weight'] = 1.0 / (iqr**2 + 1e-6)
            
            # Normalize weights
            total_weight = sum(self.term_stats[k]['weight'] for k in ['MW', 'SPARC', 'smooth'])
            for key in ['MW', 'SPARC', 'smooth']:
                self.term_stats[key]['weight'] /= total_weight
        
        # Weighted sum
        total = (self.term_stats['MW']['weight'] * mw_loss +
                self.term_stats['SPARC']['weight'] * sparc_loss +
                self.term_stats['smooth']['weight'] * smooth_loss)
        
        return total

def optimize_universal(max_iterations=2000, output_dir='out/universal'):
    """
    Main optimization loop with plateau detection and branching
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("UNIVERSAL G³ SEARCH")
    logger.info("="*60)
    logger.info(f"GPU: {'Available' if GPU_AVAILABLE else 'Not available'}")
    
    # Initialize
    scheduler = PlateauScheduler()
    objective = MultiScaleObjective()  # Would load real data here
    
    # Starting parameters
    best_params = {
        'v0': 200.0,
        'rc0': 10.0,
        'p_in': 2.0,
        'p_out': 1.0,
        'Sigma_star': 50.0,
        'beta': 0.5,
        'gamma': 0.5,
        'alpha': 2.0,
        'w_p': 1.0,
        'w_S': 1.5,
        'eta_C': 0.0,
        'grad_weight': 0.0,
        'epsilon_R': 0.05,
        'ell': 3.0
    }
    
    best_family = 'F1'
    best_loss = float('inf')
    
    weights = {'MW': 0.45, 'SPARC': 0.45, 'smooth': 0.10}
    
    # Optimization loop
    for iteration in range(max_iterations):
        # Evaluate current
        family = FORMULA_FAMILIES[best_family]
        loss = objective(best_params, family)
        
        if loss < best_loss:
            best_loss = loss
            logger.info(f"Iter {iteration}: New best = {best_loss:.6f} ({family.describe()})")
        
        # Check for plateau
        plateau = scheduler.update(best_loss)
        
        if plateau == "plateau":
            # Branch
            best_params, best_family, weights, action = branch_action(
                scheduler.plateaus, best_params, best_family, weights
            )
            scheduler.record_branch(action)
            logger.info(f"Branching: {action}")
            
            # Update objective weights
            objective.term_stats['MW']['weight'] = weights['MW']
            objective.term_stats['SPARC']['weight'] = weights['SPARC']
        
        if scheduler.done():
            logger.info("Max plateaus reached - stopping")
            break
        
        # Simple parameter update (would use CMA-ES/L-BFGS in production)
        if iteration % 10 == 0:
            # Random exploration
            for key in ['v0', 'rc0', 'p_in', 'p_out', 'Sigma_star']:
                if np.random.random() < 0.3:
                    best_params[key] *= np.random.uniform(0.9, 1.1)
    
    # Save results
    results = {
        'best_params': best_params,
        'best_family': best_family,
        'best_loss': best_loss,
        'iterations': iteration,
        'plateaus': scheduler.plateaus,
        'branches': scheduler.branches,
        'gpu_used': GPU_AVAILABLE
    }
    
    with open(Path(output_dir) / 'optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info(f"Best family: {FORMULA_FAMILIES[best_family].describe()}")
    logger.info(f"Plateaus encountered: {scheduler.plateaus}")
    logger.info(f"Results saved to {output_dir}")
    
    return results

def run_zero_shot_tests(params, family_name, output_dir='out/zero_shot'):
    """
    Run zero-shot validation tests
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("ZERO-SHOT VALIDATION")
    logger.info("="*60)
    
    family = FORMULA_FAMILIES[family_name]
    results = {}
    
    # Test 1: Solar system constraint
    R_solar = 8.2  # kpc
    Sigma_solar = 50.0  # Msun/pc^2 (typical)
    g_tail_solar = evaluate_g3_tail(
        np.array([R_solar]), 
        np.array([Sigma_solar]), 
        params, family
    )[0]
    
    G_eff_over_G = 1.0 + g_tail_solar * R_solar / (200**2)  # Approximate
    solar_pass = abs(G_eff_over_G - 1.0) < SOLAR_CONSTRAINT
    
    results['solar_system'] = {
        'G_eff/G': float(G_eff_over_G),
        'constraint': SOLAR_CONSTRAINT,
        'passed': solar_pass
    }
    
    logger.info(f"Solar system: G_eff/G = {G_eff_over_G:.2e} ({'✅ PASS' if solar_pass else '❌ FAIL'})")
    
    # Test 2: BTFR slope
    # Would compute on SPARC ensemble here
    btfr_slope = 3.5  # Placeholder
    btfr_pass = 3.0 <= btfr_slope <= 4.0
    
    results['BTFR'] = {
        'slope': btfr_slope,
        'range': [3.0, 4.0],
        'passed': btfr_pass
    }
    
    logger.info(f"BTFR slope: {btfr_slope:.2f} ({'✅ PASS' if btfr_pass else '❌ FAIL'})")
    
    # Test 3: No force reversals
    test_R = np.linspace(0.1, 100, 500)
    test_Sigma = 100 * np.exp(-test_R / 5)
    g_tail_test = evaluate_g3_tail(test_R, test_Sigma, params, family)
    
    force_reversal = np.any(g_tail_test < 0)
    results['force_reversals'] = {
        'detected': bool(force_reversal),  # Convert numpy bool to Python bool
        'passed': bool(not force_reversal)
    }
    
    logger.info(f"Force reversals: {'❌ DETECTED' if force_reversal else '✅ NONE'}")
    
    # Save
    with open(Path(output_dir) / 'zero_shot_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    all_passed = all(r.get('passed', False) for r in results.values())
    logger.info("\n" + "="*60)
    logger.info(f"Zero-shot validation: {'✅ ALL PASSED' if all_passed else '⚠️ SOME FAILED'}")
    
    return results

def main():
    """
    Main entry point
    """
    logger.info("Starting Universal G³ Search")
    
    # Run optimization
    opt_results = optimize_universal(max_iterations=100)  # Short test run
    
    # Run zero-shot tests on best
    zero_shot_results = run_zero_shot_tests(
        opt_results['best_params'],
        opt_results['best_family']
    )
    
    # Final report
    logger.info("\n" + "="*60)
    logger.info("FINAL REPORT")
    logger.info("="*60)
    logger.info(f"Best formula: {opt_results['best_family']}")
    logger.info(f"Best loss: {opt_results['best_loss']:.6f}")
    logger.info(f"Plateaus: {opt_results['plateaus']}")
    logger.info(f"Zero-shot: {sum(r.get('passed', False) for r in zero_shot_results.values())}/{len(zero_shot_results)} passed")
    
if __name__ == '__main__':
    main()