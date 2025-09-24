#!/usr/bin/env python3
"""
Unified G³ Global Law - Single formula that derives inner/outer behavior from baryon geometry.
NO per-galaxy parameters allowed - only one global parameter vector Θ.
"""

import numpy as np
import cupy as cp
import json
import hashlib
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class UnifiedG3Model:
    """
    Single global G³ law that produces inner/outer behavior from geometry alone.
    
    Global parameters Θ:
    - v0: Asymptotic velocity scale [km/s]
    - rc0: Reference core radius [kpc]
    - gamma: Half-mass radius scaling exponent
    - beta: Mean density scaling exponent
    - Sigma_star: Critical screening density [Msun/pc²]
    - alpha: Screening power
    - kappa: Screening exponent
    - eta: Transition radius factor (multiplies r_half)
    - Delta: Transition width [kpc]
    - p_in: Inner exponent (~2)
    - p_out: Outer exponent (~1)
    """
    
    def __init__(self, theta: Optional[Dict] = None):
        """Initialize with global parameters"""
        if theta is None:
            # Default starting values based on MW analysis
            self.theta = {
                'v0': 265.0,        # km/s
                'rc0': 4.0,         # kpc
                'gamma': 0.5,       # r_half scaling
                'beta': 0.3,        # density scaling
                'Sigma_star': 50.0, # Msun/pc²
                'alpha': 1.5,       # screening power
                'kappa': 1.0,       # screening exponent
                'eta': 1.2,         # transition at eta * r_half
                'Delta': 2.0,       # transition width kpc
                'p_in': 2.0,        # inner exponent
                'p_out': 1.0        # outer exponent
            }
        else:
            self.theta = theta
            
        # Lock the parameters with SHA hash
        self.theta_hash = self._compute_hash()
        
    def _compute_hash(self) -> str:
        """Compute SHA256 of parameter vector for verification"""
        theta_str = json.dumps(self.theta, sort_keys=True)
        return hashlib.sha256(theta_str.encode()).hexdigest()[:16]
    
    def compute_baryon_properties(self, R, z, Sigma_loc, M_baryon=None):
        """
        Compute galaxy properties from baryon distribution.
        
        Returns:
            r_half: Half-mass radius [kpc]
            Sigma_mean: Mean surface density [Msun/pc²]
        """
        # Compute half-mass radius (simplified - in practice use cumulative mass)
        weights = Sigma_loc * R  # Weighted by cylindrical area element
        r_half = np.sum(R * weights) / np.sum(weights)
        
        # Mean surface density within 2 * r_half
        mask = R < 2 * r_half
        if np.any(mask):
            Sigma_mean = np.mean(Sigma_loc[mask])
        else:
            Sigma_mean = np.mean(Sigma_loc)
            
        return r_half, Sigma_mean
    
    def g_tail_unified(self, R, z, Sigma_loc, gN, 
                      r_half=None, Sigma_mean=None,
                      use_gpu=False):
        """
        Unified G³ tail acceleration.
        
        Args:
            R: Galactocentric radius [kpc]
            z: Height above plane [kpc]
            Sigma_loc: Local surface density [Msun/pc²]
            gN: Newtonian acceleration [km²/s²/kpc]
            r_half: Half-mass radius [kpc] (computed if None)
            Sigma_mean: Mean surface density [Msun/pc²] (computed if None)
            use_gpu: Use GPU acceleration
            
        Returns:
            g_total: Total acceleration including G³ tail
        """
        # Move to GPU if requested
        if use_gpu:
            R = cp.asarray(R)
            z = cp.asarray(z)
            Sigma_loc = cp.asarray(Sigma_loc)
            gN = cp.asarray(gN)
            xp = cp
        else:
            xp = np
            
        # Compute baryon properties if not provided
        if r_half is None or Sigma_mean is None:
            r_half_comp, Sigma_mean_comp = self.compute_baryon_properties(
                R.get() if use_gpu else R,
                z.get() if use_gpu else z,
                Sigma_loc.get() if use_gpu else Sigma_loc
            )
            if r_half is None:
                r_half = r_half_comp
            if Sigma_mean is None:
                Sigma_mean = Sigma_mean_comp
                
        # Reference values for normalization
        r_ref = 8.0  # kpc (solar radius reference)
        Sigma0 = 50.0  # Msun/pc² (typical disk density)
        
        # 1. Geometry-aware core radius
        rc_eff = self.theta['rc0'] * (
            (r_half / r_ref) ** self.theta['gamma'] *
            (Sigma_mean / Sigma0) ** (-self.theta['beta'])
        )
        
        # 2. Radius-dependent exponent (smooth transition)
        transition_R = self.theta['eta'] * r_half
        sigmoid_arg = (transition_R - R) / self.theta['Delta']
        sigmoid = 1.0 / (1.0 + xp.exp(-sigmoid_arg))
        p_R = self.theta['p_out'] + (self.theta['p_in'] - self.theta['p_out']) * sigmoid
        
        # 3. Shape function with variable exponent
        # Use power method for numerical stability
        shape = xp.power(R, p_R) / (xp.power(R, p_R) + xp.power(rc_eff, p_R))
        
        # 4. Density screening
        Sigma_ratio = self.theta['Sigma_star'] / (Sigma_loc + 1e-6)
        screening = xp.power(1.0 + xp.power(Sigma_ratio, self.theta['kappa']), 
                            -self.theta['alpha'])
        
        # 5. Combine into tail acceleration
        g_tail = (self.theta['v0']**2 / R) * shape * screening
        
        # 6. Total acceleration
        g_total = gN + g_tail
        
        if use_gpu:
            return g_total.get()
        return g_total
    
    def predict_velocity(self, R, z, Sigma_loc, gN, 
                        r_half=None, Sigma_mean=None,
                        use_gpu=False):
        """Predict circular velocity from unified model"""
        g_total = self.g_tail_unified(R, z, Sigma_loc, gN, 
                                     r_half, Sigma_mean, use_gpu)
        v_pred = np.sqrt(np.maximum(R * g_total, 0.0))
        return v_pred
    
    def save_parameters(self, filepath):
        """Save global parameters with hash verification"""
        output = {
            'model': 'Unified G³ Global',
            'timestamp': datetime.now().isoformat(),
            'theta': self.theta,
            'theta_hash': self.theta_hash,
            'locked': True,
            'description': 'Global parameters - NO per-galaxy tuning allowed'
        }
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved global Θ with hash {self.theta_hash} to {filepath}")
        
    def load_parameters(self, filepath, verify_hash=True):
        """Load and verify global parameters"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if verify_hash:
            loaded_hash = data.get('theta_hash')
            self.theta = data['theta']
            computed_hash = self._compute_hash()
            
            if loaded_hash != computed_hash:
                raise ValueError(f"Parameter hash mismatch! File: {loaded_hash}, Computed: {computed_hash}")
                
        self.theta = data['theta']
        self.theta_hash = self._compute_hash()
        print(f"Loaded global Θ with verified hash {self.theta_hash}")
        
    def assert_no_overrides(self, **kwargs):
        """Assert that no per-galaxy parameter overrides are being passed"""
        forbidden = ['v0', 'rc', 'rc0', 'v0_inner', 'v0_outer', 'rc_inner', 'rc_outer']
        for key in forbidden:
            if key in kwargs:
                raise ValueError(f"Per-galaxy parameter '{key}' not allowed! Only global Θ permitted.")
                
    def audit_log(self, R, z, Sigma_loc, r_half, Sigma_mean, name="Galaxy"):
        """Print audit log showing derived quantities"""
        print(f"\n--- Audit Log for {name} ---")
        print(f"Hash: {self.theta_hash}")
        print(f"r_half = {r_half:.2f} kpc (from baryons)")
        print(f"Σ_mean = {Sigma_mean:.1f} Msun/pc² (from baryons)")
        
        # Sample radii
        R_sample = [0.5*r_half, r_half, 1.5*r_half, 2*r_half, 3*r_half]
        for Rs in R_sample:
            idx = np.argmin(np.abs(R - Rs))
            if idx < len(R):
                transition_R = self.theta['eta'] * r_half
                sigmoid_arg = (transition_R - Rs) / self.theta['Delta']
                sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_arg))
                p_R = self.theta['p_out'] + (self.theta['p_in'] - self.theta['p_out']) * sigmoid
                
                rc_eff = self.theta['rc0'] * (
                    (r_half / 8.0) ** self.theta['gamma'] *
                    (Sigma_mean / 50.0) ** (-self.theta['beta'])
                )
                
                print(f"  R={Rs:.1f} kpc: p(R)={p_R:.2f}, rc_eff={rc_eff:.2f}, Σ_loc={Sigma_loc[idx]:.1f}")
                
        print(f"NO per-galaxy parameters used!")


def optimize_global_theta(train_data, val_data=None, 
                         max_iter=1000, pop_size=128,
                         use_gpu=True, variant='unified'):
    """
    Optimize global parameters Θ on training data.
    NO per-galaxy parameters allowed.
    """
    import cupy as cp
    from cupyx.scipy.special import expit as cp_expit
    from scipy.stats import qmc
    
    print("\n" + "="*80)
    print(" GLOBAL PARAMETER OPTIMIZATION ")
    print("="*80)
    print(f"Training on {len(train_data['R'])} points")
    if val_data:
        print(f"Validating on {len(val_data['R'])} points")
    print("Variant: " + variant)
    print("NO per-galaxy parameters allowed!")
    
    # Parameter bounds
    bounds = [
        (150, 350),   # v0 [km/s]
        (1.0, 10.0),  # rc0 [kpc]
        (0.1, 2.0),   # gamma (r_half scaling)
        (0.0, 1.0),   # beta (density scaling)
        (10, 200),    # Sigma_star [Msun/pc²]
        (0.5, 3.0),   # alpha (screening power)
        (0.5, 2.0),   # kappa (screening exponent)
        (0.8, 2.0),   # eta (transition factor)
        (0.5, 5.0),   # Delta (transition width)
        (1.5, 2.5),   # p_in (inner exponent)
        (0.8, 1.2),   # p_out (outer exponent)
    ]
    
    n_params = len(bounds)
    param_names = ['v0', 'rc0', 'gamma', 'beta', 'Sigma_star', 'alpha', 
                   'kappa', 'eta', 'Delta', 'p_in', 'p_out']
    
    # Initialize population with Latin Hypercube
    sampler = qmc.LatinHypercube(d=n_params)
    sample = sampler.random(n=pop_size)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    population = qmc.scale(sample, lower, upper)
    
    # Move data to GPU if requested
    if use_gpu:
        R_gpu = cp.asarray(train_data['R'], dtype=cp.float32)
        z_gpu = cp.asarray(train_data['z'], dtype=cp.float32)
        v_obs_gpu = cp.asarray(train_data['v_obs'], dtype=cp.float32)
        Sigma_loc_gpu = cp.asarray(train_data['Sigma_loc'], dtype=cp.float32)
        gN_gpu = cp.asarray(train_data['gN'], dtype=cp.float32)
        
        # Precompute baryon properties
        weights = Sigma_loc_gpu * R_gpu
        r_half_gpu = cp.sum(R_gpu * weights) / cp.sum(weights)
        mask = R_gpu < 2 * r_half_gpu
        Sigma_mean_gpu = cp.mean(Sigma_loc_gpu[mask])
        
    # Evolution loop
    best_params = None
    best_loss = np.inf
    history = []
    
    for generation in range(max_iter):
        # Evaluate population
        losses = []
        for i in range(pop_size):
            theta_test = dict(zip(param_names, population[i]))
            model_test = UnifiedG3Model(theta_test)
            
            if use_gpu:
                # GPU evaluation
                v_pred = model_test.predict_velocity(
                    R_gpu.get(), z_gpu.get(), Sigma_loc_gpu.get(), gN_gpu.get(),
                    r_half_gpu.get(), Sigma_mean_gpu.get(),
                    use_gpu=False
                )
                v_pred_gpu = cp.asarray(v_pred)
                rel_error = cp.abs((v_pred_gpu - v_obs_gpu) / v_obs_gpu)
                loss = float(cp.median(rel_error))
            else:
                # CPU evaluation
                v_pred = model_test.predict_velocity(
                    train_data['R'], train_data['z'], 
                    train_data['Sigma_loc'], train_data['gN']
                )
                rel_error = np.abs((v_pred - train_data['v_obs']) / train_data['v_obs'])
                loss = np.median(rel_error)
                
            losses.append(loss)
            
            if loss < best_loss:
                best_loss = loss
                best_params = theta_test.copy()
                
        # Log progress
        if generation % 50 == 0:
            print(f"Gen {generation:4d}: Best loss = {best_loss:.4f} ({best_loss*100:.2f}% error)")
            
        history.append(best_loss)
        
        # Early stopping
        if len(history) > 200 and max(history[-200:]) - min(history[-200:]) < 1e-5:
            print(f"Converged at generation {generation}")
            break
            
        # Differential evolution update (simplified)
        for i in range(pop_size):
            if np.random.rand() < 0.5:  # 50% chance to update
                # Random mutation
                idx = np.random.choice(pop_size, 3, replace=False)
                a, b, c = population[idx]
                mutant = a + 0.8 * (b - c)
                
                # Boundary constraints
                mutant = np.clip(mutant, lower, upper)
                
                # Test mutant
                theta_mutant = dict(zip(param_names, mutant))
                model_mutant = UnifiedG3Model(theta_mutant)
                
                if use_gpu:
                    v_pred = model_mutant.predict_velocity(
                        R_gpu.get(), z_gpu.get(), Sigma_loc_gpu.get(), gN_gpu.get(),
                        r_half_gpu.get(), Sigma_mean_gpu.get(),
                        use_gpu=False
                    )
                    v_pred_gpu = cp.asarray(v_pred)
                    rel_error = cp.abs((v_pred_gpu - v_obs_gpu) / v_obs_gpu)
                    loss_mutant = float(cp.median(rel_error))
                else:
                    v_pred = model_mutant.predict_velocity(
                        train_data['R'], train_data['z'], 
                        train_data['Sigma_loc'], train_data['gN']
                    )
                    rel_error = np.abs((v_pred - train_data['v_obs']) / train_data['v_obs'])
                    loss_mutant = np.median(rel_error)
                    
                if loss_mutant < losses[i]:
                    population[i] = mutant
                    losses[i] = loss_mutant
    
    print("\n" + "="*80)
    print(" OPTIMIZATION COMPLETE ")
    print("="*80)
    print(f"Final loss: {best_loss:.4f} ({best_loss*100:.2f}% median error)")
    print("\nBest parameters:")
    for name, value in best_params.items():
        print(f"  {name:12s} = {value:.3f}")
        
    return best_params, history


def main():
    """Test the unified model"""
    print("\n" + "="*80)
    print(" UNIFIED G³ GLOBAL LAW TEST ")
    print("="*80)
    
    # Load Milky Way data
    print("\nLoading Milky Way data...")
    data = np.load('data/mw_gaia_144k.npz')
    
    train_data = {
        'R': data['R_kpc'],
        'z': data['z_kpc'],
        'v_obs': data['v_obs_kms'],
        'Sigma_loc': data['Sigma_loc_Msun_pc2'],
        'gN': data['gN_kms2_per_kpc']
    }
    
    print(f"Loaded {len(train_data['R']):,} stars")
    
    # Create model with default parameters
    model = UnifiedG3Model()
    print(f"\nModel initialized with hash: {model.theta_hash}")
    
    # Test no-override assertion
    print("\nTesting override protection...")
    try:
        model.assert_no_overrides(v0=300)  # Should fail
    except ValueError as e:
        print(f"✓ Override blocked: {e}")
        
    # Compute baryon properties
    r_half, Sigma_mean = model.compute_baryon_properties(
        train_data['R'], train_data['z'], train_data['Sigma_loc']
    )
    print(f"\nBaryon properties:")
    print(f"  r_half = {r_half:.2f} kpc")
    print(f"  Σ_mean = {Sigma_mean:.1f} Msun/pc²")
    
    # Test prediction
    print("\nTesting prediction...")
    v_pred = model.predict_velocity(
        train_data['R'][:1000], train_data['z'][:1000],
        train_data['Sigma_loc'][:1000], train_data['gN'][:1000],
        r_half, Sigma_mean
    )
    
    rel_error = np.abs((v_pred - train_data['v_obs'][:1000]) / train_data['v_obs'][:1000]) * 100
    print(f"Test on 1000 stars: {np.median(rel_error):.1f}% median error")
    
    # Audit log
    model.audit_log(train_data['R'], train_data['z'], train_data['Sigma_loc'],
                   r_half, Sigma_mean, "Milky Way")
    
    # Save parameters
    model.save_parameters('out/mw_orchestrated/unified_global_theta.json')
    
    print("\n" + "="*80)
    print(" UNIFIED MODEL TEST COMPLETE ")
    print("="*80)
    
    return model


if __name__ == '__main__':
    model = main()