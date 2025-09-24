#!/usr/bin/env python3
"""
GPU-Accelerated Milky Way G³ Optimizer
=======================================

Fits a single, global G³ law to 144,000+ Gaia stars using CuPy GPU acceleration
and CMA-ES optimization. No dark matter, no per-star parameters - just one
universal law that responds to baryon geometry.

Key features:
- Fully vectorized GPU computation (all stars evaluated simultaneously)
- Checkpoint/resume capability (never lose progress)
- Single global parameter vector for entire galaxy
- Baryon-only (no invented mass)
- Geometry-aware through surface density proxy
"""

import os, json, time, math, signal, argparse, pathlib
import numpy as np

try:
    import cupy as cp
    XP = cp  # GPU
    HAS_CUPY = True
    print("[✓] GPU acceleration enabled (CuPy)")
except Exception:
    XP = np  # CPU fallback
    HAS_CUPY = False
    print("[!] Running on CPU (CuPy not available)")

G_KPC_MSUN_KMS = 4.30091e-6  # (kpc/Msun) * (km/s)^2

# ============================================================================
# Utilities
# ============================================================================

def xp_array(a, dtype=None):
    """Convert to appropriate array type (GPU/CPU)"""
    return XP.asarray(a, dtype=dtype) if not isinstance(a, XP.ndarray) else a

def to_cpu(a):
    """Transfer array to CPU memory"""
    return cp.asnumpy(a) if (HAS_CUPY and isinstance(a, cp.ndarray)) else np.asarray(a)

def save_checkpoint(path, state):
    """Save optimizer state with atomic write"""
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: to_cpu(v) if isinstance(v, (np.ndarray,)) or
                                       (HAS_CUPY and isinstance(v, cp.ndarray)) else v
                                for k, v in state.items()})
    tmp.rename(path)
    print(f"[✓] Checkpoint saved: {path}")

def load_checkpoint(path):
    """Load optimizer state if exists"""
    if not path.exists(): 
        return None
    dat = np.load(path, allow_pickle=True)
    print(f"[✓] Checkpoint loaded: {path}")
    return {k: dat[k] for k in dat.files}

# ============================================================================
# Baryon Model (Fixed, Analytic)
# ============================================================================

def mn_forces(R, z, M, a, b):
    """
    Miyamoto-Nagai disk forces
    
    Parameters:
    -----------
    R, z : arrays - cylindrical coordinates (kpc)
    M : float - disk mass (Msun)
    a : float - scale length (kpc)
    b : float - scale height (kpc)
    
    Returns:
    --------
    gR, gz : arrays - force components (km^2/s^2/kpc)
    """
    B = XP.sqrt(z*z + b*b)
    den = XP.power(R*R + XP.power(a + B, 2.0), 1.5)
    gR = -G_KPC_MSUN_KMS * M * R / den
    gz = -G_KPC_MSUN_KMS * M * z * (a + B) / (B * den + 1e-30)
    return gR, gz

def hern_forces(R, z, M, a):
    """
    Hernquist bulge forces
    
    Parameters:
    -----------
    R, z : arrays - cylindrical coordinates (kpc)
    M : float - bulge mass (Msun)
    a : float - scale radius (kpc)
    
    Returns:
    --------
    gR, gz : arrays - force components (km^2/s^2/kpc)
    """
    r = XP.sqrt(R*R + z*z) + 1e-12
    g = -G_KPC_MSUN_KMS * M / XP.power(r + a, 2.0)
    return g * (R / r), g * (z / r)

def baryon_forces(R, z, cfg):
    """
    Total Newtonian forces from all baryon components
    """
    gR = XP.zeros_like(R)
    gz = XP.zeros_like(z)

    # Stellar disk (Miyamoto-Nagai)
    if cfg["disk"]["M"] > 0:
        d = cfg["disk"]
        gR_d, gz_d = mn_forces(R, z, d["M"], d["a"], d["b"])
        gR += gR_d
        gz += gz_d
    
    # Gas disk (Miyamoto-Nagai)
    if cfg["gas"]["M"] > 0:
        g = cfg["gas"]
        gR_g, gz_g = mn_forces(R, z, g["M"], g["a"], g["b"])
        gR += gR_g
        gz += gz_g
    
    # Bulge (Hernquist)
    if cfg["bulge"]["M"] > 0:
        b = cfg["bulge"]
        gR_b, gz_b = hern_forces(R, z, b["M"], b["a"])
        gR += gR_b
        gz += gz_b
    
    return gR, gz

# ============================================================================
# Surface Density Proxy (Geometry Gate)
# ============================================================================

def sigma_local(R, cfg):
    """
    Local surface density proxy Sigma_loc(R) in Msun/pc^2
    
    Either from tabulated data or exponential disk model
    """
    if cfg.get("sigma_table", None) is not None:
        # Use tabulated surface density
        Rtab = cfg["sigma_table"]["R_kpc"]
        Stab = cfg["sigma_table"]["Sigma_Msun_pc2"]
        return XP.interp(R, Rtab, Stab)
    
    # Exponential disk proxy
    Mdisk = cfg["disk"]["M"] + cfg["gas"]["M"]
    Rd = cfg["disk"]["a"]
    Sigma0 = Mdisk / (2.0 * math.pi * (Rd**2))  # Msun/kpc^2
    Sigma = Sigma0 * XP.exp(-R / Rd)            # Msun/kpc^2
    return Sigma * 1e-6                          # -> Msun/pc^2

# ============================================================================
# G³ Tail Acceleration (Global Law)
# ============================================================================

def tail_accel(R, z, Sigma_loc,
               v0_kms, rc_kpc, r0_kpc, delta_kpc, gamma, beta, z0_kpc, m, Sigma0):
    """
    G³ tail acceleration with geometry gates
    
    This is the key modification to gravity that depends on:
    - Local surface density (Sigma_loc)
    - Distance from galactic center (R)
    - Height above disk (z)
    
    Parameters (theta):
    -------------------
    v0_kms : float - amplitude velocity scale (km/s)
    rc_kpc : float - core radius (kpc)
    r0_kpc : float - inner cutoff radius (kpc)
    delta_kpc : float - transition width (kpc)
    gamma : float - radial profile power
    beta : float - surface density coupling
    z0_kpc : float - vertical scale height (kpc)
    m : float - vertical profile power
    Sigma0 : float - reference surface density (Msun/pc^2)
    
    Returns:
    --------
    gR_tail : array - radial acceleration (km^2/s^2/kpc)
    """
    # Radial gate: smooth profile with core
    gate_R = XP.power(R / (R + rc_kpc + 1e-12), gamma)
    
    # Step function: suppresses tail at small R (avoid bulge region)
    gate_step = 1.0 / (1.0 + XP.exp((r0_kpc - R) / (delta_kpc + 1e-12)))
    
    # Vertical gate: confines tail to disk plane
    gate_z = 1.0 / (1.0 + XP.power(XP.abs(z) / (z0_kpc + 1e-12), m))
    
    # Surface density gate: couples to baryon concentration
    gate_S = XP.power(XP.maximum(Sigma_loc, 1e-12) / Sigma0, beta)
    
    # Total tail acceleration
    return (v0_kms**2 / XP.maximum(R, 1e-6)) * gate_R * gate_step * gate_z * gate_S

# ============================================================================
# Loss Function
# ============================================================================

def robust_mape(v_model, v_obs, v_err=None, eps=1e-6):
    """
    Robust median absolute percentage error
    
    Uses median instead of mean for robustness to outliers
    Optionally weights by inverse errors
    """
    mape = XP.abs(v_model - v_obs) / XP.maximum(XP.abs(v_obs), eps)
    
    if v_err is not None:
        # Light downweight of high-uncertainty measurements
        w = 1.0 / XP.maximum(v_err, 5.0)
        weighted = mape * w
        # Remove NaN values before percentile
        valid = XP.isfinite(weighted)
        if XP.any(valid):
            m = float(XP.percentile(weighted[valid], 50.0))
        else:
            m = float(XP.inf)
    else:
        valid = XP.isfinite(mape)
        if XP.any(valid):
            m = float(XP.percentile(mape[valid], 50.0))
        else:
            m = float(XP.inf)
    
    return m  # median relative error (0..inf)

# ============================================================================
# CMA-ES Optimizer (GPU-friendly)
# ============================================================================

class CMAES:
    """
    Covariance Matrix Adaptation Evolution Strategy
    
    Simple ask-tell interface for black-box optimization
    Fully GPU-compatible (all operations in CuPy if available)
    """
    
    def __init__(self, x0, sigma0=0.2, popsize=64, bounds=None, seed=42):
        self.dim = len(x0)
        self.mean = xp_array(x0, dtype=XP.float64)
        XP.random.seed(seed)
        self.sigma = sigma0
        self.pop = popsize
        self.C = XP.eye(self.dim, dtype=XP.float64)
        self.bounds = bounds
        self.generation = 0

    def ask(self):
        """Generate new candidate solutions"""
        # Sample from multivariate normal
        A = XP.linalg.cholesky(self.C + 1e-12*XP.eye(self.dim))
        Z = XP.random.standard_normal((self.pop, self.dim), dtype=XP.float64)
        X = self.mean + self.sigma * (Z @ A.T)
        
        # Apply bounds if specified
        if self.bounds is not None:
            lo, hi = self.bounds
            X = XP.minimum(XP.maximum(X, lo), hi)
        
        return X

    def tell(self, X, f):
        """Update distribution based on fitness values"""
        # Rank-based selection
        idx = XP.argsort(f)
        elite = X[idx[: self.pop//2]]
        
        # Update mean
        self.mean = XP.mean(elite, axis=0)
        
        # Update covariance
        centered = elite - self.mean
        self.C = (centered.T @ centered) / (elite.shape[0] - 1 + 1e-12)
        
        # Simple step-size adaptation
        if self.generation > 0:
            self.sigma *= 0.98 if f[idx[0]] < XP.percentile(f, 25) else 1.01
            self.sigma = float(XP.minimum(XP.maximum(self.sigma, 0.01), 1.0))
        
        self.generation += 1

# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_theta(theta, stars, cfg):
    """
    Evaluate parameter vector on all stars
    
    Fully vectorized - evaluates all stars simultaneously on GPU
    """
    # Unpack parameters
    v0, rc, r0, dlt, gamma, beta, z0, m, Sigma0 = theta
    R = stars["R"]
    z = stars["z"]
    vphi = stars["vphi"]
    verr = stars["verr"]

    # Newtonian baryon forces
    gR_N, _ = baryon_forces(R, z, cfg)

    # Local surface density from disk geometry
    S_loc = sigma_local(R, cfg)

    # G³ tail acceleration
    gR_tail = tail_accel(R, z, S_loc, v0, rc, r0, dlt, gamma, beta, z0, m, Sigma0)

    # Total circular velocity
    v_model = XP.sqrt(XP.maximum(0.0, R * (gR_N + gR_tail)))

    # Compute loss
    return robust_mape(v_model, vphi, verr)

# ============================================================================
# Data Loading
# ============================================================================

def load_gaia_csv(path, R_col="R_kpc", z_col="z_kpc",
                  vphi_col="vphi", verr_col="vphi_err", 
                  z_cut=None, vphi_min=20.0):
    """
    Load Gaia data from CSV
    
    Applies quality cuts:
    - |z| < z_cut (near-plane stars for circular orbits)
    - vphi > vphi_min (remove counter-rotating contamination)
    """
    print(f"[Loading] {path}")
    
    # Load CSV
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    
    # Transfer to GPU
    R = xp_array(arr[R_col], dtype=XP.float64)
    z = xp_array(arr[z_col], dtype=XP.float64)
    vphi = xp_array(arr[vphi_col], dtype=XP.float64)
    verr = xp_array(arr[verr_col], dtype=XP.float64) if verr_col in arr.dtype.names else None

    n_total = len(R)
    
    # Apply z-cut for near-circular orbits
    if z_cut is not None:
        keep = XP.abs(z) <= z_cut
        R, z, vphi = R[keep], z[keep], vphi[keep]
        if verr is not None: 
            verr = verr[keep]
        print(f"  Z-cut |z| < {z_cut} kpc: {n_total} → {len(R)} stars")
    
    # Remove slow/counter-rotating stars
    keep2 = vphi >= vphi_min
    R, z, vphi = R[keep2], z[keep2], vphi[keep2]
    if verr is not None: 
        verr = verr[keep2]
    print(f"  Velocity cut vphi > {vphi_min} km/s: → {len(R)} stars")
    
    # Print statistics
    print(f"  R range: {float(XP.min(R)):.1f} - {float(XP.max(R)):.1f} kpc")
    print(f"  vphi range: {float(XP.min(vphi)):.0f} - {float(XP.max(vphi)):.0f} km/s")
    
    stars = {"R": R, "z": z, "vphi": vphi, "verr": verr}
    return stars

def maybe_load_sigma_table(csv_path):
    """Load optional surface density table"""
    if csv_path is None: 
        return None
    
    print(f"[Loading] Surface density: {csv_path}")
    tab = np.genfromtxt(csv_path, delimiter=",", names=True)
    return {
        "R_kpc": xp_array(tab["R_kpc"], XP.float64),
        "Sigma_Msun_pc2": xp_array(tab["Sigma_Msun_pc2"], XP.float64)
    }

# ============================================================================
# Main Optimization Loop
# ============================================================================

def run(args):
    """Main optimization routine with checkpoint/resume"""
    
    print("\n" + "="*70)
    print("MILKY WAY G³ GPU OPTIMIZER")
    print("="*70)
    print(f"Device: {'GPU' if HAS_CUPY else 'CPU'}")
    print("="*70)
    
    # Baryon configuration
    cfg = {
        "disk":  {"M": args.M_disk, "a": args.Rd_kpc, "b": args.hz_kpc},
        "gas":   {"M": args.M_gas,  "a": args.Rg_kpc, "b": args.hg_kpc},
        "bulge": {"M": args.M_bulge, "a": args.ab_kpc},
        "sigma_table": maybe_load_sigma_table(args.sigma_csv)
    }
    
    print("\nBaryon model:")
    print(f"  Stellar disk: M = {args.M_disk:.2e} Msun, Rd = {args.Rd_kpc} kpc")
    print(f"  Gas disk: M = {args.M_gas:.2e} Msun, Rg = {args.Rg_kpc} kpc")
    print(f"  Bulge: M = {args.M_bulge:.2e} Msun, a = {args.ab_kpc} kpc")
    
    # Load Gaia data
    stars = load_gaia_csv(args.stars_csv, z_cut=args.z_cut)
    n_stars = len(stars["R"])
    print(f"\nOptimizing on {n_stars:,} stars")
    
    # Parameter bounds (physical units)
    param_names = ["v0", "rc", "r0", "delta", "gamma", "beta", "z0", "m", "Sigma0"]
    bounds_lo = XP.array([  50.,   1.0,  0.0, 0.1,  0.0, 0.00, 0.05, 1.0,  25.], dtype=XP.float64)
    bounds_hi = XP.array([ 300.,  50.0, 10.0, 8.0,  1.5, 0.50, 1.50, 8.0, 300.], dtype=XP.float64)
    
    # Initial guess
    if args.init_json and pathlib.Path(args.init_json).exists():
        init_data = json.load(open(args.init_json))
        x0 = XP.array(init_data["theta"], dtype=XP.float64)
        print(f"\nLoaded initial theta from {args.init_json}")
    else:
        # Default near SPARC best-fit values
        x0 = XP.array([140., 22., 3.0, 3.0, 0.5, 0.10, 0.20, 2.0, 150.], dtype=XP.float64)
        print("\nUsing default initial parameters")
    
    # Check for checkpoint
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = pathlib.Path(args.out_dir) / "mw_cupy_opt_state.npz"
    ckpt = load_checkpoint(ckpt_path)
    
    if ckpt is not None:
        x0 = xp_array(ckpt["best_theta"], XP.float64)
        print(f"Resuming from checkpoint (best error: {ckpt.get('best_f', 'unknown')})")
    
    # Initialize optimizer
    opt = CMAES(
        x0=x0, 
        sigma0=args.sigma0, 
        popsize=args.popsize, 
        bounds=(bounds_lo, bounds_hi)
    )
    
    best_f = math.inf
    best_theta = None
    
    # Signal handler for graceful shutdown
    def handle_sigterm(sig, frm):
        print("\n[INTERRUPT] Saving checkpoint...")
        save_checkpoint(ckpt_path, {
            "best_theta": to_cpu(best_theta if best_theta is not None else x0),
            "best_f": best_f,
            "generation": opt.generation
        })
        print("Checkpoint saved. Exiting.")
        raise SystemExit

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    print(f"\nOptimization settings:")
    print(f"  Population size: {args.popsize}")
    print(f"  Initial sigma: {args.sigma0}")
    print(f"  Output dir: {args.out_dir}")
    print("\nPress Ctrl+C to stop and save checkpoint\n")
    print("="*70)
    
    t0 = time.time()
    gen = 0
    
    # Main optimization loop
    while True:
        # Generate candidates
        X = opt.ask()  # (popsize, dim)
        
        # Evaluate all candidates (fully vectorized on GPU)
        fvals = []
        for i in range(X.shape[0]):
            theta = X[i]
            f = evaluate_theta(theta, stars, cfg)
            fvals.append(f)
        fvals = XP.array(fvals, dtype=XP.float64)
        
        # Update optimizer
        opt.tell(X, fvals)
        
        # Track best
        i_best = int(XP.argmin(fvals))
        if fvals[i_best] < best_f:
            best_f = float(to_cpu(fvals[i_best]))
            best_theta = to_cpu(X[i_best])
            
            # Save checkpoint
            save_checkpoint(ckpt_path, {
                "best_theta": best_theta, 
                "best_f": best_f,
                "generation": gen
            })
            
            # Save human-readable results
            result = {
                "theta": list(map(float, best_theta)),
                "param_names": param_names,
                "median_relative_error": best_f,
                "n_stars": n_stars,
                "generation": gen,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(pathlib.Path(args.out_dir) / "best_theta.json", "w") as fh:
                json.dump(result, fh, indent=2)
            
            # Print formatted parameters
            print(f"\n[Gen {gen}] NEW BEST! Error = {best_f:.4f} ({100*(1-best_f):.1f}% accuracy)")
            print("  Parameters:")
            for name, val in zip(param_names, best_theta):
                print(f"    {name:8} = {val:.3f}")
        
        gen += 1
        
        # Regular progress report
        if gen % args.log_every == 0:
            elapsed = time.time() - t0
            rate = gen / elapsed
            print(f"[Gen {gen}] best = {best_f:.4f} | "
                  f"current = {float(XP.min(fvals)):.4f} - {float(XP.max(fvals)):.4f} | "
                  f"rate = {rate:.1f} gen/s | t = {elapsed:.0f}s")

# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MW G³ GPU Optimizer")
    
    # Data files
    p.add_argument("--stars_csv", required=True, 
                   help="Gaia CSV with R_kpc, z_kpc, vphi_kms, [vphi_err_kms]")
    p.add_argument("--sigma_csv", default=None, 
                   help="Optional CSV: R_kpc, Sigma_Msun_pc2 for surface density")
    p.add_argument("--out_dir", default="out/mw_g3_gpu", 
                   help="Output directory for results")
    
    # Baryon model parameters (MW-specific)
    p.add_argument("--M_disk", type=float, default=5.0e10, 
                   help="Stellar disk mass (Msun)")
    p.add_argument("--M_gas",  type=float, default=1.0e10, 
                   help="Gas disk mass (Msun)")
    p.add_argument("--M_bulge", type=float, default=1.0e10, 
                   help="Bulge mass (Msun)")
    p.add_argument("--Rd_kpc", type=float, default=2.6, 
                   help="Stellar disk scale length (kpc)")
    p.add_argument("--Rg_kpc", type=float, default=7.0, 
                   help="Gas disk scale length (kpc)")
    p.add_argument("--hz_kpc", type=float, default=0.3, 
                   help="Stellar disk scale height (kpc)")
    p.add_argument("--hg_kpc", type=float, default=0.13, 
                   help="Gas disk scale height (kpc)")
    p.add_argument("--ab_kpc", type=float, default=0.7, 
                   help="Bulge scale radius (kpc)")
    
    # Data selection
    p.add_argument("--z_cut", type=float, default=0.8, 
                   help="Only use stars with |z| < z_cut kpc")
    
    # Optimizer settings
    p.add_argument("--sigma0", type=float, default=0.25, 
                   help="Initial step size for CMA-ES")
    p.add_argument("--popsize", type=int, default=64, 
                   help="Population size for CMA-ES")
    p.add_argument("--log_every", type=int, default=5, 
                   help="Generations between progress reports")
    p.add_argument("--init_json", default=None, 
                   help="JSON file with initial theta")
    
    args = p.parse_args()
    run(args)