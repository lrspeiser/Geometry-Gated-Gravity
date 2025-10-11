
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Toy "many-path gravity" model for a Milky Way-like galaxy, GPU-ready with CuPy.

Author: ChatGPT (GPT-5 Pro)
License: MIT (for this toy example)

Concept
-------
We model the gravitational acceleration from a baryonic mass distribution
(e.g., a Milky Way-like stellar disk and optional bulge) at target points
(stars in the outer disk). In addition to the Newtonian 1/r^2 force, we
add a *phenomenological* "many-path multiplier" M(d, geometry) intended
to mimic the idea that gravity could effectively sum contributions from
numerous, longer, gently-curved paths around/over/under the disk on
galactic scales, while remaining negligible on small (solar-system) scales.

This *toy* multiplier is designed to:
- Vanish at small separations ("Solar System safe"),
- Grow with separation across kiloparsecs,
- Prefer paths near the disk plane (anisotropy),
- Optionally include a "ring-winding" term that mimics contributions from
  long azimuthal paths (with exponential suppression),
- Be controlled by a small, explicit set of parameters that you can tune,
- Preserve stable, bound orbits in practice by keeping M bounded.

Nothing here is a physical theory; it's a sandbox for numerical exploration.

Units
-----
- Distance: kpc
- Mass: Msun
- Velocity: km/s
- G = 4.30091e-6  [kpc * (km/s)^2 / Msun]

Dependencies
------------
- CuPy (preferred) for GPU:   pip install cupy-cuda12x   # match your CUDA
- NumPy fallback for CPU

Run
---
python toy_many_path_gravity.py --n_sources 200000 --targets 5 40 20 --use_bulge 1 --gpu 1

This will:
- Sample 2e5 baryonic "source" particles for the disk (and bulge if enabled),
- Evaluate accelerations for 20 targets from R=5 to 40 kpc, z=0,
- Plot rotation curves (Newtonian vs. many-path).

You can increase --n_sources on your GPU (e.g., 1e6). If memory is tight,
reduce --batch_size to keep per-chunk pairwise arrays smaller.

Gaia comparison (hook)
----------------------
To compare with Gaia-derived profiles later, compute predicted:
- v_c(R) in the plane,
- vertical frequency near z=0 from finite-differencing a_z(z),
then cross-match with published Gaia kinematic curves for appropriate tracer populations.

"""
import argparse
import math

# Try CuPy first; fallback to NumPy
try:
    import cupy as cp
    _USING_CUPY = True
except Exception:
    import numpy as cp  # type: ignore
    _USING_CUPY = False

# Constants
G = 4.30091e-6  # kpc * (km/s)^2 / Msun

# ------------------------------
# Utility
# ------------------------------
def to_cpu(a):
    """Return a NumPy array view/copy from cp/np array."""
    if _USING_CUPY:
        return cp.asnumpy(a)
    return a

def xp_array(a, dtype=None):
    return cp.array(a, dtype=dtype) if _USING_CUPY else cp.array(a, dtype=dtype)

def xp_zeros(shape, dtype=None):
    return cp.zeros(shape, dtype=dtype) if _USING_CUPY else cp.zeros(shape, dtype=dtype)

def xp_rand(seed=None):
    if seed is not None:
        if _USING_CUPY:
            cp.random.seed(seed)
        else:
            cp.random.seed(seed)
    return cp.random

# ------------------------------
# Source sampling (Milky Way-like)
# ------------------------------
def sample_exponential_disk(n, M_disk=5e10, R_d=2.6, z_d=0.3, R_max=30.0, seed=42):
    """
    Sample an exponential disk: Sigma(R) ~ exp(-R/R_d), vertical ~ exp(-|z|/z_d).
    Returns (pos[N,3], mass_per_particle). Positions in kpc.
    """
    rng = xp_rand(seed)

    # Inverse-transform for R with truncation at R_max
    # CDF_trunc(u) = 1 - exp(-(R_max - R)/R_d) / (1 - exp(-R_max/R_d)) ???
    # Simpler: sample untruncated then clip/retry to <= R_max.
    # Efficient rejection sampling:
    R = xp_zeros(n, dtype=cp.float64)
    i = 0
    while i < n:
        batch = n - i
        u = rng.random(batch)
        # R' = -R_d * ln(1 - u), exponential distribution
        R_try = -R_d * cp.log(1.0 - u)
        if _USING_CUPY:
            mask = (R_try <= R_max)
        else:
            mask = (R_try <= R_max)
        k = int(mask.sum())
        if k > 0:
            R[i:i+k] = R_try[mask][:k]
            i += k

    phi = 2.0 * math.pi * rng.random(n)
    # Vertical: double-exponential, sample |z| from Exp(z_d) then assign random sign
    uz = rng.random(n)
    z = -z_d * cp.log(1.0 - uz)  # positive
    sign = cp.where(rng.random(n) < 0.5, -1.0, 1.0)
    z = z * sign

    x = R * cp.cos(phi)
    y = R * cp.sin(phi)
    pos = cp.stack([x, y, z], axis=1)
    m = M_disk / float(n)
    return pos, m

def sample_hernquist_bulge(n, M_bulge=1e10, a=0.7, seed=123):
    """
    Sample a spherical Hernquist bulge:
    rho(r) = (M_bulge / (2*pi)) * a / [r (r+a)^3]
    CDF inversion for r: r = a * u / (1 - u), with u in (0,1) for R<inf.
    """
    rng = xp_rand(seed)
    u = cp.clip(rng.random(n), 1e-12, 1.0 - 1e-12)
    r = a * u / (1.0 - u)
    # Isotropic
    cos_t = 2.0 * rng.random(n) - 1.0
    sin_t = cp.sqrt(1.0 - cos_t**2)
    phi = 2.0 * math.pi * rng.random(n)
    x = r * sin_t * cp.cos(phi)
    y = r * sin_t * cp.sin(phi)
    z = r * cos_t
    pos = cp.stack([x, y, z], axis=1)
    m = M_bulge / float(n)
    return pos, m

# ------------------------------
# Many-path multiplier kernel
# ------------------------------
def many_path_multiplier(d, Rs, zs, Rt, zt, params, bulge_frac=None):
    """
    Compute a phenomenological "many-path" multiplier M(d, geometry).

    Arguments (pairwise, broadcastable):
    - d   : pairwise separation |r_t - r_s| [kpc], shape [Ns, Nt]
    - Rs  : source cylindrical radius [kpc], shape [Ns,]
    - zs  : source z [kpc], shape [Ns,]
    - Rt  : target cylindrical radius [kpc], shape [Nt,]
    - zt  : target z [kpc], shape [Nt,]
    - params: dict of parameters (see defaults in default_params())
    - bulge_frac: optional array of bulge fraction at target radii [Nt,], 
                  used to suppress ring-winding in bulge-dominated regions

    Returns:
    - M: dimensionless multiplier, shape [Ns, Nt], to multiply the Newtonian force.
    """
    # Unpack parameters
    eta    = params.get("eta", 0.5)          # overall amplitude
    R_gate = params.get("R_gate", 0.5)       # kpc, gating scale so Solar System << R_gate => M ~ 0
    p_gate = params.get("p_gate", 4.0)       # sharpness of gate
    R0     = params.get("R0", 5.0)           # kpc, onset scale for growth
    p      = params.get("p", 2.0)            # growth power
    R1     = params.get("R1", 80.0)          # kpc, saturation/roll scale
    q      = params.get("q", 2.0)            # saturation steepness
    Z0     = params.get("Z0", 1.0)           # kpc, plane-preference
    k_an   = params.get("k_an", 1.0)         # anisotropy exponent
    ring_amp = params.get("ring_amp", 0.2)   # amplitude of ring-winding term
    lam    = params.get("lambda_ring", 20.0) # kpc, ring winding scale
    M_max  = params.get("M_max", 5.0)        # cap to avoid blow-ups

    # Distances are [Ns, Nt]
    # Gating: negligible for d << R_gate
    g1 = 1.0 - cp.exp(- (d / R_gate)**p_gate)

    # Distance-growth with soft saturation / roll-over
    f_d = (d / R0)**p / (1.0 + (d / R1)**q)

    # Plane preference: prefer smaller z-avg (paths hugging the disk)
    # Build z_avg with broadcasting: zs[:,None], zt[None,:]
    # Use radially-modulated anisotropy: stronger near solar circle, weaker far out
    
    # Radial modulation parameters
    R_lag = params.get("R_lag", 8.0)      # kpc, center of lag enhancement
    w_lag = params.get("w_lag", 2.0)      # kpc, width of transition
    k_boost = params.get("k_boost", 0.6)  # extra anisotropy bump near R_lag
    Z0_in = params.get("Z0_in", 1.1)      # stronger planar pref inside ~R_lag
    Z0_out = params.get("Z0_out", 1.6)    # milder planar pref far out
    
    # Mid-radius (already computed for ring term)
    Rmid = 0.5 * (cp.abs(Rs)[:, None] + cp.abs(Rt)[None, :])
    
    # Smooth transition: inner → Z0_in, outer → Z0_out
    s = 0.5 * (1.0 + cp.tanh((R_lag - Rmid) / w_lag))
    Zeff = Z0_out * (1.0 - s) + Z0_in * s
    
    # Effective anisotropy exponent: base k_an plus Gaussian bump at R_lag
    k_eff = k_an + k_boost * cp.exp(-((Rmid - R_lag) / w_lag)**2)
    
    zavg = 0.5 * (cp.abs(zs)[:, None] + cp.abs(zt)[None, :])
    plane_pref = (Zeff**2 / (Zeff**2 + zavg**2))**k_eff

    # Ring-winding term: depends on mid-radius
    Rmid = 0.5 * (cp.abs(Rs)[:, None] + cp.abs(Rt)[None, :])
    # geometric series sum exp(-2π Rmid / lam) + ...  -> e^{-x} / (1 - e^{-x})
    x = (2.0 * math.pi * Rmid) / lam
    ex = cp.exp(-x)
    ring_term_base = ring_amp * (ex / cp.maximum(1e-20, 1.0 - ex))
    
    # Bulge gating: suppress ring winding in bulge-dominated regions
    if bulge_frac is not None:
        # bulge_frac is [Nt,], broadcast to [1, Nt]
        bulge_gate_power = params.get("bulge_gate_power", 2.0)
        # Suppress ring term where bulge_frac is high
        # gate = (1 - bulge_frac)^bulge_gate_power
        bulge_gate = (1.0 - cp.minimum(bulge_frac[None, :], 1.0))**bulge_gate_power
        ring_term = ring_term_base * bulge_gate
    else:
        ring_term = ring_term_base

    M = eta * g1 * f_d * plane_pref * (1.0 + ring_term)
    if M_max is not None:
        M = cp.minimum(M, M_max)
    return M

def default_params():
    return dict(
        eta=0.6,          # overall amplitude of the multiplier
        R_gate=0.5,       # kpc; << 1 kpc => solar system effects vanish
        p_gate=4.0,       # sharpness of "turn on"
        R0=5.0,           # kpc; onset for growth across the disk
        p=2.0,            # growth power
        R1=80.0,          # kpc; roll/saturation
        q=2.0,            # roll steepness
        Z0=1.0,           # planar preference scale (legacy, now uses Z0_in/Z0_out)
        k_an=1.0,         # anisotropy strength (base level)
        ring_amp=0.2,     # azimuthal winding contribution
        lambda_ring=20.0, # kpc
        M_max=4.0,        # cap
        # Radially-modulated anisotropy (new)
        R_lag=8.0,        # kpc; center of vertical lag enhancement
        w_lag=2.0,        # kpc; width of radial transition
        k_boost=0.6,      # extra anisotropy bump near R_lag
        Z0_in=1.1,        # kpc; stronger planar pref inside ~R_lag
        Z0_out=1.6,       # kpc; milder planar pref far out
        # Bulge gating
        bulge_gate_power=2.0,  # power for suppressing ring winding in bulge-dominated regions
    )

# ------------------------------
# Acceleration (batched pairwise sum)
# ------------------------------
def compute_accel_batched(src_pos, src_m, tgt_pos, eps=0.05, params=None, batch_size=100_000, use_multiplier=True, bulge_frac=None):
    """
    Compute acceleration at target positions from source particles using batched pairwise summation.
    - src_pos: [Ns,3]
    - src_m: scalar mass per particle or [Ns]
    - tgt_pos: [Nt,3]
    - eps: Plummer-like softening length [kpc] to avoid singularities
    - params: dict for many-path multiplier
    - batch_size: number of source particles per chunk
    - use_multiplier: if False => pure Newtonian force
    - bulge_frac: optional array [Nt] of bulge fraction at target positions

    Returns:
    - acc: [Nt,3]
    """
    Ns = src_pos.shape[0]
    Nt = tgt_pos.shape[0]
    acc = xp_zeros((Nt, 3), dtype=cp.float64)

    # Prepare source cylindrical radii and z for anisotropy
    Rs = cp.sqrt(src_pos[:, 0]**2 + src_pos[:, 1]**2)
    zs = src_pos[:, 2]
    # Target cylindrical R and z
    Rt = cp.sqrt(tgt_pos[:, 0]**2 + tgt_pos[:, 1]**2)
    zt = tgt_pos[:, 2]

    if isinstance(src_m, (float, int)):
        src_m_arr = xp_zeros(Ns, dtype=cp.float64) + float(src_m)
    else:
        src_m_arr = src_m.astype(cp.float64)

    eps2 = eps * eps

    for i0 in range(0, Ns, batch_size):
        i1 = min(i0 + batch_size, Ns)
        s = src_pos[i0:i1]          # [B,3]
        m = src_m_arr[i0:i1]        # [B]
        Rs_b = Rs[i0:i1]            # [B]
        zs_b = zs[i0:i1]            # [B]

        # Pairwise differences: dvec = tgt[None, :, :] - src[:, None, :]
        dvec = tgt_pos[None, :, :] - s[:, None, :]  # [B, Nt, 3]
        r2 = (dvec**2).sum(axis=2) + eps2          # [B, Nt]
        inv_r3 = r2**(-1.5)                        # [B, Nt]

        # Multiplier
        if use_multiplier and (params is not None):
            d = cp.sqrt(r2)                         # [B, Nt]
            M = many_path_multiplier(d, Rs_b, zs_b, Rt, zt, params, bulge_frac=bulge_frac)  # [B, Nt]
            factor = (1.0 + M)                     # [B, Nt]
        else:
            factor = 1.0

        # Contribution to acceleration: G m r / r^3 with sign (-), summed over sources
        # Shape handling: (B, Nt, 1) * (B, Nt, 3) -> (B, Nt, 3)
        contrib = -G * (m[:, None] * inv_r3 * factor)[:, :, None] * dvec  # [B, Nt, 3]
        # Sum over batch dimension B
        acc += contrib.sum(axis=0)

        # Free temporary memory on GPU between batches
        if _USING_CUPY:
            del dvec, r2, inv_r3, contrib
            cp._default_memory_pool.free_all_blocks()

    return acc

# ------------------------------
# Helpers: rotation curve & vertical frequency
# ------------------------------
def rotation_curve(src_pos, src_m, R_vals, z=0.0, eps=0.05, params=None, use_multiplier=True, batch_size=100_000, bulge_frac=None):
    """
    Compute circular velocity v_c(R) in the plane z (default z=0).
    
    - bulge_frac: optional array [len(R_vals)] of bulge fraction at each radius
    """
    Nt = len(R_vals)
    tgt = xp_zeros((Nt, 3), dtype=cp.float64)
    tgt[:, 0] = R_vals  # x-axis
    tgt[:, 1] = 0.0
    tgt[:, 2] = z

    acc = compute_accel_batched(src_pos, src_m, tgt, eps=eps, params=params,
                                batch_size=batch_size, use_multiplier=use_multiplier, bulge_frac=bulge_frac)
    # radial component along x (since y=0)
    a_R = acc[:, 0]
    v_c = cp.sqrt(cp.maximum(0.0, R_vals * (-a_R)))
    return v_c, a_R

def vertical_frequency(src_pos, src_m, R, dz=0.05, eps=0.05, params=None, use_multiplier=True, batch_size=100_000):
    """
    Approximate vertical frequency nu_z^2 = (∂a_z/∂z)|_{z=0} by finite difference.
    """
    tgt1 = xp_array([[R, 0.0, dz]], dtype=cp.float64)
    tgt2 = xp_array([[R, 0.0, -dz]], dtype=cp.float64)
    a1 = compute_accel_batched(src_pos, src_m, tgt1, eps=eps, params=params,
                               batch_size=batch_size, use_multiplier=use_multiplier)[0, 2]
    a2 = compute_accel_batched(src_pos, src_m, tgt2, eps=eps, params=params,
                               batch_size=batch_size, use_multiplier=use_multiplier)[0, 2]
    nu2 = (a1 - a2) / (2.0 * dz)  # because a_z ~ -nu^2 z near plane
    return nu2

# ------------------------------
# CLI / Demo
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Toy many-path gravity model (Milky Way-like)")
    parser.add_argument("--n_sources", type=int, default=50000, help="Number of disk source particles")
    parser.add_argument("--M_disk", type=float, default=5e10, help="Stellar disk mass [Msun]")
    parser.add_argument("--R_d", type=float, default=2.6, help="Disk scale length [kpc]")
    parser.add_argument("--z_d", type=float, default=0.3, help="Disk scale height [kpc]")
    parser.add_argument("--R_max", type=float, default=30.0, help="Disk truncation radius [kpc]")
    parser.add_argument("--use_bulge", type=int, default=1, help="Add Hernquist bulge (1=yes,0=no)")
    parser.add_argument("--n_bulge", type=int, default=10000, help="Number of bulge particles")
    parser.add_argument("--M_bulge", type=float, default=1e10, help="Bulge mass [Msun]")
    parser.add_argument("--a_bulge", type=float, default=0.7, help="Bulge scale [kpc]")
    parser.add_argument("--targets", nargs=3, type=float, default=[5.0, 40.0, 20],
                        help="Rmin Rmax Npoints for rotation curve (kpc kpc N)")
    parser.add_argument("--eps", type=float, default=0.05, help="Softening [kpc]")
    parser.add_argument("--gpu", type=int, default=1, help="Prefer GPU (CuPy) if available")
    parser.add_argument("--batch_size", type=int, default=100000, help="Pairwise batch size")
    parser.add_argument("--plot", type=int, default=1, help="Make a rotation-curve plot")
    args = parser.parse_args()

    global _USING_CUPY
    if args.gpu == 0:
        _USING_CUPY = False

    # Sample sources
    disk_pos, m_disk = sample_exponential_disk(args.n_sources, M_disk=args.M_disk, R_d=args.R_d,
                                               z_d=args.z_d, R_max=args.R_max, seed=42)
    if args.use_bulge:
        bulge_pos, m_bulge = sample_hernquist_bulge(args.n_bulge, M_bulge=args.M_bulge,
                                                    a=args.a_bulge, seed=123)
        src_pos = cp.concatenate([disk_pos, bulge_pos], axis=0)
        src_mass = cp.concatenate([
            xp_zeros(disk_pos.shape[0]) + m_disk,
            xp_zeros(bulge_pos.shape[0]) + m_bulge
        ])
    else:
        src_pos = disk_pos
        src_mass = xp_zeros(disk_pos.shape[0]) + m_disk

    # Targets
    Rmin, Rmax, Np = args.targets
    Np = int(Np)
    R_vals = cp.linspace(Rmin, Rmax, Np, dtype=cp.float64)

    # Params
    params = default_params()

    # Newtonian baseline
    vN, aR_N = rotation_curve(src_pos, src_mass, R_vals, z=0.0, eps=args.eps, params=None,
                              use_multiplier=False, batch_size=args.batch_size)

    # Many-path variant
    vM, aR_M = rotation_curve(src_pos, src_mass, R_vals, z=0.0, eps=args.eps, params=params,
                              use_multiplier=True, batch_size=args.batch_size)

    # Print a Solar System sanity check (1 AU effect)
    # 1 AU in kpc:
    AU_kpc = 4.84813681e-9 * 206265.0  # This is wrong; fix below
    # Correct: 1 pc = 206265 AU, 1 kpc = 1000 pc => 1 AU = 1 / (206265 * 1000) kpc
    AU_kpc = 1.0 / (206265.0 * 1000.0)
    d_au = xp_array([[AU_kpc]])
    # Fake args to probe multiplier at Solar-System scale:
    # Put Rs ~ Rt ~ 8 kpc, zs ~ zt ~ 0
    Rs = xp_array([8.0])
    zs = xp_array([0.0])
    Rt = xp_array([8.0])
    zt = xp_array([0.0])
    M_au = many_path_multiplier(d_au, Rs, zs, Rt, zt, params)[0,0]
    print(f"Estimated multiplier M at d=1 AU: {float(to_cpu(M_au)):.3e} (should be ~0)")

    # Output basic table
    print("R (kpc)   v_c_Newton (km/s)   v_c_manypath (km/s)   boost vM/vN")
    for R, v1, v2 in zip(to_cpu(R_vals), to_cpu(vN), to_cpu(vM)):
        boost = (v2 / v1) if (v1 > 1e-6) else float('nan')
        print(f"{R:6.2f}   {v1:16.3f}   {v2:18.3f}   {boost:10.3f}")

    # Optional plot
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(to_cpu(R_vals), to_cpu(vN), label="Newtonian")
        plt.plot(to_cpu(R_vals), to_cpu(vM), label="Many-path")
        plt.xlabel("R (kpc)")
        plt.ylabel("v_c (km/s)")
        plt.title("Rotation curve: Newtonian vs. Many-path toy model")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
