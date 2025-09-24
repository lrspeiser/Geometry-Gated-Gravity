#!/usr/bin/env python3
"""
G3Solver wrapper class for the 3D PDE solver functions.

This wraps the functional interface from solve_phi3d.py into a class-based API
that's easier to use for parameter optimization and analysis.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from solve_phi3d import (
    G3Globals, MobilityParams, BC,
    geometry_scalars_from_rho, solve_g3_3d,
    gradient_3d
)

class G3Solver:
    """Wrapper class for 3D G³ PDE solver."""
    
    def __init__(self, 
                 nx: int, ny: int, nz: int,
                 dx: float,
                 S0: float = 1.5,
                 rc: float = 10.0,
                 rc_ref: float = 10.0,
                 gamma: float = 0.3,
                 beta: float = 0.5,
                 mob_scale: float = 1.0,
                 mob_sat: float = 100.0,
                 use_sigma_screen: bool = False,
                 sigma_crit: float = 100.0,
                 screen_exp: float = 2.0,
                 bc_type: str = 'robin',
                 robin_alpha: float = 1.0,
                 tol: float = 1e-6,
                 max_iter: int = 100):
        """
        Initialize G³ solver.
        
        Args:
            nx, ny, nz: Grid dimensions
            dx: Grid spacing (assumed uniform)
            S0: Base coupling strength
            rc: Core radius in kpc
            rc_ref: Reference core radius for scaling
            gamma: Size scaling exponent
            beta: Density scaling exponent
            mob_scale: Mobility scale factor
            mob_sat: Saturation gradient
            use_sigma_screen: Whether to use surface density screening
            sigma_crit: Critical surface density for screening
            screen_exp: Screening exponent
            bc_type: Boundary condition type ('robin' or 'neumann')
            robin_alpha: Robin BC parameter
            tol: Convergence tolerance
            max_iter: Maximum iterations
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dx  # Assume uniform grid
        self.dz = dx
        
        # Store parameters
        self.S0 = S0
        self.rc = rc
        self.rc_ref = rc_ref
        self.gamma = gamma
        self.beta = beta
        self.mob_scale = mob_scale
        self.mob_sat = mob_sat
        self.use_sigma_screen = use_sigma_screen
        self.sigma_crit = sigma_crit
        self.screen_exp = screen_exp
        self.bc_type = bc_type
        self.robin_alpha = robin_alpha
        self.tol = tol
        self.max_iter = max_iter
        
    def solve(self, rho_3d: np.ndarray, r_half: float = None, 
              sigma_bar: float = None) -> np.ndarray:
        """
        Solve the G³ PDE for given density distribution.
        
        Args:
            rho_3d: 3D density array in M_sun/kpc^3
            r_half: Half-mass radius in kpc (optional, will be computed if None)
            sigma_bar: Mean surface density in M_sun/pc^2 (optional)
            
        Returns:
            phi: 3D potential field
        """
        # Compute geometry scalars if not provided
        if r_half is None or sigma_bar is None:
            r_half_comp, sigma_bar_comp = geometry_scalars_from_rho(
                rho_3d, self.dx, self.dy, self.dz
            )
            if r_half is None:
                r_half = r_half_comp
            if sigma_bar is None:
                sigma_bar = sigma_bar_comp
        
        # Apply geometry scaling to core radius
        rc_eff = self.rc * (r_half / self.rc_ref) ** self.gamma
        
        # Apply density scaling to coupling strength
        sigma_ratio = sigma_bar / 100.0  # Normalize to 100 M_sun/pc^2
        S0_eff = self.S0 * sigma_ratio ** self.beta
        
        # Set up global parameters
        globs = G3Globals(
            S0=S0_eff,
            rc_kpc=rc_eff,
            g0_kms2_per_kpc=1200.0,
            rc_gamma=self.gamma,
            rc_ref_kpc=self.rc_ref,
            sigma_beta=self.beta,
            sigma0_Msun_pc2=100.0
        )
        
        # Set up mobility parameters
        mob = MobilityParams(
            g0_kms2_per_kpc=1200.0 * self.mob_scale,
            use_saturating_mobility=True,
            g_sat_kms2_per_kpc=self.mob_sat,
            n_sat=2.0,
            use_sigma_screen=self.use_sigma_screen,
            sigma_star_Msun_per_pc2=self.sigma_crit,
            alpha_sigma=1.0,
            n_sigma=self.screen_exp
        )
        
        # Set up boundary conditions
        if self.bc_type == 'robin':
            bc = BC(kind='robin', lambda_kpc=1e6 / self.robin_alpha)
        else:
            bc = BC(kind='neumann')
        
        # Solve the PDE
        result = solve_g3_3d(
            rho=rho_3d,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            origin=(0.0, 0.0, 0.0),
            g3=globs,
            mobil=mob,
            bc=bc,
            levels=None,
            sigma_window_cells=8,
            use_sigma_screen=self.use_sigma_screen,
            max_cycles=self.max_iter,
            tol=self.tol,
            verbose=False
        )
        
        phi = result['phi']
        
        return phi
    
    def compute_gradient(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gravitational acceleration from potential.
        
        Args:
            phi: 3D potential field
            
        Returns:
            gx, gy, gz: Components of gravitational acceleration
        """
        # Gradient gives -∇φ = g
        gx, gy, gz = gradient_3d(phi, self.dx, self.dy, self.dz)
        
        # Return negative gradient (acceleration points inward)
        return -gx, -gy, -gz