import numpy as np

class GeometricExponentGravity:
    """
    Scale-dependent geometric exponent model.
    For rotation curves: v_tot^2 = v_bar^2 × (1 + fX)^{β(Σ, R)}
    For lensing (proxy):  Σ_eff = Σ_bar × (1 + fX)^{β(Σ, R)}

    This variant includes an interior-anchored export term so dense cores boost
    deflection at larger radii in a controlled way.
    """
    def __init__(self, a: float, b: float, d: float, gamma1: float, gamma2: float,
                 R_scale_kpc: float = 100.0, beta_clip=(1.0, 4.0),
                 # Interior-anchored export knobs (defaults provide mild boost):
                 A_core: float = 0.25, p_core: float = 2.0,
                 Sigma0_hat: float = 0.0, beta_core: float = 0.4,
                 smooth_R_kpc: float = 5.0):
        self.a = float(a)
        self.b = float(b)
        self.d = float(d)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.R_scale = float(R_scale_kpc)
        self.beta_clip = beta_clip
        # Interior export parameters
        self.A_core = float(A_core)
        self.p_core = float(p_core)
        self.Sigma0_hat = float(Sigma0_hat)
        self.beta_core = float(beta_core)
        self.smooth_R = float(smooth_R_kpc)

    @staticmethod
    def _safe_clip(x, lo=1e-9):
        return np.maximum(x, lo)

    @staticmethod
    def sigma_hat_from_Sigma_pc2(Sigma_pc2):
        Sigma_pc2 = np.asarray(Sigma_pc2, float)
        return np.log10(np.maximum(Sigma_pc2, 1e-30) / 100.0)

    @staticmethod
    def _cum_enclosed_mean(R_kpc, Sigma_kpc2):
        R = np.asarray(R_kpc, float)
        S = np.asarray(Sigma_kpc2, float)
        # cumulative mass and area
        Mcum = np.cumtrapz(2.0*np.pi*R*S, R, initial=0.0)
        Acum = np.pi*np.maximum(R, 1e-9)**2
        return np.divide(Mcum, Acum, out=np.zeros_like(Mcum), where=Acum>0)

    @staticmethod
    def _rollmax(x):
        return np.maximum.accumulate(np.asarray(x, float))

    def _smooth(self, x, R):
        x = np.asarray(x, float); R = np.asarray(R, float)
        if x.size < 5 or self.smooth_R <= 0:
            return x
        try:
            from numpy.lib.stride_tricks import sliding_window_view as swv
            w = max(3, int(np.ceil(self.smooth_R / max(np.median(np.diff(R)), 1e-6))))
            if w % 2 == 0: w += 1
            pad = w//2
            xp = np.pad(x, (pad, pad), mode='edge')
            return np.median(swv(xp, w), axis=-1)
        except Exception:
            # fallback simple running mean
            w = max(3, int(np.ceil(self.smooth_R / max(np.median(np.diff(R)), 1e-6))))
            if w % 2 == 0: w += 1
            k = w//2
            out = np.copy(x)
            for i in range(x.size):
                lo = max(0, i-k); hi = min(x.size, i+k+1)
                out[i] = np.mean(x[lo:hi])
            return out

    @staticmethod
    def grad_ln_Sigma(R_kpc, Sigma_kpc2):
        R = np.asarray(R_kpc, float)
        S = np.maximum(np.asarray(Sigma_kpc2, float), 1e-30)
        dS = np.gradient(S, R, edge_order=2)
        return (R / S) * dS

    def compute_fX(self, R_kpc, Rd_kpc, Sigma_hat, grad_ln_Sigma):
        R_kpc = np.asarray(R_kpc, float)
        Rd_kpc = max(float(Rd_kpc), 1e-6)
        x = R_kpc / Rd_kpc
        # Avoid runaway suppression from noisy gradients; keep a soft floor relative to 'a'
        denom = self.a - self.b * Sigma_hat - self.d * np.minimum(np.abs(grad_ln_Sigma), 3.0)
        denom = np.maximum(denom, 0.2 * self.a)
        fX = (x * x) / denom
        return np.maximum(fX, 0.0)

    def compute_beta(self, Sigma_hat, R_kpc):
        Sigma_hat = np.asarray(Sigma_hat, float)
        R_kpc = np.asarray(R_kpc, float)
        base = 1.0 + self.gamma1 * np.abs(Sigma_hat) + self.gamma2 * np.log10(self._safe_clip(R_kpc / self.R_scale, 1e-6))
        lo, hi = self.beta_clip
        return np.clip(base, max(lo, 1.0), hi)

    def Sigma_effective(self, R_kpc, Sigma_bar_kpc2, Rd_kpc):
        """Compute Σ_eff from Σ_bar with local + interior-anchored boost.

        Returns (Sigma_eff, fX_total, beta_effective) for compatibility; callers
        typically only need the first element.
        """
        R = np.asarray(R_kpc, float)
        S = np.asarray(Sigma_bar_kpc2, float)
        # Local diagnostics (smoothed gradient)
        Shat_local = self.sigma_hat_from_Sigma_pc2((S/1e6))
        glnS_local = self.grad_ln_Sigma(R, self._smooth(S, R))
        # Interior diagnostics (mean & peak)
        Sbar_int  = self._cum_enclosed_mean(R, S)
        Shat_int  = self.sigma_hat_from_Sigma_pc2((Sbar_int/1e6))
        Shat_peak = self._rollmax(Shat_local)
        # Local boost
        fX_local   = self.compute_fX(R, Rd_kpc, Shat_local, glnS_local)
        beta_local = self.compute_beta(Shat_int, R)  # exponent driven by interior mean
        # Interior-anchored export
        core_signal = np.maximum(Shat_peak - self.Sigma0_hat, 0.0)
        fX_core   = self.A_core * (np.maximum(R, 1e-9)/max(Rd_kpc,1e-6))**self.p_core * core_signal
        beta_core = np.clip(self.beta_core, 0.0, self.beta_clip[1])
        # Combine multiplicatively (guarantee Σ_eff>=Σ_bar)
        M_factor = np.power(1.0 + fX_local, beta_local)
        if beta_core > 0:
            M_factor = M_factor * np.power(1.0 + fX_core, beta_core)
        Sigma_eff = S * M_factor
        # Return combined diagnostics (for downstream debugging)
        fX_total = fX_local + fX_core
        beta_eff = beta_local + (beta_core if beta_core>0 else 0.0)
        return Sigma_eff, fX_total, beta_eff

    def Sigma_effective_xp(self, xp, R_kpc, Sigma_bar_kpc2, Rd_kpc):
        """Compute Σ_eff on the provided array backend (CuPy/NumPy) with identical physics to CPU path."""
        # Backend arrays
        R = xp.asarray(R_kpc)
        Sb = xp.asarray(Sigma_bar_kpc2)

        # Sigma_hat from Σ in pc^-2 (same as CPU): Σ_pc2 = Σ_kpc2 / 1e6; Σ̂ = log10(max(Σ_pc2, 1e-30)/100)
        Sigma_pc2 = Sb / 1e6
        Sigma_hat = xp.log10(xp.maximum(Sigma_pc2, 1e-30) / 100.0)

        # grad ln Σ = (R/Σ) dΣ/dR
        dS = xp.gradient(Sb, R)
        glnS = (R / xp.maximum(Sb, 1e-30)) * dS

        # fX = (R/Rd)^2 / clip(a - b Σ̂ - d |∇ln Σ|, 1e-6)
        Rd = float(Rd_kpc)
        x = R / max(Rd, 1e-6)
        denom = self.a - self.b * Sigma_hat - self.d * xp.abs(glnS)
        denom = xp.maximum(denom, 1e-6)
        fX = xp.maximum((x * x) / denom, 0.0)

        # β = clip(1 + γ1 |Σ̂| + γ2 log10(clip(R/R_scale, 1e-6)), beta_clip)
        base = 1.0 + self.gamma1 * xp.abs(Sigma_hat) + self.gamma2 * xp.log10(xp.maximum(R / self.R_scale, 1e-6))
        lo, hi = self.beta_clip
        beta = xp.clip(base, lo, hi)

        # Σ_eff = Σ_bar × (1 + fX)^β
        M_factor = xp.power(1.0 + fX, beta)
        return Sb * M_factor, fX, beta
