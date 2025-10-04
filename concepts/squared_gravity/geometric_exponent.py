import numpy as np

class GeometricExponentGravity:
    """
    Scale-dependent geometric exponent model.
    For rotation curves: v_tot^2 = v_bar^2 × (1 + fX)^{β(Σ, R)}
    For lensing (proxy):  Σ_eff = Σ_bar × (1 + fX)^{β(Σ, R)}
    """
    def __init__(self, a: float, b: float, d: float, gamma1: float, gamma2: float,
                 R_scale_kpc: float = 100.0, beta_clip=(1.0, 4.0)):
        self.a = float(a)
        self.b = float(b)
        self.d = float(d)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.R_scale = float(R_scale_kpc)
        self.beta_clip = beta_clip

    @staticmethod
    def _safe_clip(x, lo=1e-9):
        return np.maximum(x, lo)

    @staticmethod
    def sigma_hat_from_Sigma_pc2(Sigma_pc2):
        Sigma_pc2 = np.asarray(Sigma_pc2, float)
        return np.log10(np.maximum(Sigma_pc2, 1e-30) / 100.0)

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
        denom = self.a - self.b * Sigma_hat - self.d * np.abs(grad_ln_Sigma)
        denom = self._safe_clip(denom, 1e-6)
        fX = (x * x) / denom
        return np.maximum(fX, 0.0)

    def compute_beta(self, Sigma_hat, R_kpc):
        Sigma_hat = np.asarray(Sigma_hat, float)
        R_kpc = np.asarray(R_kpc, float)
        base = 1.0 + self.gamma1 * np.abs(Sigma_hat) + self.gamma2 * np.log10(self._safe_clip(R_kpc / self.R_scale, 1e-6))
        lo, hi = self.beta_clip
        return np.clip(base, lo, hi)

    def Sigma_effective(self, R_kpc, Sigma_bar_kpc2, Rd_kpc):
        """Compute Σ_eff proxy from Σ_bar with exponent model."""
        Sigma_pc2 = Sigma_bar_kpc2 / 1e6
        Sigma_hat = self.sigma_hat_from_Sigma_pc2(Sigma_pc2)
        glnS = self.grad_ln_Sigma(R_kpc, Sigma_bar_kpc2)
        fX = self.compute_fX(R_kpc, Rd_kpc, Sigma_hat, glnS)
        beta = self.compute_beta(Sigma_hat, R_kpc)
        M_factor = np.power(1.0 + fX, beta)
        return Sigma_bar_kpc2 * M_factor, fX, beta
