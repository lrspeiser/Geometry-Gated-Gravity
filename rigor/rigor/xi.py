
from __future__ import annotations
# JAX-friendly xi functions (fall back to numpy if JAX is absent)
try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore

def _sigmoid(z):
    return 1.0/(1.0 + jnp.exp(-z))

def shell_logistic_radius(R_kpc, Sigma_bar, Mbar, params):
    R = jnp.asarray(R_kpc)
    lnR = jnp.log(jnp.maximum(R, 1e-6))
    xi_max = jnp.maximum(params.get("xi_max", 3.0), 1.0)
    lnR0_base = params.get("lnR0_base", jnp.log(3.0))
    width = jnp.maximum(params.get("width", 0.6), 1e-3)
    alpha_M = params.get("alpha_M", -0.2)
    Mref = params.get("Mref", 1e10)
    Mbar = Mbar if (Mbar is not None) else Mref
    lnR0_g = lnR0_base + alpha_M * jnp.log(jnp.maximum((Mbar)/Mref, 1e-12))
    xi = 1.0 + (xi_max - 1.0) * _sigmoid((lnR - lnR0_g)/width)
    return xi

def logistic_density(R_kpc, Sigma_bar, Mbar, params):
    S = jnp.asarray(Sigma_bar) if Sigma_bar is not None else None
    if S is None:
        return jnp.ones_like(jnp.asarray(R_kpc))
    xi_max = jnp.maximum(params.get("xi_max", 3.0), 1.0)
    lnSigma_c = params.get("lnSigma_c", jnp.log(10.0))
    width_sigma = jnp.maximum(params.get("width_sigma", 0.6), 1e-3)
    n = jnp.maximum(params.get("n_sigma", 1.0), 1e-3)
    lnS = jnp.log(jnp.maximum(S, 1e-8))
    z = (lnSigma_c - lnS)/width_sigma
    xi = 1.0 + (xi_max - 1.0) * jnp.power(_sigmoid(z), n)
    xi = jnp.where(jnp.isfinite(S), xi, 1.0)  # where S is nan (padding), neutralize
    return xi

def logistic_radius_power(R_kpc, Sigma_bar, Mbar, params):
    R = jnp.asarray(R_kpc)
    xi_max = jnp.maximum(params.get("xi_max", 3.0), 1.0)
    R0_base = jnp.maximum(params.get("R0_base", 3.0), 1e-6)
    m = jnp.maximum(params.get("m", 2.0), 1e-3)
    alpha_M = params.get("alpha_M", -0.2)
    Mref = params.get("Mref", 1e10)
    Mbar = Mbar if (Mbar is not None) else Mref
    R0_g = R0_base * jnp.power((Mbar/Mref), alpha_M)
    return 1.0 + (xi_max - 1.0) / (1.0 + jnp.power(jnp.maximum(R, 1e-6)/R0_g, m))

def constant(R_kpc, Sigma_bar, Mbar, params):
    return jnp.full_like(jnp.asarray(R_kpc), jnp.maximum(params.get("xi_const", 1.0), 1.0))

XI_REGISTRY = {
    "shell_logistic_radius": shell_logistic_radius,
    "logistic_density": logistic_density,
    "logistic_radius_power": logistic_radius_power,
    "constant": constant,
}
