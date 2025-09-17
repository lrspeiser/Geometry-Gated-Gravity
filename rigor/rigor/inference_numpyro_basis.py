
from __future__ import annotations
import os, json, math
from typing import Dict, List, Optional, Tuple
import numpy as np

def fit_sparse_logit_basis(dataset, n_knots=12, rng_key=0,
                           use_outer_only=False, num_warmup=2000, num_samples=2000, num_chains=4,
                           target_accept=0.85, platform="gpu", save_dir="results_basis"):

    import jax, jax.numpy as jnp
    from jax import random, vmap
    import numpyro
    from numpyro import distributions as dist
    from numpyro.infer import MCMC, NUTS

    # Pack padded arrays
    G = len(dataset.galaxies)
    maxJ = max(len(g.R_kpc) for g in dataset.galaxies)
    R = np.full((G, maxJ), np.nan, dtype=np.float32)
    Vobs = np.full_like(R, np.nan)
    eV = np.full_like(R, np.nan)
    Vbar = np.full_like(R, np.nan)
    mask_valid = np.zeros_like(R, dtype=bool)
    mask_outer = np.zeros_like(R, dtype=bool)
    names = []
    for g_idx, g in enumerate(dataset.galaxies):
        J = len(g.R_kpc)
        R[g_idx,:J] = g.R_kpc
        Vobs[g_idx,:J] = g.Vobs_kms
        eV[g_idx,:J] = g.eVobs_kms
        Vbar[g_idx,:J] = g.Vbar_kms
        mask_valid[g_idx,:J] = True
        mask_outer[g_idx,:J] = g.outer_mask
        names.append(g.name)

    R = jnp.asarray(R); Vobs=jnp.asarray(Vobs); eV=jnp.asarray(eV); Vbar=jnp.asarray(Vbar)
    mask_valid_j = jnp.asarray(mask_valid)
    mask_outer_j = jnp.asarray(mask_outer)
    data_mask = mask_valid_j & (mask_outer_j if use_outer_only else jnp.ones_like(mask_valid_j, dtype=bool))

    # Create knot centers in ln R, shared across galaxies
    lnR_min = jnp.log(jnp.maximum(jnp.nanmin(R), 1e-3))
    lnR_max = jnp.log(jnp.maximum(jnp.nanmax(R), 1e-3))
    centers = jnp.linspace(lnR_min, lnR_max, n_knots)

    def model(R, Vobs, eV, Vbar, data_mask):
        # Global parameters
        xi_max = numpyro.sample("xi_max", dist.TruncatedNormal(2.5, 1.5, low=1.0, high=10.0))
        width = numpyro.sample("width", dist.TruncatedNormal(0.7, 0.5, low=0.05, high=3.0))
        # Sparse coefficients for a monotone logit of xi
        # xi = 1 + (xi_max-1) * sigmoid( b0 + sum_k beta_k * relu(lnR - c_k) )
        b0 = numpyro.sample("b0", dist.Normal(0.0, 2.0))
        tau = numpyro.sample("tau", dist.HalfCauchy(1.0))  # global shrinkage
        with numpyro.plate("knots", n_knots):
            lam = numpyro.sample("lambda", dist.HalfCauchy(1.0))
            beta = numpyro.sample("beta", dist.Normal(0.0, tau*lam))
        # Hyper
        sigma_ML = numpyro.sample("sigma_ML", dist.TruncatedNormal(0.2, 0.2, low=0.01, high=0.8))
        nu = numpyro.sample("nu", dist.TruncatedNormal(6.0, 3.0, low=2.0, high=50.0))

        G = R.shape[0]
        with numpyro.plate("galaxies", G):
            dlogML_g = numpyro.sample("dlogML_g", dist.Normal(0.0, sigma_ML))
            sigma_int_g = numpyro.sample("sigma_int_g", dist.HalfCauchy(5.0))

            Vbar_adj = Vbar * jnp.exp(0.5 * dlogML_g)[:,None]

            lnR = jnp.log(jnp.maximum(R, 1e-6))
            # ReLU basis
            phi = jnp.maximum(lnR[...,None] - centers[None,None,:], 0.0)  # [G,J,K]
            logit_xi = b0 + jnp.sum(phi * beta[None,None,:], axis=-1) / jnp.maximum(width, 1e-6)
            xi = 1.0 + (xi_max - 1.0) * (1.0/(1.0+jnp.exp(-logit_xi)))
            Vpred = Vbar_adj * jnp.sqrt(jnp.clip(xi, 1.0, 100.0))

            sigma_eff = jnp.sqrt(jnp.square(eV) + jnp.square(sigma_int_g)[:,None])
            numpyro.sample("obs", dist.StudentT(nu, Vpred, sigma_eff).mask(data_mask), obs=Vobs)

    if platform == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
    elif platform == "mps":
        os.environ["JAX_PLATFORMS"] = "metal"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["JAX_ENABLE_X64"] = "true"

    kernel = NUTS(model, target_accept_prob=target_accept, dense_mass=True)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True)
    rng = random.PRNGKey(rng_key)
    mcmc.run(rng, R, Vobs, eV, Vbar, data_mask)
    mcmc.print_summary()

    os.makedirs(save_dir, exist_ok=True)
    samples = mcmc.get_samples(group_by_chain=False)
    np.savez(os.path.join(save_dir, "posterior_samples_basis.npz"), **{k: np.asarray(v) for k,v in samples.items()})
    return samples, names
