
from __future__ import annotations
import os, json, math
from typing import Dict, List, Optional, Tuple
import numpy as np

def fit_hierarchical(dataset, xi_name="shell_logistic_radius", rng_key=0,
                     use_outer_only=False, num_warmup=1500, num_samples=1500, num_chains=4,
                     target_accept=0.8, platform="gpu", save_dir="results_numpyro"):

    # Select backend BEFORE importing jax
    if platform == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("JAX_ENABLE_X64", "true")
    elif platform == "mps":
        os.environ["JAX_PLATFORMS"] = "metal"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("JAX_ENABLE_X64", "true")
    else:
        # Force CPU to avoid accidental Metal selection when jax-metal is installed
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("JAX_ENABLE_X64", "true")

    import jax, jax.numpy as jnp
    from jax import random, vmap
    import numpyro
    from numpyro import distributions as dist
    from numpyro.infer import MCMC, NUTS
    from .xi import XI_REGISTRY, shell_logistic_radius, logistic_density

    if xi_name not in XI_REGISTRY:
        raise KeyError(f"Unknown xi_name '{xi_name}'. Available: {list(XI_REGISTRY)}")
    # We will combine radius and density logistics multiplicatively (minus-one parts),
    # which typically avoids overshoot while remaining smooth.

    # Pack padded arrays
    G = len(dataset.galaxies)
    maxJ = max(len(g.R_kpc) for g in dataset.galaxies)
    R = np.full((G, maxJ), np.nan, dtype=np.float32)
    Vobs = np.full_like(R, np.nan)
    eV = np.full_like(R, np.nan)
    Vbar = np.full_like(R, np.nan)
    Sigma = np.full_like(R, np.nan)
    mask_valid = np.zeros_like(R, dtype=bool)
    mask_outer = np.zeros_like(R, dtype=bool)
    Mbar = np.zeros((G,), dtype=np.float32)

    names = []
    for g_idx, g in enumerate(dataset.galaxies):
        J = len(g.R_kpc)
        R[g_idx,:J] = g.R_kpc
        Vobs[g_idx,:J] = g.Vobs_kms
        eV[g_idx,:J] = g.eVobs_kms
        Vbar[g_idx,:J] = g.Vbar_kms
        if g.Sigma_bar is not None:
            Sigma[g_idx,:J] = g.Sigma_bar
        mask_valid[g_idx,:J] = True
        mask_outer[g_idx,:J] = g.outer_mask
        Mbar[g_idx] = g.Mbar_Msun if (g.Mbar_Msun and np.isfinite(g.Mbar_Msun)) else 1e10
        names.append(g.name)

    # To JAX
    R = jnp.asarray(R); Vobs=jnp.asarray(Vobs); eV=jnp.asarray(eV); Vbar=jnp.asarray(Vbar); Sigma=jnp.asarray(Sigma)
    mask_valid = jnp.asarray(mask_valid); mask_outer=jnp.asarray(mask_outer)
    Mbar = jnp.asarray(Mbar)

    data_mask = mask_valid & (mask_outer if use_outer_only else jnp.ones_like(mask_valid, dtype=bool))

    def model(R, Vobs, eV, Vbar, Sigma, data_mask, Mbar):
        # Global xi parameters
        xi_max = numpyro.sample("xi_max", dist.TruncatedNormal(2.5, 1.5, low=1.0, high=10.0))
        lnR0_base = numpyro.sample("lnR0_base", dist.Normal(jnp.log(3.0), 1.5))
        width = numpyro.sample("width", dist.TruncatedNormal(0.6, 0.5, low=0.05, high=3.0))
        alpha_M = numpyro.sample("alpha_M", dist.Normal(-0.2, 0.5))

        lnSigma_c = numpyro.sample("lnSigma_c", dist.Normal(jnp.log(10.0), 2.0))
        width_sigma = numpyro.sample("width_sigma", dist.TruncatedNormal(0.6, 0.5, low=0.05, high=3.0))
        n_sigma = numpyro.sample("n_sigma", dist.TruncatedNormal(1.0, 1.0, low=0.1, high=10.0))

        # Hyperpriors
        sigma_ML = numpyro.sample("sigma_ML", dist.TruncatedNormal(0.2, 0.2, low=0.01, high=0.8))
        nu = numpyro.sample("nu", dist.TruncatedNormal(6.0, 3.0, low=2.0, high=50.0))

        G = R.shape[0]; J = R.shape[1]
        # Per-galaxy nuisance parameters
        with numpyro.plate("galaxies", G):
            dlogML_g = numpyro.sample("dlogML_g", dist.Normal(0.0, sigma_ML))
            sigma_int_g = numpyro.sample("sigma_int_g", dist.HalfCauchy(5.0))

        # Adjusted baryonic speed per galaxy (broadcast over radii)
        Vbar_adj = Vbar * jnp.exp(0.5 * dlogML_g)[:, None]

        # xi parameters dict (all jnp scalars)
        params = {
            "xi_max": xi_max,
            "lnR0_base": lnR0_base,
            "width": width,
            "alpha_M": alpha_M,
            "lnSigma_c": lnSigma_c,
            "width_sigma": width_sigma,
            "n_sigma": n_sigma,
            "Mref": 1e10,
        }

        # Compute xi for each galaxy row with vectorized calls
        def xi_for_g(g):
            xi_r = shell_logistic_radius(R[g], None, Mbar[g], params)
            xi_d = logistic_density(R[g], Sigma[g], Mbar[g], params)
            xi = jnp.minimum(xi_r * xi_d, 10.0)
            return xi

        xi = vmap(xi_for_g)(jnp.arange(G))
        Vpred = Vbar_adj * jnp.sqrt(jnp.clip(xi, 1.0, 100.0))

        sigma_eff = jnp.sqrt(jnp.square(eV) + jnp.square(sigma_int_g)[:, None])
        # Sanitize arrays to avoid NaNs/Inf in distribution parameters outside the mask
        Vpred = jnp.where(data_mask, Vpred, 0.0)
        Vobs_c = jnp.where(data_mask, jnp.nan_to_num(Vobs, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        sigma_eff = jnp.nan_to_num(sigma_eff, nan=1.0, posinf=1e6, neginf=1e6)
        sigma_eff = jnp.where(data_mask, sigma_eff, 1.0)
        # Use masked observation to avoid dynamic indexing under JAX transformations
        numpyro.sample(
            "obs",
            dist.StudentT(nu, Vpred, sigma_eff).mask(data_mask),
            obs=Vobs_c,
        )

    # Backend already selected above before importing jax

    kernel = NUTS(model, target_accept_prob=target_accept, dense_mass=True)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True)
    rng = random.PRNGKey(rng_key)
    mcmc.run(rng, R, Vobs, eV, Vbar, Sigma, data_mask, Mbar)
    mcmc.print_summary()

    os.makedirs(save_dir, exist_ok=True)
    samples = mcmc.get_samples(group_by_chain=False)
    np.savez(os.path.join(save_dir, "posterior_samples.npz"), **{k: np.asarray(v) for k,v in samples.items()})
    with open(os.path.join(save_dir, "galaxy_names.json"), "w") as f:
        json.dump(list(names), f, indent=2)
    return samples, names
