import numpy as np
import pytest
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma

try:
    from rigor.rigor.data import load_sparc
except Exception:
    load_sparc = None


def test_feature_shapes_and_finiteness():
    # Fallback synthetic if SPARC not present; in repo it should be present
    if load_sparc is None:
        R = np.linspace(0.1, 10, 100)
        S = 100.0 / (1.0 + (R/5)**2)
    else:
        ds = load_sparc()
        assert len(ds.galaxies) > 0
        g = ds.galaxies[0]
        R = g.R_kpc
        S = g.Sigma_bar if g.Sigma_bar is not None else np.maximum(1.0, np.exp(-R))
    x = dimensionless_radius(R, Rd=3.0)
    Sh = sigma_hat(S)
    dlnS = grad_log_sigma(R, S)
    for arr in (x, Sh, dlnS):
        assert np.all(np.isfinite(np.asarray(arr)))
        assert arr.shape == R.shape
