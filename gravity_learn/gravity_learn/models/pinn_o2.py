from __future__ import annotations
import json, os, datetime
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception as e:  # pragma: no cover
    jax = None
    jnp = None


class O2PINN:
    """A small parametric gating model for the tail acceleration g_tail(R).
    g_tail(R) = a * softplus((R/R0)^b) / (1 + (R/R1)^c)
    Parameters a,b,c,R0,R1 are constrained positive.
    """
    def __init__(self, seed: int = 0, lr: float = 1e-2):
        if jax is None:
            raise ImportError("JAX is required for O2PINN; install jax (or jax[cuda12]) per README.")
        self.key = jax.random.PRNGKey(seed)
        self.params = {
            "a": jnp.array(1e-2),
            "b": jnp.array(1.0),
            "c": jnp.array(2.0),
            "R0": jnp.array(3.0),
            "R1": jnp.array(10.0),
        }
        self.opt = {"lr": lr, "m": {k: jnp.zeros_like(v) for k, v in self.params.items()}, "v": {k: jnp.zeros_like(v) for k, v in self.params.items()}, "t": 0}

    @staticmethod
    def _softplus(x):
        return jnp.log1p(jnp.exp(x))

    def predict(self, R):
        p = self.params
        a = jnp.abs(p["a"]) + 1e-12
        b = jnp.abs(p["b"]) + 1e-6
        c = jnp.abs(p["c"]) + 1e-6
        R0 = jnp.abs(p["R0"]) + 1e-6
        R1 = jnp.abs(p["R1"]) + 1e-6
        x0 = (R / R0) ** b
        x1 = (R / R1) ** c
        return a * self._softplus(x0) / (1.0 + x1)

    def loss(self, R, gX, sigma_g=None, weight_smooth=1e-3, weight_positive=1e-3):
        pred = self.predict(R)
        if sigma_g is None:
            sigma_g = jnp.ones_like(pred)
        chi2 = jnp.mean(((pred - gX) / (sigma_g + 1e-12)) ** 2)
        # Smoothness: penalize curvature in ln R space
        eps = 1e-12
        lnR = jnp.log(jnp.maximum(R, eps))
        d_pred = (pred[2:] - pred[:-2]) / (lnR[2:] - lnR[:-2] + eps)
        smooth = jnp.mean(d_pred * d_pred)
        # Positivity (soft) — enforce g_tail >= 0
        pos = jnp.mean(jnp.square(jnp.minimum(pred, 0.0)))
        return chi2 + weight_smooth * smooth + weight_positive * pos

    def _adam_step(self, grads):
        # Minimal Adam
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.opt["t"] += 1
        t = self.opt["t"]
        lr = self.opt["lr"]
        for k in self.params.keys():
            g = grads[k]
            self.opt["m"][k] = b1 * self.opt["m"][k] + (1 - b1) * g
            self.opt["v"][k] = b2 * self.opt["v"][k] + (1 - b2) * (g * g)
            m_hat = self.opt["m"][k] / (1 - b1 ** t)
            v_hat = self.opt["v"][k] / (1 - b2 ** t)
            self.params[k] = self.params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    def fit(self, R, gX, sigma_g=None, epochs: int = 60, weight_smooth: float = 1e-3, weight_positive: float = 1e-3):
        R = jnp.asarray(R)
        gX = jnp.asarray(gX)
        sigma_g = None if sigma_g is None else jnp.asarray(sigma_g)

        def loss_wrapped(params):
            # bind params into self temporarily
            old = self.params
            self.params = params
            val = self.loss(R, gX, sigma_g, weight_smooth, weight_positive)
            self.params = old
            return val

        @jax.jit
        def grad_fn(params):
            return jax.grad(lambda p: loss_wrapped(p))(params)

        for _ in range(epochs):
            grads = grad_fn(self.params)
            self._adam_step(grads)
        return self

    def save_artifacts(self, out_dir: str, R_eval=None):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Params
        with open(os.path.join(out_dir, f"params_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in self.params.items()}, f, indent=2)
        # Optional predictions
        if R_eval is not None:
            pred = np.asarray(self.predict(jnp.asarray(R_eval)))
            np.savetxt(os.path.join(out_dir, f"pred_{ts}.csv"), np.c_[R_eval, pred], delimiter=",", header="R_kpc,g_tail", comments="")
