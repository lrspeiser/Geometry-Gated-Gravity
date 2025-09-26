from __future__ import annotations

class O3PINN:
    """Placeholder for an O3 learner targeting lensing ΔΣ residuals.
    TODO: Ingest cluster profiles (ACCEPT etc.), weak/strong lensing, and provide
    a differentiable projection path. For now, this is a stub.
    """
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        raise NotImplementedError("O3PINN is a stub; implement cluster ΔΣ training.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError
