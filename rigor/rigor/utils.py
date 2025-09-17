
import numpy as np

def safe_div(a, b, eps=1e-12):
    return a / np.maximum(b, eps)

def finite_mask(*arrays):
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask

def to_numpy(x):
    import numpy as np
    return np.asarray(x)

def groupby_first(xs, keys):
    out = {}
    for x,k in zip(xs, keys):
        out.setdefault(k, x)
    return out

def broadcast_like(x, ref):
    import numpy as np
    return np.broadcast_to(x, ref.shape)

def logit(p, eps=1e-8):
    import numpy as np
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def inv_logit(x):
    import numpy as np
    return 1.0/(1.0+np.exp(-x))
