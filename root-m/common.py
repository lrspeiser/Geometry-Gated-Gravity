# -*- coding: utf-8 -*-
import numpy as np

# Soft Root-M tail v^2 term: mass-aware amplitude times a single global shape r/(r+rc)
# v_tail^2(r) = A^2 * sqrt(Mb(<r)/Mref) * r/(r+rc)
# Returns v^2 in (km/s)^2

def v_tail2_rootm_soft(r_kpc, Mb_enc_Msun, A_kms=140.0, Mref=6.0e10, rc_kpc=15.0):
    r = np.asarray(r_kpc, dtype=float)
    Mb = np.asarray(Mb_enc_Msun, dtype=float)
    amp = (A_kms**2) * np.sqrt(np.clip(Mb / max(Mref, 1e-30), 0.0, None))
    shape = r / np.maximum(r + rc_kpc, 1e-12)
    return amp * shape