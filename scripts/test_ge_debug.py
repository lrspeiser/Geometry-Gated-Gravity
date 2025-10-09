#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lensing_utils import alpha_fun_GE
import traceback

try:
    result = alpha_fun_GE('MACSJ0416', 0.396, 2.0,
                         a=3.0, b=0.2, d=0.1,
                         gamma1=0.2, gamma2=0.1,
                         Rd_kpc=1000.0, R_scale_kpc=100.0,
                         beta_clip=(1.0, 5.0),
                         A_core=0.25, p_core=2.0,
                         Sigma0_hat=0.0, beta_core=0.4,
                         smooth_R_kpc=5.0)
    print(f"Result: {result}")
    if result is not None:
        print(f"Test: alpha(10) = {result(10.0)}")
except Exception as e:
    print(f"Exception: {e}")
    traceback.print_exc()
