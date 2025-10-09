import pandas as pd
import numpy as np

g = pd.read_csv(r'C:\Users\henry\dev\GravityCalculator\data\clusters\MACSJ0416\gas_profile.csv')
MU_E = 1.17
M_P_G = 1.67262192369e-24
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.988409870698051e33

rho = MU_E * M_P_G * g['n_e_cm3'].values * (KPC_CM**3) / MSUN_G

print(f'n_e range: {g["n_e_cm3"].min():.2e} to {g["n_e_cm3"].max():.2e} cm-3')
print(f'rho range: {rho.min():.2e} to {rho.max():.2e} Msun/kpc3')
print(f'First 5 rho values: {rho[:5]}')
print(f'First 5 r values: {g["r_kpc"].values[:5]}')
