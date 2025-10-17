"""Check observed theta_E vs M_500 to see if Tier 3 is anomalous"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

catalog = pd.read_csv(Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv')

# Exclude MACS0717 (merger), A1689, MACS1149 (holdouts)
train = catalog[~catalog['cluster_name'].isin(['MACS0717', 'A1689', 'MACS1149'])].copy()

# Sort by mass
train = train.sort_values('M_500_Msun')

print("Observed theta_E vs M_500:")
print(f"{'Cluster':<12} {'Tier':>4} {'M_500':>8} {'R_500':>7} {'theta_E':>8} {'Dynamical':>10}")
for _, c in train.iterrows():
    m = c['M_500_Msun'] / 1e15
    r = c['R_500_kpc']
    te = c['theta_E_obs_arcsec']
    print(f"{c['cluster_name']:<12} {c['tier']:>4} {m:>8.2f} {r:>7.0f} {te:>8.1f} {c['dynamical_state']:>10}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for tier in [1, 2, 3]:
    tier_data = train[train['tier'] == tier]
    ax.scatter(tier_data['M_500_Msun']/1e15, tier_data['theta_E_obs_arcsec'], 
               label=f'Tier {tier}', s=100, alpha=0.7)
    for _, c in tier_data.iterrows():
        ax.text(c['M_500_Msun']/1e15, c['theta_E_obs_arcsec']+1.5, c['cluster_name'], 
                fontsize=8, ha='center')

ax.set_xlabel('M_500 [10^15 Msun]')
ax.set_ylabel('theta_E_obs [arcsec]')
ax.set_title('Observed Einstein Radii vs Mass')
ax.legend()
ax.grid(True, alpha=0.3)

output = Path(__file__).parent.parent / 'output' / 'theta_E_vs_mass.png'
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=150, bbox_inches='tight')
print(f"\nSaved: {output}")

# Compute correlation
corr = np.corrcoef(train['M_500_Msun'], train['theta_E_obs_arcsec'])[0,1]
print(f"\nPearson correlation (M_500 vs theta_E): {corr:.3f}")

# Check if Tier 3 is systematically different
tier12 = train[train['tier'].isin([1,2])]
tier3 = train[train['tier'] == 3]

print(f"\nTier 1-2: <M_500>={tier12['M_500_Msun'].mean()/1e15:.2f}, <theta_E>={tier12['theta_E_obs_arcsec'].mean():.1f}\"")
print(f"Tier 3:   <M_500>={tier3['M_500_Msun'].mean()/1e15:.2f}, <theta_E>={tier3['theta_E_obs_arcsec'].mean():.1f}\"")

# Expected theta_E from simple M ~ theta_E^2 scaling
print("\nIf theta_E ~ sqrt(M):")
for _, c in tier3.iterrows():
    expected = tier12['theta_E_obs_arcsec'].mean() * np.sqrt(c['M_500_Msun'] / tier12['M_500_Msun'].mean())
    ratio = c['theta_E_obs_arcsec'] / expected
    print(f"  {c['cluster_name']:<12}: obs={c['theta_E_obs_arcsec']:5.1f}\", expect~{expected:5.1f}\", ratio={ratio:.2f}")
