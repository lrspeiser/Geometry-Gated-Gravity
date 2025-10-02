"""
generate_figure4_lensing.py

Creates Figure 4: 3-panel cluster lensing failure demonstration
- Panel (a): Abell 1689 - κ and γ profiles
- Panel (b): Bullet Cluster - κ and γ profiles
- Panel (c): Coma Cluster - κ and γ profiles

Shows that O2 ratio_curv fails to reproduce observed strong lensing.
Uses pre-generated lensing diagnostic plots from best-fit results.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
LENS_DIR = BASE_DIR / "results" / "best_fit" / "mape_median_20250926_2259" / "lensing_o2"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Source images - kappa_gamma plots for 3 clusters
clusters = {
    'Abell_1689': LENS_DIR / "Abell_1689" / "kappa_gamma_o2.png",
    'Bullet': LENS_DIR / "Bullet" / "kappa_gamma_o2.png",
    'Coma': LENS_DIR / "Coma" / "kappa_gamma_o2.png",
}

# Check if all files exist
missing = [k for k, v in clusters.items() if not v.exists()]
if missing:
    print(f"Missing cluster lensing files: {missing}")
    print(f"Searched in: {LENS_DIR}")
    print("\nAvailable cluster folders:")
    for d in LENS_DIR.glob("*"):
        if d.is_dir():
            print(f"  {d.name}")
            for f in d.glob("*.png"):
                print(f"    {f.name}")
    exit(1)

# Load images
imgs = {k: np.array(Image.open(v)) for k, v in clusters.items()}

# Create figure - 1 row, 3 columns
fig = plt.figure(figsize=(18, 6))
gs = GridSpec(1, 3, figure=fig, hspace=0.15, wspace=0.25, 
              left=0.05, right=0.98, top=0.90, bottom=0.10)

# Panel labels
labels = ['(a) Abell 1689', '(b) Bullet', '(c) Coma']
cluster_names = ['Abell_1689', 'Bullet', 'Coma']

for i, (label, cluster_name) in enumerate(zip(labels, cluster_names)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(imgs[cluster_name])
    ax.axis('off')
    ax.text(0.02, 0.98, label, transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Title
fig.suptitle('Figure 4: Cluster Lensing Failure - O2 ratio_curv Cannot Reproduce Strong Lensing',
             fontsize=16, fontweight='bold', y=0.97)

# Save
output_png = FIG_DIR / "Figure4_Cluster_Lensing.png"
output_pdf = FIG_DIR / "Figure4_Cluster_Lensing.pdf"

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"✅ Figure 4 generated:")
print(f"   {output_png}")
print(f"   {output_pdf}")

plt.close()
