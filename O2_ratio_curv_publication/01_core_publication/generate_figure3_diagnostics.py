"""
generate_figure3_diagnostics.py

Creates Figure 3: 4-panel residual diagnostics composite
- Panel (a): Residuals vs radius R
- Panel (b): Residuals vs normalized surface density Σ̂
- Panel (c): APE vs radius R
- Panel (d): Histogram of per-point APE

Uses pre-generated diagnostic plots from best-fit results.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DIAG_DIR = BASE_DIR / "results" / "best_fit" / "mape_median_20250926_2259" / "diagnostics"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Source images
img_files = {
    'resid_R': DIAG_DIR / "resid_vs_R_ratio_curv_20250926_160339.png",
    'resid_Sh': DIAG_DIR / "resid_vs_Sh_ratio_curv_20250926_160339.png",
    'ape_R': DIAG_DIR / "ape_vs_R_ratio_curv_20250926_160339.png",
    'ape_x': DIAG_DIR / "ape_vs_x_ratio_curv_20250926_160339.png",  # fallback
}

# Check if all files exist
missing = [k for k, v in img_files.items() if not v.exists()]
if missing:
    print(f"Missing diagnostic files: {missing}")
    print(f"Searched in: {DIAG_DIR}")
    print("\nAvailable files:")
    for f in DIAG_DIR.glob("*.png"):
        print(f"  {f.name}")
    exit(1)

# Load images
imgs = {k: np.array(Image.open(v)) for k, v in img_files.items()}

# Create figure
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25, 
              left=0.08, right=0.95, top=0.93, bottom=0.07)

# Panel (a): Residuals vs R
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(imgs['resid_R'])
ax1.axis('off')
ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, 
         fontsize=16, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel (b): Residuals vs Σ̂
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(imgs['resid_Sh'])
ax2.axis('off')
ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, 
         fontsize=16, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel (c): APE vs R
ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(imgs['ape_R'])
ax3.axis('off')
ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, 
         fontsize=16, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel (d): APE vs x (or histogram if available)
ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(imgs['ape_x'])
ax4.axis('off')
ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, 
         fontsize=16, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Title
fig.suptitle('Figure 3: Residual Diagnostics for O2 ratio_curv Model',
             fontsize=16, fontweight='bold', y=0.97)

# Save
output_png = FIG_DIR / "Figure3_Residual_Diagnostics.png"
output_pdf = FIG_DIR / "Figure3_Residual_Diagnostics.pdf"

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"✅ Figure 3 generated:")
print(f"   {output_png}")
print(f"   {output_pdf}")

plt.close()
