# Analysis Workflows Concept

This folder groups one-off and composite analysis scripts that orchestrate multiple components (optimizers, PDE solvers, validators) to produce figures, summaries, or comparisons.

Typical inputs
- SPARC Parquet/CSV under data/
- MW Gaia CSV/NPZ under data/

Common outputs
- Figures and CSV/JSON summaries next to scripts or under out/

Run examples
- python concepts/analysis_workflows/analyze_actual_predictions.py
- python concepts/analysis_workflows/optimize_g3_all_datasets.py
