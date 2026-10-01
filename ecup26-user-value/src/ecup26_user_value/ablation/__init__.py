"""Ablation matrix for E-CUP T3 user-value (T3.6.12).

Per .clinerules/02-ecup-rules.md:
- R1: open-source; lightgbm (Apache 2.0), lifetimes (Apache 2.0), sklearn (BSD)
- R4: no internet at runtime; all CPU models, no HF download
- R6: seeded synthetic data, deterministic time-CV
- R7: CPU tabular models, no LLM or inference server
- R8: lightgbm lives in [dependency-groups].dev, NOT [project].dependencies,
  so it never reaches the submission Docker image. Ablation is in scripts/,
  never imported by run.py.
- R10: writes docs/results/*.csv, never submit.csv.

v1: 4 models (btyd_global, btyd_segmented, lgbm_mse, lgbm_ziln)
    x 2 feature groups (base_features, btyd_features)
    x leave-one-out (3 combos)
    x folds (default 2)
    = 24 rows CSV.
"""

from __future__ import annotations
