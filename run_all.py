"""
run_all.py
==========
Master Reproduction Pipeline for REVISION_PAPER_PRO_MAX
Executes all modules sequentially, produces all metrics, tables, models, and 300 DPI figures.

Usage:
    python run_all.py
"""

import sys
import subprocess
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR / 'code'

scripts = [
    ("Module 1 (Structural MRI)", CODE_DIR / "01_smri_module.py"),
    ("Module 2 (Questionnaires)", CODE_DIR / "02_questionnaire_module.py"),
    ("Module 3 (Phenotypic)", CODE_DIR / "03_static_module.py"),
    ("Multimodal Fusion & Multiplicity Audit", CODE_DIR / "04_multimodal_fusion.py"),
    ("SHAP Explainability Analysis", CODE_DIR / "05_shap_analysis.py"),
    ("Feature Selection Audit", CODE_DIR / "06_feature_selection_audit.py"),
    ("Publication Figure Generation", CODE_DIR / "generate_figures.py"),
]

print("=" * 85)
print("REVISION_PAPER_PRO_MAX: END-TO-END PIPELINE REPRODUCTION")
print("=" * 85)

start_time = time.time()

for name, script_path in scripts:
    print(f"\n>>> EXECUTING: {name} ({script_path.name}) <<<")
    res = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
    if res.returncode != 0:
        print(f"\n[ERROR] Step {name} failed with return code {res.returncode}. Aborting.")
        sys.exit(res.returncode)

elapsed = time.time() - start_time
print("\n" + "=" * 85)
print(f"PIPELINE COMPLETE IN {elapsed:.2f} SECONDS!")
print("All tables saved in:   REVISION_PAPER_PRO_MAX/results/")
print("All figures saved in:  REVISION_PAPER_PRO_MAX/figures/")
print("All models saved in:   REVISION_PAPER_PRO_MAX/models/")
print("=" * 85)
