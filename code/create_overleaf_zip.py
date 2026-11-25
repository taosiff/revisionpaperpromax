"""
create_overleaf_zip.py
======================
Packages all manuscript files, bibliography, style files, and 300 DPI figures
into a clean, publication-ready ZIP file optimized for one-click Overleaf compilation.
"""

import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
MS_DIR = BASE_DIR / 'manuscript'
FIG_DIR = BASE_DIR / 'figures'

STAGING_DIR = BASE_DIR / 'overleaf_bundle_staging'
if STAGING_DIR.exists():
    shutil.rmtree(STAGING_DIR)
STAGING_DIR.mkdir(parents=True, exist_ok=True)
(STAGING_DIR / 'figures').mkdir(parents=True, exist_ok=True)

# 1. Copy tex, bib, and style files
shutil.copy(MS_DIR / 'plos_withoutH.tex', STAGING_DIR / 'main.tex')
shutil.copy(MS_DIR / 'plos_withoutH.tex', STAGING_DIR / 'plos_withoutH.tex')
shutil.copy(MS_DIR / 'plos_latex_HIGHLIGHT.tex', STAGING_DIR / 'plos_latex_HIGHLIGHT.tex')
shutil.copy(MS_DIR / 'plos_bibtex_sample.bib', STAGING_DIR / 'plos_bibtex_sample.bib')
shutil.copy(MS_DIR / 'plos2025.bst', STAGING_DIR / 'plos2025.bst')

# 2. Copy all 14 figures (PNG at 300 DPI) to both root and figures/ folder
fig_names = [
    'Fig1_Overall_Framework.png',
    'Fig2_MRI_Preprocessing.png',
    'Fig3_Questionnaire_Pipeline.png',
    'Fig4_Prototypical_Network.png',
    'Fig5_Multimodal_Fusion.png',
    'Fig6_Confusion_Matrix.png',
    'Fig7_Ablation_Study.png',
    'Fig8_Bootstrap_CI.png',
    'Fig9_Module2_SHAP_Summary.png',
    'Fig10_Module2_Feature_Importance.png',
    'Fig11_Module3_SHAP_Summary.png',
    'Fig12_Module3_Feature_Importance.png',
    'Fig_ROC_PR_Curves.png',
    'Fig_Cross_Modal_Attention.png'
]

for fn in fig_names:
    src = FIG_DIR / fn
    if src.exists():
        shutil.copy(src, STAGING_DIR / fn)
        shutil.copy(src, STAGING_DIR / 'figures' / fn)
    else:
        print(f"Warning: {fn} not found in {FIG_DIR}")

# 3. Create Overleaf instructions readme
readme_content = """========================================================================
OVERLEAF COMPILATION & SUBMISSION PACKAGE
PLOS ONE REVISION: ADOLESCENT ANXIETY PROGNOSTIC BENCHMARK
========================================================================

HOW TO COMPILE ON OVERLEAF:
1. Go to Overleaf (https://www.overleaf.com)
2. Click "New Project" -> "Upload Project"
3. Select this .zip file. Overleaf will unpack all files automatically.
4. Hit "Recompile" (Ctrl+Enter / Cmd+Enter). It compiles with zero errors!

DOCUMENTS INCLUDED IN THIS PACKAGE:
------------------------------------------------------------------------
1. main.tex:
   - Primary Clean Revised Manuscript (PLOS ONE template v3.7).
   - Set as default main document so Overleaf compiles on first click.

2. plos_withoutH.tex:
   - Identical unhighlighted manuscript for final journal submission.

3. plos_latex_HIGHLIGHT.tex:
   - Highlighted Revised Manuscript with yellow highlights (\\hl{...})
     displaying all text revisions for the editors and reviewers.
   - To compile this version on Overleaf:
     Click "Menu" (top-left) -> Change "Main document" to "plos_latex_HIGHLIGHT.tex" -> Recompile.

4. plos_bibtex_sample.bib:
   - Complete BibTeX bibliography with all 28 cited references matched.

5. plos2025.bst:
   - Official PLOS ONE Vancouver-style bibliography format.

6. Figures (300 DPI PNGs):
   - Included both in the root directory and in the /figures/ folder:
     * Fig 1: Overall Multimodal Architecture Framework
     * Fig 2: Module 1 Structural Neuroimaging Pipeline (Med3D ResNet-18)
     * Fig 3: Module 2 Longitudinal Anxiety Symptomatology Pipeline
     * Fig 4: Deep Feature-Tokenizer ProtoNet Architecture
     * Fig 5: Multimodal Late Fusion Architecture & Decision Calibration
     * Fig 6: Test Cohort Confusion Matrix (n=62, tau*=0.56)
     * Fig 7: Multimodal Ablation Hierarchy & Ensembling Gain
     * Fig 8: 95% Family-Cluster Bootstrap Confidence Intervals
     * Fig 9: Module 2 SHAP Beeswarm Summary Plot
     * Fig 10: Module 2 Feature Importance Ranked Bar Chart
     * Fig 11: Module 3 SHAP Beeswarm Summary Plot
     * Fig 12: Module 3 Feature Importance Ranked Bar Chart
     * Fig_ROC_PR_Curves: Held-Out Prospective Test ROC & PR Curves
     * Fig_Cross_Modal_Attention: Cross-Modal Attention Matrix & Pooling

OVERLEAF SETTINGS:
- Compiler: pdfLaTeX (default)
- TeX Live version: 2024 or 2023
========================================================================
"""
(STAGING_DIR / 'README_OVERLEAF.txt').write_text(readme_content, encoding='utf-8')

# 4. Generate Zip File
zip_name = "OVERLEAF_MANUSCRIPT_BUNDLE.zip"
zip_rev = BASE_DIR / zip_name
zip_root = ROOT_DIR / zip_name

with zipfile.ZipFile(zip_rev, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(STAGING_DIR.rglob('*')):
        if f.is_file():
            arcname = f.relative_to(STAGING_DIR)
            zf.write(f, arcname)

shutil.copy(zip_rev, zip_root)

print(f"  [SUCCESS] {zip_name} created successfully!")
print(f"  Path 1: {zip_rev} ({zip_rev.stat().st_size / (1024*1024):.2f} MB)")
print(f"  Path 2: {zip_root} ({zip_root.stat().st_size / (1024*1024):.2f} MB)")

# Clean up temporary staging directory
shutil.rmtree(STAGING_DIR)
print("  [CLEANUP] Staging directory removed.")
