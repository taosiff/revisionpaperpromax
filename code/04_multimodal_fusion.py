"""
04_multimodal_fusion.py
=======================
Multimodal Fusion Pipeline & Statistical Multiplicity Audit
REVISION_PAPER_PRO_MAX (Strict Zero-Leakage & Transparent Q1 Revision)

Methodological Compliance:
1. Multi-Paradigm Comparison:
   - Dawes' Parameter-Free Equal Weighting (1/3, 1/3, 1/3)
   - Train-OOF Calibrated Threshold Fusion (Threshold selected on Train OOF)
   - Rank Averaging (Percentile Rank Ensembling)
   - Train-OOF SNR-Weighted Fusion (derived purely from Train CV AUC)
   - Train-OOF Stacking Meta-Learner (fit on Train OOF, zero val/test leakage)
   - Parameterized Overfitting Baselines: 186-Grid Val-Tuned Late Fusion & Cross-Modal Attention
2. Strict Decision Threshold Policy:
   - Decision threshold tau is chosen strictly on Train Out-Of-Fold (OOF) predictions
     or fixed a priori (tau = 0.50). Zero threshold search on validation or test labels.
3. Family-Cluster Bootstrapping:
   - 5,000 resamples at the family cluster level to avoid underestimating CI width.
   - Rigorous multiplicity control: Benjamini-Hochberg FDR and Bonferroni adjustments.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    brier_score_loss,
    precision_score,
    recall_score
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
PROCESSED_DIR = DATA_DIR / 'processed'
OUT_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("04_MULTIMODAL_FUSION: ZERO-LEAKAGE FUSION & MULTIPLICITY CONTROL AUDIT")
print("=" * 80)

# 1. Load Partitions and Predictions
val_df = pd.read_csv(SPLITS_DIR / 'val.csv')
test_df = pd.read_csv(SPLITS_DIR / 'test.csv')
train_df = pd.read_csv(SPLITS_DIR / 'train.csv')

y_train = train_df['internalizing_incident'].values.astype(int)
y_val = val_df['internalizing_incident'].values.astype(int)
y_test = test_df['internalizing_incident'].values.astype(int)

test_fams = test_df['family_id'].values

# Load Unimodal Predictions
p_tr_m1 = np.load(PROCESSED_DIR / 'm1_train_probs.npy')
p_va_m1 = np.load(PROCESSED_DIR / 'm1_val_probs.npy')
p_te_m1 = np.load(PROCESSED_DIR / 'm1_test_probs.npy')
oof_m1 = np.load(PROCESSED_DIR / 'm1_oof_probs.npy')

p_va_m2 = np.load(PROCESSED_DIR / 'm2_val_probs.npy')
p_te_m2 = np.load(PROCESSED_DIR / 'm2_test_probs.npy')
oof_m2 = np.load(PROCESSED_DIR / 'm2_oof_probs.npy')

p_va_m3 = np.load(PROCESSED_DIR / 'm3_val_probs.npy')
p_te_m3 = np.load(PROCESSED_DIR / 'm3_test_probs.npy')
oof_m3 = np.load(PROCESSED_DIR / 'm3_oof_probs.npy')

# Baseline FT-Proto Predictions (for parameterization contrast)
p_va_m2_proto = np.load(PROCESSED_DIR / 'm2_ftproto_val_probs.npy')
p_te_m2_proto = np.load(PROCESSED_DIR / 'm2_ftproto_test_probs.npy')

p_va_m3_proto = np.load(PROCESSED_DIR / 'm3_ftproto_val_probs.npy')
p_te_m3_proto = np.load(PROCESSED_DIR / 'm3_ftproto_test_probs.npy')

print(f"Loaded unimodal predictions for Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

# Helper: Tune threshold strictly on Train OOF
def get_train_optimal_f1_threshold(y_true_tr, oof_probs_tr):
    best_th = 0.50
    best_f1 = 0.0
    for th in np.linspace(0.10, 0.90, 81):
        preds = (oof_probs_tr >= th).astype(int)
        if preds.sum() == 0:
            continue
        f = f1_score(y_true_tr, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
    return best_th, best_f1

def eval_full(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, preds, zero_division=0)
    brier = brier_score_loss(y_true, probs)
    nns = (tp + fp) / tp if tp > 0 else np.nan
    return {
        'AUC': auc,
        'PR_AUC': pr_auc,
        'F1': f1,
        'Sensitivity': sens,
        'Specificity': spec,
        'Precision': prec,
        'NPV': npv,
        'Brier': brier,
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'NNS': nns
    }

# =========================================================================
# FUSION PARADIGMS
# =========================================================================

# 1. Dawes' Parameter-Free Equal Average (1/3, 1/3, 1/3)
oof_eq = (oof_m1 + oof_m2 + oof_m3) / 3.0
p_va_eq = (p_va_m1 + p_va_m2 + p_va_m3) / 3.0
p_te_eq = (p_te_m1 + p_te_m2 + p_te_m3) / 3.0
th_eq_train, _ = get_train_optimal_f1_threshold(y_train, oof_eq)

# 2. Rank Averaging (Percentile Rank Ensembling)
def rank_tf(p):
    return rankdata(p) / len(p)

oof_rank = (rank_tf(oof_m1) + rank_tf(oof_m2) + rank_tf(oof_m3)) / 3.0
p_va_rank = (rank_tf(p_va_m1) + rank_tf(p_va_m2) + rank_tf(p_va_m3)) / 3.0
p_te_rank = (rank_tf(p_te_m1) + rank_tf(p_te_m2) + rank_tf(p_te_m3)) / 3.0
th_rank_train, _ = get_train_optimal_f1_threshold(y_train, oof_rank)

# 3. Train-OOF SNR-Weighted Fusion
snr1 = max(0.0, roc_auc_score(y_train, oof_m1) - 0.50)
snr2 = max(0.0, roc_auc_score(y_train, oof_m2) - 0.50)
snr3 = max(0.0, roc_auc_score(y_train, oof_m3) - 0.50)
snr_sum = snr1 + snr2 + snr3
w_snr = [snr1 / snr_sum, snr2 / snr_sum, snr3 / snr_sum]

oof_snr = w_snr[0] * oof_m1 + w_snr[1] * oof_m2 + w_snr[2] * oof_m3
p_va_snr = w_snr[0] * p_va_m1 + w_snr[1] * p_va_m2 + w_snr[2] * p_va_m3
p_te_snr = w_snr[0] * p_te_m1 + w_snr[1] * p_te_m2 + w_snr[2] * p_te_m3
th_snr_train, _ = get_train_optimal_f1_threshold(y_train, oof_snr)

# 4. Train-OOF Stacking Meta-Learner (Ridge fit on Train OOF)
X_oof_st = np.column_stack([oof_m1, oof_m2, oof_m3])
X_va_st = np.column_stack([p_va_m1, p_va_m2, p_va_m3])
X_te_st = np.column_stack([p_te_m1, p_te_m2, p_te_m3])

meta_clf = LogisticRegression(C=0.1, penalty='l2', class_weight='balanced', random_state=SEED)
meta_clf.fit(X_oof_st, y_train)

oof_stack = meta_clf.predict_proba(X_oof_st)[:, 1]
p_va_stack = meta_clf.predict_proba(X_va_st)[:, 1]
p_te_stack = meta_clf.predict_proba(X_te_st)[:, 1]
th_stack_train, _ = get_train_optimal_f1_threshold(y_train, oof_stack)

# 5. Overfitted Parameterized Baselines (For Comparison)
# 186-Grid Validation-Tuned Weights
best_va_grid_auc = 0.0
best_w_grid = (1/3, 1/3, 1/3)
for w1 in np.linspace(0.15, 0.45, 16):
    for w2 in np.linspace(0.35, 0.65, 16):
        w3 = 1.0 - w1 - w2
        if 0.10 <= w3 <= 0.45:
            comb_va = w1 * p_va_m1 + w2 * p_va_m2_proto + w3 * p_va_m3_proto
            auc_v = roc_auc_score(y_val, comb_va)
            if auc_v > best_va_grid_auc:
                best_va_grid_auc = auc_v
                best_w_grid = (w1, w2, w3)

p_va_grid = best_w_grid[0] * p_va_m1 + best_w_grid[1] * p_va_m2_proto + best_w_grid[2] * p_va_m3_proto
p_te_grid = best_w_grid[0] * p_te_m1 + best_w_grid[1] * p_te_m2_proto + best_w_grid[2] * p_te_m3_proto

# FT-Proto Baseline Average
p_va_base_avg = (p_va_m1 + p_va_m2_proto + p_va_m3_proto) / 3.0
p_te_base_avg = (p_te_m1 + p_te_m2_proto + p_te_m3_proto) / 3.0

# Master Model Registry
models_dict = {
    "Module 1: Structural MRI (Pretrained ResNet-18)": (p_va_m1, p_te_m1, 0.50),
    "Module 2: Anxiety Symptomatology (Regularized)": (p_va_m2, p_te_m2, 0.50),
    "Module 2: Anxiety Symptomatology (FT-ProtoNet Baseline)": (p_va_m2_proto, p_te_m2_proto, 0.50),
    "Module 3: Static Phenotypic (Regularized)": (p_va_m3, p_te_m3, 0.50),
    "Module 3: Static Phenotypic (FT-ProtoNet Baseline)": (p_va_m3_proto, p_te_m3_proto, 0.50),
    "Ablation: MRI + Questionnaire": ((p_va_m1 + p_va_m2)/2, (p_te_m1 + p_te_m2)/2, 0.50),
    "Ablation: MRI + Phenotypic": ((p_va_m1 + p_va_m3)/2, (p_te_m1 + p_te_m3)/2, 0.50),
    "Ablation: Questionnaire + Phenotypic": ((p_va_m2 + p_va_m3)/2, (p_te_m2 + p_te_m3)/2, 0.50),
    "Multimodal: Dawes' Equal Average (Thresh=0.50)": (p_va_eq, p_te_eq, 0.50),
    "Multimodal: Dawes' Equal Average (Train-OOF Thresh)": (p_va_eq, p_te_eq, th_eq_train),
    "Multimodal: Rank Averaging (Screening Thresh=0.50)": (p_va_rank, p_te_rank, 0.50),
    "Multimodal: Rank Averaging (Train-OOF Thresh)": (p_va_rank, p_te_rank, th_rank_train),
    "Multimodal: SNR-Weighted Fusion": (p_va_snr, p_te_snr, th_snr_train),
    "Multimodal: Stacking Meta-Learner": (p_va_stack, p_te_stack, th_stack_train),
    "Overfit Baseline: 186-Grid Late Fusion": (p_va_grid, p_te_grid, 0.50),
    "Overfit Baseline: FT-Proto Average (1/3, 1/3, 1/3)": (p_va_base_avg, p_te_base_avg, 0.50),
}

# Compile Table
table_rows = []
for name, (p_va, p_te, th) in models_dict.items():
    vm = eval_full(y_val, p_va, threshold=th)
    tm = eval_full(y_test, p_te, threshold=th)
    table_rows.append({
        'Framework / Model': name,
        'Threshold': th,
        'Val AUC': round(vm['AUC'], 4),
        'Test AUC': round(tm['AUC'], 4),
        'Overfit Gap': round(vm['AUC'] - tm['AUC'], 4),
        'Test F1': round(tm['F1'], 4),
        'Sens': round(tm['Sensitivity'], 4),
        'Spec': round(tm['Specificity'], 4),
        'Precision': round(tm['Precision'], 4),
        'NPV': round(tm['NPV'], 4),
        'Brier': round(tm['Brier'], 4),
        'NNS': round(tm['NNS'], 2) if not np.isnan(tm['NNS']) else np.nan,
        'TP': tm['TP'],
        'FP': tm['FP'],
        'TN': tm['TN'],
        'FN': tm['FN']
    })

df_fusion = pd.DataFrame(table_rows)
df_fusion.to_csv(OUT_DIR / 'multimodal_fusion_table.csv', index=False)

print("\n" + "=" * 105)
print("MULTIMODAL FUSION RESULTS TABLE:")
print("=" * 105)
print(df_fusion[['Framework / Model', 'Threshold', 'Val AUC', 'Test AUC', 'Overfit Gap', 'Test F1', 'Sens', 'Spec', 'Precision', 'NNS']].to_string(index=False))

# =========================================================================
# 5,000 FAMILY-CLUSTER BOOTSTRAP SIGNIFICANCE & MULTIPLICITY CONTROL
# =========================================================================
print("\n--- Running 5,000 Family-Cluster Bootstraps for Multiplicity Control ---")
unique_fams = np.unique(test_fams)
fam_to_idx = {f: np.where(test_fams == f)[0] for f in unique_fams}
n_fams = len(unique_fams)

rng = np.random.RandomState(SEED)
models_to_test_mult = [
    ("Module 1: Structural MRI", p_te_m1),
    ("Module 2: Anxiety Symptomatology (Regularized)", p_te_m2),
    ("Module 3: Static Phenotypic (Regularized)", p_te_m3),
    ("Multimodal: Dawes' Equal Average", p_te_eq),
    ("Multimodal: Rank Averaging", p_te_rank),
    ("Multimodal: SNR-Weighted Fusion", p_te_snr),
    ("Multimodal: Stacking Meta-Learner", p_te_stack),
    ("Overfit Baseline: 186-Grid Late Fusion", p_te_grid)
]

boot_results = {name: [] for name, _ in models_to_test_mult}
boot_diffs_m1 = {name: [] for name, _ in models_to_test_mult}

for _ in range(5000):
    samp_fams = rng.choice(unique_fams, size=n_fams, replace=True)
    samp_idx = np.concatenate([fam_to_idx[f] for f in samp_fams])
    y_b = y_test[samp_idx]
    if len(np.unique(y_b)) < 2:
        continue
    
    a_m1 = roc_auc_score(y_b, p_te_m1[samp_idx])
    for name, p_te in models_to_test_mult:
        a = roc_auc_score(y_b, p_te[samp_idx])
        boot_results[name].append(a)
        boot_diffs_m1[name].append(a - a_m1)

mult_rows = []
for name, p_te in models_to_test_mult:
    obs_auc = roc_auc_score(y_test, p_te)
    b_arr = np.array(boot_results[name])
    ci_low, ci_high = np.percentile(b_arr, [2.5, 97.5])
    
    # p-value vs Chance (0.50)
    p_null = 2.0 * min(np.mean(b_arr <= 0.50), np.mean(b_arr >= 0.50))
    p_null = min(1.0, max(0.0001, p_null))
    
    # p-value vs M1
    if "Module 1" in name:
        p_m1 = 1.0
        delta_m1 = 0.0
    else:
        diff_arr = np.array(boot_diffs_m1[name])
        p_m1 = 2.0 * min(np.mean(diff_arr <= 0.0), np.mean(diff_arr >= 0.0))
        p_m1 = min(1.0, max(0.0001, p_m1))
        delta_m1 = obs_auc - roc_auc_score(y_test, p_te_m1)
        
    mult_rows.append({
        'Model': name,
        'Test AUC': round(obs_auc, 4),
        '95% Cluster CI': f"[{ci_low:.4f}, {ci_high:.4f}]",
        'Delta vs MRI': round(delta_m1, 4),
        'Raw p (vs Chance)': round(p_null, 4),
        'Raw p (vs MRI)': round(p_m1, 4)
    })

df_mult = pd.DataFrame(mult_rows)

# Benjamini-Hochberg FDR and Bonferroni adjustments
m_tests = len(df_mult)
p_vals = df_mult['Raw p (vs Chance)'].values

# Bonferroni
df_mult['Bonferroni p (vs Chance)'] = np.minimum(1.0, p_vals * m_tests).round(4)

# Benjamini-Hochberg FDR
sorted_idx = np.argsort(p_vals)
fdr_q = np.zeros(m_tests)
cum_min = 1.0
for rank_i, idx in reversed(list(enumerate(sorted_idx, 1))):
    q_val = p_vals[idx] * m_tests / rank_i
    cum_min = min(cum_min, q_val)
    fdr_q[idx] = cum_min
df_mult['FDR q (vs Chance)'] = fdr_q.round(4)

df_mult.to_csv(OUT_DIR / 'multiplicity_control_table.csv', index=False)
print("\nMULTIPLICITY CONTROL AUDIT TABLE:")
print(df_mult[['Model', 'Test AUC', '95% Cluster CI', 'Delta vs MRI', 'Raw p (vs Chance)', 'FDR q (vs Chance)', 'Raw p (vs MRI)']].to_string(index=False))

# Save processed arrays for figures
np.save(PROCESSED_DIR / 'p_te_eq.npy', p_te_eq)
np.save(PROCESSED_DIR / 'p_te_rank.npy', p_te_rank)
np.save(PROCESSED_DIR / 'p_te_grid.npy', p_te_grid)
np.save(PROCESSED_DIR / 'p_te_base_avg.npy', p_te_base_avg)

print(f"\n[OK] 04_multimodal_fusion.py completed successfully. Results saved in {OUT_DIR}")
