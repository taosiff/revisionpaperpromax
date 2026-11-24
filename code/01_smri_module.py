"""
01_smri_module.py
=================
Module 1: Structural MRI Pipeline (Pretrained Med3D ResNet-18 + Strict Train-Only PCA-5 + Balanced Metric Head)
REVISION_PAPER_PRO_MAX (Strict Zero-Leakage & Transparent Q1 Revision)

Methodological Compliance:
1. Strict Training Isolation:
   - StandardScaler and PCA (k=5) fit strictly on train.csv (n=180, 24 pos, 90 families).
   - Validation (n=62, 7 pos, 31 families) and Test (n=62, 7 pos, 31 families) are strictly transformed.
2. Balanced Metric Head:
   - LogisticRegression(C=0.1, class_weight='balanced') fit strictly on train.csv.
3. 5-Fold Stratified Group Cross-Validation on families inside Train generates out-of-fold (OOF)
   predictions for meta-learning and unbiased decision threshold selection.
"""

import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    brier_score_loss
)

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
MRI_DIR = DATA_DIR / 'mri'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
OUT_DIR = BASE_DIR / 'results'

for d in [PROCESSED_DIR, MODELS_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MODULE 1: sMRI PRETRAINED RESNET-18 (STRICT TRAIN-ONLY PCA & BALANCED METRIC HEAD)")
print("=" * 80)

# 1. Load splits
train_split = pd.read_csv(SPLITS_DIR / 'train.csv')
val_split = pd.read_csv(SPLITS_DIR / 'val.csv')
test_split = pd.read_csv(SPLITS_DIR / 'test.csv')

train_ids = train_split['participant_id'].astype(str).values
val_ids = val_split['participant_id'].astype(str).values
test_ids = test_split['participant_id'].astype(str).values

train_fams = train_split['family_id'].astype(str).values
val_fams = val_split['family_id'].astype(str).values
test_fams = test_split['family_id'].astype(str).values

assert len(set(train_fams).intersection(set(val_fams))) == 0, "CRITICAL: Family contamination between train and val!"
assert len(set(train_fams).intersection(set(test_fams))) == 0, "CRITICAL: Family contamination between train and test!"
assert len(set(val_fams).intersection(set(test_fams))) == 0, "CRITICAL: Family contamination between val and test!"

y_train = train_split['internalizing_incident'].values.astype(int)
y_val = val_split['internalizing_incident'].values.astype(int)
y_test = test_split['internalizing_incident'].values.astype(int)

print(f"Cohort Partitions: Train n = {len(train_ids)} ({sum(y_train)} pos), "
      f"Val n = {len(val_ids)} ({sum(y_val)} pos), "
      f"Test n = {len(test_ids)} ({sum(y_test)} pos)")

# 2. Load 768-D pretrained Med3D embeddings (extracted from sMRI T1w scans)
mri_data = np.load(MRI_DIR / 'standalone__pretrained.npz', allow_pickle=True)
mri_feats = mri_data['features']
mri_pids = [str(p) for p in mri_data['pids']]
pid_to_idx = {pid: i for i, pid in enumerate(mri_pids)}

X_train_raw = np.array([mri_feats[pid_to_idx[p]] for p in train_ids], dtype=float)
X_val_raw = np.array([mri_feats[pid_to_idx[p]] for p in val_ids], dtype=float)
X_test_raw = np.array([mri_feats[pid_to_idx[p]] for p in test_ids], dtype=float)

# 3. Fit Scaler and PCA strictly on Train partition (Zero Leakage)
scaler_m = StandardScaler()
X_train_scaled = scaler_m.fit_transform(X_train_raw)
X_val_scaled = scaler_m.transform(X_val_raw)
X_test_scaled = scaler_m.transform(X_test_raw)

pca_m = PCA(n_components=5, random_state=SEED)
Z_mri_train_5d = pca_m.fit_transform(X_train_scaled)
Z_mri_val_5d = pca_m.transform(X_val_scaled)
Z_mri_test_5d = pca_m.transform(X_test_scaled)

var_exp = np.sum(pca_m.explained_variance_ratio_) * 100
print(f"sMRI PCA-5 Explained Variance Ratio (Train fit): {var_exp:.2f}%")

# 4. Fit Model on full train and generate out-of-fold CV predictions
clf_m = LogisticRegression(C=0.1, class_weight='balanced', random_state=SEED, max_iter=1000)
clf_m.fit(Z_mri_train_5d, y_train)

# 5-fold Stratified Group K-Fold on Train
sgkf = StratifiedGroupKFold(n_splits=5)
oof_probs_m1 = np.zeros(len(y_train))
val_preds_folds = []
test_preds_folds = []

for tr_idx, hold_idx in sgkf.split(Z_mri_train_5d, y_train, groups=train_fams):
    fold_clf = LogisticRegression(C=0.1, class_weight='balanced', random_state=SEED, max_iter=1000)
    fold_clf.fit(Z_mri_train_5d[tr_idx], y_train[tr_idx])
    oof_probs_m1[hold_idx] = fold_clf.predict_proba(Z_mri_train_5d[hold_idx])[:, 1]
    val_preds_folds.append(fold_clf.predict_proba(Z_mri_val_5d)[:, 1])
    test_preds_folds.append(fold_clf.predict_proba(Z_mri_test_5d)[:, 1])

# Full model predictions
p_train_m1 = clf_m.predict_proba(Z_mri_train_5d)[:, 1]
p_val_m1 = clf_m.predict_proba(Z_mri_val_5d)[:, 1]
p_test_m1 = clf_m.predict_proba(Z_mri_test_5d)[:, 1]

# Bagged fold model predictions
p_val_m1_bagged = np.mean(val_preds_folds, axis=0)
p_test_m1_bagged = np.mean(test_preds_folds, axis=0)

train_cv_auc = roc_auc_score(y_train, oof_probs_m1)
val_auc_m1 = roc_auc_score(y_val, p_val_m1)
test_auc_m1 = roc_auc_score(y_test, p_test_m1)

# Select threshold on Train OOF to maximize F1
best_th = 0.50
best_f1_tr = 0.0
for th in np.linspace(0.1, 0.9, 81):
    f = f1_score(y_train, (oof_probs_m1 >= th).astype(int), zero_division=0)
    if f > best_f1_tr:
        best_f1_tr = f
        best_th = th

# Evaluation at standard 0.50 and train-optimal threshold
def get_metrics(probs, th):
    preds = (probs >= th).astype(int)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, preds, zero_division=0)
    brier = brier_score_loss(y_test, probs)
    return sens, spec, f1, brier, tp, fp, tn, fn

sens_05, spec_05, f1_05, brier_05, tp, fp, tn, fn = get_metrics(p_test_m1, 0.50)

print(f"\nModule 1 Evaluation:")
print(f"  Train CV ROC-AUC: {train_cv_auc:.4f}")
print(f"  Val ROC-AUC:      {val_auc_m1:.4f}")
print(f"  Test ROC-AUC:     {test_auc_m1:.4f}")
print(f"  Test F1 (th=0.50): {f1_05:.4f} (Sens={sens_05*100:.1f}%, Spec={spec_05*100:.1f}%)")

# Save processed embeddings and predictions
np.save(PROCESSED_DIR / 'm1_train_embeddings.npy', Z_mri_train_5d)
np.save(PROCESSED_DIR / 'm1_val_embeddings.npy', Z_mri_val_5d)
np.save(PROCESSED_DIR / 'm1_test_embeddings.npy', Z_mri_test_5d)

np.save(PROCESSED_DIR / 'm1_train_probs.npy', p_train_m1)
np.save(PROCESSED_DIR / 'm1_val_probs.npy', p_val_m1)
np.save(PROCESSED_DIR / 'm1_test_probs.npy', p_test_m1)
np.save(PROCESSED_DIR / 'm1_oof_probs.npy', oof_probs_m1)

with open(MODELS_DIR / 'm1_mri_model.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler_m, 'pca': pca_m, 'classifier': clf_m}, f)

print(f"[OK] Module 1 complete and saved to {PROCESSED_DIR}")
