"""Train and save only the missing XGBoost and LightGBM models"""
import pandas as pd
import numpy as np
import pickle
import gc
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
X_train_df = pd.read_csv('data/train/X_train_balanced.csv')
y_train = pd.read_csv('data/train/y_train_balanced.csv').values.ravel()
X_val_df = pd.read_csv('data/validation/X_val.csv')
y_val = pd.read_csv('data/validation/y_val.csv').values.ravel()

X_train_df = X_train_df.fillna(X_train_df.median())
X_val_df = X_val_df.fillna(X_train_df.median())

feature_names = X_train_df.columns.tolist()
X_train = X_train_df.values
X_val = X_val_df.values
del X_train_df, X_val_df
gc.collect()

print(f"✓ Data loaded: {X_train.shape}\n")

# XGBoost Default
print("1/4: XGBoost Default...")
xgb_def = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
xgb_def.fit(X_train, y_train)
auc = roc_auc_score(y_val, xgb_def.predict_proba(X_val)[:, 1])
print(f"✓ AUC: {auc:.4f}")
with open('outputs/manual_models/XGBoost_Default.pkl', 'wb') as f:
    pickle.dump(xgb_def, f)
print("✓ Saved\n")
gc.collect()

# XGBoost Tuned
print("2/4: XGBoost AutoML-Tuned...")
xgb_tuned = XGBClassifier(n_estimators=150, learning_rate=0.018, max_depth=10,
                          colsample_bytree=0.69, min_child_weight=0.6, random_state=42, n_jobs=-1)
xgb_tuned.fit(X_train, y_train)
auc = roc_auc_score(y_val, xgb_tuned.predict_proba(X_val)[:, 1])
print(f"✓ AUC: {auc:.4f}")
with open('outputs/manual_models/XGBoost_Tuned.pkl', 'wb') as f:
    pickle.dump(xgb_tuned, f)
print("✓ Saved\n")
gc.collect()

# LightGBM Default
print("3/4: LightGBM Default...")
lgbm_def = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
lgbm_def.fit(X_train, y_train)
auc = roc_auc_score(y_val, lgbm_def.predict_proba(X_val)[:, 1])
print(f"✓ AUC: {auc:.4f}")
with open('outputs/manual_models/LightGBM_Default.pkl', 'wb') as f:
    pickle.dump(lgbm_def, f)
print("✓ Saved\n")
gc.collect()

# LightGBM Tuned
print("4/4: LightGBM AutoML-Tuned...")
lgbm_tuned = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=100,
                           feature_fraction=0.9, max_depth=-1, random_state=42, n_jobs=-1, verbose=-1)
lgbm_tuned.fit(X_train, y_train)
auc = roc_auc_score(y_val, lgbm_tuned.predict_proba(X_val)[:, 1])
print(f"✓ AUC: {auc:.4f}")
with open('outputs/manual_models/LightGBM_Tuned.pkl', 'wb') as f:
    pickle.dump(lgbm_tuned, f)
print("✓ Saved\n")

print("="*60)
print("✓ All 4 missing models trained and saved successfully!")
print("="*60)
