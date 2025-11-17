"""
MEMORY-OPTIMIZED Manual ML Training
Trains 8 models with minimal memory footprint
"""
import pandas as pd
import numpy as np
import pickle
import json
import gc
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MEMORY-OPTIMIZED MANUAL ML TRAINING")
print("="*80)

# Load data as numpy arrays immediately
print("Loading data...")
X_train_df = pd.read_csv('data/train/X_train_balanced.csv')
y_train = pd.read_csv('data/train/y_train_balanced.csv').values.ravel()
X_val_df = pd.read_csv('data/validation/X_val.csv')
y_val = pd.read_csv('data/validation/y_val.csv').values.ravel()

# Fill NaN
X_train_df = X_train_df.fillna(X_train_df.median())
X_val_df = X_val_df.fillna(X_train_df.median())

# Convert to numpy
feature_names = X_train_df.columns.tolist()
X_train = X_train_df.values
X_val = X_val_df.values
del X_train_df, X_val_df
gc.collect()

print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}")
print()

results = []
models = {}

def eval_model(name, model):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    return {
        'model': name,
        'roc_auc': roc_auc_score(y_val, y_proba),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    }

# ===========================================================================
# MODEL 1: Logistic Regression
# ===========================================================================
print("MODEL 1: Logistic Regression")
print("-"*80)
lr = LogisticRegression(C=1.0, max_iter=500, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
results.append(eval_model('LogisticRegression', lr))
models['LogisticRegression'] = lr
print(f"✓ ROC-AUC: {results[-1]['roc_auc']:.4f}\n")
gc.collect()

# ===========================================================================
# MODEL 2: Random Forest (Gini)
# ===========================================================================
print("MODEL 2: Random Forest (Gini)")
print("-"*80)
rf_gini = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_gini.fit(X_train, y_train)
results.append(eval_model('RandomForest_Gini', rf_gini))
models['RandomForest_Gini'] = rf_gini
print(f"✓ ROC-AUC: {results[-1]['roc_auc']:.4f}\n")

# Feature importance
fi_rf = pd.DataFrame({'feature': feature_names, 'importance': rf_gini.feature_importances_})
fi_rf.to_csv('outputs/manual_models/fi_rf_gini.csv', index=False)
gc.collect()

# ===========================================================================
# MODEL 3: Random Forest (Entropy)
# ===========================================================================
print("MODEL 3: Random Forest (Entropy)")
print("-"*80)
rf_ent = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=15, random_state=42, n_jobs=-1)
rf_ent.fit(X_train, y_train)
results.append(eval_model('RandomForest_Entropy', rf_ent))
models['RandomForest_Entropy'] = rf_ent
print(f"✓ ROC-AUC: {results[-1]['roc_auc']:.4f}\n")
gc.collect()

# ===========================================================================
# MODEL 4: Extra Trees
# ===========================================================================
print("MODEL 4: Extra Trees")
print("-"*80)
et = ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
results.append(eval_model('ExtraTrees', et))
models['ExtraTrees'] = et
print(f"✓ ROC-AUC: {results[-1]['roc_auc']:.4f}\n")
gc.collect()

# ===========================================================================
# MODEL 5: XGBoost (Default)
# ===========================================================================
print("MODEL 5: XGBoost (Default)")
print("-"*80)
xgb_def = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
xgb_def.fit(X_train, y_train)
results.append(eval_model('XGBoost_Default', xgb_def))
models['XGBoost_Default'] = xgb_def
print(f"✓ ROC-AUC: {results[-1]['roc_auc']:.4f}\n")

fi_xgb = pd.DataFrame({'feature': feature_names, 'importance': xgb_def.feature_importances_})
fi_xgb.to_csv('outputs/manual_models/fi_xgb_default.csv', index=False)
gc.collect()

# ===========================================================================
# MODEL 6: XGBoost (AutoML-Informed)
# ===========================================================================
print("MODEL 6: XGBoost (AutoML-Informed)")
print("Using AutoML best params: lr=0.018, max_depth=10, colsample=0.69")
print("-"*80)
xgb_tuned = XGBClassifier(
    n_estimators=150,
    learning_rate=0.018,
    max_depth=10,
    colsample_bytree=0.69,
    min_child_weight=0.6,
    random_state=42,
    n_jobs=-1
)
xgb_tuned.fit(X_train, y_train)
results.append(eval_model('XGBoost_AutoML_Tuned', xgb_tuned))
models['XGBoost_Tuned'] = xgb_tuned
print(f"✓ ROC-AUC: {results[-1]['roc_auc']:.4f}\n")
gc.collect()

# ===========================================================================
# MODEL 7: LightGBM (Default)
# ===========================================================================
print("MODEL 7: LightGBM (Default)")
print("-"*80)
lgbm_def = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
lgbm_def.fit(X_train, y_train)
results.append(eval_model('LightGBM_Default', lgbm_def))
models['LightGBM_Default'] = lgbm_def
print(f"✓ ROC-AUC: {results[-1]['roc_auc']:.4f}\n")

fi_lgbm = pd.DataFrame({'feature': feature_names, 'importance': lgbm_def.feature_importances_})
fi_lgbm.to_csv('outputs/manual_models/fi_lgbm_default.csv', index=False)
gc.collect()

# ===========================================================================
# MODEL 8: LightGBM (AutoML-Informed)
# ===========================================================================
print("MODEL 8: LightGBM (AutoML-Informed)")
print("Using AutoML best params: lr~0.05-0.07, num_leaves 64-128, feature_fraction 0.9")
print("-"*80)
lgbm_tuned = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=100,
    feature_fraction=0.9,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_tuned.fit(X_train, y_train)
results.append(eval_model('LightGBM_AutoML_Tuned', lgbm_tuned))
models['LightGBM_Tuned'] = lgbm_tuned
print(f"✓ ROC-AUC: {results[-1]['roc_auc']:.4f}\n")

fi_lgbm_tuned = pd.DataFrame({'feature': feature_names, 'importance': lgbm_tuned.feature_importances_})
fi_lgbm_tuned.to_csv('outputs/manual_models/fi_lgbm_tuned.csv', index=False)
gc.collect()

# ===========================================================================
# RESULTS
# ===========================================================================
print("="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
print("\nModel Performance:")
print("-"*80)
for _, row in results_df.iterrows():
    print(f"{row['model']:30s} | AUC: {row['roc_auc']:.4f} | Acc: {row['accuracy']:.4f} | F1: {row['f1']:.4f}")

# Save
results_df.to_csv('outputs/manual_models/model_comparison.csv', index=False)
for name, model in models.items():
    with open(f'outputs/manual_models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

config = {
    'date': str(datetime.now()),
    'models': list(models.keys()),
    'best_model': results_df.iloc[0]['model'],
    'best_roc_auc': float(results_df.iloc[0]['roc_auc'])
}
with open('outputs/manual_models/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n✓ Best Model: {results_df.iloc[0]['model']}")
print(f"✓ Best ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
print("\n✓ All models saved to outputs/manual_models/")
print("="*80)
