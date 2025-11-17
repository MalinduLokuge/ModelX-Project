"""
MANUAL ML MODEL TRAINING - DEMENTIA PREDICTION
Train 8 binary classification models with comprehensive documentation
Memory-optimized for low-RAM systems
"""

import pandas as pd
import numpy as np
import pickle
import json
import gc
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MANUAL ML TRAINING - DEMENTIA PREDICTION")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("STEP 1: Loading Data...")
print("-"*80)

X_train = pd.read_csv('data/train/X_train_balanced.csv')
y_train = pd.read_csv('data/train/y_train_balanced.csv').values.ravel()
X_val = pd.read_csv('data/validation/X_val.csv')
y_val = pd.read_csv('data/validation/y_val.csv').values.ravel()

# Handle any remaining NaN
X_train = X_train.fillna(X_train.median())
X_val = X_val.fillna(X_train.median())  # Use train median

# MEMORY OPTIMIZATION: Convert to numpy arrays
feature_names = X_train.columns.tolist()
X_train_np = X_train.values  # Convert to numpy
X_val_np = X_val.values
del X_train, X_val  # Free memory
gc.collect()

print(f"✓ Train: {X_train_np.shape} (balanced with SMOTE)")
print(f"✓ Val:   {X_val_np.shape}")
print(f"✓ Features: {X_train_np.shape[1]}")
print(f"✓ Train class dist: {np.bincount(y_train)}")
print(f"✓ Val class dist: {np.bincount(y_val)}")
print()

# ============================================================================
# SETUP
# ============================================================================
# Storage for results
results = []
models_trained = {}
feature_importance = {}

# CV strategy (same as AutoML: 3-fold stratified)
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Evaluation function
def evaluate_model(name, model, X_val, y_val):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        'model': name,
        'roc_auc': roc_auc_score(y_val, y_proba),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    }

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================
print("="*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*80)
print("Justification:")
print("  - Baseline linear model for binary classification")
print("  - Fast training and inference")
print("  - Interpretable coefficients")
print("  - Good probability calibration")
print("  - Handles high-dimensional data well")
print()

# Default configuration
print("Training with DEFAULT parameters...")
lr_default_params = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1
}
print(f"Default params: {lr_default_params}")

lr_default = LogisticRegression(**lr_default_params)
lr_default.fit(X_train_np, y_train)
results.append(evaluate_model('LogisticRegression_default', lr_default, X_val_np, y_val))
print(f"✓ Default ROC-AUC: {results[-1]['roc_auc']:.4f}")

# Hyperparameter tuning
print("\nHyperparameter Tuning...")
print("  - Search Method: RandomizedSearchCV")
print("  - CV Strategy: 3-fold Stratified")
print("  - Iterations: 10")
print("  - Metric: ROC-AUC")

lr_param_space = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [500, 1000]
}

lr_search = RandomizedSearchCV(
    LogisticRegression(random_state=42, n_jobs=-1),
    lr_param_space,
    n_iter=10,
    cv=cv_strategy,
    scoring='roc_auc',
    random_state=42,
    n_jobs=1,  # MEMORY: Sequential CV
    verbose=0
)
lr_search.fit(X_train_np, y_train)
print(f"✓ Best params: {lr_search.best_params_}")
print(f"✓ Best CV ROC-AUC: {lr_search.best_score_:.4f}")

results.append(evaluate_model('LogisticRegression_tuned', lr_search.best_estimator_, X_val_np, y_val))
print(f"✓ Tuned Val ROC-AUC: {results[-1]['roc_auc']:.4f}")

models_trained['LogisticRegression'] = {
    'default': lr_default,
    'tuned': lr_search.best_estimator_,
    'best_params': lr_search.best_params_,
    'default_params': lr_default_params
}

del lr_search, lr_default
gc.collect()
print("✓ Model 1 Complete\n")

# ============================================================================
# MODEL 2: RANDOM FOREST (GINI)
# ============================================================================
print("="*80)
print("MODEL 2: RANDOM FOREST (GINI)")
print("="*80)
print("Justification:")
print("  - Robust ensemble method")
print("  - Handles non-linear relationships")
print("  - Resistant to overfitting")
print("  - Provides feature importance")
print("  - No feature scaling needed")
print()

# Default configuration
print("Training with DEFAULT parameters...")
rf_default_params = {
    'n_estimators': 100,  # Reduced from 300 for memory
    'criterion': 'gini',
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}
print(f"Default params: {rf_default_params}")

rf_gini = RandomForestClassifier(**rf_default_params)
rf_gini.fit(X_train, y_train)
results.append(evaluate_model('RandomForest_gini_default', rf_gini, X_val, y_val))
print(f"✓ Default ROC-AUC: {results[-1]['roc_auc']:.4f}")

# Feature importance
feature_importance['RandomForest_gini'] = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_gini.feature_importances_
}).sort_values('importance', ascending=False)

# Hyperparameter tuning
print("\nHyperparameter Tuning...")
rf_param_space = {
    'n_estimators': [50, 100, 150],  # Memory-limited
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1),
    rf_param_space,
    n_iter=15,
    cv=cv_strategy,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_search.fit(X_train, y_train)
print(f"✓ Best params: {rf_search.best_params_}")
print(f"✓ Best CV ROC-AUC: {rf_search.best_score_:.4f}")

results.append(evaluate_model('RandomForest_gini_tuned', rf_search.best_estimator_, X_val, y_val))
print(f"✓ Tuned Val ROC-AUC: {results[-1]['roc_auc']:.4f}")

models_trained['RandomForest_gini'] = {
    'default': rf_gini,
    'tuned': rf_search.best_estimator_,
    'best_params': rf_search.best_params_,
    'default_params': rf_default_params
}

del rf_search, rf_gini
gc.collect()
print("✓ Model 2 Complete\n")

# ============================================================================
# MODEL 3: RANDOM FOREST (ENTROPY)
# ============================================================================
print("="*80)
print("MODEL 3: RANDOM FOREST (ENTROPY)")
print("="*80)
print("Justification:")
print("  - Alternative split criterion to Gini")
print("  - May capture different patterns")
print("  - Often better for imbalanced data")
print("  - Same benefits as Gini RF")
print()

print("Training with DEFAULT parameters...")
rf_entropy = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    random_state=42,
    n_jobs=-1
)
rf_entropy.fit(X_train, y_train)
results.append(evaluate_model('RandomForest_entropy', rf_entropy, X_val, y_val))
print(f"✓ Default ROC-AUC: {results[-1]['roc_auc']:.4f}")

feature_importance['RandomForest_entropy'] = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_entropy.feature_importances_
}).sort_values('importance', ascending=False)

models_trained['RandomForest_entropy'] = {
    'default': rf_entropy,
    'default_params': {'n_estimators': 100, 'criterion': 'entropy'}
}

del rf_entropy
gc.collect()
print("✓ Model 3 Complete\n")

# ============================================================================
# MODEL 4: EXTRA TREES
# ============================================================================
print("="*80)
print("MODEL 4: EXTRA TREES")
print("="*80)
print("Justification:")
print("  - More randomized than Random Forest")
print("  - Faster training")
print("  - Better variance reduction")
print("  - Good for high-dimensional data")
print()

print("Training with DEFAULT parameters...")
et = ExtraTreesClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=42,
    n_jobs=-1
)
et.fit(X_train, y_train)
results.append(evaluate_model('ExtraTrees', et, X_val, y_val))
print(f"✓ Default ROC-AUC: {results[-1]['roc_auc']:.4f}")

feature_importance['ExtraTrees'] = pd.DataFrame({
    'feature': X_train.columns,
    'importance': et.feature_importances_
}).sort_values('importance', ascending=False)

models_trained['ExtraTrees'] = {
    'default': et,
    'default_params': {'n_estimators': 100, 'criterion': 'gini'}
}

del et
gc.collect()
print("✓ Model 4 Complete\n")

# ============================================================================
# MODEL 5: XGBOOST (DEFAULT)
# ============================================================================
print("="*80)
print("MODEL 5: XGBOOST (DEFAULT)")
print("="*80)
print("Justification:")
print("  - State-of-the-art gradient boosting")
print("  - Excellent performance on structured data")
print("  - Built-in regularization (L1/L2)")
print("  - Handles missing values")
print("  - Fast training with early stopping")
print()

print("Training with DEFAULT parameters...")
xgb_default_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}
print(f"Default params: {xgb_default_params}")

xgb_default = XGBClassifier(**xgb_default_params)
xgb_default.fit(X_train, y_train)
results.append(evaluate_model('XGBoost_default', xgb_default, X_val, y_val))
print(f"✓ Default ROC-AUC: {results[-1]['roc_auc']:.4f}")

feature_importance['XGBoost_default'] = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_default.feature_importances_
}).sort_values('importance', ascending=False)

models_trained['XGBoost_default'] = {
    'default': xgb_default,
    'default_params': xgb_default_params
}

del xgb_default
gc.collect()
print("✓ Model 5 Complete\n")

# ============================================================================
# MODEL 6: XGBOOST (TUNED - AutoML Informed)
# ============================================================================
print("="*80)
print("MODEL 6: XGBOOST (TUNED - AutoML Informed)")
print("="*80)
print("Justification:")
print("  - Use AutoML insights for smart tuning")
print("  - AutoML best: learning_rate=0.018, max_depth=10, colsample_bytree=0.69")
print("  - Search around these values")
print()

print("Hyperparameter Tuning (AutoML-informed)...")
xgb_param_space = {
    'learning_rate': [0.01, 0.018, 0.03, 0.05],  # Around AutoML best: 0.018
    'max_depth': [8, 10, 12],                     # Around AutoML best: 10
    'colsample_bytree': [0.65, 0.69, 0.75],       # AutoML best: 0.69
    'min_child_weight': [0.5, 0.6, 1.0],          # AutoML best: 0.6
    'n_estimators': [100, 150, 200],
    'subsample': [0.8, 0.9, 1.0]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
    xgb_param_space,
    n_iter=20,
    cv=cv_strategy,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
xgb_search.fit(X_train, y_train)
print(f"✓ Best params: {xgb_search.best_params_}")
print(f"✓ Best CV ROC-AUC: {xgb_search.best_score_:.4f}")

results.append(evaluate_model('XGBoost_tuned', xgb_search.best_estimator_, X_val, y_val))
print(f"✓ Tuned Val ROC-AUC: {results[-1]['roc_auc']:.4f}")

feature_importance['XGBoost_tuned'] = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_search.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

models_trained['XGBoost_tuned'] = {
    'tuned': xgb_search.best_estimator_,
    'best_params': xgb_search.best_params_
}

del xgb_search
gc.collect()
print("✓ Model 6 Complete\n")

# ============================================================================
# MODEL 7: LIGHTGBM (DEFAULT)
# ============================================================================
print("="*80)
print("MODEL 7: LIGHTGBM (DEFAULT)")
print("="*80)
print("Justification:")
print("  - AutoML top performer (0.9811 ROC-AUC)")
print("  - Fast and memory-efficient")
print("  - Leaf-wise tree growth (better accuracy)")
print("  - Handles categorical features natively")
print("  - Excellent for large datasets")
print()

print("Training with DEFAULT parameters...")
lgbm_default_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
print(f"Default params: {lgbm_default_params}")

lgbm_default = LGBMClassifier(**lgbm_default_params)
lgbm_default.fit(X_train, y_train)
results.append(evaluate_model('LightGBM_default', lgbm_default, X_val, y_val))
print(f"✓ Default ROC-AUC: {results[-1]['roc_auc']:.4f}")

feature_importance['LightGBM_default'] = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgbm_default.feature_importances_
}).sort_values('importance', ascending=False)

models_trained['LightGBM_default'] = {
    'default': lgbm_default,
    'default_params': lgbm_default_params
}

del lgbm_default
gc.collect()
print("✓ Model 7 Complete\n")

# ============================================================================
# MODEL 8: LIGHTGBM (TUNED - AutoML Informed)
# ============================================================================
print("="*80)
print("MODEL 8: LIGHTGBM (TUNED - AutoML Informed)")
print("="*80)
print("Justification:")
print("  - Best AutoML L1 model: LightGBM_r131 (0.9811 ROC-AUC)")
print("  - AutoML insights: learning_rate ~0.05-0.07, num_leaves 64-128")
print("  - Search around these optimal values")
print()

print("Hyperparameter Tuning (AutoML-informed)...")
lgbm_param_space = {
    'learning_rate': [0.03, 0.05, 0.07, 0.1],     # Around AutoML best: 0.05-0.07
    'num_leaves': [50, 64, 100, 128],             # Around AutoML best: 64-128
    'max_depth': [-1, 10, 15],
    'feature_fraction': [0.8, 0.9, 1.0],          # AutoML best: 0.9
    'n_estimators': [100, 150, 200],
    'min_child_samples': [10, 20, 30]
}

lgbm_search = RandomizedSearchCV(
    LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    lgbm_param_space,
    n_iter=20,
    cv=cv_strategy,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
lgbm_search.fit(X_train, y_train)
print(f"✓ Best params: {lgbm_search.best_params_}")
print(f"✓ Best CV ROC-AUC: {lgbm_search.best_score_:.4f}")

results.append(evaluate_model('LightGBM_tuned', lgbm_search.best_estimator_, X_val, y_val))
print(f"✓ Tuned Val ROC-AUC: {results[-1]['roc_auc']:.4f}")

feature_importance['LightGBM_tuned'] = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgbm_search.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

models_trained['LightGBM_tuned'] = {
    'tuned': lgbm_search.best_estimator_,
    'best_params': lgbm_search.best_params_
}

del lgbm_search
gc.collect()
print("✓ Model 8 Complete\n")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("="*80)
print("TRAINING COMPLETE - RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('roc_auc', ascending=False)

print("\nModel Performance (Sorted by ROC-AUC):")
print("-"*80)
for idx, row in results_df.iterrows():
    print(f"{row['model']:40s} | ROC-AUC: {row['roc_auc']:.4f} | Acc: {row['accuracy']:.4f} | F1: {row['f1']:.4f}")

print(f"\n✓ Best Model: {results_df.iloc[0]['model']}")
print(f"✓ Best ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

import os
os.makedirs('outputs/manual_models', exist_ok=True)

# Save results CSV
results_df.to_csv('outputs/manual_models/model_comparison.csv', index=False)
print("✓ Saved: outputs/manual_models/model_comparison.csv")

# Save models
for name, model_dict in models_trained.items():
    if 'tuned' in model_dict:
        with open(f'outputs/manual_models/{name}_tuned.pkl', 'wb') as f:
            pickle.dump(model_dict['tuned'], f)
    if 'default' in model_dict:
        with open(f'outputs/manual_models/{name}_default.pkl', 'wb') as f:
            pickle.dump(model_dict['default'], f)
print("✓ Saved: All trained models")

# Save feature importance
for name, df in feature_importance.items():
    df.to_csv(f'outputs/manual_models/feature_importance_{name}.csv', index=False)
print("✓ Saved: Feature importance files")

# Save complete config
config = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models_trained': list(models_trained.keys()),
    'cv_strategy': '3-fold Stratified',
    'data': {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'features': X_train.shape[1],
        'balanced': 'SMOTE applied'
    },
    'results': results
}

with open('outputs/manual_models/training_config.json', 'w') as f:
    json.dump(config, f, indent=2, default=str)
print("✓ Saved: outputs/manual_models/training_config.json")

print("\n" + "="*80)
print("✓ ALL TASKS COMPLETE")
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output Directory: outputs/manual_models/")
print("="*80)
