# HYPERPARAMETER TUNING REPORT
Generated: 2025-11-17 16:31:30

## 📊 Optimization Summary

**Configuration:**
- Trials per model: 100
- CV Folds: 5
- Strategy: Random (50) → TPE Bayesian (150)
- Pruning: MedianPruner (bottom 70%)

---

## 🏆 Best Models

### LIGHTGBM
**CV Score:** 0.9192 ± 0.0014
**Brier Score:** 0.1114

**Best Parameters:**
```json
{
  "num_leaves": 118,
  "learning_rate": 0.07021241609365873,
  "n_estimators": 300,
  "feature_fraction": 0.940101850374288,
  "bagging_fraction": 0.8885985829209233,
  "bagging_freq": 6,
  "min_child_samples": 37,
  "lambda_l1": 0.36509477727193046,
  "lambda_l2": 6.5995032440709265,
  "max_depth": 17
}
```

### XGBOOST
**CV Score:** 0.9239 ± 0.0013
**Brier Score:** 0.1086

**Best Parameters:**
```json
{
  "max_depth": 12,
  "learning_rate": 0.05667130382026883,
  "n_estimators": 300,
  "subsample": 0.9238708511843299,
  "colsample_bytree": 0.7167064094317155,
  "gamma": 0.39553155471798773,
  "min_child_weight": 1.6556267200701262,
  "reg_alpha": 0.10565672942362851,
  "reg_lambda": 1.179347020355599
}
```

---

## 🎯 Ensemble Performance

**Stacked Ensemble CV AUC:** 0.9237 ± 0.0013

**Architecture:**
- Layer 1: 2 base models (LightGBM, XGBoost)
- Layer 2: Logistic Regression meta-learner
- Calibration: Isotonic regression

**Ensemble Weights (LogReg coefficients):**
- lightgbm: 1.8139
- xgboost: 5.0979

---

## 🔍 Key Insights

**Top Hyperparameters (by importance):**

**LIGHTGBM:**
  learning_rate: 0.784
  min_child_samples: 0.096
  bagging_fraction: 0.048
  lambda_l1: 0.034
  num_leaves: 0.015

**XGBOOST:**
  learning_rate: 0.373
  max_depth: 0.270
  n_estimators: 0.134
  colsample_bytree: 0.096
  gamma: 0.048

---

## 💾 Saved Artifacts

```
tuning_results/
├── lightgbm_study.pkl
├── xgb_study.pkl
├── best_lightgbm_model.pkl
├── best_xgboost_model.pkl
├── best_lightgbm_params.json
├── best_xgboost_params.json
├── ensemble_stacker.pkl
├── ensemble_calibrated.pkl (⭐ USE THIS)
└── tuning_report.md
```

---

## 🚀 Usage

```python
import pickle
with open('tuning_results/ensemble_calibrated.pkl', 'rb') as f:
    ensemble = pickle.load(f)

# Predict
base_preds = np.column_stack([
    model.predict_proba(X_new)[:, 1]
    for model in ensemble['base_models'].values()
])
final_pred = ensemble['meta_learner_calibrated'].predict_proba(base_preds)[:, 1]
```
