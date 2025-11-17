# HYPERPARAMETER TUNING - EXECUTION PLAN

## 🎯 Objective
Maximize dementia prediction AUC through systematic hyperparameter optimization + ensemble stacking

---

## 📊 Current Baseline
| Model | AUC | Status |
|-------|-----|--------|
| LightGBM_Tuned | 0.7947 | Best single model |
| XGBoost_Tuned | 0.7896 | Strong contender |
| LightGBM_Default | 0.7882 | Good baseline |

**Target:** AUC > 0.80 via tuning + ensemble

---

## 🔧 Implementation Architecture

### Step 1: Data Preparation
```
✓ Use existing: data/train/X_train_balanced.csv (192K samples)
✓ Create internal validation: 5-fold Stratified CV
✓ Keep test set untouched
✓ Preprocessing: median imputation (already done)
```

### Step 2: Model-Specific Search Spaces

**LightGBM:**
```python
{
    'num_leaves': [31, 50, 70, 100, 127],  # tree complexity
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],  # convergence speed
    'n_estimators': [100, 150, 200, 300],  # ensemble size
    'feature_fraction': [0.7, 0.8, 0.9, 1.0],  # regularization
    'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
    'bagging_freq': [3, 5, 7],
    'min_child_samples': [10, 20, 30, 50],
    'lambda_l1': [0, 0.1, 1.0, 10],  # L1 reg
    'lambda_l2': [0, 0.1, 1.0, 10],  # L2 reg
    'max_depth': [-1, 10, 15, 20]  # depth limit
}
```

**XGBoost:**
```python
{
    'max_depth': [6, 8, 10, 12, 15],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
    'n_estimators': [100, 150, 200, 300],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.5, 1.0],  # min split loss
    'min_child_weight': [0.5, 1, 3, 5],
    'reg_alpha': [0, 0.1, 1.0, 10],  # L1
    'reg_lambda': [0, 1, 10, 50]  # L2
}
```

### Step 3: Three-Stage Optimization

**Stage 1: Random Exploration (50 trials/model)**
- Purpose: Map parameter landscape
- Method: Uniform sampling
- Early stop: Prune if AUC < 0.75 after 100 trees
- Output: Heatmap of param importance

**Stage 2: Bayesian TPE (150 trials/model)**
- Purpose: Exploit promising regions
- Method: Optuna TPE sampler
- Pruning: MedianPruner (bottom 70%)
- Metric: Mean CV AUC + penalty for std > 0.02
- Multi-objective: AUC (weight=0.8) + calibration (0.2)

**Stage 3: Fine-Tuning (20 trials)**
- Take top 5 configs from Stage 2
- Narrow search around best params (±10%)
- Full training (no early stop)
- Record calibration curves

### Step 4: Cross-Validation Protocol
```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
For each fold:
    ✓ Fit only on train_fold (no val/test leakage)
    ✓ Evaluate on val_fold
    ✓ Record: AUC, F1, Brier score, train_time
Report: mean, std, min, max
```

### Step 5: Ensemble Stacking
```
Layer 1: Top 5 tuned models (diverse architectures)
    - Best LightGBM
    - Best XGBoost
    - 2nd best LGBM (for diversity)
    - 2nd best XGB
    - Best RF (if AUC > 0.77)

Layer 2: Meta-learner options
    Option A: LogisticRegression(C=[0.1, 1, 10])
    Option B: LightGBM(num_leaves=7, lr=0.05, n_est=50)
    Option C: Weighted average (tune weights via CV)

Tune meta-learner using 5-fold CV on Layer 1 predictions
```

### Step 6: Calibration
```python
CalibratedClassifierCV(method='isotonic', cv=5)
Apply to final ensemble
Verify Brier score improvement
```

---

## 📈 Evaluation Metrics

**Primary:** ROC-AUC (maximize)
**Secondary:**
- Brier Score (calibration)
- F1 Score @ optimal threshold
- Precision/Recall tradeoff

**Stability:** CV std < 0.02
**Speed:** Inference < 10ms/sample

---

## 💾 Output Structure

```
tuning_results/
├── lgbm_study.db          # Optuna study database
├── xgb_study.db
├── best_lgbm_params.json
├── best_xgb_params.json
├── best_lgbm_model.pkl
├── best_xgb_model.pkl
├── ensemble_stacker.pkl
├── cv_results.csv         # All trial results
├── calibration_curves.png
├── feature_importance_comparison.csv
└── tuning_report.md       # Summary + insights
```

---

## ⚡ Computational Optimization

1. **Early Stopping:** Prune trials scoring < median after 1/3 iterations
2. **Parallel:** Run 4 trials concurrently (if CPU allows)
3. **Caching:** Store CV fold indices (avoid re-split)
4. **Pruning:** Use Optuna MedianPruner(n_startup_trials=10, n_warmup_steps=20)

---

## 🚀 Execution Command

```bash
python tune_hyperparameters.py --model all --trials 200 --cv 5 --n_jobs -1
```

**Estimated Time:**
- LightGBM: 60-90 min
- XGBoost: 90-120 min
- Ensemble: 20 min
- **Total: ~3 hours**

---

## ✅ Success Criteria

- [ ] AUC > 0.80 (ensemble)
- [ ] AUC > 0.795 (best single model)
- [ ] Brier score < 0.16
- [ ] CV std < 0.02 (stable)
- [ ] All models saved + reproducible
- [ ] Tuning report generated

---

## 🔍 Analysis Deliverables

1. **Parameter Importance Plot** (Optuna built-in)
2. **Optimization History** (AUC over trials)
3. **Parallel Coordinate Plot** (param interactions)
4. **Slice Plot** (param vs score)
5. **Confusion Matrix** (best model)
6. **Calibration Curve** (ensemble)
7. **Feature Importance** (top 20 features)

---

## 🎓 Expected Insights

**Questions to Answer:**
- Which hyperparameters matter most? (learning_rate, num_leaves, depth?)
- Is LightGBM or XGBoost better for this task?
- Does ensemble improve AUC by >1%?
- Are models well-calibrated?
- What's the overfitting gap?
- Can we simplify without losing performance?

---

## 📝 Next Steps

**After tuning completes:**
1. Select final model (ensemble or best single)
2. Retrain on full train+val data
3. Evaluate once on test set
4. Generate competition submission
5. Document final model card
