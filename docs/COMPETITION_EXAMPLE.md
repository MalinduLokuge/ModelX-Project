# Competition Example: Step-by-Step Walkthrough

## Scenario: Kaggle-Style Classification Competition

**Competition:** Predict customer churn
**Dataset:** 50,000 rows, 25 features
**Time Available:** 4 hours
**Goal:** Top 10% leaderboard

---

## Hour 1: Quick Validation (5 min) + Auto Run (55 min)

### Step 1: Quick Test (5 minutes)
```bash
# Place data in data/raw/
data/raw/train.csv
data/raw/test.csv

# Run quick test to validate data loads correctly
scripts\quick_test.bat data/raw/train.csv
```

**Output:**
- ✓ Data loads successfully
- ✓ Problem type detected: classification
- ✓ Basic model trained
- ✓ No critical errors

### Step 2: Competition Run (55 minutes)
```bash
# Run full competition mode (2 hours allocated, finishes in ~55 min)
scripts\competition_run.bat data/raw/train.csv data/raw/test.csv
```

**What Happens:**
1. [5 min] Data loading, validation, EDA
2. [10 min] Preprocessing (missing values, encoding, scaling)
3. [15 min] Feature engineering (interactions, polynomials, competition tricks)
4. [20 min] Model training (AutoGluon with stacking + bagging)
5. [5 min] Generate outputs (submission, recipe, model package)

**Output Files:**
```
outputs/20250115_143022/
├── submission.csv              ← Upload this to competition!
├── recipe.txt                  ← What was done
├── model_20250115_143022/      ← Deployment package
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── feature_engineer.pkl
│   ├── metadata.json
│   └── inference.py
└── eda_plots/                  ← Analysis charts
```

---

## Hour 2: Analysis & First Submission (60 min)

### Step 3: Review Results (15 minutes)
```bash
# Read recipe to understand what worked
type outputs\20250115_143022\recipe.txt
```

**Key Findings from Recipe:**
- Best model: WeightedEnsemble_L2 (stacked model)
- CV Score: 0.8542 AUC
- 47 features created (25 original → 47 after FE)
- Competition tricks applied: target encoding, frequency encoding
- Important features: age_squared, income_x_tenure, city_target_enc

### Step 4: Submit to Leaderboard (5 minutes)
```bash
# Upload outputs/20250115_143022/submission.csv
```

**Leaderboard Score:** 0.8498 AUC (Public)
**Rank:** 245 / 2,500 (Top 10%)

### Step 5: Identify Improvements (40 minutes)

Check recipe for potential improvements:
- ✓ Target encoding enabled
- ✓ Feature interactions created
- ✗ Polynomial features disabled (could help)
- ✗ Only 5 bag folds (could increase to 8)

---

## Hour 3: Iteration (60 min)

### Step 6: Enable More Features (30 minutes)

Edit `configs/competition.yaml`:
```yaml
polynomial_features: true        # Changed from false
ag_num_bag_folds: 10            # Changed from 8
ag_num_stack_levels: 3          # Changed from 2
```

Run again:
```bash
scripts\competition_run.bat data/raw/train.csv data/raw/test.csv
```

**New CV Score:** 0.8587 AUC (+0.0045 improvement!)

### Step 7: Manual Feature Engineering (30 minutes)

Based on domain knowledge, create custom features:

```python
# templates/code/feature_engineering_template.py
# Add domain-specific features
df_features['tenure_per_age'] = df['tenure'] / (df['age'] + 1)
df_features['total_value'] = df['monthly_charges'] * df['tenure']
df_features['is_senior_citizen'] = (df['age'] > 65).astype(int)
```

---

## Hour 4: Final Optimization (60 min)

### Step 8: Ensemble Multiple Runs (40 minutes)

Create 3 models with different random seeds:
```bash
# Model 1 (seed=42)
python main.py --train data/raw/train.csv --test data/raw/test.csv --config configs/competition.yaml

# Model 2 (seed=123)
python main.py --train data/raw/train.csv --test data/raw/test.csv --config configs/competition.yaml --seed 123

# Model 3 (seed=999)
python main.py --train data/raw/train.csv --test data/raw/test.csv --config configs/competition.yaml --seed 999
```

Average predictions:
```python
import pandas as pd
pred1 = pd.read_csv('outputs/run1/submission.csv')
pred2 = pd.read_csv('outputs/run2/submission.csv')
pred3 = pd.read_csv('outputs/run3/submission.csv')

ensemble = pred1.copy()
ensemble['prediction'] = (pred1['prediction'] + pred2['prediction'] + pred3['prediction']) / 3
ensemble.to_csv('outputs/final_ensemble_submission.csv', index=False)
```

### Step 9: Final Submission (5 minutes)
Upload `final_ensemble_submission.csv`

**Final Leaderboard Score:** 0.8621 AUC
**Final Rank:** 89 / 2,500 (Top 4%)

### Step 10: Documentation (15 minutes)

Document what worked for future competitions:
- Target encoding gave +2% boost
- Polynomial features gave +0.5% boost
- Ensemble of 3 seeds gave +0.3% boost
- Custom domain features gave +1% boost

---

## Results Summary

| Step | Action | CV Score | LB Score | Time |
|------|--------|----------|----------|------|
| 1 | Auto baseline | 0.8542 | 0.8498 | 60 min |
| 2 | Enable polynomials | 0.8587 | 0.8543 | 30 min |
| 3 | Custom features | 0.8612 | 0.8598 | 30 min |
| 4 | Ensemble 3 seeds | - | 0.8621 | 40 min |

**Total Time:** 2h 40min / 4h available
**Achievement:** Top 4% 🏆

---

## Key Lessons

1. **Start with automation** - CompeteML gets you to top 10% quickly
2. **Review recipe** - Understand what worked before manual tuning
3. **Incremental improvements** - Each small change adds up
4. **Ensemble helps** - Simple averaging of different seeds boosts score
5. **Save time** - Let AutoML find good baseline, spend time on domain features

---

## Competition Checklist

Before submission:
- ✓ Validate submission format matches sample_submission.csv
- ✓ Check no missing predictions
- ✓ Verify ID column matches
- ✓ Save recipe and code for reproducibility
- ✓ Test inference script works on new data
- ✓ Document what worked for future reference
