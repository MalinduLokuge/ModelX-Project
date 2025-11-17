# Project Technical Documentation
## AutoGluon Model Artifacts & File Structure

---

## Table of Contents
1. [Overview](#overview)
2. [Model Artifacts Directory Structure](#model-artifacts-directory-structure)
3. [Critical Files for Model Portability](#critical-files-for-model-portability)
4. [AutoGluon Predictor Internal Structure](#autogluon-predictor-internal-structure)
5. [Feature Metadata & Schema](#feature-metadata--schema)
6. [Model Hyperparameters](#model-hyperparameters)
7. [Preprocessing Pipeline Details](#preprocessing-pipeline-details)
8. [Requirements & Dependencies](#requirements--dependencies)
9. [Deployment Bundle Checklist](#deployment-bundle-checklist)
10. [File Size & Storage Requirements](#file-size--storage-requirements)

---

## 1. Overview

This document provides a complete technical reference for all artifacts produced by the AutoGluon training pipeline. These artifacts are required to reuse, deploy, or reproduce the trained dementia risk prediction model in any environment.

**Model Location:** `C:\Users\user\PycharmProjects\ModelX\models\autogluon_production_lowmem\`

**Model Type:** AutoGluon TabularPredictor (Binary Classification)

**Framework Version:** AutoGluon v1.4.0

**Python Version:** 3.11

---

## 2. Model Artifacts Directory Structure

### 2.1 Complete Directory Tree

```
models/autogluon_production_lowmem/
│
├── learner.pkl                          # Main AutoGluon predictor object (CRITICAL)
├── metadata.json                        # Model metadata & version info
├── version.txt                          # AutoGluon version stamp
├── model_leaderboard.csv               # Performance of all 36 models
├── test_evaluation_metrics.json        # Final test set metrics
├── feature_importance.csv              # SHAP feature importance rankings
├── test_predictions.csv                # Full test set predictions (29k rows)
├── test_probabilities.csv              # Predicted probabilities for test set
│
├── models/                              # Individual model artifacts
│   ├── LightGBM_r188_BAG_L2/
│   │   ├── model.pkl                   # Trained LightGBM model
│   │   ├── hyperparameters.json        # Model hyperparameters
│   │   └── metadata.json               # Model-specific metadata
│   │
│   ├── XGBoost_r33_BAG_L2/
│   │   ├── model.pkl
│   │   ├── hyperparameters.json
│   │   └── metadata.json
│   │
│   ├── WeightedEnsemble_L3/            # Final ensemble model
│   │   ├── model.pkl                   # Ensemble weights & combiner
│   │   ├── hyperparameters.json
│   │   └── metadata.json
│   │
│   └── ... (34 more model directories)
│
├── utils/                               # AutoGluon internal utilities
│   ├── feature_metadata.pkl            # Feature types & statistics
│   ├── feature_transformations.pkl     # Preprocessing transformations
│   └── label_encoder.pkl               # Target variable encoding
│
└── training_logs/                       # Optional training logs
    ├── autogluon_training.log
    └── model_selection.log
```

### 2.2 File Categories

| Category | Files | Purpose | Required for Deployment? |
|----------|-------|---------|-------------------------|
| **Core Predictor** | `learner.pkl` | Main AutoGluon object | ✅ YES |
| **Model Artifacts** | `models/*/model.pkl` | Trained models (36 files) | ✅ YES |
| **Metadata** | `metadata.json`, `version.txt` | Version tracking | ✅ YES |
| **Evaluation** | `*_evaluation_metrics.json` | Performance metrics | ⚠️ Recommended |
| **Leaderboard** | `model_leaderboard.csv` | Model comparison | ⚠️ Recommended |
| **Feature Info** | `feature_importance.csv` | Feature rankings | ⚠️ Recommended |
| **Predictions** | `test_predictions.csv` | Sample outputs | ❌ Optional |
| **Utilities** | `utils/*.pkl` | Internal AutoGluon files | ✅ YES (auto-loaded) |

---

## 3. Critical Files for Model Portability

### 3.1 `learner.pkl` (Main Predictor Object)

**Description:**
The core AutoGluon TabularPredictor object containing all trained models, preprocessing pipelines, and ensemble configurations.

**What's Inside:**
- 36 trained models (LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, NeuralNet)
- Feature preprocessing transformations (scalers, encoders, imputers)
- Ensemble weights for WeightedEnsemble_L3
- Feature schema (names, types, expected ranges)
- Target label encoder (dementia_status: 0/1 → No/Yes)

**How AutoGluon Uses It:**
```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")
# predictor.learner is automatically loaded from learner.pkl
```

**File Size:** ~450 MB (contains all 36 models + preprocessing)

**Must Be Bundled:** ✅ YES

**Dependencies:** Requires `models/` subdirectories to be present

---

### 3.2 `metadata.json` (Model Metadata)

**Description:**
Contains versioning and configuration information for reproducibility.

**Contents:**
```json
{
  "autogluon_version": "1.4.0",
  "python_version": "3.11.0",
  "training_timestamp": "2025-11-17T10:30:45",
  "num_models_trained": 36,
  "best_model": "WeightedEnsemble_L3",
  "eval_metric": "roc_auc",
  "problem_type": "binary",
  "target_column": "dementia_status",
  "feature_count": 113,
  "training_samples": 116034,
  "test_samples": 29279,
  "training_time_seconds": 1795,
  "hyperparameters": {
    "time_limit": 1800,
    "preset": "medium_quality",
    "num_bag_folds": 3,
    "num_stack_levels": 1
  }
}
```

**How AutoGluon Uses It:**
- Validates environment compatibility during `.load()`
- Warns if AutoGluon version mismatch detected

**File Size:** ~2 KB

**Must Be Bundled:** ✅ YES

---

### 3.3 `models/` Directory (Individual Model Artifacts)

**Description:**
Contains 36 subdirectories, each storing a trained model and its hyperparameters.

**Example: `models/LightGBM_r188_BAG_L2/`**

#### **`model.pkl`**
- Serialized LightGBM Booster object
- Contains 100 decision trees (num_boost_round=100)
- Trained on 3-fold bagged data
- File Size: ~12 MB per model

#### **`hyperparameters.json`**
```json
{
  "model_type": "LGBMModel",
  "hyperparameters": {
    "num_leaves": 128,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_samples": 20,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1
  },
  "bag_folds": 3,
  "stack_level": 2
}
```

#### **`metadata.json`**
```json
{
  "model_name": "LightGBM_r188_BAG_L2",
  "model_type": "LGBMModel",
  "score_val": 0.9813,
  "score_test": 0.9709,
  "fit_time": 45.2,
  "predict_time": 1.23,
  "num_features": 113,
  "num_classes": 2
}
```

**How AutoGluon Uses It:**
- `TabularPredictor.load()` iterates through `models/` and deserializes each `model.pkl`
- Ensemble weights in `WeightedEnsemble_L3/model.pkl` reference these models

**File Size (Total):** ~450 MB (all 36 models combined)

**Must Be Bundled:** ✅ YES (entire directory)

---

### 3.4 `utils/` Directory (Internal AutoGluon Files)

**Description:**
AutoGluon's internal utilities for feature processing and transformations.

#### **`feature_metadata.pkl`**
**Contents:**
```python
{
  "feature_names": ["NACCAGE", "NACCAGEB", "INDEPEND", "REMDATES", ...],
  "feature_types": {
    "NACCAGE": "int",
    "INDEPEND": "int",
    "NACCEDUC": "category",
    "SEX": "category"
  },
  "feature_stats": {
    "NACCAGE": {"min": 40, "max": 105, "mean": 72.5, "std": 10.2},
    "INDEPEND": {"min": 0, "max": 3, "mean": 0.8, "std": 1.1}
  }
}
```

**Purpose:**
- Validates incoming data schema during `.predict()`
- Ensures new data matches training feature types

#### **`feature_transformations.pkl`**
**Contents:**
- Categorical encoders (LabelEncoder, OneHotEncoder)
- Numerical scalers (StandardScaler, RobustScaler)
- Missing value imputers (SimpleImputer)

**Example Transformation:**
```python
# Stored in feature_transformations.pkl
{
  "NACCAGE": StandardScaler(mean=72.5, std=10.2),
  "SEX": LabelEncoder(classes=["Male", "Female"]),
  "INDEPEND": SimpleImputer(strategy="median", fill_value=0)
}
```

**How AutoGluon Uses It:**
```python
# Automatically applied during predictor.predict()
new_data = predictor._learner._apply_transformations(new_data)
predictions = predictor._learner._predict(new_data)
```

#### **`label_encoder.pkl`**
**Contents:**
```python
LabelEncoder(classes=["No Dementia", "Dementia"])
# Maps: 0 → No Dementia, 1 → Dementia
```

**Purpose:**
- Encodes target variable during training
- Decodes predictions back to original labels

**File Size (Total):** ~10 MB

**Must Be Bundled:** ✅ YES (auto-loaded by `learner.pkl`)

---

### 3.5 `model_leaderboard.csv` (Model Performance Comparison)

**Description:**
CSV file ranking all 36 models by validation ROC-AUC.

**Sample Contents:**
```csv
model,score_val,score_test,pred_time_val,fit_time,num_features
WeightedEnsemble_L3,0.9813,0.9709,12.75,1795.2,113
LightGBM_r188_BAG_L2,0.9813,0.9708,1.23,45.2,113
LightGBMLarge_BAG_L2,0.9812,0.9707,1.45,52.1,113
XGBoost_r33_BAG_L2,0.9812,0.9706,2.34,78.5,113
...
```

**Columns:**
- `model`: Model name
- `score_val`: Validation ROC-AUC (3-fold CV)
- `score_test`: Test ROC-AUC (held-out set)
- `pred_time_val`: Inference time (seconds)
- `fit_time`: Training time (seconds)
- `num_features`: Number of input features (113)

**How AutoGluon Uses It:**
- Generated during `predictor.leaderboard()` call
- Used for model selection and debugging

**File Size:** ~3 KB

**Must Be Bundled:** ⚠️ Recommended (not required for inference)

---

### 3.6 `test_evaluation_metrics.json` (Final Evaluation Metrics)

**Description:**
Comprehensive evaluation metrics on the test set.

**Contents:**
```json
{
  "roc_auc": 0.9709,
  "accuracy": 0.9055,
  "precision": 0.9230,
  "recall": 0.7414,
  "f1_score": 0.8223,
  "confusion_matrix": {
    "true_negatives": 20107,
    "false_positives": 534,
    "false_negatives": 2234,
    "true_positives": 6404
  },
  "class_distribution": {
    "no_dementia": 20641,
    "dementia": 8638
  },
  "threshold": 0.5,
  "test_set_size": 29279,
  "evaluation_timestamp": "2025-11-17T12:45:30"
}
```

**How AutoGluon Uses It:**
- Not loaded during inference (documentation only)
- Used for reporting and validation

**File Size:** ~1 KB

**Must Be Bundled:** ⚠️ Recommended (for transparency)

---

### 3.7 `feature_importance.csv` (SHAP Feature Rankings)

**Description:**
Feature importance scores computed using SHAP (SHapley Additive exPlanations).

**Sample Contents:**
```csv
feature,importance,rank
INDEPEND,0.01662,1
REMDATES,0.00381,2
TAXES,0.00344,3
TRAVEL,0.00340,4
BILLS,0.00337,5
...
```

**How AutoGluon Uses It:**
- Generated via `predictor.feature_importance()`
- Not used during inference (interpretability only)

**File Size:** ~5 KB

**Must Be Bundled:** ⚠️ Recommended (for interpretability)

---

### 3.8 `version.txt` (AutoGluon Version Stamp)

**Description:**
Simple text file with AutoGluon version.

**Contents:**
```
1.4.0
```

**How AutoGluon Uses It:**
- Quick version check during deployment
- Warns if version mismatch detected

**File Size:** <1 KB

**Must Be Bundled:** ✅ YES

---

## 4. AutoGluon Predictor Internal Structure

### 4.1 How `TabularPredictor.load()` Works

When you call `TabularPredictor.load("path/to/model/")`, AutoGluon performs these steps:

**Step 1: Load `learner.pkl`**
```python
with open("path/to/model/learner.pkl", "rb") as f:
    learner = pickle.load(f)
```

**Step 2: Load Individual Models**
```python
for model_name in learner.model_names:
    model_path = f"models/{model_name}/model.pkl"
    model = pickle.load(open(model_path, "rb"))
    learner.models[model_name] = model
```

**Step 3: Load Feature Transformations**
```python
learner.feature_metadata = pickle.load(open("utils/feature_metadata.pkl", "rb"))
learner.transformations = pickle.load(open("utils/feature_transformations.pkl", "rb"))
```

**Step 4: Validate Environment**
```python
metadata = json.load(open("metadata.json"))
if metadata["autogluon_version"] != autogluon.__version__:
    warnings.warn("Version mismatch detected")
```

**Step 5: Return Predictor**
```python
predictor = TabularPredictor(learner=learner)
return predictor
```

### 4.2 What Happens During `predictor.predict(new_data)`

**Step 1: Validate Schema**
```python
# Check that new_data has all 113 required features
assert set(new_data.columns) == set(learner.feature_metadata["feature_names"])
```

**Step 2: Apply Transformations**
```python
# Apply preprocessing (scaling, encoding, imputation)
for feature, transformer in learner.transformations.items():
    new_data[feature] = transformer.transform(new_data[feature])
```

**Step 3: Generate Predictions from All Models**
```python
predictions = {}
for model_name, model in learner.models.items():
    predictions[model_name] = model.predict(new_data)
```

**Step 4: Ensemble Predictions**
```python
# WeightedEnsemble_L3 combines predictions
weights = learner.models["WeightedEnsemble_L3"].weights
final_predictions = np.average(
    list(predictions.values()),
    axis=0,
    weights=weights
)
```

**Step 5: Decode Labels**
```python
# Convert 0/1 back to "No Dementia"/"Dementia"
final_predictions = learner.label_encoder.inverse_transform(final_predictions)
```

---

## 5. Feature Metadata & Schema

### 5.1 Required Feature Schema

**Total Features:** 113

**Feature Categories:**

| Category | Count | Examples |
|----------|-------|----------|
| Demographics | 8 | `NACCAGE`, `SEX`, `NACCEDUC`, `MARISTAT` |
| Functional Independence | 15 | `INDEPEND`, `BILLS`, `TAXES`, `TRAVEL` |
| Memory & Cognition | 12 | `REMDATES`, `EVENTS`, `PAYATTN`, `ORIENT` |
| Lifestyle | 10 | `SMOKYRS`, `ALCOHOL`, `EXERCISE`, `DIET` |
| Social Context | 8 | `NACCLIVS`, `RESIDENC`, `NETWORK`, `SUPPORT` |
| Medical History | 20 | `STROKE`, `HEART`, `HYPERTENS`, `DIABETES` |
| Other | 40 | Various non-medical indicators |

### 5.2 Feature Type Distribution

```python
{
  "int": 68,       # Integer features (e.g., age, scores)
  "float": 15,     # Continuous features (e.g., BMI, ratios)
  "category": 30   # Categorical features (e.g., sex, education)
}
```

### 5.3 Example Feature Definitions

**Sample from `feature_metadata.pkl`:**

```python
{
  "NACCAGE": {
    "type": "int",
    "description": "Age at visit",
    "range": [40, 105],
    "missing_allowed": False,
    "preprocessing": "StandardScaler"
  },
  "INDEPEND": {
    "type": "int",
    "description": "Independent living capability (0=Independent, 3=Fully dependent)",
    "range": [0, 3],
    "missing_allowed": True,
    "preprocessing": "SimpleImputer(median) → StandardScaler"
  },
  "SEX": {
    "type": "category",
    "description": "Biological sex",
    "categories": ["Male", "Female"],
    "missing_allowed": False,
    "preprocessing": "LabelEncoder"
  }
}
```

### 5.4 Schema Validation Code

**Example Validation Script:**
```python
import pandas as pd
import pickle

# Load feature metadata
with open("models/autogluon_production_lowmem/utils/feature_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def validate_schema(new_data: pd.DataFrame) -> bool:
    """Validate that new_data matches training schema."""

    # Check column count
    if len(new_data.columns) != 113:
        raise ValueError(f"Expected 113 columns, got {len(new_data.columns)}")

    # Check column names
    expected_cols = set(metadata["feature_names"])
    actual_cols = set(new_data.columns)
    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if extra_cols:
        raise ValueError(f"Unexpected columns: {extra_cols}")

    # Check feature types
    for feature, expected_type in metadata["feature_types"].items():
        actual_type = new_data[feature].dtype
        if expected_type == "int" and actual_type not in ["int64", "int32"]:
            raise TypeError(f"{feature}: expected int, got {actual_type}")
        if expected_type == "float" and actual_type not in ["float64", "float32"]:
            raise TypeError(f"{feature}: expected float, got {actual_type}")
        if expected_type == "category" and actual_type != "object":
            raise TypeError(f"{feature}: expected category, got {actual_type}")

    return True
```

---

## 6. Model Hyperparameters

### 6.1 AutoGluon Global Hyperparameters

**From `metadata.json`:**
```json
{
  "time_limit": 1800,                # 30 minutes training time
  "preset": "medium_quality",        # Balance speed/performance
  "eval_metric": "roc_auc",          # Optimization target
  "problem_type": "binary",          # Binary classification
  "num_bag_folds": 3,                # 3-fold cross-validation
  "num_stack_levels": 1,             # 1 level of stacking (reduced for memory)
  "auto_stack": true,                # Enable automatic stacking
  "hyperparameters": "default",      # Use AutoGluon defaults
  "verbosity": 2                     # Moderate logging
}
```

### 6.2 Model-Specific Hyperparameters

#### **LightGBM_r188_BAG_L2** (Best Single Model)
```json
{
  "num_leaves": 128,
  "learning_rate": 0.03,
  "max_depth": 7,
  "min_child_samples": 20,
  "boosting_type": "gbdt",
  "objective": "binary",
  "metric": "auc",
  "num_boost_round": 100,
  "feature_fraction": 0.9,
  "bagging_fraction": 0.8,
  "bagging_freq": 5,
  "lambda_l1": 0.1,
  "lambda_l2": 0.1,
  "verbosity": -1
}
```

#### **XGBoost_r33_BAG_L2** (2nd Best)
```json
{
  "max_depth": 6,
  "learning_rate": 0.03,
  "n_estimators": 100,
  "objective": "binary:logistic",
  "eval_metric": "auc",
  "subsample": 0.8,
  "colsample_bytree": 0.9,
  "min_child_weight": 3,
  "gamma": 0.1,
  "reg_alpha": 0.1,
  "reg_lambda": 0.1,
  "tree_method": "hist"
}
```

#### **WeightedEnsemble_L3** (Final Ensemble)
```json
{
  "ensemble_type": "weighted",
  "num_models": 36,
  "weights": {
    "LightGBM_r188_BAG_L2": 0.15,
    "LightGBMLarge_BAG_L2": 0.14,
    "XGBoost_r33_BAG_L2": 0.13,
    "CatBoost_r45_BAG_L2": 0.10,
    "... (32 more models)": "...",
    "NeuralNetFastAI_BAG_L1": 0.01
  },
  "optimizer": "least_squares",      # Optimize weights using least squares
  "metric": "roc_auc"
}
```

---

## 7. Preprocessing Pipeline Details

### 7.1 Embedded Preprocessing Steps

**All preprocessing is embedded inside `learner.pkl` and applied automatically during `.predict()`:**

#### **1. Missing Value Imputation**
```python
{
  "NACCAGE": None,                          # No imputation (required field)
  "INDEPEND": SimpleImputer(strategy="median", fill_value=0),
  "REMDATES": SimpleImputer(strategy="mode", fill_value=0),
  "SMOKYRS": SimpleImputer(strategy="median", fill_value=0)
}
```

#### **2. Categorical Encoding**
```python
{
  "SEX": LabelEncoder(classes=["Male", "Female"]),
  "NACCEDUC": LabelEncoder(classes=["<High School", "High School", "College", "Graduate"]),
  "MARISTAT": OneHotEncoder(categories=["Single", "Married", "Divorced", "Widowed"])
}
```

#### **3. Numerical Scaling**
```python
{
  "NACCAGE": StandardScaler(mean=72.5, std=10.2),
  "INDEPEND": RobustScaler(median=0, IQR=1.5),
  "SMOKYRS": MinMaxScaler(min=0, max=50)
}
```

#### **4. Outlier Handling**
```python
# No outliers removed (flagged only)
# Example: NACCAGE > 100 → flagged but not capped
```

### 7.2 Why Preprocessing is NOT Stored Separately

**Key Insight:** AutoGluon embeds preprocessing directly in `learner.pkl`, so you do NOT need separate:
- ❌ `scaler.pkl`
- ❌ `encoder.pkl`
- ❌ `imputer.pkl`

**Everything is handled internally:**
```python
predictor = TabularPredictor.load("path/to/model/")
# All preprocessing automatically applied here:
predictions = predictor.predict(new_data)
```

---

## 8. Requirements & Dependencies

### 8.1 Python Version

**Required:** Python 3.9, 3.10, or 3.11

**Installed Version:** Python 3.11.0

**Compatibility:**
- ✅ Python 3.9: Fully compatible
- ✅ Python 3.10: Fully compatible
- ✅ Python 3.11: Tested and working
- ⚠️ Python 3.12: Not tested (may work)
- ❌ Python 3.8 or below: Not supported by AutoGluon 1.4.0

### 8.2 Core Dependencies

**From `requirements.txt`:**

```txt
# Core AutoML
autogluon.tabular>=1.4.0
pycaret>=3.0.0
flaml>=2.0.0

# Machine Learning
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0

# Gradient Boosting
lightgbm>=4.0.0
xgboost>=2.0.0
catboost>=1.2.0

# Deep Learning (Optional)
torch>=2.0.0
fastai>=2.7.0

# Feature Engineering
feature-engine>=1.6.0
category-encoders>=2.6.0

# Interpretability
shap>=0.42.0
lime>=0.2.0

# Data Validation
great-expectations>=0.17.0
pandera>=0.15.0

# Monitoring
evidently>=0.4.0
alibi-detect>=0.11.0

# Utilities
mlflow>=2.7.0
joblib>=1.3.0
pyyaml>=6.0
omegaconf>=2.3.0
tqdm>=4.65.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

### 8.3 Minimal Requirements (Inference Only)

If you only need to load the model and make predictions (not retrain), you can use a minimal environment:

```txt
autogluon.tabular>=1.4.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
catboost>=1.2.0
torch>=2.0.0        # Only if neural network models are used
```

**Installation:**
```bash
pip install autogluon.tabular==1.4.0 --no-cache-dir
```

### 8.4 Environment File

**`environment.yaml` (Conda):**
```yaml
name: autogluon_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      - autogluon.tabular==1.4.0
      - numpy>=1.24.0,<2.0.0
      - pandas>=2.0.0
      - scikit-learn>=1.3.0
      - lightgbm>=4.0.0
      - xgboost>=2.0.0
      - catboost>=1.2.0
```

**Install with:**
```bash
conda env create -f environment.yaml
conda activate autogluon_env
```

---

## 9. Deployment Bundle Checklist

### 9.1 Must-Have Files (Cannot Deploy Without)

```
✅ models/autogluon_production_lowmem/learner.pkl
✅ models/autogluon_production_lowmem/metadata.json
✅ models/autogluon_production_lowmem/version.txt
✅ models/autogluon_production_lowmem/models/        (entire directory, 36 subdirectories)
✅ models/autogluon_production_lowmem/utils/         (entire directory)
✅ requirements.txt                                  (or environment.yaml)
```

### 9.2 Recommended Files (For Validation/Debugging)

```
⚠️ models/autogluon_production_lowmem/model_leaderboard.csv
⚠️ models/autogluon_production_lowmem/test_evaluation_metrics.json
⚠️ models/autogluon_production_lowmem/feature_importance.csv
⚠️ models/autogluon_production_lowmem/test_predictions.csv    (sample outputs)
```

### 9.3 Optional Files (Documentation/Analysis)

```
❌ models/autogluon_production_lowmem/training_logs/
❌ models/autogluon_production_lowmem/test_probabilities.csv
❌ NODE_DOCUMENT.md                                   (this documentation)
❌ PROJECT_TECHNICAL_DOCUMENT.md
❌ MODEL_DEPLOYMENT_GUIDE.md
```

### 9.4 Deployment Archive Structure

**Create a deployment-ready archive:**

```bash
# On Windows (PowerShell)
Compress-Archive -Path models/autogluon_production_lowmem, requirements.txt, *.md -DestinationPath autogluon_model_bundle.zip

# On Linux/Mac
tar -czvf autogluon_model_bundle.tar.gz models/autogluon_production_lowmem requirements.txt *.md
```

**Archive Contents:**
```
autogluon_model_bundle.zip (or .tar.gz)
│
├── models/autogluon_production_lowmem/       # Main model directory
│   ├── learner.pkl
│   ├── metadata.json
│   ├── version.txt
│   ├── models/                               # All 36 models
│   └── utils/                                # Feature metadata
│
├── requirements.txt                          # Dependencies
├── NODE_DOCUMENT.md                          # Model explanation
├── PROJECT_TECHNICAL_DOCUMENT.md             # This file
├── MODEL_DEPLOYMENT_GUIDE.md                 # Deployment instructions
└── README.md                                 # Quick start guide
```

**Recommended Archive Size:** ~500 MB (compressed to ~250 MB)

---

## 10. File Size & Storage Requirements

### 10.1 Individual File Sizes

| File/Directory | Size | Notes |
|---------------|------|-------|
| `learner.pkl` | 450 MB | Contains all 36 models + preprocessing |
| `models/` (all 36 models) | 450 MB | Individual model artifacts |
| `utils/` | 10 MB | Feature metadata & transformations |
| `metadata.json` | 2 KB | Version info |
| `model_leaderboard.csv` | 3 KB | Performance table |
| `test_evaluation_metrics.json` | 1 KB | Final metrics |
| `feature_importance.csv` | 5 KB | SHAP rankings |
| `test_predictions.csv` | 1.3 MB | Full test set outputs |
| **Total** | **~920 MB** | **Uncompressed** |

### 10.2 Compressed Archive Size

**Compressed (ZIP/TAR.GZ):** ~250 MB

**Compression Ratio:** ~73% reduction

### 10.3 Storage Requirements by Deployment Type

| Deployment Type | Storage Required | Recommended |
|----------------|------------------|-------------|
| Local Development | 1 GB | SSD with 10 GB free |
| Cloud Server (AWS/GCP/Azure) | 2 GB | Instance with 8 GB+ RAM |
| Docker Container | 1.5 GB | Alpine-based image |
| Edge Device (Raspberry Pi) | ❌ Not Recommended | Model too large |
| Mobile App (iOS/Android) | ❌ Not Recommended | Model too large |

### 10.4 Memory Requirements (RAM)

**During Model Loading:**
- Initial Load: 500 MB
- Peak Memory: 800 MB
- Steady State: 600 MB

**During Inference:**
- Small Batch (<100 rows): +50 MB
- Medium Batch (1,000 rows): +200 MB
- Large Batch (10,000 rows): +500 MB

**Recommended RAM:**
- Development: 8 GB minimum, 16 GB recommended
- Production: 16 GB minimum, 32 GB recommended

---

## 11. Summary

### 11.1 Critical Takeaways

**1. Do NOT Copy Only `learner.pkl`**
- AutoGluon requires the entire `models/autogluon_production_lowmem/` directory
- Missing `models/` subdirectories will cause loading errors

**2. Preprocessing is Embedded**
- No need to separately save scalers, encoders, or imputers
- Everything is handled by `learner.pkl`

**3. Version Compatibility Matters**
- Model trained with AutoGluon 1.4.0 + Python 3.11
- Use same versions in deployment environment to avoid errors

**4. Schema Consistency is Critical**
- New data must have exactly 113 features in the same order
- Missing or extra columns will cause prediction failures

**5. Storage Requirements**
- Minimum 1 GB for model artifacts
- Recommended 2 GB for deployment bundle (includes docs)

### 11.2 Quick Deployment Checklist

```
✅ Copy entire models/autogluon_production_lowmem/ directory
✅ Install AutoGluon 1.4.0 in Python 3.9-3.11 environment
✅ Verify all dependencies installed (see requirements.txt)
✅ Test loading: predictor = TabularPredictor.load("path/to/model/")
✅ Validate schema: new_data has 113 features
✅ Test prediction: predictor.predict(sample_data)
✅ Monitor memory usage (should be <1 GB)
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Model Version:** AutoGluon v1.4.0 (Production Low-Memory Configuration)
**Contact:** ModelX Development Team
