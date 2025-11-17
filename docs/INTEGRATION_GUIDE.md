# Integration Guide
## Step-by-Step: Integrating AutoGluon Model into New Project

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Loading the Model](#loading-the-model)
4. [Validating Input Data](#validating-input-data)
5. [Running Predictions](#running-predictions)
6. [Interpreting Results](#interpreting-results)
7. [Troubleshooting](#troubleshooting)
8. [Advanced: Continuing Training](#advanced-continuing-training)

---

## 1. Prerequisites

**Before starting, ensure you have:**
- ✅ Python 3.9, 3.10, or 3.11 installed
- ✅ At least 8 GB RAM available
- ✅ 2 GB free disk space
- ✅ Complete model directory: `models/autogluon_production_lowmem/`

---

## 2. Installation

### Step 1: Create Virtual Environment

```bash
# Create environment
python -m venv autogluon_env

# Activate (Linux/Mac)
source autogluon_env/bin/activate

# Activate (Windows)
autogluon_env\Scripts\activate
```

### Step 2: Install AutoGluon

```bash
pip install autogluon.tabular==1.4.0
```

**Installation time:** 5-10 minutes

### Step 3: Verify Installation

```python
import autogluon
print(f"AutoGluon version: {autogluon.__version__}")
# Expected: 1.4.0
```

---

## 3. Loading the Model

### Step 1: Import Libraries

```python
from autogluon.tabular import TabularPredictor
import pandas as pd
```

### Step 2: Load Predictor

```python
# Load model from directory
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

print(f"Model loaded: {predictor.problem_type}")  # binary
print(f"Best model: {predictor.model_best}")      # WeightedEnsemble_L3
```

**Loading time:** 10-15 seconds

### Step 3: Inspect Model

```python
# View leaderboard
leaderboard = predictor.leaderboard()
print(leaderboard.head())

# Check feature count
features = predictor.feature_metadata_in.get_features()
print(f"Required features: {len(features)}")  # 113
```

---

## 4. Validating Input Data

### Step 1: Load Your Data

```python
# Load new patient data
new_data = pd.read_csv("your_patient_data.csv")
print(f"Loaded {len(new_data)} samples")
```

### Step 2: Validate Schema

```python
def validate_schema(df, predictor):
    """Check if data matches model requirements."""

    expected_features = predictor.feature_metadata_in.get_features()

    # Check column count
    if len(df.columns) != len(expected_features):
        print(f"❌ Column count mismatch: expected {len(expected_features)}, got {len(df.columns)}")
        return False

    # Check for missing columns
    missing = set(expected_features) - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False

    # Check for extra columns
    extra = set(df.columns) - set(expected_features)
    if extra:
        print(f"⚠️ Extra columns (will be ignored): {extra}")
        df = df.drop(columns=extra)

    print("✅ Schema validation passed")
    return True

# Validate
if validate_schema(new_data, predictor):
    print("Ready for prediction")
```

### Step 3: Fix Schema Issues (If Needed)

```python
# Add missing columns
expected_features = predictor.feature_metadata_in.get_features()
for col in expected_features:
    if col not in new_data.columns:
        new_data[col] = 0  # Default value

# Remove extra columns
new_data = new_data[expected_features]

# Reorder columns to match training order
new_data = new_data[expected_features]
```

---

## 5. Running Predictions

### Step 1: Predict Class Labels

```python
# Predict dementia status (binary: "No Dementia" or "Dementia")
predictions = predictor.predict(new_data)

print(predictions.head())
# 0    No Dementia
# 1    Dementia
# 2    No Dementia
```

### Step 2: Predict Probabilities

```python
# Get probability estimates
probabilities = predictor.predict_proba(new_data)

print(probabilities.head())
#      No Dementia  Dementia
# 0         0.92      0.08
# 1         0.23      0.77
# 2         0.95      0.05
```

### Step 3: Combine Results

```python
# Add predictions to original data
new_data["predicted_class"] = predictions
new_data["dementia_risk_probability"] = probabilities["Dementia"]

# Save results
new_data.to_csv("predictions_output.csv", index=False)
print("✅ Predictions saved to predictions_output.csv")
```

---

## 6. Interpreting Results

### Probability Interpretation

```python
def interpret_risk(probability):
    """Convert probability to risk category."""
    if probability < 0.10:
        return "Very Low Risk"
    elif probability < 0.30:
        return "Low Risk"
    elif probability < 0.50:
        return "Moderate Risk"
    elif probability < 0.70:
        return "High Risk"
    else:
        return "Very High Risk"

# Apply to results
new_data["risk_category"] = probabilities["Dementia"].apply(interpret_risk)

print(new_data[["predicted_class", "dementia_risk_probability", "risk_category"]].head())
#   predicted_class  dementia_risk_probability  risk_category
# 0  No Dementia                  0.08          Very Low Risk
# 1  Dementia                     0.77          Very High Risk
```

### Summary Statistics

```python
# Overall risk distribution
print(f"Mean Risk: {probabilities['Dementia'].mean():.2%}")
print(f"Median Risk: {probabilities['Dementia'].median():.2%}")
print(f"High Risk (>70%): {(probabilities['Dementia'] > 0.7).sum()} patients")
```

---

## 7. Troubleshooting

### Issue 1: Model Not Loading

**Error:** `FileNotFoundError`

**Solution:**
```python
import os

model_path = "models/autogluon_production_lowmem/"
if not os.path.exists(model_path):
    print(f"❌ Model directory not found: {model_path}")
    print("Ensure you copied the entire 'models/autogluon_production_lowmem/' directory")
else:
    print("✅ Model directory exists")
```

### Issue 2: Missing Columns

**Error:** `ValueError: Missing required features`

**Solution:**
```python
# Generate template with all required columns
template = pd.DataFrame(columns=predictor.feature_metadata_in.get_features())
template.to_csv("input_template.csv", index=False)
print("✅ Template saved to input_template.csv")
```

### Issue 3: Type Errors

**Error:** `TypeError: Cannot convert...`

**Solution:**
```python
# Fix data types
new_data["NACCAGE"] = pd.to_numeric(new_data["NACCAGE"], errors="coerce")
new_data["INDEPEND"] = pd.to_numeric(new_data["INDEPEND"], errors="coerce").astype("Int64")
```

---

## 8. Advanced: Continuing Training

### Option 1: Add New Data to Existing Model

```python
# Load original predictor
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

# Load new training data
new_train_data = pd.read_csv("new_training_data.csv")

# Continue training (incremental learning)
predictor = predictor.fit(
    train_data=new_train_data,
    time_limit=600,  # 10 minutes
    presets="medium_quality",
    keep_only_best=True
)

# Save updated model
predictor.save("models/autogluon_updated/")
```

### Option 2: Retrain from Scratch

```python
from autogluon.tabular import TabularPredictor

# Combine old + new data
combined_data = pd.concat([old_train_data, new_train_data])

# Train new predictor
new_predictor = TabularPredictor(label="dementia_status", eval_metric="roc_auc")
new_predictor.fit(
    train_data=combined_data,
    time_limit=1800,  # 30 minutes
    presets="medium_quality",
    num_bag_folds=3,
    num_stack_levels=1
)

# Save new model
new_predictor.save("models/autogluon_retrained/")
```

---

## Complete Integration Example

```python
#!/usr/bin/env python3
"""
Complete example: Load model, validate data, predict, save results.
"""

from autogluon.tabular import TabularPredictor
import pandas as pd

# 1. Load model
print("Loading model...")
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

# 2. Load data
print("Loading data...")
new_data = pd.read_csv("patient_data.csv")

# 3. Validate schema
expected_features = predictor.feature_metadata_in.get_features()
missing = set(expected_features) - set(new_data.columns)
if missing:
    print(f"Adding missing columns: {missing}")
    for col in missing:
        new_data[col] = 0

new_data = new_data[expected_features]

# 4. Predict
print(f"Predicting for {len(new_data)} samples...")
predictions = predictor.predict(new_data)
probabilities = predictor.predict_proba(new_data)

# 5. Save results
new_data["predicted_class"] = predictions
new_data["dementia_risk"] = probabilities["Dementia"]
new_data.to_csv("predictions_output.csv", index=False)

print(f"✅ Done! Results saved to predictions_output.csv")
print(f"   Mean risk: {probabilities['Dementia'].mean():.2%}")
print(f"   High risk (>70%): {(probabilities['Dementia'] > 0.7).sum()} patients")
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
