# Inference Documentation
## AutoGluon Pipeline Internal Processing

---

## Table of Contents
1. [Inference Pipeline Overview](#inference-pipeline-overview)
2. [Preprocessing During Inference](#preprocessing-during-inference)
3. [Missing Value Handling](#missing-value-handling)
4. [Categorical Encoding](#categorical-encoding)
5. [Numerical Normalization](#numerical-normalization)
6. [Schema Validation](#schema-validation)
7. [Troubleshooting Column Mismatches](#troubleshooting-column-mismatches)

---

## 1. Inference Pipeline Overview

### 1.1 Inference Flow

```
New Data (CSV/DataFrame)
    ↓
Schema Validation (113 features check)
    ↓
Missing Value Imputation (Auto-applied from stored imputers)
    ↓
Categorical Encoding (LabelEncoder/OneHotEncoder)
    ↓
Numerical Scaling (StandardScaler/RobustScaler)
    ↓
Prediction by 36 Models (LightGBM, XGBoost, CatBoost, etc.)
    ↓
Ensemble Aggregation (Weighted voting)
    ↓
Final Prediction (Class + Probability)
```

### 1.2 Key Principle

**All preprocessing steps are stored inside `learner.pkl` and applied automatically during `predictor.predict()`**

You do NOT need to manually:
- Impute missing values
- Encode categorical features
- Scale numerical features

AutoGluon handles everything internally.

---

## 2. Preprocessing During Inference

### 2.1 Stored Preprocessing Objects

**Location:** `models/autogluon_production_lowmem/utils/feature_transformations.pkl`

**Contents:**
```python
{
  "imputers": {
    "INDEPEND": SimpleImputer(strategy="median", fill_value=0),
    "REMDATES": SimpleImputer(strategy="mode", fill_value=0),
    "SMOKYRS": SimpleImputer(strategy="median", fill_value=0)
  },
  "encoders": {
    "SEX": LabelEncoder(classes=["Male", "Female"]),
    "NACCEDUC": LabelEncoder(classes=["<HS", "HS", "College", "Grad"])
  },
  "scalers": {
    "NACCAGE": StandardScaler(mean=72.5, std=10.2),
    "INDEPEND": RobustScaler(median=0, IQR=1.5)
  }
}
```

### 2.2 Automatic Application

```python
# When you call predictor.predict()
new_data = pd.DataFrame([{
    "NACCAGE": 72,
    "INDEPEND": None,  # Missing value
    "SEX": "Male",     # Categorical
    ...
}])

predictions = predictor.predict(new_data)

# Internally, AutoGluon runs:
# 1. new_data["INDEPEND"] = imputer.transform(new_data["INDEPEND"])  → 0
# 2. new_data["SEX"] = encoder.transform(new_data["SEX"])  → 0
# 3. new_data["NACCAGE"] = scaler.transform(new_data["NACCAGE"])  → -0.05
# 4. predictions = ensemble.predict(transformed_data)
```

---

## 3. Missing Value Handling

### 3.1 Imputation Strategies

**AutoGluon applies different strategies by feature type:**

| Feature Type | Strategy | Example |
|-------------|----------|---------|
| Numerical (continuous) | Median imputation | `NACCAGE`: missing → 72 (median) |
| Numerical (discrete) | Median imputation | `INDEPEND`: missing → 0 (median) |
| Categorical | Mode imputation | `SEX`: missing → "Male" (most frequent) |
| Binary | Mode imputation | `STROKE`: missing → 0 (no stroke) |

### 3.2 Example: Missing Values Auto-Handled

```python
# Input data with missing values
new_data = pd.DataFrame([{
    "NACCAGE": 72,
    "INDEPEND": None,     # Missing
    "REMDATES": None,     # Missing
    "TAXES": 1,
    "SEX": "Male",
    "SMOKYRS": None,      # Missing
    ...
}])

# AutoGluon automatically imputes:
# INDEPEND: None → 0 (median from training data)
# REMDATES: None → 0 (median from training data)
# SMOKYRS: None → 0 (median from training data)

predictions = predictor.predict(new_data)  # Works without errors
```

### 3.3 Missing Value Flags

**AutoGluon may create indicator columns for missingness:**

```python
# If INDEPEND has >10% missing in training, AutoGluon creates:
# INDEPEND_missing (0/1 flag indicating whether value was missing)

# During inference, if INDEPEND is None:
# INDEPEND → 0 (imputed)
# INDEPEND_missing → 1 (flagged as missing)
```

**You don't need to create these flags manually** - AutoGluon handles it automatically.

---

## 4. Categorical Encoding

### 4.1 Encoding Methods Used

**AutoGluon uses two encoding strategies:**

#### **LabelEncoder** (for binary/ordinal features)
```python
# Example: SEX
# Training: {"Male": 0, "Female": 1}
# Inference: "Male" → 0, "Female" → 1

# Example: NACCEDUC (education)
# Training: {"<HS": 0, "HS": 1, "College": 2, "Grad": 3}
# Inference: "College" → 2
```

#### **OneHotEncoder** (for multi-class features)
```python
# Example: MARISTAT (marital status)
# Training: Creates columns [MARISTAT_Single, MARISTAT_Married, MARISTAT_Divorced]
# Inference: "Married" → [0, 1, 0]
```

### 4.2 Handling Unknown Categories

**If new data contains categories not seen during training:**

```python
# Training data: SEX = ["Male", "Female"]
# New data: SEX = "Other"

# AutoGluon behavior:
# Option 1: Raise error (strict mode)
# Option 2: Map to most frequent class (default) → "Male"
```

**Recommended:** Ensure new data categories match training data.

---

## 5. Numerical Normalization

### 5.1 Scaling Methods

**AutoGluon applies feature-specific scaling:**

| Feature | Scaler | Transformation |
|---------|--------|---------------|
| NACCAGE | StandardScaler | `(x - 72.5) / 10.2` |
| INDEPEND | RobustScaler | `(x - median) / IQR` |
| SMOKYRS | MinMaxScaler | `(x - 0) / (50 - 0)` |

### 5.2 Example: Automatic Scaling

```python
# Input data
new_data = pd.DataFrame([{
    "NACCAGE": 82,        # Raw value
    "INDEPEND": 2,        # Raw value
    "SMOKYRS": 10,        # Raw value
    ...
}])

# AutoGluon applies scaling automatically:
# NACCAGE: 82 → (82 - 72.5) / 10.2 = 0.93 (standardized)
# INDEPEND: 2 → (2 - 0) / 1.5 = 1.33 (robust scaled)
# SMOKYRS: 10 → 10 / 50 = 0.20 (min-max scaled)

predictions = predictor.predict(new_data)
```

**You receive predictions without seeing the transformed values** - AutoGluon handles scaling internally.

### 5.3 Stored Scaler Parameters

**Location:** `utils/feature_transformations.pkl`

**Example:**
```python
StandardScaler(mean=72.5, std=10.2)  # Fitted on training data

# During inference:
scaled_value = (new_value - 72.5) / 10.2
```

---

## 6. Schema Validation

### 6.1 Required Schema

**Critical Requirements:**
- ✅ Exactly 113 features
- ✅ Feature names match training data (case-sensitive)
- ✅ Feature types match (int, float, category)
- ❌ No extra columns allowed

### 6.2 Validation Code

```python
def validate_schema(new_data, predictor):
    """Ensure new_data matches training schema."""

    # Get expected features
    expected_features = predictor.feature_metadata_in.get_features()

    # Check column count
    if len(new_data.columns) != len(expected_features):
        raise ValueError(f"Expected {len(expected_features)} columns, got {len(new_data.columns)}")

    # Check column names
    missing = set(expected_features) - set(new_data.columns)
    extra = set(new_data.columns) - set(expected_features)

    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if extra:
        print(f"Warning: Extra columns will be ignored: {extra}")
        new_data = new_data.drop(columns=extra)

    # Reorder columns to match training order
    new_data = new_data[expected_features]

    return new_data

# Usage
new_data = validate_schema(new_data, predictor)
predictions = predictor.predict(new_data)
```

### 6.3 Feature Type Validation

```python
# Check that feature types match training data
feature_types = predictor.feature_metadata_in.get_type_map_raw()

for feature, expected_type in feature_types.items():
    actual_type = new_data[feature].dtype

    if expected_type == "int" and actual_type not in ["int64", "int32", "Int64"]:
        raise TypeError(f"{feature}: expected int, got {actual_type}")

    if expected_type == "float" and actual_type not in ["float64", "float32"]:
        raise TypeError(f"{feature}: expected float, got {actual_type}")

    if expected_type == "category" and actual_type != "object":
        raise TypeError(f"{feature}: expected category, got {actual_type}")
```

---

## 7. Troubleshooting Column Mismatches

### 7.1 Error: Missing Required Columns

**Symptom:**
```
ValueError: Missing required features: ['INDEPEND', 'REMDATES']
```

**Solution 1: Add missing columns with default values**
```python
new_data["INDEPEND"] = 0
new_data["REMDATES"] = 0
```

**Solution 2: Use template**
```python
# Generate template with all required columns
template = pd.DataFrame(columns=predictor.feature_metadata_in.get_features())
template.to_csv("input_template.csv", index=False)
```

### 7.2 Error: Unexpected Columns

**Symptom:**
```
ValueError: Unexpected features (remove these): {'patient_id', 'timestamp'}
```

**Solution: Drop extra columns**
```python
new_data = new_data.drop(columns=["patient_id", "timestamp"])
```

### 7.3 Error: Column Type Mismatch

**Symptom:**
```
TypeError: Cannot convert column 'NACCAGE' to int
```

**Solution: Convert to correct type**
```python
new_data["NACCAGE"] = pd.to_numeric(new_data["NACCAGE"], errors="coerce")
```

### 7.4 Error: Column Order Mismatch

**Symptom:**
Predictions are incorrect (no error message)

**Solution: Reorder columns**
```python
expected_order = predictor.feature_metadata_in.get_features()
new_data = new_data[expected_order]
```

### 7.5 Complete Validation Function

```python
def prepare_data_for_inference(new_data, predictor):
    """Comprehensive data preparation for inference."""

    # 1. Get expected schema
    expected_features = predictor.feature_metadata_in.get_features()
    feature_types = predictor.feature_metadata_in.get_type_map_raw()

    # 2. Handle missing columns
    missing = set(expected_features) - set(new_data.columns)
    if missing:
        print(f"Warning: Adding missing columns with default values: {missing}")
        for col in missing:
            new_data[col] = 0  # Or use appropriate default

    # 3. Remove extra columns
    extra = set(new_data.columns) - set(expected_features)
    if extra:
        print(f"Warning: Removing extra columns: {extra}")
        new_data = new_data.drop(columns=extra)

    # 4. Reorder columns
    new_data = new_data[expected_features]

    # 5. Fix data types
    for feature, expected_type in feature_types.items():
        if expected_type == "int":
            new_data[feature] = pd.to_numeric(new_data[feature], errors="coerce").fillna(0).astype("Int64")
        elif expected_type == "float":
            new_data[feature] = pd.to_numeric(new_data[feature], errors="coerce").astype("float64")
        elif expected_type == "category":
            new_data[feature] = new_data[feature].astype("str")

    return new_data

# Usage
new_data = prepare_data_for_inference(new_data, predictor)
predictions = predictor.predict(new_data)
```

---

## Summary

**Key Takeaways:**

1. **Preprocessing is Automatic**: All imputation, encoding, and scaling applied by AutoGluon during `predict()`
2. **Missing Values Handled**: Median/mode imputation automatically applied
3. **Categorical Encoding**: LabelEncoder/OneHotEncoder stored in `learner.pkl`
4. **Numerical Scaling**: StandardScaler/RobustScaler applied internally
5. **Schema Critical**: Must have exactly 113 features in correct order
6. **Validation Essential**: Check schema before prediction to avoid errors

**Recommended Workflow:**
```python
# 1. Load model
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

# 2. Validate and prepare data
new_data = prepare_data_for_inference(new_data, predictor)

# 3. Predict (preprocessing applied automatically)
predictions = predictor.predict(new_data)
probabilities = predictor.predict_proba(new_data)
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
