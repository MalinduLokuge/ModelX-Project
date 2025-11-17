# Model Reuse Philosophy
## Best Practices for Reusing AutoGluon Pipelines

---

## Why AutoML Pipelines Must Be Reused Carefully

### 1. Complexity of AutoML Artifacts

**AutoML models are NOT simple pickle files:**
- Traditional Model: Single `model.pkl` file (~10 MB)
- AutoML Model: Entire directory with 36+ models, metadata, preprocessing pipelines (~920 MB)

**What Can Go Wrong:**
❌ Copying only `learner.pkl` → Missing individual model files → Load failure
❌ Copying models without `utils/` → Missing preprocessors → Prediction errors
❌ Mismatched versions → Silent errors or incorrect predictions

**Correct Approach:**
✅ Copy entire `models/autogluon_production_lowmem/` directory
✅ Preserve directory structure exactly as-is
✅ Verify all subdirectories (`models/`, `utils/`) present

---

### 2. Hidden Dependencies

**AutoML models have many invisible dependencies:**

| Dependency Type | Example | Impact if Missing |
|----------------|---------|-------------------|
| Feature transformations | `StandardScaler(mean=72.5, std=10.2)` | Predictions off by orders of magnitude |
| Categorical encoders | `LabelEncoder(classes=["Male", "Female"])` | Unknown category errors |
| Missing value imputers | `SimpleImputer(strategy="median")` | NaN propagation → prediction failures |
| Feature schema | 113 columns in specific order | Schema mismatch errors |
| Model ensemble weights | `WeightedEnsemble_L3` weights | Suboptimal predictions |

**Key Principle:** Never assume you can cherry-pick parts of an AutoML model. It's an integrated system.

---

### 3. Version Sensitivity

**AutoML frameworks are version-sensitive:**

**Scenario:** Model trained with AutoGluon 1.4.0, deployed with AutoGluon 1.3.0

**Potential Issues:**
- ❌ Load failures (incompatible pickle protocol)
- ❌ Different preprocessing logic (e.g., missing value handling changed)
- ❌ Silent errors (predictions appear to work but are incorrect)

**Mitigation:**
1. **Lock versions:** Use exact version in `requirements.txt`
   ```
   autogluon.tabular==1.4.0  # NOT autogluon.tabular>=1.4.0
   ```

2. **Version checking code:**
   ```python
   import autogluon
   assert autogluon.__version__ == "1.4.0", "Version mismatch!"
   ```

3. **Document Python version:** Model trained with Python 3.11 → Deploy with Python 3.11

---

## Why Data Schema Consistency is Critical

### 1. Exact Feature Match Required

**AutoGluon expects EXACTLY 113 features in EXACTLY the same order:**

**Scenario 1: Missing Column**
```python
# Training: 113 features including "INDEPEND"
# Inference: 112 features (missing "INDEPEND")
# Result: ValueError: Missing required features
```

**Scenario 2: Extra Column**
```python
# Training: 113 features
# Inference: 114 features (added "patient_id")
# Result: AutoGluon ignores or raises error (depending on version)
```

**Scenario 3: Wrong Order**
```python
# Training: [NACCAGE, INDEPEND, REMDATES, ...]
# Inference: [INDEPEND, NACCAGE, REMDATES, ...]
# Result: Silent error - predictions incorrect
```

**Best Practice:**
```python
# Always reorder columns to match training schema
expected_features = predictor.feature_metadata_in.get_features()
new_data = new_data[expected_features]
```

---

### 2. Feature Type Consistency

**Data types must match exactly:**

| Feature | Training Type | Inference Type | Result |
|---------|--------------|---------------|--------|
| NACCAGE | `int64` | `float64` | ⚠️ May work but risky |
| SEX | `object` (category) | `int64` | ❌ Encoding error |
| INDEPEND | `Int64` (nullable int) | `float64` | ⚠️ Imputation issues |

**Solution:**
```python
# Enforce types
new_data["NACCAGE"] = new_data["NACCAGE"].astype("int64")
new_data["SEX"] = new_data["SEX"].astype("str")
```

---

### 3. Feature Value Ranges

**AutoGluon expects values within training distribution:**

**Example: NACCAGE (age)**
- Training Range: 40-105
- Inference Value: 150 (typo or error)
- Result: ⚠️ Model extrapolates (unreliable prediction)

**Best Practice:**
```python
# Validate value ranges
assert (new_data["NACCAGE"] >= 40).all() and (new_data["NACCAGE"] <= 105).all(), "Age out of range"
```

---

## How AutoGluon Stores Preprocessing Steps

### 1. Embedded Preprocessing Architecture

**Traditional ML Pipeline (Scikit-Learn):**
```python
# Preprocessing stored separately
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_test)

# Save separately
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "model.pkl")

# Deploy: Must load BOTH files
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
```

**AutoGluon Pipeline:**
```python
# Preprocessing embedded in predictor
predictor = TabularPredictor(label="target")
predictor.fit(train_data)  # Automatically fits preprocessing + models

# Save: Single directory contains everything
predictor.save("models/autogluon/")

# Deploy: Single load command
predictor = TabularPredictor.load("models/autogluon/")
# Preprocessing automatically applied during predict()
```

**Advantage:** No risk of forgetting to load preprocessing objects

**Disadvantage:** Cannot inspect or modify preprocessing easily

---

### 2. Where Preprocessing is Stored

**Location:** `models/autogluon_production_lowmem/utils/feature_transformations.pkl`

**Contents:**
```python
{
  "feature_metadata": {
    "feature_names": ["NACCAGE", "INDEPEND", ...],
    "feature_types": {"NACCAGE": "int", "SEX": "category", ...}
  },
  "transformations": {
    "NACCAGE": StandardScaler(mean=72.5, std=10.2),
    "INDEPEND": RobustScaler(median=0, IQR=1.5),
    "SEX": LabelEncoder(classes=["Male", "Female"])
  },
  "imputers": {
    "INDEPEND": SimpleImputer(strategy="median", fill_value=0)
  }
}
```

**How AutoGluon Uses It:**
```python
# Internally during predictor.predict(new_data):
for feature, transformer in feature_transformations.items():
    new_data[feature] = transformer.transform(new_data[feature])
```

---

### 3. Inspecting Preprocessing

**You CAN inspect preprocessing (but not modify easily):**

```python
# Load predictor
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

# Inspect feature metadata
features = predictor.feature_metadata_in.get_features()
print(f"Required features: {features}")

# Inspect feature types
types = predictor.feature_metadata_in.get_type_map_raw()
print(f"Feature types: {types}")

# Note: Cannot easily access individual scalers/encoders
# They are embedded in learner internal state
```

---

## Why Using Only `model.pkl` is Insufficient

### Case Study: What Happens If You Copy Only `learner.pkl`

**Scenario:**
```bash
# User copies only the main file
cp models/autogluon_production_lowmem/learner.pkl /new/project/
```

**Attempt to Load:**
```python
from autogluon.tabular import TabularPredictor

# This will FAIL
predictor = TabularPredictor.load("/new/project/learner.pkl")
```

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/new/project/models/LightGBM_r188_BAG_L2/model.pkl'
```

**Why:**
- `learner.pkl` contains references to 36 individual models
- Those models are stored in `models/` subdirectories
- Without `models/`, AutoGluon cannot load the full ensemble

---

### What's Actually Needed

**Minimum Required Files:**
```
models/autogluon_production_lowmem/
├── learner.pkl                  # Main predictor
├── metadata.json                # Version info
├── version.txt                  # AutoGluon version
├── models/                      # Individual models (36 subdirectories)
│   ├── WeightedEnsemble_L3/
│   ├── LightGBM_r188_BAG_L2/
│   └── ... (34 more)
└── utils/                       # Preprocessing transformations
    ├── feature_metadata.pkl
    └── feature_transformations.pkl
```

**Copy Command:**
```bash
# Correct: Copy entire directory
cp -r models/autogluon_production_lowmem /new/project/models/

# Verify
ls /new/project/models/autogluon_production_lowmem/models/  # Should show 36 subdirectories
```

---

## The Importance of Copying the Entire Predictor Directory

### 1. Directory Structure is Sacred

**AutoGluon relies on specific directory structure:**

```
autogluon_production_lowmem/
├── learner.pkl                  # References models/ and utils/
├── models/
│   ├── WeightedEnsemble_L3/     # Final ensemble
│   ├── LightGBM_r188_BAG_L2/    # Best base model
│   └── ...
└── utils/
    ├── feature_metadata.pkl     # Schema validation
    └── feature_transformations.pkl  # Preprocessing
```

**Internal References (learner.pkl):**
```python
# learner.pkl contains paths like:
self.models_path = "models/"
self.model_files = {
    "WeightedEnsemble_L3": "models/WeightedEnsemble_L3/model.pkl",
    "LightGBM_r188_BAG_L2": "models/LightGBM_r188_BAG_L2/model.pkl",
    ...
}
```

**If you move files around:**
- AutoGluon cannot find models → Load failure
- Preprocessing cannot be applied → Prediction errors

---

### 2. Atomic Copy Requirement

**Treat the directory as atomic (indivisible):**

```bash
# ✅ CORRECT: Copy entire directory as-is
cp -r models/autogluon_production_lowmem /new/project/models/

# ❌ WRONG: Copy files individually
cp models/autogluon_production_lowmem/learner.pkl /new/project/
cp models/autogluon_production_lowmem/models/*.pkl /new/project/
# This breaks internal references

# ❌ WRONG: Reorganize directory structure
cp -r models/autogluon_production_lowmem /new/project/my_custom_model_folder/
# AutoGluon expects specific structure
```

---

### 3. Verification After Copy

**Always verify integrity:**

```bash
# Check directory structure
ls /new/project/models/autogluon_production_lowmem/

# Expected output:
# learner.pkl  metadata.json  version.txt  models/  utils/

# Check model count
ls /new/project/models/autogluon_production_lowmem/models/ | wc -l
# Expected: 36

# Test loading
python -c "from autogluon.tabular import TabularPredictor; predictor = TabularPredictor.load('/new/project/models/autogluon_production_lowmem/'); print('✅ Loaded successfully')"
```

---

## Summary: Golden Rules for AutoML Reuse

### Rule 1: Copy Everything
✅ Always copy the entire model directory
❌ Never copy individual files

### Rule 2: Preserve Structure
✅ Maintain exact directory structure
❌ Never reorganize or rename subdirectories

### Rule 3: Match Versions
✅ Use same AutoGluon version (1.4.0)
✅ Use same Python version (3.11)
❌ Never mix versions

### Rule 4: Validate Schema
✅ Ensure new data has exact same 113 features
✅ Verify feature types and order
❌ Never assume schema flexibility

### Rule 5: Test Before Production
✅ Load model in test environment
✅ Run predictions on sample data
✅ Verify outputs match expected results
❌ Never deploy without testing

### Rule 6: Document Dependencies
✅ Lock all dependency versions in requirements.txt
✅ Document system requirements (RAM, storage)
❌ Never use flexible version ranges (e.g., `>=1.4.0`)

### Rule 7: Monitor Performance
✅ Track inference time, memory usage
✅ Log predictions for audit trails
✅ Monitor for model drift
❌ Never assume model works forever without monitoring

---

## Reuse Checklist

**Before deploying AutoML model in new project:**

```
✅ Copied entire models/autogluon_production_lowmem/ directory
✅ Verified all subdirectories present (models/, utils/)
✅ Installed AutoGluon 1.4.0 in Python 3.11 environment
✅ Tested model loading: TabularPredictor.load()
✅ Validated feature schema (113 features, correct types)
✅ Ran test predictions on sample data
✅ Verified predictions match expected format
✅ Set up logging and monitoring
✅ Documented deployment process
✅ Created rollback plan if issues arise
```

---

## Common Mistakes to Avoid

| Mistake | Impact | Solution |
|---------|--------|----------|
| Copying only `learner.pkl` | Load failure | Copy entire directory |
| Using AutoGluon 1.3.0 instead of 1.4.0 | Silent errors | Lock version to 1.4.0 |
| Adding extra columns to new data | Schema mismatch | Drop extra columns before prediction |
| Assuming missing values handled externally | NaN errors | Let AutoGluon handle imputation |
| Reorganizing directory structure | Load failure | Preserve exact structure |
| Not testing in staging environment | Production failures | Always test before deploy |

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Key Takeaway:** AutoML models are complex systems - treat them as immutable, atomic units for deployment.
