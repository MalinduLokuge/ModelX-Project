# Model Deployment Guide
## Reusing AutoGluon Dementia Risk Prediction Model in New Projects

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Loading the Model](#loading-the-model)
4. [Making Predictions](#making-predictions)
5. [Input Data Requirements](#input-data-requirements)
6. [Understanding Predictions](#understanding-predictions)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance-optimization)
9. [Reproducibility](#reproducibility)
10. [Common Deployment Scenarios](#common-deployment-scenarios)
11. [Troubleshooting Guide](#troubleshooting-guide)

---

## 1. Quick Start

### 1.1 Minimal Working Example

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load the trained model
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

# Prepare new data (must have 113 features matching training schema)
new_data = pd.read_csv("new_patient_data.csv")

# Make predictions
predictions = predictor.predict(new_data)
print(predictions)  # Output: ['No Dementia', 'Dementia', 'No Dementia', ...]

# Get prediction probabilities
probabilities = predictor.predict_proba(new_data)
print(probabilities)
#      No Dementia  Dementia
# 0         0.92      0.08
# 1         0.23      0.77
# 2         0.95      0.05
```

### 1.2 Expected Output

```
Input: 3 patients with 113 features each
Output:
  Patient 1: No Dementia (92% confidence)
  Patient 2: Dementia (77% confidence)
  Patient 3: No Dementia (95% confidence)
```

---

## 2. Environment Setup

### 2.1 Installation Steps

#### **Option A: Using pip (Recommended)**

```bash
# Create virtual environment
python -m venv autogluon_env
source autogluon_env/bin/activate  # On Linux/Mac
# OR
autogluon_env\Scripts\activate     # On Windows

# Install AutoGluon
pip install autogluon.tabular==1.4.0

# Install additional dependencies
pip install numpy pandas scikit-learn
```

#### **Option B: Using Conda**

```bash
# Create environment
conda create -n autogluon_env python=3.11
conda activate autogluon_env

# Install AutoGluon
pip install autogluon.tabular==1.4.0

# Install additional dependencies
conda install numpy pandas scikit-learn
```

#### **Option C: Using Docker**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install AutoGluon
RUN pip install --no-cache-dir autogluon.tabular==1.4.0

# Copy model artifacts
COPY models/autogluon_production_lowmem /app/models/autogluon_production_lowmem

# Copy inference script
COPY inference.py /app/

CMD ["python", "inference.py"]
```

**Build and run:**
```bash
docker build -t autogluon-model .
docker run -v $(pwd)/data:/app/data autogluon-model
```

### 2.2 Verify Installation

```python
import autogluon
from autogluon.tabular import TabularPredictor

print(f"AutoGluon Version: {autogluon.__version__}")
# Expected Output: AutoGluon Version: 1.4.0

# Test model loading
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")
print(f"Model Loaded: {predictor.class_labels}")
# Expected Output: Model Loaded: ['No Dementia', 'Dementia']
```

### 2.3 System Requirements

**Minimum:**
- Python: 3.9, 3.10, or 3.11
- RAM: 8 GB
- Storage: 2 GB free space
- OS: Windows, Linux, macOS

**Recommended:**
- Python: 3.11
- RAM: 16 GB
- Storage: 5 GB free space
- CPU: 4+ cores

---

## 3. Loading the Model

### 3.1 Basic Loading

```python
from autogluon.tabular import TabularPredictor

# Load predictor from saved directory
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

# Check model info
print(f"Problem Type: {predictor.problem_type}")        # binary
print(f"Eval Metric: {predictor.eval_metric}")          # roc_auc
print(f"Number of Models: {len(predictor.model_names())}")  # 36
print(f"Best Model: {predictor.model_best}")            # WeightedEnsemble_L3
```

### 3.2 Loading from Different Locations

```python
# Absolute path
predictor = TabularPredictor.load("C:/models/autogluon_production_lowmem/")

# Relative path
predictor = TabularPredictor.load("../models/autogluon_production_lowmem/")

# From cloud storage (after downloading)
import os
predictor = TabularPredictor.load(os.path.join(os.getcwd(), "models", "autogluon_production_lowmem"))
```

### 3.3 Validating Model After Loading

```python
# Check that model loaded correctly
assert predictor is not None, "Model failed to load"
assert predictor.problem_type == "binary", "Unexpected problem type"
assert len(predictor.model_names()) == 36, "Missing models"

# Inspect leaderboard
leaderboard = predictor.leaderboard(silent=True)
print(leaderboard.head(5))
#                     model  score_val  score_test  pred_time_val  fit_time
# 0  WeightedEnsemble_L3     0.9813      0.9709        12.75       1795.2
# 1  LightGBM_r188_BAG_L2    0.9813      0.9708         1.23         45.2
# ...
```

### 3.4 Loading Performance

**Expected Loading Time:**
- Cold Start (first load): 10-15 seconds
- Warm Start (cached): 2-5 seconds

**Memory Usage:**
- Initial Load: ~500 MB
- Steady State: ~600 MB

---

## 4. Making Predictions

### 4.1 Predict Class Labels

```python
import pandas as pd

# Load new patient data
new_data = pd.read_csv("new_patients.csv")

# Predict class labels (binary: "No Dementia" or "Dementia")
predictions = predictor.predict(new_data)

# Output
print(predictions)
# 0    No Dementia
# 1    Dementia
# 2    No Dementia
# dtype: object
```

### 4.2 Predict Probabilities

```python
# Get probability estimates for each class
probabilities = predictor.predict_proba(new_data)

# Output: DataFrame with columns ["No Dementia", "Dementia"]
print(probabilities)
#      No Dementia  Dementia
# 0         0.92      0.08      ← 92% confident NO dementia
# 1         0.23      0.77      ← 77% confident YES dementia
# 2         0.95      0.05      ← 95% confident NO dementia

# Extract dementia risk probability (column "Dementia")
dementia_risk = probabilities["Dementia"]
print(dementia_risk)
# 0    0.08
# 1    0.77
# 2    0.05
# dtype: float64
```

### 4.3 Combine Predictions with Input Data

```python
# Add predictions and probabilities to original data
new_data["predicted_class"] = predictions
new_data["dementia_risk_probability"] = probabilities["Dementia"]

# Save results
new_data.to_csv("predictions_output.csv", index=False)

# Example output:
#   NACCAGE  INDEPEND  ...  predicted_class  dementia_risk_probability
# 0    72        0     ...  No Dementia                0.08
# 1    85        2     ...  Dementia                   0.77
# 2    65        0     ...  No Dementia                0.05
```

### 4.4 Batch Prediction

```python
# Predict in batches for large datasets
chunk_size = 1000
predictions_list = []

for chunk in pd.read_csv("large_dataset.csv", chunksize=chunk_size):
    preds = predictor.predict(chunk)
    predictions_list.append(preds)

# Combine all predictions
all_predictions = pd.concat(predictions_list)
```

### 4.5 Single Patient Prediction

```python
# Create single patient record (must be DataFrame, not Series)
single_patient = pd.DataFrame([{
    "NACCAGE": 72,
    "INDEPEND": 0,
    "REMDATES": 0,
    "TAXES": 1,
    "BILLS": 0,
    # ... (all 113 features required)
}])

# Predict
prediction = predictor.predict(single_patient)
probability = predictor.predict_proba(single_patient)

print(f"Prediction: {prediction[0]}")
print(f"Dementia Risk: {probability['Dementia'][0]:.2%}")
# Output:
#   Prediction: No Dementia
#   Dementia Risk: 8.23%
```

---

## 5. Input Data Requirements

### 5.1 Required Schema

**Critical Requirements:**
1. **Exactly 113 features** (columns) must be present
2. **Feature names must match training data exactly** (case-sensitive)
3. **Feature types must match** (int, float, category)
4. **No extra columns allowed** (remove before prediction)

### 5.2 Feature Name Validation

```python
# Load expected feature names from model
expected_features = predictor.feature_metadata_in.get_features()

# Validate new data
def validate_features(new_data):
    """Ensure new_data has correct features."""
    missing = set(expected_features) - set(new_data.columns)
    extra = set(new_data.columns) - set(expected_features)

    if missing:
        raise ValueError(f"Missing required features: {missing}")
    if extra:
        raise ValueError(f"Unexpected features (remove these): {extra}")

    # Reorder columns to match training order
    new_data = new_data[expected_features]
    return new_data

# Apply validation
new_data = validate_features(new_data)
```

### 5.3 Handling Missing Values

**AutoGluon automatically handles missing values during inference:**

```python
# Example: Some features are missing (NaN)
new_data = pd.DataFrame([{
    "NACCAGE": 72,
    "INDEPEND": None,      # Missing value
    "REMDATES": 0,
    "TAXES": None,         # Missing value
    # ... (rest of features)
}])

# AutoGluon will impute missing values using stored imputers
predictions = predictor.predict(new_data)  # Works correctly
```

**Manual Missing Value Handling (Optional):**
```python
# Fill missing values before prediction
new_data["INDEPEND"].fillna(0, inplace=True)
new_data["TAXES"].fillna(new_data["TAXES"].median(), inplace=True)
```

### 5.4 Feature Type Conversion

```python
# Ensure correct data types
def fix_data_types(new_data):
    """Convert features to expected types."""

    # Integer features
    int_features = ["NACCAGE", "INDEPEND", "REMDATES", "TAXES"]
    for col in int_features:
        new_data[col] = new_data[col].astype("Int64")  # Nullable int

    # Float features
    float_features = ["BMI", "WEIGHT"]
    for col in float_features:
        new_data[col] = new_data[col].astype("float64")

    # Categorical features
    cat_features = ["SEX", "NACCEDUC", "MARISTAT"]
    for col in cat_features:
        new_data[col] = new_data[col].astype("str")

    return new_data

# Apply type conversion
new_data = fix_data_types(new_data)
```

### 5.5 Sample Input Template

**Download Template:**
```python
# Generate template with correct schema
template = pd.DataFrame(columns=predictor.feature_metadata_in.get_features())
template.to_csv("input_template.csv", index=False)
```

**Example Template (First 10 Features):**
```csv
NACCAGE,NACCAGEB,INDEPEND,REMDATES,TAXES,TRAVEL,BILLS,PAYATTN,EVENTS,GAMES,...
72,2,0,0,1,0,0,0,0,1,...
85,3,2,2,3,1,1,1,2,0,...
65,2,0,0,0,0,0,0,0,0,...
```

---

## 6. Understanding Predictions

### 6.1 Interpreting Class Predictions

```python
predictions = predictor.predict(new_data)

# Output: "No Dementia" or "Dementia"
for i, pred in enumerate(predictions):
    if pred == "Dementia":
        print(f"Patient {i}: AT RISK of dementia")
    else:
        print(f"Patient {i}: NOT at risk of dementia")
```

### 6.2 Interpreting Probability Scores

```python
probabilities = predictor.predict_proba(new_data)

# Example output:
#      No Dementia  Dementia
# 0         0.92      0.08

# Interpretation:
# - Patient 0 has 8% probability of dementia (LOW RISK)
# - Patient 0 has 92% probability of NO dementia (HIGH CONFIDENCE)

# Risk categories based on probability
def categorize_risk(dementia_prob):
    if dementia_prob < 0.10:
        return "Very Low Risk"
    elif dementia_prob < 0.30:
        return "Low Risk"
    elif dementia_prob < 0.50:
        return "Moderate Risk"
    elif dementia_prob < 0.70:
        return "High Risk"
    else:
        return "Very High Risk"

for i, prob in enumerate(probabilities["Dementia"]):
    risk_level = categorize_risk(prob)
    print(f"Patient {i}: {prob:.2%} dementia risk ({risk_level})")

# Output:
# Patient 0: 8.00% dementia risk (Very Low Risk)
# Patient 1: 77.00% dementia risk (Very High Risk)
# Patient 2: 5.00% dementia risk (Very Low Risk)
```

### 6.3 Adjusting Prediction Threshold

**Default threshold: 0.5** (predict "Dementia" if probability > 0.5)

```python
# Custom threshold for higher sensitivity
threshold = 0.3  # Flag anyone with >30% risk

probabilities = predictor.predict_proba(new_data)
custom_predictions = (probabilities["Dementia"] > threshold).map({True: "Dementia", False: "No Dementia"})

print(custom_predictions)
# 0    No Dementia   (8% < 30%)
# 1    Dementia      (77% > 30%)
# 2    No Dementia   (5% < 30%)
```

**Use Cases for Custom Thresholds:**
- **Threshold = 0.3**: Early screening (catch more cases, accept more false alarms)
- **Threshold = 0.5**: Balanced (default)
- **Threshold = 0.7**: Conservative (only flag high-confidence cases)

### 6.4 Feature Importance for Individual Predictions

```python
# Explain why a specific patient was classified as high-risk
import shap

# Get SHAP explainer (requires additional setup)
# Note: SHAP explanations may take time for large models
explainer = shap.TreeExplainer(predictor.model_best)
shap_values = explainer.shap_values(new_data.iloc[[0]])  # Patient 0

# Plot explanation
shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                      base_values=explainer.expected_value,
                                      data=new_data.iloc[0]))
```

---

## 7. Error Handling

### 7.1 Common Errors and Solutions

#### **Error 1: Missing Features**
```python
# Error Message:
# ValueError: Missing required features: ['INDEPEND', 'REMDATES']

# Solution: Add missing columns
new_data["INDEPEND"] = 0  # Default value
new_data["REMDATES"] = 0
```

#### **Error 2: Extra Features**
```python
# Error Message:
# ValueError: Unexpected features (remove these): {'patient_id', 'timestamp'}

# Solution: Drop extra columns
new_data = new_data.drop(columns=["patient_id", "timestamp"])
```

#### **Error 3: Type Mismatch**
```python
# Error Message:
# TypeError: Cannot convert column 'NACCAGE' to int

# Solution: Fix data types
new_data["NACCAGE"] = pd.to_numeric(new_data["NACCAGE"], errors="coerce")
```

#### **Error 4: Model Not Found**
```python
# Error Message:
# FileNotFoundError: [Errno 2] No such file or directory: 'models/...'

# Solution: Check path
import os
model_path = "models/autogluon_production_lowmem/"
assert os.path.exists(model_path), f"Model not found at {model_path}"
```

### 7.2 Robust Prediction Wrapper

```python
def safe_predict(predictor, new_data):
    """Predict with comprehensive error handling."""
    try:
        # Validate schema
        expected_features = predictor.feature_metadata_in.get_features()
        missing = set(expected_features) - set(new_data.columns)
        extra = set(new_data.columns) - set(expected_features)

        if missing:
            raise ValueError(f"Missing features: {missing}")
        if extra:
            new_data = new_data.drop(columns=extra)  # Auto-remove extra columns

        # Reorder columns
        new_data = new_data[expected_features]

        # Make predictions
        predictions = predictor.predict(new_data)
        probabilities = predictor.predict_proba(new_data)

        return predictions, probabilities

    except Exception as e:
        print(f"Prediction failed: {e}")
        return None, None

# Usage
predictions, probabilities = safe_predict(predictor, new_data)
if predictions is not None:
    print("Predictions successful!")
```

### 7.3 Logging Errors

```python
import logging

logging.basicConfig(level=logging.INFO, filename="predictions.log")

try:
    predictions = predictor.predict(new_data)
    logging.info(f"Predicted {len(predictions)} samples successfully")
except Exception as e:
    logging.error(f"Prediction failed: {e}")
    raise
```

---

## 8. Performance Optimization

### 8.1 Inference Speed

**Expected Inference Times:**
| Dataset Size | Inference Time | Throughput |
|-------------|---------------|-----------|
| 1 sample | 0.5 seconds | 2 samples/sec |
| 100 samples | 2 seconds | 50 samples/sec |
| 1,000 samples | 12 seconds | 83 samples/sec |
| 10,000 samples | 90 seconds | 111 samples/sec |

### 8.2 Using Lightweight Models for Faster Inference

```python
# Load only the best single model (LightGBM) instead of full ensemble
# This is 10x faster but -0.01 ROC-AUC performance

# Note: Requires retraining with persist_models=True
# predictor_fast = TabularPredictor.load("models/autogluon_production_lowmem/")
# predictor_fast.set_model_best("LightGBM_r188_BAG_L2")

# For now, use full ensemble (most accurate)
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")
```

### 8.3 Batch Prediction for Large Datasets

```python
# Process large datasets in chunks
def batch_predict(predictor, data_path, chunk_size=1000, output_path="predictions.csv"):
    """Predict in batches to manage memory."""

    chunks = pd.read_csv(data_path, chunksize=chunk_size)
    first_chunk = True

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}...")

        # Predict
        predictions = predictor.predict(chunk)
        probabilities = predictor.predict_proba(chunk)

        # Combine results
        chunk["predicted_class"] = predictions
        chunk["dementia_risk"] = probabilities["Dementia"]

        # Write to CSV (append mode)
        chunk.to_csv(output_path, mode="a", header=first_chunk, index=False)
        first_chunk = False

    print(f"Predictions saved to {output_path}")

# Usage
batch_predict(predictor, "large_dataset.csv", chunk_size=1000)
```

### 8.4 Parallelization

```python
# AutoGluon uses multi-threading automatically
# Set number of CPU cores to use
import os
os.environ["OMP_NUM_THREADS"] = "4"  # Use 4 cores

predictor = TabularPredictor.load("models/autogluon_production_lowmem/")
predictions = predictor.predict(new_data)  # Automatically parallelized
```

---

## 9. Reproducibility

### 9.1 Ensuring Consistent Predictions

**AutoGluon guarantees deterministic predictions if:**
1. Same model version (v1.4.0)
2. Same Python version (3.11)
3. Same input data

```python
# Set random seed (if using models with randomness)
import numpy as np
np.random.seed(42)

# Load model
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

# Predict
predictions_run1 = predictor.predict(new_data)
predictions_run2 = predictor.predict(new_data)

# Verify consistency
assert (predictions_run1 == predictions_run2).all(), "Predictions are not reproducible!"
```

### 9.2 Version Checking

```python
import autogluon

# Check AutoGluon version
assert autogluon.__version__ == "1.4.0", f"Version mismatch: {autogluon.__version__}"

# Check model metadata
import json
with open("models/autogluon_production_lowmem/metadata.json") as f:
    metadata = json.load(f)

assert metadata["autogluon_version"] == "1.4.0", "Model trained with different version"
print(f"Model and environment versions match: {metadata['autogluon_version']}")
```

### 9.3 Logging Predictions for Auditability

```python
import datetime

# Create audit log
def log_prediction(patient_id, prediction, probability):
    """Log predictions for audit trail."""
    with open("prediction_audit.log", "a") as f:
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"{timestamp},{patient_id},{prediction},{probability:.4f}\n")

# Usage
for i, (pred, prob) in enumerate(zip(predictions, probabilities["Dementia"])):
    log_prediction(patient_id=i, prediction=pred, probability=prob)
```

---

## 10. Common Deployment Scenarios

### 10.1 Scenario 1: Web Application (Flask API)

```python
from flask import Flask, request, jsonify
from autogluon.tabular import TabularPredictor
import pandas as pd

app = Flask(__name__)

# Load model on startup
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for predictions."""
    try:
        # Get JSON input
        data = request.get_json()
        new_data = pd.DataFrame([data])

        # Predict
        prediction = predictor.predict(new_data)[0]
        probability = predictor.predict_proba(new_data)["Dementia"][0]

        # Return result
        return jsonify({
            "prediction": prediction,
            "dementia_risk_probability": float(probability),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

**Test API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"NACCAGE": 72, "INDEPEND": 0, "REMDATES": 0, ...}'
```

### 10.2 Scenario 2: Batch Processing Script

```python
#!/usr/bin/env python3
"""
Batch processing script for dementia risk prediction.
Usage: python batch_predict.py input.csv output.csv
"""

import sys
import pandas as pd
from autogluon.tabular import TabularPredictor

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_predict.py input.csv output.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load model
    print("Loading model...")
    predictor = TabularPredictor.load("models/autogluon_production_lowmem/")

    # Load data
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file)

    # Predict
    print(f"Predicting for {len(data)} samples...")
    predictions = predictor.predict(data)
    probabilities = predictor.predict_proba(data)

    # Save results
    data["predicted_class"] = predictions
    data["dementia_risk"] = probabilities["Dementia"]
    data.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
```

### 10.3 Scenario 3: Cloud Deployment (AWS Lambda)

```python
import json
import pandas as pd
from autogluon.tabular import TabularPredictor

# Load model once (cold start)
predictor = TabularPredictor.load("/tmp/autogluon_model/")

def lambda_handler(event, context):
    """AWS Lambda handler for predictions."""
    try:
        # Parse input
        data = json.loads(event["body"])
        new_data = pd.DataFrame([data])

        # Predict
        prediction = predictor.predict(new_data)[0]
        probability = float(predictor.predict_proba(new_data)["Dementia"][0])

        # Return response
        return {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": prediction,
                "dementia_risk": probability
            })
        }

    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
```

### 10.4 Scenario 4: Jupyter Notebook Analysis

```python
# Load model
from autogluon.tabular import TabularPredictor
import pandas as pd
import matplotlib.pyplot as plt

predictor = TabularPredictor.load("../models/autogluon_production_lowmem/")

# Load new data
new_data = pd.read_csv("new_patients.csv")

# Predict
predictions = predictor.predict(new_data)
probabilities = predictor.predict_proba(new_data)

# Visualize risk distribution
plt.figure(figsize=(10, 6))
plt.hist(probabilities["Dementia"], bins=50, edgecolor="black")
plt.xlabel("Dementia Risk Probability")
plt.ylabel("Number of Patients")
plt.title("Distribution of Dementia Risk Scores")
plt.savefig("risk_distribution.png")
plt.show()

# Summary statistics
print(f"Mean Risk: {probabilities['Dementia'].mean():.2%}")
print(f"Median Risk: {probabilities['Dementia'].median():.2%}")
print(f"High Risk (>70%): {(probabilities['Dementia'] > 0.7).sum()} patients")
```

---

## 11. Troubleshooting Guide

### 11.1 Model Fails to Load

**Symptoms:**
```
FileNotFoundError: No such file or directory: 'models/autogluon_production_lowmem/learner.pkl'
```

**Solutions:**
1. Check that entire `models/autogluon_production_lowmem/` directory was copied
2. Verify file permissions (read access required)
3. Check path separators (`/` on Linux/Mac, `\` on Windows)

```python
import os
model_path = "models/autogluon_production_lowmem/"
assert os.path.exists(model_path), f"Directory not found: {model_path}"
assert os.path.exists(os.path.join(model_path, "learner.pkl")), "learner.pkl missing"
```

### 11.2 Version Mismatch Warnings

**Symptoms:**
```
UserWarning: Model trained with AutoGluon 1.4.0, but current version is 1.3.0
```

**Solutions:**
1. Install matching version: `pip install autogluon.tabular==1.4.0`
2. Upgrade AutoGluon: `pip install --upgrade autogluon.tabular`

### 11.3 Feature Schema Errors

**Symptoms:**
```
ValueError: Missing required features: ['INDEPEND']
```

**Solutions:**
```python
# Check expected features
expected = predictor.feature_metadata_in.get_features()
actual = list(new_data.columns)

missing = set(expected) - set(actual)
print(f"Missing features: {missing}")

# Add missing features with default values
for feature in missing:
    new_data[feature] = 0  # Or appropriate default
```

### 11.4 Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate X GB for array
```

**Solutions:**
1. Reduce batch size:
```python
# Instead of predicting all at once
predictions = predictor.predict(large_data)  # May crash

# Predict in batches
batch_size = 100
for i in range(0, len(large_data), batch_size):
    batch = large_data.iloc[i:i+batch_size]
    predictions = predictor.predict(batch)
```

2. Increase system RAM or use cloud instance with more memory

### 11.5 Slow Inference

**Symptoms:**
- Prediction takes >60 seconds for 1,000 samples

**Solutions:**
1. Use lightweight model (single LightGBM instead of ensemble)
2. Enable multi-threading:
```python
import os
os.environ["OMP_NUM_THREADS"] = "8"  # Use 8 CPU cores
```
3. Use GPU acceleration (if neural network models included):
```python
# Requires GPU-enabled AutoGluon installation
predictor = TabularPredictor.load("...", use_gpu=True)
```

### 11.6 Getting Help

**Resources:**
- AutoGluon Documentation: https://auto.gluon.ai/stable/index.html
- AutoGluon GitHub Issues: https://github.com/autogluon/autogluon/issues
- Model Documentation: See `NODE_DOCUMENT.md` and `PROJECT_TECHNICAL_DOCUMENT.md`

**Reporting Issues:**
Include:
1. AutoGluon version (`autogluon.__version__`)
2. Python version (`python --version`)
3. Operating system
4. Error message and stack trace
5. Minimal reproducible example

---

## 12. Summary

### 12.1 Deployment Checklist

```
✅ Install AutoGluon 1.4.0 in Python 3.9-3.11
✅ Copy entire models/autogluon_production_lowmem/ directory
✅ Verify model loads: predictor = TabularPredictor.load("...")
✅ Validate input schema: 113 features, correct names and types
✅ Test prediction: predictor.predict(sample_data)
✅ Handle errors: missing features, type mismatches
✅ Monitor performance: inference time, memory usage
✅ Document predictions: save outputs with timestamps
```

### 12.2 Key Takeaways

1. **Loading**: Use `TabularPredictor.load("path/to/model/")`
2. **Prediction**: `predictor.predict(new_data)` for classes, `predictor.predict_proba()` for probabilities
3. **Schema**: Must have exactly 113 features matching training data
4. **Error Handling**: Validate schema before prediction, handle missing values
5. **Performance**: 10-15 seconds for 1,000 samples, batch processing recommended for large datasets

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Model Version:** AutoGluon v1.4.0 (Production Low-Memory Configuration)
**Contact:** ModelX Development Team
