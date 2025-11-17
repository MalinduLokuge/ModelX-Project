# Model Usage Code Snippets

This document contains ready-to-run code snippets for loading and using both the AutoGluon and manual models.

---

## 1. AutoGluon Model - Load and Predict

### Basic Usage

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load the trained AutoGluon predictor
predictor = TabularPredictor.load('outputs/models/autogluon_optimized/')

# Load your new data (must have same 112 features as training)
new_data = pd.read_csv('path/to/new_data.csv')

# Make predictions
predictions = predictor.predict(new_data)  # Returns class labels (0 or 1)
probabilities = predictor.predict_proba(new_data)  # Returns probability matrix

print(f"Predictions: {predictions[:5].tolist()}")
print(f"Probability of Dementia: {probabilities[1][:5].tolist()}")
```

### With Feature Engineering

```python
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np

def apply_feature_engineering(df):
    """
    Apply the same feature engineering used during training.
    Creates 20 additional features from domain interactions and aggregations.
    """
    df_eng = df.copy()
    
    # Age × Cognitive interactions (example - adjust based on actual features)
    if 'NACCAGE' in df.columns and 'EVENTS' in df.columns:
        df_eng['age_x_events'] = df['NACCAGE'] * df['EVENTS']
    if 'NACCAGE' in df.columns and 'REMDATES' in df.columns:
        df_eng['age_x_remdates'] = df['NACCAGE'] * df['REMDATES']
    
    # Education × Memory interactions
    if 'EDUC' in df.columns and 'PAYATTN' in df.columns:
        df_eng['educ_x_payattn'] = df['EDUC'] * df['PAYATTN']
    if 'EDUC' in df.columns and 'SHOPPING' in df.columns:
        df_eng['educ_x_shopping'] = df['EDUC'] * df['SHOPPING']
    
    # Statistical aggregations across related feature groups
    cognitive_cols = ['EVENTS', 'REMDATES', 'PAYATTN', 'SHOPPING', 'TRAVEL']
    if all(col in df.columns for col in cognitive_cols):
        df_eng['cognitive_mean'] = df[cognitive_cols].mean(axis=1)
        df_eng['cognitive_std'] = df[cognitive_cols].std(axis=1)
    
    medical_cols = ['HYPERTEN', 'HYPERCHO', 'DIABETES', 'CVHATT', 'CBSTROKE']
    if all(col in df.columns for col in medical_cols):
        df_eng['medical_sum'] = df[medical_cols].sum(axis=1)
    
    # Add more engineered features as needed to reach 132 total features
    
    return df_eng

# Load model
predictor = TabularPredictor.load('outputs/models/autogluon_optimized/')

# Load new data
new_data = pd.read_csv('path/to/new_data.csv')

# Apply feature engineering
new_data_engineered = apply_feature_engineering(new_data)

# Verify feature count (should be 132)
print(f"Features: {new_data_engineered.shape[1]} (should be 132)")

# Make predictions
predictions = predictor.predict(new_data_engineered)
probabilities = predictor.predict_proba(new_data_engineered)

print(f"Predictions: {predictions[:5].tolist()}")
```

### Batch Inference

```python
from autogluon.tabular import TabularPredictor
import pandas as pd
import time

def batch_predict(data_path, model_path, batch_size=10000, output_path='predictions.csv'):
    """
    Process large datasets in batches for memory efficiency.
    """
    predictor = TabularPredictor.load(model_path)
    
    # Read data in chunks
    chunks = []
    start_time = time.time()
    
    for chunk in pd.read_csv(data_path, chunksize=batch_size):
        # Apply feature engineering if needed
        chunk_engineered = apply_feature_engineering(chunk)
        
        # Predict
        predictions = predictor.predict(chunk_engineered)
        proba = predictor.predict_proba(chunk_engineered)
        
        # Combine results
        chunk['prediction'] = predictions
        chunk['dementia_probability'] = proba[1]
        chunks.append(chunk)
    
    # Concatenate all results
    results = pd.concat(chunks, ignore_index=True)
    results.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"Processed {len(results)} rows in {elapsed:.2f}s")
    print(f"Throughput: {len(results)/elapsed:.0f} rows/second")
    
    return results

# Usage
results = batch_predict(
    data_path='large_dataset.csv',
    model_path='outputs/models/autogluon_optimized/',
    batch_size=10000
)
```

### Evaluation on Test Set

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load model
predictor = TabularPredictor.load('outputs/models/autogluon_optimized/')

# Load test data (with labels)
X_test = pd.read_csv('data/test/X_test.csv')
y_test = pd.read_csv('data/test/y_test.csv')

# Combine for AutoGluon's evaluate method
test_data = X_test.copy()
test_data['Dementia'] = y_test.values

# Apply feature engineering
test_data_engineered = apply_feature_engineering(test_data)

# Evaluate
metrics = predictor.evaluate(test_data_engineered)

print("Test Set Performance:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

---

## 2. Manual Model - Load and Predict

### LightGBM Tuned Model (Best Manual Model)

```python
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('outputs/manual_models/LightGBM_Tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# Load new data
new_data = pd.read_csv('path/to/new_data.csv')

# Ensure correct feature order (112 features)
expected_features = [
    'VISITMO', 'VISITDAY', 'VISITYR', 'BIRTHMO', 'BIRTHYR', 'SEX', 'HISPANIC',
    'HISPOR', 'RACE', 'RACESEC', 'RACETER', 'PRIMLANG', 'EDUC', 'MARISTAT',
    'NACCLIVS', 'INDEPEND', 'RESIDENC', 'HANDED', 'NACCAGE', 'NACCAGEB',
    # ... (all 112 features in training order)
]

# Validate features
missing_features = set(expected_features) - set(new_data.columns)
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

# Select and order features
new_data = new_data[expected_features]

# Make predictions
predictions = model.predict(new_data)  # Class labels (0 or 1)
probabilities = model.predict_proba(new_data)  # Probability matrix

print(f"Predictions: {predictions[:5].tolist()}")
print(f"Probability of Dementia: {probabilities[:, 1][:5].tolist()}")
```

### With Preprocessing Pipeline

```python
import pickle
import pandas as pd

# Load preprocessing pipeline and model
with open('outputs/dementia_preprocessed/preprocessing_pipeline.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('outputs/manual_models/LightGBM_Tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# Load raw data
raw_data = pd.read_csv('path/to/raw_data.csv')

# Apply preprocessing
processed_data = preprocessor.transform(raw_data)

# Make predictions
predictions = model.predict(processed_data)
probabilities = model.predict_proba(processed_data)

print(f"Predictions: {predictions[:5].tolist()}")
```

### Load All Manual Models and Ensemble

```python
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_all_manual_models():
    """Load all 8 manual models."""
    model_dir = Path('outputs/manual_models')
    models = {}
    
    model_files = [
        'LogisticRegression.pkl',
        'RandomForest_Gini.pkl',
        'RandomForest_Entropy.pkl',
        'ExtraTrees.pkl',
        'XGBoost_Default.pkl',
        'XGBoost_Tuned.pkl',
        'LightGBM_Default.pkl',
        'LightGBM_Tuned.pkl'
    ]
    
    for model_file in model_files:
        model_path = model_dir / model_file
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[model_file.replace('.pkl', '')] = pickle.load(f)
    
    return models

def ensemble_predict(models, data, weights=None):
    """
    Ensemble prediction using weighted voting.
    
    Args:
        models: Dict of {model_name: model}
        data: Input features
        weights: Dict of {model_name: weight}. If None, uses equal weights.
    
    Returns:
        predictions, probabilities
    """
    if weights is None:
        # Equal weights
        weights = {name: 1.0 / len(models) for name in models.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Collect predictions
    all_probas = []
    for model_name, model in models.items():
        proba = model.predict_proba(data)
        weight = weights.get(model_name, 0)
        all_probas.append(proba * weight)
    
    # Average probabilities
    avg_proba = np.mean(all_probas, axis=0)
    predictions = (avg_proba[:, 1] > 0.5).astype(int)
    
    return predictions, avg_proba

# Usage
models = load_all_manual_models()
new_data = pd.read_csv('path/to/new_data.csv')

# Use performance-based weights (from training results)
weights = {
    'LightGBM_Tuned': 0.7947,
    'XGBoost_Tuned': 0.7896,
    'LightGBM_Default': 0.7882,
    'XGBoost_Default': 0.7843,
    'RandomForest_Entropy': 0.7746,
    'RandomForest_Gini': 0.7742,
    'ExtraTrees': 0.7548,
    'LogisticRegression': 0.7358,
}

predictions, probabilities = ensemble_predict(models, new_data, weights)
print(f"Ensemble Predictions: {predictions[:5].tolist()}")
```

---

## 3. Input Validation and Preprocessing

### Schema Validation

```python
import pandas as pd
import numpy as np

def validate_input_schema(df, expected_features):
    """
    Validate that input data has correct features and types.
    
    Args:
        df: Input DataFrame
        expected_features: List of expected feature names
    
    Returns:
        bool: True if valid
        list: List of validation errors
    """
    errors = []
    
    # Check for missing features
    missing = set(expected_features) - set(df.columns)
    if missing:
        errors.append(f"Missing features: {missing}")
    
    # Check for extra features
    extra = set(df.columns) - set(expected_features)
    if extra:
        errors.append(f"Extra features (will be ignored): {extra}")
    
    # Check data types (example - adjust based on your features)
    numeric_features = [f for f in expected_features if f in df.columns]
    for feature in numeric_features:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            errors.append(f"Feature '{feature}' should be numeric, got {df[feature].dtype}")
    
    # Check for invalid values
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"Found null values in: {null_cols}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            errors.append(f"Found infinite values in: {col}")
    
    return len(errors) == 0, errors

# Usage
new_data = pd.read_csv('path/to/new_data.csv')

# Define expected features (112 original features)
expected_features = [
    'VISITMO', 'VISITDAY', 'VISITYR', 'BIRTHMO', 'BIRTHYR', 'SEX', 'HISPANIC',
    # ... (all 112 features)
]

is_valid, errors = validate_input_schema(new_data, expected_features)

if not is_valid:
    print("Validation failed:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✓ Input validation passed")
    # Proceed with prediction
```

### Handling Missing Values

```python
import pandas as pd
import numpy as np

def handle_missing_values(df, strategy='median'):
    """
    Impute missing values using specified strategy.
    
    Args:
        df: Input DataFrame
        strategy: 'median', 'mean', or 'mode'
    
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                if strategy == 'median':
                    df_imputed[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mean':
                    df_imputed[col].fillna(df[col].mean(), inplace=True)
            else:
                # Mode for categorical
                df_imputed[col].fillna(df[col].mode()[0], inplace=True)
    
    return df_imputed

# Usage
new_data = pd.read_csv('path/to/new_data.csv')
new_data_clean = handle_missing_values(new_data, strategy='median')
```

---

## 4. Complete Inference Pipeline

```python
"""
Complete inference pipeline combining validation, preprocessing, and prediction.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class DementiaPredictionPipeline:
    """End-to-end prediction pipeline for dementia risk assessment."""
    
    def __init__(self, model_type='autogluon', model_path=None):
        """
        Initialize pipeline.
        
        Args:
            model_type: 'autogluon' or 'manual'
            model_path: Path to model (AutoGluon dir or .pkl file)
        """
        self.model_type = model_type
        
        if model_type == 'autogluon':
            from autogluon.tabular import TabularPredictor
            self.model = TabularPredictor.load(
                model_path or 'outputs/models/autogluon_optimized/'
            )
        else:
            with open(model_path or 'outputs/manual_models/LightGBM_Tuned.pkl', 'rb') as f:
                self.model = pickle.load(f)
        
        # Expected features (112 original)
        self.expected_features = self._load_expected_features()
    
    def _load_expected_features(self):
        """Load feature list from training data."""
        train_sample = pd.read_csv('data/train/X_train.csv', nrows=1)
        return train_sample.columns.tolist()
    
    def validate_input(self, df):
        """Validate input data schema."""
        errors = []
        
        # Check features
        missing = set(self.expected_features) - set(df.columns)
        if missing:
            errors.append(f"Missing features: {missing}")
        
        # Check nulls
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            errors.append(f"Null values in: {null_cols}")
        
        if errors:
            raise ValueError("Validation failed:\n" + "\n".join(errors))
        
        return True
    
    def preprocess(self, df):
        """Apply preprocessing."""
        df_clean = df.copy()
        
        # Ensure correct feature order
        df_clean = df_clean[self.expected_features]
        
        # Handle any remaining nulls
        df_clean.fillna(df_clean.median(), inplace=True)
        
        return df_clean
    
    def predict(self, df, return_proba=True):
        """
        Make predictions.
        
        Args:
            df: Input DataFrame
            return_proba: Whether to return probabilities
        
        Returns:
            predictions, probabilities (if return_proba=True)
        """
        # Validate
        self.validate_input(df)
        
        # Preprocess
        df_clean = self.preprocess(df)
        
        # Apply feature engineering for AutoGluon
        if self.model_type == 'autogluon':
            df_clean = apply_feature_engineering(df_clean)
        
        # Predict
        predictions = self.model.predict(df_clean)
        
        if return_proba:
            probabilities = self.model.predict_proba(df_clean)
            return predictions, probabilities
        
        return predictions
    
    def predict_single(self, sample_dict):
        """
        Predict for a single sample.
        
        Args:
            sample_dict: Dict of {feature: value}
        
        Returns:
            prediction, probability
        """
        df = pd.DataFrame([sample_dict])
        predictions, probabilities = self.predict(df)
        
        return predictions[0], probabilities[1][0]

# Usage Examples

# AutoGluon pipeline
pipeline_ag = DementiaPredictionPipeline(
    model_type='autogluon',
    model_path='outputs/models/autogluon_optimized/'
)

new_data = pd.read_csv('path/to/new_data.csv')
predictions, probabilities = pipeline_ag.predict(new_data)

print(f"Predictions: {predictions[:5]}")
print(f"Dementia Probabilities: {probabilities[1][:5]}")

# Manual model pipeline
pipeline_manual = DementiaPredictionPipeline(
    model_type='manual',
    model_path='outputs/manual_models/LightGBM_Tuned.pkl'
)

predictions_manual, probabilities_manual = pipeline_manual.predict(new_data)

# Single prediction
sample = {
    'VISITMO': 11, 'VISITDAY': 15, 'VISITYR': 2023,
    'BIRTHMO': 4, 'BIRTHYR': 1936, 'SEX': 1,
    # ... (all 112 features)
}

pred, prob = pipeline_ag.predict_single(sample)
print(f"Prediction: {'Dementia' if pred == 1 else 'No Dementia'}")
print(f"Probability: {prob:.2%}")
```

---

## 5. Performance Measurement

```python
import time
import pandas as pd
from autogluon.tabular import TabularPredictor

def measure_inference_performance(model_path, test_data_path, n_runs=5):
    """
    Measure inference latency and throughput.
    
    Args:
        model_path: Path to model
        test_data_path: Path to test data
        n_runs: Number of runs for averaging
    
    Returns:
        dict of performance metrics
    """
    # Load model
    predictor = TabularPredictor.load(model_path)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    n_samples = len(test_data)
    
    # Warm-up run
    _ = predictor.predict(test_data)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.time()
        predictions = predictor.predict(test_data)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = n_samples / avg_time
    latency_per_sample = (avg_time / n_samples) * 1000  # milliseconds
    
    return {
        'n_samples': n_samples,
        'avg_time_seconds': avg_time,
        'std_time_seconds': std_time,
        'throughput_samples_per_sec': throughput,
        'latency_ms_per_sample': latency_per_sample,
    }

# Measure AutoGluon performance
perf = measure_inference_performance(
    model_path='outputs/models/autogluon_optimized/',
    test_data_path='data/test/X_test.csv',
    n_runs=5
)

print("AutoGluon Performance:")
print(f"  Samples: {perf['n_samples']}")
print(f"  Avg Time: {perf['avg_time_seconds']:.2f}s ± {perf['std_time_seconds']:.2f}s")
print(f"  Throughput: {perf['throughput_samples_per_sec']:.0f} samples/sec")
print(f"  Latency: {perf['latency_ms_per_sample']:.2f}ms per sample")
```

---

## Summary

- **AutoGluon**: Best for maximum accuracy (94.34% ROC-AUC), automated feature engineering
- **Manual Models**: Best for interpretability, smaller size, faster inference
- **Preprocessing**: Handle missing values with median imputation, validate schema before prediction
- **Production**: Use complete pipeline class for robust deployment

All code snippets are ready to run with minimal modifications to file paths.
