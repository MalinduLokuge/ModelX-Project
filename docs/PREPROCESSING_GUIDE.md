# Dementia Risk Prediction - Preprocessing Pipeline Guide

## Overview

This guide explains the comprehensive preprocessing pipeline designed specifically for dementia risk prediction using **non-medical features only**.

The pipeline follows competition requirements: build a model using only information that normal people know about themselves, excluding medical/diagnostic variables.

## Pipeline Architecture

```
Raw Data (143 features)
    ↓
[1] Feature Selection → Non-medical features only
    ↓
[2] Missing Value Handling → NACC-specific codes
    ↓
[3] Feature Engineering → Domain-specific features
    ↓
[4] Outlier Handling → IQR + Capping
    ↓
[5] Categorical Encoding → Label + One-Hot
    ↓
[6] Feature Scaling → RobustScaler
    ↓
[7] Final Selection → Correlation + Variance
    ↓
Clean Data (ready for modeling)
```

## Quick Start

### 1. Basic Usage

```python
from src.preprocessing.dementia_preprocessing_pipeline import DementiaPreprocessingPipeline
from src.reporting.preprocessing_report_generator import PreprocessingReportGenerator
import pandas as pd

# Load your data
df = pd.read_csv('data/nacc_dataset.csv')

# Initialize pipeline
pipeline = DementiaPreprocessingPipeline(target_col='dementia')

# Run preprocessing
X_processed, y, report = pipeline.fit_transform(df)

# Generate report
report_gen = PreprocessingReportGenerator(report)
report_gen.save_all_reports('outputs/reports/')

# Save processed data
X_processed.to_csv('outputs/X_processed.csv', index=False)
y.to_csv('outputs/y.csv', index=False)
```

### 2. Command Line Usage

```bash
python scripts/run_dementia_preprocessing.py \
    --data data/nacc_dataset.csv \
    --target dementia \
    --output outputs/preprocessed/
```

## Pipeline Components

### 1. Feature Selection - Non-Medical Only

**Purpose**: Select only features that people know about themselves.

**Categories Included**:
- Demographics (age, sex, education, race, marital status)
- Lifestyle (smoking, alcohol consumption)
- Simple medical history (heart attack, stroke - things people know)
- Functional capacity (ability to do daily tasks)
- Social factors (living situation, relationships)
- Sensory (vision, hearing - self-assessed)

**Categories Excluded**:
- Cognitive test scores
- Medical scans/imaging
- Lab results
- Detailed clinical assessments
- Diagnostic measurements

**Implementation**:
```python
from src.preprocessing.dementia_feature_selector import DementiaFeatureSelector

selector = DementiaFeatureSelector()
X_selected = selector.select_features(df, target_col='dementia')
report = selector.get_selection_report()
```

### 2. Missing Value Handling

**Purpose**: Handle NACC dataset's special missing value codes.

**Special Codes**:
- `-4`: Not available (form didn't collect data)
- `8, 88, 888, 8888`: Not applicable
- `9, 99, 999, 9999`: Unknown

**Strategies**:

| Missing % | Strategy |
|-----------|----------|
| 0-5% | Median/Mode imputation |
| 5-20% | Median/Mode imputation |
| 20-50% | Median/Mode + Missing indicator |
| 50-80% | Median/Mode + Missing indicator |
| >80% | Drop column |

**Implementation**:
```python
from src.preprocessing.nacc_missing_handler import NACCMissingValueHandler

handler = NACCMissingValueHandler()
X_clean, report = handler.fit_transform(X)
```

### 3. Feature Engineering

**Purpose**: Create domain-specific features based on dementia research.

**Features Created**:

1. **Cardiovascular Risk Score**
   - Formula: Weighted sum of heart conditions
   - Rationale: CV disease increases dementia risk

2. **Cerebrovascular Risk Score**
   - Formula: Stroke (3 points) + TIA (2 points) + multiples
   - Rationale: Stroke strongly predicts dementia

3. **Lifestyle Risk Score**
   - Formula: Smoking + alcohol abuse + substance abuse
   - Rationale: Lifestyle factors affect brain health

4. **Functional Impairment Score**
   - Formula: Sum of difficulties across 10 daily activities
   - Rationale: Early sign of cognitive decline

5. **Age Features**
   - `age_squared`: Non-linear age effect
   - `age_65plus`, `age_75plus`, `age_85plus`: Risk thresholds

6. **Education Features**
   - `low_education`: < 12 years (risk factor)
   - `high_education`: >= 16 years (protective factor)

7. **Social Features**
   - `lives_alone`: Social isolation indicator
   - `widowed`, `never_married`: Relationship status

8. **Smoking Features**
   - `pack_years`: Standard smoking exposure measure
   - `years_since_quit`: Time since cessation

9. **Comorbidity Features**
   - `total_comorbidities`: Count of all conditions

10. **Sensory Features**
    - `vision_impaired`, `hearing_impaired`
    - `dual_sensory_impairment`: Both impaired

**Implementation**:
```python
from src.feature_engineering.dementia_features import DementiaFeatureEngineer

engineer = DementiaFeatureEngineer()
X_engineered, report = engineer.engineer_features(X)
```

### 4. Outlier Handling

**Method**: IQR (Interquartile Range)
- **Detection**: Values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
- **Treatment**: Capping (Winsorization)
- **Rationale**: Medical data has legitimate extremes; capping preserves distribution

### 5. Categorical Encoding

**Strategy**: Mixed approach based on cardinality

| Cardinality | Method |
|-------------|--------|
| Binary (2 values) | Label Encoding |
| Medium (3-10 values) | One-Hot Encoding |
| High (>10 values) | Label Encoding |

### 6. Feature Scaling

**Method**: RobustScaler
- **Formula**: `(x - median) / IQR`
- **Rationale**: Robust to outliers, uses median and IQR instead of mean and std
- **Applied to**: All numerical features

### 7. Final Feature Selection

**Methods**:
1. **Variance Threshold**: Remove features with variance < 0.01
2. **Correlation Analysis**: Remove one feature from pairs with correlation > 0.95

## Output Files

After running the pipeline, you'll get:

```
outputs/preprocessed/
├── X_processed.csv                 # Processed features
├── y.csv                           # Target variable
├── PREPROCESSING_REPORT.md         # Detailed markdown report
├── preprocessing_report.xlsx       # Excel report with multiple sheets
├── preprocessing_pipeline.pkl      # Saved pipeline for inference
└── preprocessing_report.json       # JSON report for programmatic access
```

## Understanding the Reports

### Markdown Report

Comprehensive documentation including:
- Feature selection justification
- Missing value strategies
- Feature engineering details (formula, rationale, expected impact)
- Outlier handling summary
- Encoding methods
- Scaling approach
- Final feature set

### Excel Report

Multiple sheets:
- **Overview**: Key metrics and summary
- **Feature Engineering**: All created features with definitions
- **Missing Values**: Imputation strategies per feature

### JSON Report

Machine-readable format for:
- Programmatic analysis
- Dashboard integration
- Reproducibility tracking

## Customization

### Change Missing Value Strategy

```python
from src.preprocessing.nacc_missing_handler import NACCMissingValueHandler

# Custom thresholds
handler = NACCMissingValueHandler()
# Modify thresholds in _recommend_strategy method
```

### Add Custom Features

```python
from src.feature_engineering.dementia_features import DementiaFeatureEngineer

engineer = DementiaFeatureEngineer()

# Extend engineer_features method
def custom_features(self, df):
    df['my_feature'] = df['col1'] * df['col2']
    return df
```

### Change Scaling Method

```python
# In dementia_preprocessing_pipeline.py
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Replace RobustScaler with StandardScaler
self.scaler = StandardScaler()  # instead of RobustScaler()
```

## Best Practices

### 1. Data Quality Checks

Before preprocessing:
```python
# Check for completely empty columns
empty_cols = df.columns[df.isnull().all()].tolist()

# Check for single-value columns
single_value = [col for col in df.columns if df[col].nunique() == 1]

# Check target distribution
print(df['dementia'].value_counts())
```

### 2. Train-Test Split

Always split **before** preprocessing:
```python
from sklearn.model_selection import train_test_split

# Split first
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Then preprocess
pipeline = DementiaPreprocessingPipeline()
X_train, y_train, _ = pipeline.fit_transform(train_df)
X_test, y_test, _ = pipeline.transform(test_df)
```

### 3. Save Pipeline

Always save the fitted pipeline for inference:
```python
pipeline.save_pipeline('outputs/models/')

# Later, load for inference
import pickle
with open('outputs/models/preprocessing_pipeline.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

X_new_processed = loaded_pipeline.transform(new_data)
```

## Troubleshooting

### Issue: "Column not found" error

**Solution**: Some features may not be in your dataset. The pipeline handles this gracefully by checking for column existence before processing.

### Issue: Memory error with large datasets

**Solution**: Process in chunks:
```python
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    X_chunk, y_chunk, _ = pipeline.fit_transform(chunk)
```

### Issue: Feature names don't match expected

**Solution**: Check the data dictionary and verify column names match NACC format.

## Technical Details

### Dependencies

```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
openpyxl >= 3.0.0  # for Excel reports
```

### Performance

- **Small datasets** (<10K rows): < 1 minute
- **Medium datasets** (10K-100K rows): 1-5 minutes
- **Large datasets** (>100K rows): 5-15 minutes

### Memory Usage

Approximate memory requirements:
- 10K rows: ~100 MB
- 100K rows: ~1 GB
- 1M rows: ~10 GB

## References

This preprocessing pipeline is based on:

1. NACC Data Dictionary specifications
2. Established dementia risk factors from medical literature
3. Best practices in ML preprocessing
4. Competition requirements for non-medical features

## Support

For issues or questions:
- Check the generated reports for detailed information
- Review the code comments in each module
- Examine example outputs in `docs/examples/`
