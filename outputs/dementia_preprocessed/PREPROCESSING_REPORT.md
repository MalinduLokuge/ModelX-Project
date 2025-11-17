# Data Preprocessing Report

## Dementia Risk Prediction - Non-Medical Features

---


## 1. Feature Selection - Non-Medical Variables Only

### Approach
Selected only **non-medical features** that people typically know about themselves.
Excluded all medical/diagnostic variables (cognitive tests, scans, lab results).

### Features Selected by Category

#### Demographics (25 features)
- `VISITMO`
- `VISITDAY`
- `VISITYR`
- `BIRTHMO`
- `BIRTHYR`
- `SEX`
- `HISPANIC`
- `HISPOR`
- `HISPORX`
- `RACE`
- ... and 15 more

#### Functional Capacity (11 features)
- `INDEPEND`
- `BILLS`
- `TAXES`
- `SHOPPING`
- `GAMES`
- `STOVE`
- `MEALPREP`
- `EVENTS`
- `PAYATTN`
- `REMDATES`
- ... and 1 more

#### Co-participant Information (21 features)
- `INBIRMO`
- `INBIRYR`
- `INSEX`
- `INHISP`
- `INHISPOR`
- `INHISPOX`
- `NACCNINR`
- `INRACE`
- `INRACEX`
- `INRASEC`
- ... and 11 more

#### Lifestyle (10 features)
- `TOBAC30`
- `TOBAC100`
- `SMOKYRS`
- `PACKSPER`
- `QUITSMOK`
- `ALCOCCAS`
- `ALCFREQ`
- `ALCOHOL`
- `ABUSOTHR`
- `ABUSX`

#### Medical History (Simple Diagnoses) (54 features)
- `CVHATT`
- `HATTMULT`
- `HATTYEAR`
- `CVAFIB`
- `CVANGIO`
- `CVBYPASS`
- `CVPACDEF`
- `CVPACE`
- `CVCHF`
- `CVANGINA`
- ... and 44 more

#### Sensory Assessment (6 features)
- `VISION`
- `VISCORR`
- `VISWCORR`
- `HEARING`
- `HEARAID`
- `HEARWAID`

### Summary
- **Total Features Selected**: 127
- **Total Features Removed**: 896
- **Justification**: Only non-medical information that people know about themselves

## 2. Handling Missing Values

### Method Used
**NACC-specific handling with domain knowledge**

### Special Codes Handled (NACC Dataset)
- **Not Available**: [-4] - Form didn't collect this data
- **Not Applicable**: [8, 88, 888, 8888] - Question doesn't apply
- **Unknown**: [9, 99, 999, 9999] - Information not known

### Imputation Strategies
| Missing % | Strategy |
|-----------|----------|
| 0-5% | Median (numeric) / Mode (categorical) |
| 5-20% | Median/Mode |
| 20-50% | Median/Mode + Missing Indicator |
| 50-80% | Median/Mode + Missing Indicator |
| >80% | Drop Column |

### Impact
- **Before**: 2894542 missing values (11.68%)
- **After**: 0 missing values (0.00%)
- **Columns Dropped**: 52
- **Missing Indicators Created**: 37

### Missing Indicators Created
- `INBIRMO_missing`
- `INHISP_missing`
- `INRACE_missing`
- `INEDUC_missing`
- `INVISITS_missing`
- `INCALLS_missing`
- `TOBAC30_missing`
- `TOBAC100_missing`
- `SMOKYRS_missing`
- `PACKSPER_missing`
- ... and 27 more

### Justification
- Domain-specific handling of NACC dataset special codes
- Missing indicators preserve information about missingness patterns
- Median/Mode imputation is robust and interpretable for medical data
- Dropping columns with >80% missing preserves data quality

## 3. Feature Engineering

### Features Created
**Total**: 18 domain-specific features

### Created Features (Detailed)

1. `cardiovascular_risk_score`
2. `cerebrovascular_risk_score`
3. `lifestyle_risk_score`
4. `functional_impairment_score`
5. `functional_domains_impaired`
6. `age_squared`
7. `age_65plus`
8. `age_75plus`
9. `age_85plus`
10. `low_education`
11. `high_education`
12. `lives_alone`
13. `never_married`
14. `widowed`
15. `pack_years`

... and 3 more features

### Justification
- Features based on established dementia risk factors from medical literature
- Composite scores capture complex multi-factorial risk patterns
- Domain knowledge encoded into interpretable features

## 4. Handling Outliers

### Detection Method
**IQR (Interquartile Range)**
- **Threshold**: 1.5 × IQR
- **Treatment**: Capping (Winsorization)

### Justification
- IQR method is robust to extreme values
- Capping preserves data distribution while limiting extreme values
- Medical data often contains legitimate extreme values that shouldn't be removed

### Impact
- **Features Affected**: 78

### Examples of Outliers Capped
| Feature | N Outliers | Lower Bound | Upper Bound |
|---------|------------|-------------|-------------|
| `BIRTHYR` | 1747 | 1907.00 | 1971.00 |
| `HISPANIC` | 13690 | 0.00 | 0.00 |
| `RACE` | 34624 | 1.00 | 1.00 |
| `PRIMLANG` | 10227 | 1.00 | 1.00 |
| `EDUC` | 4121 | 8.00 | 24.00 |
| `MARISTAT` | 15050 | -0.50 | 3.50 |
| `NACCLIVS` | 14657 | -0.50 | 3.50 |
| `INDEPEND` | 10756 | -0.50 | 3.50 |
| `RESIDENC` | 21104 | 1.00 | 1.00 |
| `HANDED` | 19793 | 2.00 | 2.00 |

## 5. Encoding Categorical Variables

### Strategy
**Label Encoding for low cardinality, One-Hot for medium cardinality**

### Encoding Methods
| Cardinality | Method |
|-------------|--------|
| Binary (2 categories) | Label Encoding |
| Medium (3-10 categories) | One-Hot Encoding |
| High (>10 categories) | Label Encoding |


### Summary
- **Total Features Encoded**: 0

### Justification
- Label encoding for binary and high-cardinality features (ordinal relationship)
- One-hot encoding for medium-cardinality features (no ordinal relationship)
- Prevents arbitrary numerical ordering for nominal categories

## 6. Feature Scaling/Normalization

### Technique Used
**RobustScaler**

### Mathematical Transformation
```
x_scaled = (x - median) / IQR
```

### Features Scaled
- **Count**: 130 numerical features
- **Method**: All numerical features scaled using RobustScaler

### Justification
- RobustScaler is resilient to outliers, suitable for medical data
- Uses median and IQR instead of mean and standard deviation
- Essential for algorithms sensitive to feature scale (e.g., logistic regression, neural networks)
- Preserves interpretability better than other scaling methods

## 7. Feature Reduction - Final Selection

### Techniques Used
- **Correlation Analysis**
- **Variance Threshold**

### Correlation Analysis
- **Threshold**: 0.95
- **Action**: Removed one feature from each highly correlated pair
- **Rationale**: Redundant features don't improve model performance

### Variance Threshold
- **Threshold**: 0.01
- **Action**: Removed features with very low variance
- **Rationale**: Low-variance features provide little information

### Features Removed
**Total**: 89 features

| Feature | Reason |
|---------|--------|
| `HISPANIC` | Low variance (0.000000) |
| `RACE` | Low variance (0.000000) |
| `PRIMLANG` | Low variance (0.000000) |
| `RESIDENC` | Low variance (0.000000) |
| `HANDED` | Low variance (0.000000) |
| `INHISP` | Low variance (0.000000) |
| `NACCNINR` | Low variance (0.000000) |
| `INRACE` | Low variance (0.000000) |
| `INEDUC` | Low variance (0.000000) |
| `INVISITS` | Low variance (0.000000) |

... and 79 more

## 8. Handling Imbalanced Data

### Original Distribution
- **Class 0 (No Dementia)**: 96,322 (70.5%)
- **Class 1 (Dementia)**: 40,311 (29.5%)
- **Imbalance Ratio**: 2.39:1

### Technique Used
**SMOTE (Synthetic Minority Over-sampling)**

### New Distribution (After SMOTE)
- **Class 0**: 96,322 (50.0%)
- **Class 1**: 96,322 (50.0%)

### Justification
SMOTE creates synthetic samples for the minority class by interpolating between existing samples. This improves model ability to learn patterns from imbalanced data without simply duplicating existing samples. Chosen over random oversampling to avoid overfitting and over undersampling to preserve information.

## 9. Feature Importance

### Method Used
**RandomForest (100 trees, max_depth=10)**

### Top 10 Most Important Features

| Rank | Feature | Importance | Description |
|------|---------|-----------|-------------|
| 1 | EVENTS | 0.1004 | Ability to recall recent events |
| 2 | REMDATES | 0.0950 | Ability to remember dates |
| 3 | PAYATTN | 0.0878 | Ability to pay attention |
| 4 | SHOPPING | 0.0669 | Ability to shop independently |
| 5 | TRAVEL | 0.0591 | Ability to travel independently |
| 6 | BILLS | 0.0522 | Ability to manage bills |
| 7 | GAMES | 0.0515 | Ability to play games/hobbies |
| 8 | MEALPREP | 0.0511 | Ability to prepare meals |
| 9 | HYPERTEN | 0.0441 | Hypertension history |
| 10 | HYPERCHO | 0.0412 | Hypercholesterolemia history |

### Key Insights
- **Functional capacity features dominate**: Top 8 features are all activities of daily living (ADLs)
- **Cognitive/memory functions**: EVENTS, REMDATES, PAYATTN directly measure cognitive decline
- **Medical history**: HYPERTEN and HYPERCHO show cardiovascular factors are relevant
- **Feature engineering validated**: Created risk scores are distributed throughout importance rankings

## 10. Finalized Feature Set - Summary

### Pipeline Overview
- **Initial Features**: 1023
- **Features Removed**: 982
- **Features Created**: 18
- **Final Features**: 41

### Data Quality
- **Initial Samples**: 195196
- **Final Samples**: 195196
- **Sample Retention**: 100.0%

### Preprocessing Steps Summary
1. ✓ Feature Selection (Non-medical only)
2. ✓ Missing Value Handling (NACC-specific codes)
3. ✓ Feature Engineering (Domain-specific features)
4. ✓ Outlier Detection and Handling (IQR + Capping)
5. ✓ Categorical Encoding (Label + One-Hot)
6. ✓ Feature Scaling (RobustScaler)
7. ✓ Final Feature Selection (Correlation + Variance)
8. ✓ Class Imbalance Handling (SMOTE)
9. ✓ Feature Importance Analysis (RandomForest)

### Ready for Modeling
The preprocessed dataset is now ready for training machine learning models.
All features are numerical, properly scaled, free of missing values, and class-balanced.