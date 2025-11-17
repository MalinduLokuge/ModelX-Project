# Data Preprocessing Report

## Dementia Risk Prediction - Non-Medical Features

---


## 1. Feature Selection - Non-Medical Variables Only

### Approach
Selected only **non-medical features** that people typically know about themselves.
Excluded all medical/diagnostic variables (cognitive tests, scans, lab results).

### Features Selected by Category

#### Demographics (8 features)
- `NACCAGE`
- `SEX`
- `EDUC`
- `RACE`
- `HISPANIC`
- `MARISTAT`
- `NACCLIVS`
- `HANDED`

#### Lifestyle (6 features)
- `TOBAC100`
- `TOBAC30`
- `SMOKYRS`
- `PACKSPER`
- `ALCOCCAS`
- `ALCFREQ`

#### Medical History (Simple Diagnoses) (6 features)
- `CVHATT`
- `CBSTROKE`
- `CBTIA`
- `DIABETES`
- `HYPERTEN`
- `HYPERCHO`

#### Functional Capacity (10 features)
- `BILLS`
- `TAXES`
- `SHOPPING`
- `GAMES`
- `STOVE`
- `MEALPREP`
- `EVENTS`
- `PAYATTN`
- `REMDATES`
- `TRAVEL`

#### Sensory Assessment (4 features)
- `VISION`
- `VISWCORR`
- `HEARING`
- `HEARWAID`

### Summary
- **Total Features Selected**: 34
- **Total Features Removed**: 2
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
- **Before**: 0 missing values (0.00%)
- **After**: 0 missing values (0.00%)
- **Columns Dropped**: 0
- **Missing Indicators Created**: 15

### Missing Indicators Created
- `SMOKYRS_missing`
- `PACKSPER_missing`
- `ALCFREQ_missing`
- `BILLS_missing`
- `TAXES_missing`
- `SHOPPING_missing`
- `GAMES_missing`
- `STOVE_missing`
- `MEALPREP_missing`
- `EVENTS_missing`
- ... and 5 more

### Justification
- Domain-specific handling of NACC dataset special codes
- Missing indicators preserve information about missingness patterns
- Median/Mode imputation is robust and interpretable for medical data
- Dropping columns with >80% missing preserves data quality

## 3. Feature Engineering

### Features Created
**Total**: 19 domain-specific features

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

... and 4 more features

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
- **Features Affected**: 22

### Examples of Outliers Capped
| Feature | N Outliers | Lower Bound | Upper Bound |
|---------|------------|-------------|-------------|
| `HISPANIC` | 44 | 0.00 | 0.00 |
| `HANDED` | 37 | 2.00 | 2.00 |
| `TOBAC30` | 36 | 0.00 | 0.00 |
| `PACKSPER` | 22 | 0.12 | 5.12 |
| `ALCFREQ` | 21 | 0.50 | 4.50 |
| `CVHATT` | 48 | 0.00 | 0.00 |
| `CBSTROKE` | 39 | 0.00 | 0.00 |
| `CBTIA` | 44 | 0.00 | 0.00 |
| `VISWCORR` | 45 | 1.00 | 1.00 |
| `HEARWAID` | 45 | 0.00 | 0.00 |

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
- **Count**: 68 numerical features
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
**Total**: 17 features

| Feature | Reason |
|---------|--------|
| `HISPANIC` | Low variance (0.000000) |
| `HANDED` | Low variance (0.000000) |
| `TOBAC30` | Low variance (0.000000) |
| `CVHATT` | Low variance (0.000000) |
| `CBSTROKE` | Low variance (0.000000) |
| `CBTIA` | Low variance (0.000000) |
| `VISWCORR` | Low variance (0.000000) |
| `HEARWAID` | Low variance (0.000000) |
| `SMOKYRS_missing` | Low variance (0.000000) |
| `age_85plus` | Low variance (0.000000) |

... and 7 more

## 8. Finalized Feature Set - Summary

### Pipeline Overview
- **Initial Features**: 36
- **Features Removed**: -15
- **Features Created**: 19
- **Final Features**: 51

### Data Quality
- **Initial Samples**: 200
- **Final Samples**: 200
- **Sample Retention**: 100.0%

### Preprocessing Steps Summary
1. ✓ Feature Selection (Non-medical only)
2. ✓ Missing Value Handling (NACC-specific codes)
3. ✓ Feature Engineering (Domain-specific features)
4. ✓ Outlier Detection and Handling (IQR + Capping)
5. ✓ Categorical Encoding (Label + One-Hot)
6. ✓ Feature Scaling (RobustScaler)
7. ✓ Final Feature Selection (Correlation + Variance)

### Ready for Modeling
The preprocessed dataset is now ready for training machine learning models.
All features are numerical, properly scaled, and free of missing values.