"""
Generate Complete EDA Report for Competition
Includes: target distribution, missing values, correlations, distributions, outliers
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

def load_data():
    """Load preprocessed data"""
    print("[1/7] Loading preprocessed data...")
    X = pd.read_csv('data/train/X_train_balanced.csv')
    y = pd.read_csv('data/train/y_train_balanced.csv').iloc[:, 0]

    # Combine for EDA
    df = X.copy()
    df['dementia_risk'] = y

    print(f"✓ Data loaded: {df.shape}")
    return df, X, y

def create_output_dir():
    """Create EDA output directory"""
    output_dir = Path('outputs/eda')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_target_distribution(df, output_dir):
    """Plot target variable distribution with percentages"""
    print("[2/7] Plotting target distribution...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Count plot
    value_counts = df['dementia_risk'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(value_counts.index, value_counts.values, color=colors, alpha=0.7, edgecolor='black')

    # Add percentages
    total = len(df)
    for i, (idx, val) in enumerate(value_counts.items()):
        pct = val / total * 100
        ax.text(idx, val + 500, f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_xlabel('Dementia Risk (0 = No, 1 = Yes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Risk (0)', 'At Risk (1)'])

    plt.tight_layout()
    plt.savefig(output_dir / 'target_distribution.png', bbox_inches='tight')
    plt.close()

    print(f"✓ Target distribution saved")
    return value_counts

def analyze_missing_values(df, output_dir):
    """Analyze and visualize missing values"""
    print("[3/7] Analyzing missing values...")

    # Calculate missing values
    missing = df.drop('dementia_risk', axis=1).isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Feature': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    # Save missing value table
    missing_df.to_csv(output_dir / 'missing_values_table.csv', index=False)

    if len(missing_df) > 0:
        # Plot missing values
        fig, ax = plt.subplots(figsize=(12, max(6, len(missing_df) * 0.25)))

        y_pos = np.arange(len(missing_df))
        bars = ax.barh(y_pos, missing_df['Missing_Percentage'], color='coral', edgecolor='black')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(missing_df['Feature'], fontsize=10)
        ax.set_xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Missing Values Analysis (Top {len(missing_df)} Features)',
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for i, (idx, row) in enumerate(missing_df.iterrows()):
            ax.text(row['Missing_Percentage'] + 0.5, i, f"{row['Missing_Percentage']:.1f}%",
                   va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'missing_values_heatmap.png', bbox_inches='tight')
        plt.close()

        print(f"✓ Found {len(missing_df)} features with missing values")
    else:
        print(f"✓ No missing values found!")

    return missing_df

def identify_feature_types(X):
    """Identify numerical vs categorical features"""
    print("[4/7] Identifying feature types...")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Check for binary features (might be stored as numeric)
    binary_features = []
    for col in numeric_features:
        if X[col].nunique() == 2:
            binary_features.append(col)

    true_numeric = [col for col in numeric_features if col not in binary_features]

    print(f"✓ True Numerical: {len(true_numeric)}")
    print(f"✓ Binary: {len(binary_features)}")
    print(f"✓ Categorical: {len(categorical_features)}")

    return {
        'numeric': true_numeric,
        'binary': binary_features,
        'categorical': categorical_features
    }

def plot_correlation_heatmap(X, output_dir):
    """Plot correlation heatmap for numerical features"""
    print("[5/7] Generating correlation heatmap...")

    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        print("✗ Not enough numeric features for correlation")
        return None

    # Limit to top 30 features (more comprehensive)
    if len(numeric_cols) > 30:
        # Select features with highest variance
        variances = X[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(30).index

    corr = X[numeric_cols].corr()

    # Plot
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

    ax.set_title('Feature Correlation Heatmap (Top 30 by Variance)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', bbox_inches='tight')
    plt.close()

    # Find high correlations
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.7:
                high_corr.append({
                    'Feature1': corr.columns[i],
                    'Feature2': corr.columns[j],
                    'Correlation': corr.iloc[i, j]
                })

    if high_corr:
        pd.DataFrame(high_corr).to_csv(output_dir / 'high_correlations.csv', index=False)
        print(f"✓ Found {len(high_corr)} high correlation pairs (|r| > 0.7)")
    else:
        print(f"✓ No high correlations found")

    return corr

def plot_top_feature_distributions(X, y, output_dir, top_n=10):
    """Plot distributions for top N most important-looking features"""
    print(f"[6/7] Plotting distributions for top {top_n} features...")

    # Use variance as proxy for importance
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    variances = X[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()

    # Create subplots
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(top_features):
        ax = axes[idx]

        # Plot distributions by target
        for target_val in [0, 1]:
            data = X.loc[y == target_val, feature]
            ax.hist(data, bins=30, alpha=0.6,
                   label=f'Risk={target_val}',
                   color=['green', 'red'][target_val])

        ax.set_title(f'{feature}', fontweight='bold', fontsize=10)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide extra subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Distribution of Top {top_n} Features (by Variance)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_feature_distributions.png', bbox_inches='tight')
    plt.close()

    print(f"✓ Distribution plots saved")
    return top_features

def plot_outlier_detection(X, output_dir, top_n=8):
    """Plot box plots for outlier detection in key features"""
    print(f"[7/7] Detecting outliers in top {top_n} features...")

    # Use variance to select features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    variances = X[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()

    # Create box plots
    n_cols = 2
    n_rows = (top_n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()

    outlier_summary = []

    for idx, feature in enumerate(top_features):
        ax = axes[idx]

        # Box plot
        bp = ax.boxplot(X[feature].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', edgecolor='black'),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'))

        # Calculate outliers
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = X[(X[feature] < lower_bound) | (X[feature] > upper_bound)]
        outlier_pct = len(outliers) / len(X) * 100

        outlier_summary.append({
            'Feature': feature,
            'Outlier_Count': len(outliers),
            'Outlier_Percentage': outlier_pct,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        })

        ax.set_title(f'{feature}\n({len(outliers)} outliers, {outlier_pct:.1f}%)',
                    fontweight='bold', fontsize=10)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(alpha=0.3, axis='y')

    # Hide extra subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Outlier Detection (Box Plots)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'outlier_detection.png', bbox_inches='tight')
    plt.close()

    # Save outlier summary
    pd.DataFrame(outlier_summary).to_csv(output_dir / 'outlier_summary.csv', index=False)

    print(f"✓ Outlier detection complete")
    return outlier_summary

def generate_eda_report(df, X, y, target_counts, missing_df, feature_types,
                       top_features, outlier_summary, output_dir):
    """Generate comprehensive EDA report in Markdown"""
    print("\nGenerating EDA_REPORT.md...")

    report = f"""# EXPLORATORY DATA ANALYSIS (EDA) REPORT

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** Dementia Risk Prediction (Non-Medical Features)

---

## 1. DATASET OVERVIEW

| Metric | Value |
|--------|-------|
| Total Samples | {len(df):,} |
| Total Features | {len(X.columns)} |
| Target Variable | dementia_risk (binary: 0/1) |
| Memory Usage | {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB |

---

## 2. TARGET VARIABLE DISTRIBUTION

![Target Distribution](target_distribution.png)

### Class Balance

| Class | Count | Percentage |
|-------|-------|------------|
| No Risk (0) | {target_counts[0]:,} | {target_counts[0]/len(df)*100:.2f}% |
| At Risk (1) | {target_counts[1]:,} | {target_counts[1]/len(df)*100:.2f}% |

**Interpretation:** {"Dataset is perfectly balanced (50/50 split)" if abs(target_counts[0] - target_counts[1]) < 100 else "Dataset shows class imbalance"}

---

## 3. FEATURE TYPES

| Feature Type | Count | Examples |
|--------------|-------|----------|
| Numerical | {len(feature_types['numeric'])} | {', '.join(feature_types['numeric'][:5])}{'...' if len(feature_types['numeric']) > 5 else ''} |
| Binary | {len(feature_types['binary'])} | {', '.join(feature_types['binary'][:5])}{'...' if len(feature_types['binary']) > 5 else ''} |
| Categorical | {len(feature_types['categorical'])} | {', '.join(feature_types['categorical'][:5]) if feature_types['categorical'] else 'None'} |

---

## 4. MISSING VALUES ANALYSIS

![Missing Values](missing_values_heatmap.png)

"""

    if len(missing_df) > 0:
        report += f"""### Top Features with Missing Values

| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
"""
        for _, row in missing_df.head(10).iterrows():
            report += f"| {row['Feature']} | {row['Missing_Count']:,} | {row['Missing_Percentage']:.2f}% |\n"

        report += f"\n**Total Features with Missing Values:** {len(missing_df)}\n"
        report += f"**Overall Missing Rate:** {df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%\n"
    else:
        report += "**No missing values detected!** ✓\n"

    report += """
---

## 5. CORRELATION ANALYSIS

![Correlation Heatmap](correlation_heatmap.png)

### Key Findings

"""

    # Check for high correlations file
    high_corr_path = output_dir / 'high_correlations.csv'
    if high_corr_path.exists():
        high_corr_df = pd.read_csv(high_corr_path)
        report += f"**Found {len(high_corr_df)} feature pairs with high correlation (|r| > 0.7)**\n\n"
        report += "| Feature 1 | Feature 2 | Correlation |\n"
        report += "|-----------|-----------|-------------|\n"
        for _, row in high_corr_df.head(10).iterrows():
            report += f"| {row['Feature1']} | {row['Feature2']} | {row['Correlation']:.3f} |\n"

        report += "\n**Recommendation:** Consider removing highly correlated features to reduce multicollinearity.\n"
    else:
        report += "**No high correlations found (|r| > 0.7).** Features appear relatively independent. ✓\n"

    report += """
---

## 6. FEATURE DISTRIBUTIONS

![Top Feature Distributions](top_feature_distributions.png)

### Top 10 Features by Variance

"""

    for i, feature in enumerate(top_features, 1):
        report += f"{i}. `{feature}`\n"

    report += """
**Interpretation:** These features show the highest variance and are likely to be important for modeling.

---

## 7. OUTLIER DETECTION

![Outlier Detection](outlier_detection.png)

### Outlier Summary

| Feature | Outliers | Percentage |
|---------|----------|------------|
"""

    for item in outlier_summary:
        report += f"| {item['Feature']} | {item['Outlier_Count']:,} | {item['Outlier_Percentage']:.2f}% |\n"

    avg_outlier_pct = np.mean([item['Outlier_Percentage'] for item in outlier_summary])
    report += f"\n**Average Outlier Rate:** {avg_outlier_pct:.2f}%\n"

    if avg_outlier_pct > 5:
        report += "\n**Recommendation:** Consider outlier treatment (capping, transformation, or robust scaling).\n"
    else:
        report += "\n**Observation:** Outlier rate is acceptable. Tree-based models handle outliers well. ✓\n"

    report += """
---

## 8. KEY INSIGHTS & RECOMMENDATIONS

### Data Quality
"""

    if len(missing_df) == 0:
        report += "- ✅ **No missing values** - data is complete\n"
    else:
        report += f"- ⚠️ **Missing values present** in {len(missing_df)} features - already handled via median imputation\n"

    report += f"- {'✅' if abs(target_counts[0] - target_counts[1]) < 100 else '⚠️'} **Class balance:** "
    report += "Perfect 50/50 split\n" if abs(target_counts[0] - target_counts[1]) < 100 else "Imbalanced - handled via balancing\n"

    if high_corr_path.exists():
        report += f"- ⚠️ **Multicollinearity detected** - consider feature selection\n"
    else:
        report += "- ✅ **Low multicollinearity** - features are relatively independent\n"

    report += """
### Feature Engineering Opportunities
- Binary features could benefit from interaction terms
- High-variance features are good candidates for importance analysis
- Consider polynomial features for non-linear relationships

### Model Selection Guidance
- **Tree-based models** (LightGBM, XGBoost) are well-suited due to:
  - Mix of numerical and binary features
  - Presence of outliers (trees are robust)
  - No need for feature scaling
- **Linear models** may require:
  - Feature scaling/normalization
  - Outlier treatment
  - Feature selection (if multicollinearity present)

---

## 9. NEXT STEPS

1. ✅ EDA complete - visualizations saved to `outputs/eda/`
2. ⏭️ Feature engineering (if needed)
3. ⏭️ Model training (AutoML + Manual ML)
4. ⏭️ Hyperparameter tuning
5. ⏭️ Model evaluation and selection

---

**Report Location:** `outputs/eda/EDA_REPORT.md`
**Plots Directory:** `outputs/eda/`

"""

    # Save report
    report_path = output_dir / 'EDA_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ EDA report saved: {report_path}")
    return report_path

def main():
    """Main execution"""
    print("="*80)
    print("COMPREHENSIVE EDA GENERATION")
    print("="*80)

    # Load data
    df, X, y = load_data()

    # Create output directory
    output_dir = create_output_dir()

    # 1. Target distribution
    target_counts = plot_target_distribution(df, output_dir)

    # 2. Missing values
    missing_df = analyze_missing_values(df, output_dir)

    # 3. Feature types
    feature_types = identify_feature_types(X)

    # 4. Correlation heatmap
    corr = plot_correlation_heatmap(X, output_dir)

    # 5. Top feature distributions
    top_features = plot_top_feature_distributions(X, y, output_dir, top_n=10)

    # 6. Outlier detection
    outlier_summary = plot_outlier_detection(X, output_dir, top_n=8)

    # 7. Generate report
    report_path = generate_eda_report(
        df, X, y, target_counts, missing_df, feature_types,
        top_features, outlier_summary, output_dir
    )

    print("\n" + "="*80)
    print("EDA COMPLETE!")
    print("="*80)
    print(f"📊 Plots saved to: {output_dir}")
    print(f"📄 Report: {report_path}")
    print("="*80)

if __name__ == "__main__":
    main()
