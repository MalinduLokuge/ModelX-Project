#!/usr/bin/env python3
"""Generate visualizations for project report"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

OUT = Path('outputs/report_figures')
OUT.mkdir(parents=True, exist_ok=True)
sns.set_style('whitegrid')

print("Generating report visualizations...")

# Load data
X_train = pd.read_csv('data/train/X_train.csv')
y_train = pd.read_csv('data/train/y_train.csv')['target']
X_test = pd.read_csv('data/test/X_test.csv')
y_test = pd.read_csv('data/test/y_test.csv')['target']

# Combine for full dataset stats
X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

print(f"Data: {len(X_full):,} total samples")

# 1. Data Types Distribution
print("\n[1/6] Data types...")
dtypes_count = X_full.dtypes.value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
dtypes_count.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71'])
ax.set_title('Feature Data Types Distribution', fontweight='bold', fontsize=14)
ax.set_xlabel('Data Type', fontsize=12)
ax.set_ylabel('Number of Features', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
for i, v in enumerate(dtypes_count.values):
    ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'01_data_types.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Missing Values
print("[2/6] Missing values...")
missing = X_full.isnull().sum()
missing_pct = (missing / len(X_full) * 100).sort_values(ascending=False)
top_missing = missing_pct[missing_pct > 0].head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Missing values bar chart
if len(top_missing) > 0:
    top_missing.plot(kind='barh', ax=axes[0], color='#e74c3c')
    axes[0].set_title('Top 20 Features with Missing Values', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Missing Percentage (%)', fontsize=11)
    axes[0].set_ylabel('Feature', fontsize=11)
    axes[0].invert_yaxis()
else:
    axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
    axes[0].set_title('Missing Values', fontweight='bold')

# Missing values heatmap sample
missing_matrix = X_full.iloc[:100, :30].isnull().astype(int)
sns.heatmap(missing_matrix.T, cbar=True, cmap='RdYlGn_r', ax=axes[1],
            yticklabels=missing_matrix.columns[:30], xticklabels=False)
axes[1].set_title('Missing Values Pattern (First 100 samples, 30 features)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Sample Index', fontsize=11)
axes[1].set_ylabel('Features', fontsize=11)

plt.tight_layout()
plt.savefig(OUT/'02_missing_values.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Class Distribution
print("[3/6] Class distribution...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original distribution
class_counts = y_full.value_counts()
axes[0].pie(class_counts, labels=['No Dementia', 'Dementia'], autopct='%1.1f%%',
            startangle=90, colors=['#2ecc71', '#e74c3c'], explode=(0.05, 0))
axes[0].set_title('Full Dataset Class Distribution\n(Original)', fontweight='bold', fontsize=12)

# Training distribution
train_counts = y_train.value_counts()
axes[1].bar(['No Dementia', 'Dementia'], train_counts.values, color=['#2ecc71', '#e74c3c'])
axes[1].set_title('Training Set Class Distribution\n(Balanced)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Number of Samples', fontsize=11)
for i, v in enumerate(train_counts.values):
    axes[1].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

# Test distribution
test_counts = y_test.value_counts()
axes[2].bar(['No Dementia', 'Dementia'], test_counts.values, color=['#2ecc71', '#e74c3c'])
axes[2].set_title('Test Set Class Distribution\n(Original)', fontweight='bold', fontsize=12)
axes[2].set_ylabel('Number of Samples', fontsize=11)
for i, v in enumerate(test_counts.values):
    axes[2].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT/'03_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Statistical Summary - Key Features
print("[4/6] Statistical summary...")
# Select key features
key_features = ['NACCAGE', 'EDUC', 'VISITYR', 'BIRTHYR', 'SMOKYRS', 'PACKSPER']
key_features = [f for f in key_features if f in X_full.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feat in enumerate(key_features):
    ax = axes[idx]
    data = X_full[feat].dropna()

    # Histogram
    ax.hist(data, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_title(f'{feat} Distribution', fontweight='bold', fontsize=11)
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)

    # Add statistics text
    stats_text = f'Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

plt.tight_layout()
plt.savefig(OUT/'04_key_features_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Distribution Analysis - Box plots
print("[5/6] Distribution analysis...")
fig, ax = plt.subplots(figsize=(14, 6))
X_full[key_features].boxplot(ax=ax, vert=False, patch_artist=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2))
ax.set_title('Distribution Analysis - Box Plots (Key Features)', fontweight='bold', fontsize=14)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT/'05_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Correlation Analysis
print("[6/6] Correlation analysis...")
# Select numerical features for correlation
num_features = X_full.select_dtypes(include=[np.number]).columns[:20]
corr_matrix = X_full[num_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix (First 20 Numerical Features)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(OUT/'06_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate statistics JSON
stats = {
    'total_samples': len(X_full),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'total_features': len(X_full.columns),
    'numerical_features': len(X_full.select_dtypes(include=[np.number]).columns),
    'categorical_features': len(X_full.select_dtypes(include=['object']).columns),
    'missing_percentage': float(X_full.isnull().sum().sum() / (len(X_full) * len(X_full.columns)) * 100),
    'class_distribution': {
        'no_dementia': int(class_counts[0]),
        'dementia': int(class_counts[1]),
        'imbalance_ratio': float(class_counts[0] / class_counts[1])
    }
}

import json
with open(OUT/'data_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n{'='*80}")
print("VISUALIZATIONS COMPLETE")
print(f"{'='*80}")
print(f"✓ 6 figures saved to {OUT}/")
print(f"✓ Statistics: {OUT}/data_statistics.json")
print(f"{'='*80}")
