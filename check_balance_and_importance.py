"""
Quick script: Check class balance, apply SMOTE, get feature importance
"""
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import json
import pickle

print("="*80)
print("STEP 1: Loading data...")
print("="*80)

X_train = pd.read_csv('data/train/X_train.csv')
y_train = pd.read_csv('data/train/y_train.csv').values.ravel()

# Fill NaN with median (SMOTE doesn't accept NaN)
if X_train.isnull().any().any():
    print(f"⚠️  Found {X_train.isnull().sum().sum()} NaN values - filling with median...")
    X_train = X_train.fillna(X_train.median())

print(f"Train shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")

# Check class balance
print("\n" + "="*80)
print("STEP 2: Class Balance Analysis")
print("="*80)

class_counts = Counter(y_train)
total = len(y_train)
class_0_pct = (class_counts[0] / total) * 100
class_1_pct = (class_counts[1] / total) * 100

print(f"Class 0 (No Dementia): {class_counts[0]:,} ({class_0_pct:.2f}%)")
print(f"Class 1 (Dementia):    {class_counts[1]:,} ({class_1_pct:.2f}%)")
print(f"Imbalance Ratio: {class_counts[0]/class_counts[1]:.2f}:1")

# Determine if SMOTE needed
imbalance_ratio = max(class_counts[0], class_counts[1]) / min(class_counts[0], class_counts[1])
needs_smote = imbalance_ratio > 1.5  # If more than 60:40

balance_info = {
    'original_distribution': {
        'class_0': int(class_counts[0]),
        'class_1': int(class_counts[1]),
        'class_0_pct': round(class_0_pct, 2),
        'class_1_pct': round(class_1_pct, 2),
        'ratio': round(imbalance_ratio, 2)
    },
    'needs_balancing': needs_smote
}

# Apply SMOTE if needed
if needs_smote:
    print(f"\n⚠️  Imbalance detected ({imbalance_ratio:.2f}:1) - Applying SMOTE...")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    balanced_counts = Counter(y_train_balanced)
    balanced_0_pct = (balanced_counts[0] / len(y_train_balanced)) * 100
    balanced_1_pct = (balanced_counts[1] / len(y_train_balanced)) * 100

    print(f"✓ After SMOTE:")
    print(f"  Class 0: {balanced_counts[0]:,} ({balanced_0_pct:.2f}%)")
    print(f"  Class 1: {balanced_counts[1]:,} ({balanced_1_pct:.2f}%)")

    balance_info['technique_used'] = 'SMOTE (Synthetic Minority Over-sampling)'
    balance_info['new_distribution'] = {
        'class_0': int(balanced_counts[0]),
        'class_1': int(balanced_counts[1]),
        'class_0_pct': round(balanced_0_pct, 2),
        'class_1_pct': round(balanced_1_pct, 2)
    }
    balance_info['justification'] = 'SMOTE creates synthetic samples for minority class, improving model ability to learn from imbalanced data'

    # Save balanced data
    pd.DataFrame(X_train_balanced, columns=X_train.columns).to_csv('data/train/X_train_balanced.csv', index=False)
    pd.DataFrame(y_train_balanced, columns=['target']).to_csv('data/train/y_train_balanced.csv', index=False)
    print(f"✓ Saved balanced data to data/train/X_train_balanced.csv")

    X_model = X_train_balanced
    y_model = y_train_balanced
else:
    print("✓ Classes are reasonably balanced - no SMOTE needed")
    balance_info['technique_used'] = 'None - data already balanced'
    balance_info['justification'] = 'Class distribution is acceptable for training'
    X_model = X_train
    y_model = y_train

# Save balance info
with open('outputs/dementia_preprocessed/class_balance_report.json', 'w') as f:
    json.dump(balance_info, f, indent=2)

# Train quick RandomForest for feature importance
print("\n" + "="*80)
print("STEP 3: Training RandomForest for Feature Importance")
print("="*80)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
print("Training RandomForest (100 trees, max_depth=10)...")
rf.fit(X_model, y_model)
print("✓ Training complete")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print("-" * 60)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:40s} {row['importance']:.4f}")

# Save feature importance
feature_importance.to_csv('outputs/dementia_preprocessed/feature_importance.csv', index=False)
print(f"\n✓ Saved feature importance to outputs/dementia_preprocessed/feature_importance.csv")

# Save model
with open('outputs/dementia_preprocessed/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print(f"✓ Saved model to outputs/dementia_preprocessed/rf_model.pkl")

# Create summary
summary = {
    'class_balance': balance_info,
    'feature_importance': {
        'method': 'RandomForest (100 trees, max_depth=10)',
        'top_10_features': feature_importance.head(10).to_dict('records'),
        'total_features': len(feature_importance)
    },
    'model_info': {
        'algorithm': 'RandomForestClassifier',
        'n_estimators': 100,
        'max_depth': 10,
        'train_samples': len(X_model)
    }
}

with open('outputs/dementia_preprocessed/preprocessing_final_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✓ ALL STEPS COMPLETE")
print("="*80)
print(f"Files saved:")
print(f"  - outputs/dementia_preprocessed/class_balance_report.json")
print(f"  - outputs/dementia_preprocessed/feature_importance.csv")
print(f"  - outputs/dementia_preprocessed/rf_model.pkl")
print(f"  - outputs/dementia_preprocessed/preprocessing_final_summary.json")
if needs_smote:
    print(f"  - data/train/X_train_balanced.csv")
    print(f"  - data/train/y_train_balanced.csv")
print("="*80)
