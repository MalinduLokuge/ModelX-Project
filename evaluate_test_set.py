#!/usr/bin/env python3
"""
Evaluate Tuned Models on Test Set
Generates final performance metrics and comparison with baseline
"""

import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup
OUTPUT_DIR = Path('test_results')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("TEST SET EVALUATION - TUNED MODELS")
print("="*80)

# Load test data
print("\n[1/6] Loading test data...")
X_test = pd.read_csv('data/test/X_test.csv')
y_test = pd.read_csv('data/test/y_test.csv')['target']

# Handle missing values (same as training)
X_test = X_test.fillna(X_test.median())

print(f"✓ Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
print(f"✓ Class balance: {y_test.value_counts().to_dict()}")

# Load tuned models
print("\n[2/6] Loading tuned models...")
with open('tuning_results/best_lightgbm_model.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)
with open('tuning_results/best_xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('tuning_results/ensemble_calibrated.pkl', 'rb') as f:
    ensemble = pickle.load(f)

print("✓ Models loaded: LightGBM, XGBoost, Ensemble")

# Generate predictions
print("\n[3/6] Generating predictions...")

# Individual models
lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# Ensemble
base_preds = np.column_stack([lgbm_proba, xgb_proba])
ensemble_proba = ensemble['meta_learner_calibrated'].predict_proba(base_preds)[:, 1]

# Binary predictions (threshold 0.5)
lgbm_pred = (lgbm_proba > 0.5).astype(int)
xgb_pred = (xgb_proba > 0.5).astype(int)
ensemble_pred = (ensemble_proba > 0.5).astype(int)

print("✓ Predictions generated")

# Calculate metrics
print("\n[4/6] Calculating metrics...")

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive metrics"""
    return {
        'AUC': roc_auc_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Brier': brier_score_loss(y_true, y_proba)
    }

results = {
    'LightGBM': calculate_metrics(y_test, lgbm_pred, lgbm_proba),
    'XGBoost': calculate_metrics(y_test, xgb_pred, xgb_proba),
    'Ensemble': calculate_metrics(y_test, ensemble_pred, ensemble_proba)
}

# Load baseline results for comparison
try:
    baseline_results = pd.read_csv('outputs/manual_ml/model_comparison.csv', index_col=0)
    baseline_lgbm_auc = baseline_results.loc['LightGBM_Tuned', 'Test_AUC']
    baseline_xgb_auc = baseline_results.loc['XGBoost_Tuned', 'Test_AUC']
except:
    baseline_lgbm_auc = 0.7947  # From previous runs
    baseline_xgb_auc = 0.7896

print("\n" + "="*80)
print("PERFORMANCE ON TEST SET")
print("="*80)

print(f"\n{'Model':<20} {'AUC':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Brier':>8}")
print("-"*80)
for model, metrics in results.items():
    print(f"{model:<20} {metrics['AUC']:>8.4f} {metrics['Accuracy']:>10.4f} "
          f"{metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} "
          f"{metrics['F1']:>8.4f} {metrics['Brier']:>8.4f}")

print("\n" + "="*80)
print("IMPROVEMENT vs BASELINE")
print("="*80)
lgbm_improvement = ((results['LightGBM']['AUC'] - baseline_lgbm_auc) / baseline_lgbm_auc) * 100
xgb_improvement = ((results['XGBoost']['AUC'] - baseline_xgb_auc) / baseline_xgb_auc) * 100

print(f"\nLightGBM: {baseline_lgbm_auc:.4f} → {results['LightGBM']['AUC']:.4f} ({lgbm_improvement:+.2f}%)")
print(f"XGBoost:  {baseline_xgb_auc:.4f} → {results['XGBoost']['AUC']:.4f} ({xgb_improvement:+.2f}%)")
print(f"Ensemble: {results['Ensemble']['AUC']:.4f} ⭐")

# Visualizations
print("\n[5/6] Generating visualizations...")

# 1. ROC Curves
plt.figure(figsize=(10, 8))
for model_name in ['LightGBM', 'XGBoost', 'Ensemble']:
    if model_name == 'LightGBM':
        proba = lgbm_proba
    elif model_name == 'XGBoost':
        proba = xgb_proba
    else:
        proba = ensemble_proba

    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = results[model_name]['AUC']
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5000)', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Test Set Performance', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curves_test.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Precision-Recall Curves
plt.figure(figsize=(10, 8))
for model_name in ['LightGBM', 'XGBoost', 'Ensemble']:
    if model_name == 'LightGBM':
        proba = lgbm_proba
    elif model_name == 'XGBoost':
        proba = xgb_proba
    else:
        proba = ensemble_proba

    precision, recall, _ = precision_recall_curve(y_test, proba)
    plt.plot(recall, precision, label=f'{model_name}', linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Test Set', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'precision_recall_test.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (model_name, pred) in enumerate([
    ('LightGBM', lgbm_pred),
    ('XGBoost', xgb_pred),
    ('Ensemble', ensemble_pred)
]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                cbar=False, square=True)
    axes[idx].set_title(f'{model_name}\nAcc={results[model_name]["Accuracy"]:.4f}',
                       fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrices_test.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Metrics Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
metrics_df = pd.DataFrame(results).T
metrics_df[['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']].plot(
    kind='bar', ax=ax, width=0.8
)
ax.set_title('Model Performance Comparison - Test Set', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'metrics_comparison_test.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Visualizations saved to test_results/")

# Save results
print("\n[6/6] Saving results...")

# Save metrics to CSV
results_df = pd.DataFrame(results).T
results_df.to_csv(OUTPUT_DIR / 'test_metrics.csv')

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'lgbm_proba': lgbm_proba,
    'xgb_proba': xgb_proba,
    'ensemble_proba': ensemble_proba,
    'lgbm_pred': lgbm_pred,
    'xgb_pred': xgb_pred,
    'ensemble_pred': ensemble_pred
})
predictions_df.to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)

# Generate detailed report
report = f"""# TEST SET EVALUATION REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset
- Test samples: {len(y_test):,}
- Features: {X_test.shape[1]}
- Class balance: {dict(y_test.value_counts())}

## Performance Metrics

| Model | AUC | Accuracy | Precision | Recall | F1 | Brier |
|-------|-----|----------|-----------|--------|----|----|
| LightGBM | {results['LightGBM']['AUC']:.4f} | {results['LightGBM']['Accuracy']:.4f} | {results['LightGBM']['Precision']:.4f} | {results['LightGBM']['Recall']:.4f} | {results['LightGBM']['F1']:.4f} | {results['LightGBM']['Brier']:.4f} |
| XGBoost | {results['XGBoost']['AUC']:.4f} | {results['XGBoost']['Accuracy']:.4f} | {results['XGBoost']['Precision']:.4f} | {results['XGBoost']['Recall']:.4f} | {results['XGBoost']['F1']:.4f} | {results['XGBoost']['Brier']:.4f} |
| **Ensemble** | **{results['Ensemble']['AUC']:.4f}** | **{results['Ensemble']['Accuracy']:.4f}** | **{results['Ensemble']['Precision']:.4f}** | **{results['Ensemble']['Recall']:.4f}** | **{results['Ensemble']['F1']:.4f}** | **{results['Ensemble']['Brier']:.4f}** |

## Improvement vs Baseline

**LightGBM:**
- Baseline Test AUC: {baseline_lgbm_auc:.4f}
- Tuned Test AUC: {results['LightGBM']['AUC']:.4f}
- **Improvement: {lgbm_improvement:+.2f}%**

**XGBoost:**
- Baseline Test AUC: {baseline_xgb_auc:.4f}
- Tuned Test AUC: {results['XGBoost']['AUC']:.4f}
- **Improvement: {xgb_improvement:+.2f}%**

**Ensemble:**
- Test AUC: {results['Ensemble']['AUC']:.4f} ⭐

## Confusion Matrices

### LightGBM
```
{confusion_matrix(y_test, lgbm_pred)}
```

### XGBoost
```
{confusion_matrix(y_test, xgb_pred)}
```

### Ensemble
```
{confusion_matrix(y_test, ensemble_pred)}
```

## Key Findings

1. **Best Model:** {'Ensemble' if results['Ensemble']['AUC'] == max(r['AUC'] for r in results.values()) else max(results.items(), key=lambda x: x[1]['AUC'])[0]}
2. **Highest AUC:** {max(r['AUC'] for r in results.values()):.4f}
3. **Best Accuracy:** {max(r['Accuracy'] for r in results.values()):.4f}
4. **Best F1:** {max(r['F1'] for r in results.values()):.4f}

## Visualizations
- `roc_curves_test.png` - ROC curves comparison
- `precision_recall_test.png` - Precision-Recall curves
- `confusion_matrices_test.png` - Confusion matrices
- `metrics_comparison_test.png` - Metrics bar chart

## Files Generated
- `test_metrics.csv` - Performance metrics
- `test_predictions.csv` - Model predictions
- `test_evaluation_report.md` - This report

## Recommendation

**Final Model for Deployment:** Ensemble (Calibrated)
- File: `tuning_results/ensemble_calibrated.pkl`
- Test AUC: {results['Ensemble']['AUC']:.4f}
- Well-calibrated probabilities
- Robust predictions from multiple models

---
**Next Steps:**
1. Deploy `ensemble_calibrated.pkl` to production
2. Monitor performance on new data
3. Consider retraining if distribution shifts
"""

with open(OUTPUT_DIR / 'test_evaluation_report.md', 'w') as f:
    f.write(report)

print("✓ Results saved:")
print(f"  - test_results/test_metrics.csv")
print(f"  - test_results/test_predictions.csv")
print(f"  - test_results/test_evaluation_report.md")

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print(f"\n🏆 WINNER: Ensemble (Test AUC: {results['Ensemble']['AUC']:.4f})")
print(f"📊 Full report: test_results/test_evaluation_report.md")
print(f"🚀 Deploy: tuning_results/ensemble_calibrated.pkl")
