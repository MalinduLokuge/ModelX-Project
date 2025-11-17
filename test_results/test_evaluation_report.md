# TEST SET EVALUATION REPORT
Generated: 2025-11-17 18:07:44

## Dataset
- Test samples: 29,279
- Features: 112
- Class balance: {0: np.int64(20641), 1: np.int64(8638)}

## Performance Metrics

| Model | AUC | Accuracy | Precision | Recall | F1 | Brier |
|-------|-----|----------|-----------|--------|----|----|
| LightGBM | 0.8086 | 0.7666 | 0.6647 | 0.4212 | 0.5156 | 0.1553 |
| XGBoost | 0.8173 | 0.7711 | 0.6715 | 0.4385 | 0.5306 | 0.1521 |
| **Ensemble** | **0.8171** | **0.7589** | **0.8031** | **0.2422** | **0.3721** | **0.1594** |

## Improvement vs Baseline

**LightGBM:**
- Baseline Test AUC: 0.7947
- Tuned Test AUC: 0.8086
- **Improvement: +1.75%**

**XGBoost:**
- Baseline Test AUC: 0.7896
- Tuned Test AUC: 0.8173
- **Improvement: +3.51%**

**Ensemble:**
- Test AUC: 0.8171 ⭐

## Confusion Matrices

### LightGBM
```
[[18806  1835]
 [ 5000  3638]]
```

### XGBoost
```
[[18788  1853]
 [ 4850  3788]]
```

### Ensemble
```
[[20128   513]
 [ 6546  2092]]
```

## Key Findings

1. **Best Model:** XGBoost
2. **Highest AUC:** 0.8173
3. **Best Accuracy:** 0.7711
4. **Best F1:** 0.5306

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
- Test AUC: 0.8171
- Well-calibrated probabilities
- Robust predictions from multiple models

---
**Next Steps:**
1. Deploy `ensemble_calibrated.pkl` to production
2. Monitor performance on new data
3. Consider retraining if distribution shifts
