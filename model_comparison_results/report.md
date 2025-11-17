# MODEL COMPARISON REPORT

## WINNER: XGBoost (AUC: 0.8173)

### Performance:
| Metric | XGBoost | Ensemble | Difference |
|--------|---------|----------|------------|
| AUC | 0.8173 | 0.8171 | +0.01% |
| Accuracy | 0.7711 | 0.7589 | +1.22% |
| Precision | 0.6715 | 0.8031 | -13.16% |
| Recall | 0.4385 | 0.2422 | +19.63% |
| F1 | 0.5306 | 0.3721 | +15.84% |

### Trade-offs:
- **XGBoost:** Single model, 0.252s inference, AUC 0.8173
- **Ensemble:** 2-model stack + calibration, 0.299s inference, AUC 0.8171
- **Speed:** XGBoost is 1.2x faster

### Final Selection: XGBoost

**Rationale:**
1. **Performance:** AUC 0.8173 (Excellent: 0.8-0.9 range)
2. **Generalization:** Validation→Test consistent
3. **Interpretability:** Simple (single model)
4. **Speed:** 0.252s for 29,279 samples = 0.01ms/sample
5. **Deployment:** Easier (1 model)

### Test Set Performance (XGBoost):
- **Accuracy:** 0.7711 (77.1%)
- **Precision:** 0.6715 (67.2%)
- **Recall:** 0.4385 (43.9%)
- **F1-Score:** 0.5306
- **AUC-ROC:** 0.8173 ⭐

### Strengths (XGBoost):
- ✓ Simpler architecture
- ✓ Faster inference
- ✓ Easier deployment
- ✓ Excellent discrimination (AUC 0.8173)

### Limitations (XGBoost):
- ⚠ Single model bias
- ⚠ No ensemble robustness
- ⚠ Class imbalance (70.5% vs 29.5%)
- ⚠ Lower precision than ensemble
