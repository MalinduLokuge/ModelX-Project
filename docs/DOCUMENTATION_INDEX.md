# AutoGluon Model Documentation Index

Complete documentation package for reusing the dementia risk prediction AutoML pipeline.

---

## Quick Start

1. **Read First:** [STAKEHOLDER_SUMMARY.md](STAKEHOLDER_SUMMARY.md) - Executive overview
2. **Deploy:** [MODEL_DEPLOYMENT_GUIDE.md](MODEL_DEPLOYMENT_GUIDE.md) - Step-by-step deployment
3. **Integrate:** [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Code examples

---

## Documentation Files

### For Business Stakeholders
- **[STAKEHOLDER_SUMMARY.md](STAKEHOLDER_SUMMARY.md)** - Business value, use cases, ROI, risks

### For Engineers & Data Scientists
- **[MODEL_DEPLOYMENT_GUIDE.md](MODEL_DEPLOYMENT_GUIDE.md)** - How to load and use the model
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Step-by-step integration instructions
- **[INFERENCE_DOCUMENTATION.md](INFERENCE_DOCUMENTATION.md)** - How inference works internally
- **[PROJECT_TECHNICAL_DOCUMENT.md](PROJECT_TECHNICAL_DOCUMENT.md)** - Detailed artifact descriptions
- **[REQUIRED_FILES_CHECKLIST.md](REQUIRED_FILES_CHECKLIST.md)** - What files to copy

### For Model Understanding
- **[NODE_DOCUMENT.md](NODE_DOCUMENT.md)** - Complete model explanation & reasoning
- **[MODEL_REUSE_PHILOSOPHY.md](MODEL_REUSE_PHILOSOPHY.md)** - Best practices & common mistakes

---

## Model Quick Facts

- **Type:** Binary Classification (Dementia vs No Dementia)
- **Performance:** 97.09% ROC-AUC, 90.55% Accuracy
- **Features:** 113 non-medical features
- **Framework:** AutoGluon 1.4.0
- **Models:** 36-model ensemble (WeightedEnsemble_L3)
- **Size:** ~920 MB uncompressed, ~250 MB compressed

---

## Essential Commands

**Load Model:**
```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor.load("models/autogluon_production_lowmem/")
```

**Predict:**
```python
predictions = predictor.predict(new_data)
probabilities = predictor.predict_proba(new_data)
```

**Validate Schema:**
```python
expected_features = predictor.feature_metadata_in.get_features()
new_data = new_data[expected_features]
```

---

## Support

- **Technical Questions:** See troubleshooting sections in deployment guides
- **Integration Help:** Review [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Model Questions:** See [NODE_DOCUMENT.md](NODE_DOCUMENT.md)

---

**Generated:** 2025-11-17
**Model Version:** AutoGluon v1.4.0
