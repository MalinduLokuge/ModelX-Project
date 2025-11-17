# ModelX-Project 🏆

**ML Model for ModelX Dementia Risk Prediction Competition**

An automated machine learning system built specifically for predicting dementia risk using non-medical features.

## 🎉 **Production Model: 94.34% ROC-AUC Achieved!**

**AutoGluon AutoML Model - Production Ready**
- ✅ **Validation ROC-AUC: 94.34%** (WeightedEnsemble_L4)
- ✅ **+14.87 pp improvement** over best manual model (79.47%)
- ✅ **42 models trained** with 4-level stacking
- ✅ **1,299 rows/second** inference speed
- ✅ **132 engineered features** (112 original + 20 automated)
- 📊 **See `AUTOML_TRAINING_REPORT.md` for complete details**

**Quick Use:**
```python
from autogluon.tabular import TabularPredictor

# Load production model
predictor = TabularPredictor.load('outputs/models/autogluon_optimized/')

# Make predictions
predictions = predictor.predict(new_data)
probabilities = predictor.predict_proba(new_data)
```

---

## 🎯 What is This Project?

This is a **production-ready ML system** for dementia risk prediction achieving **94.34% ROC-AUC** using AutoML. The system handles the complete pipeline from raw data to trained models with explainability analysis.

**Key Features:**
- ✅ **Automated Pipeline**: One command runs everything
- ✅ **High Performance**: 94.34% ROC-AUC with AutoGluon (42 models, 4-level stacking)
- ✅ **Multiple Models**: Compare AutoML vs 8 traditional ML models
- ✅ **Explainability**: SHAP, LIME, feature importance analysis
- ✅ **Production Ready**: Complete documentation, checksums, deployment guides

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MalinduLokuge/ModelX-Project.git
cd ModelX-Project

# Install dependencies
pip install -r requirements.txt
```

### 🚀 Run Complete Pipeline (Recommended)

**Option 1: Full Pipeline (Preprocessing + Training)**
```bash
# Run everything from scratch (~45 minutes)
python run_complete_pipeline.py
```

**Option 2: Quick Test (5 minutes)**
```bash
# Fast validation run
python run_complete_pipeline.py --quick-test
```

**Option 3: Skip Preprocessing (Training Only)**
```bash
# If data already preprocessed (~30 minutes)
python run_complete_pipeline.py --skip-preprocessing
```

### 📊 Individual Components

**If you want to run specific parts:**

```bash
# 1. Data Preprocessing Only
python run_preprocessing_simple.py

# 2. Train Manual Models Only (8 models)
python train_manual_lowmem.py

# 3. Train AutoML Model Only (42 models, 4-level stacking)
python train_autogluon_optimized.py

# 4. Generate Model Comparison Report
python model_comparison_final.py

# 5. Generate Explainability Analysis
python generate_xai_analysis.py
```

### 🎯 What Each Script Does

| Script | Time | Output | Description |
|--------|------|--------|-------------|
| `run_complete_pipeline.py` | 45 min | All models + reports | **Main entry point** - runs everything |
| `run_preprocessing_simple.py` | 5 min | `data/train/`, `data/test/` | Data cleaning, splitting, balancing |
| `train_manual_lowmem.py` | 10 min | `outputs/manual_models/` | 8 traditional ML models |
| `train_autogluon_optimized.py` | 30 min | `outputs/models/autogluon_optimized/` | AutoML with 94.34% ROC-AUC |
| `model_comparison_final.py` | 2 min | `model_comparison_results/` | Compare all models |
| `generate_xai_analysis.py` | 5 min | `outputs/xai/` | Feature importance, SHAP, LIME |

### ✅ Pipeline Output

After running `run_complete_pipeline.py`, you'll get:

```
├── data/
│   ├── train/              # Preprocessed training data
│   │   ├── X_train_balanced.csv
│   │   └── y_train_balanced.csv
│   └── test/               # Preprocessed test data
│       ├── X_test.csv
│       └── y_test.csv
│
├── outputs/
│   ├── manual_models/      # 8 traditional ML models
│   │   ├── LightGBM_Tuned.pkl
│   │   ├── XGBoost_Tuned.pkl
│   │   └── model_comparison.csv
│   ├── models/
│   │   └── autogluon_optimized/  # Production AutoML model (94.34% ROC-AUC)
│   └── xai/                # Explainability analysis
│       ├── XAI_DOCUMENTATION.md
│       └── *.png (visualizations)
│
├── model_comparison_results/  # Model comparison reports
│   ├── roc_curves_test.png
│   ├── confusion_matrices_test.png
│   └── metrics_comparison_test.png
│
└── Documentation/
    ├── MODEL_README.md              # Complete model documentation
    ├── AUTOML_TRAINING_REPORT.md    # Training details
    ├── USAGE_SNIPPETS.md            # Code examples
    └── CHECKSUMS.md                 # Model verification
```

## 📁 Project Structure

```
CompeteML/
├── src/
│   ├── core/              # Core system (orchestrator, config, logger)
│   ├── preprocessing/     # Data cleaning & preprocessing
│   ├── modeling/          # AutoML training (AutoGluon)
│   ├── evaluation/        # Metrics & evaluation
│   └── reporting/         # Submission & report generation
│
├── configs/               # Configuration presets
│   ├── default.yaml      # 1-hour balanced run
│   ├── competition.yaml  # 2-hour high-performance
│   └── quick_test.yaml   # 5-minute test
│
├── data/                 # Your datasets
│   ├── raw/              # Original competition data
│   └── sample/           # Sample datasets for testing
│
├── outputs/              # All results
│   └── <run_id>/
│       ├── submissions/  # Competition submission files
│       ├── models/       # Trained models
│       ├── recipe.txt    # What was done
│       └── logs/         # Execution logs
│
├── main.py              # CLI entry point
└── requirements.txt     # Dependencies
```

## 🎮 Usage

### Basic Commands

```bash
# Run pipeline
python main.py run --train <train.csv> --test <test.csv>

# Quick exploration
python main.py explore --train <train.csv>

# System info
python main.py info
```

### Advanced Options

```bash
python main.py run \
  --train data/train.csv \
  --test data/test.csv \
  --target price \
  --id-col id \
  --time-limit 7200 \
  --output-dir my_results \
  --preset competition
```

### Using Custom Config

```bash
# Create custom config
cp configs/default.yaml my_config.yaml
# Edit my_config.yaml...

# Run with custom config
python main.py run --train data/train.csv --config my_config.yaml
```

## 🔧 Configuration Presets

### Quick Test (5 minutes)
- Fast preprocessing only
- No feature engineering
- Quick model training
- Perfect for testing

```bash
--preset quick
```

### Default (1 hour)
- Full preprocessing
- Auto feature engineering
- Medium quality models
- Balanced speed/performance

```bash
--preset default
```

### Competition (2 hours)
- Full preprocessing
- Advanced feature engineering
- Best quality models (8-fold bagging, 2-level stacking)
- Maximum performance

```bash
--preset competition
```

## 🤖 What CompeteML Does Automatically

### 1. Smart Data Loading
- Auto-detects file format (CSV, Excel, Parquet, JSON)
- Auto-detects target column
- Auto-detects problem type (classification/regression)
- Auto-detects ID columns

### 2. Data Preprocessing
- **Missing Values**: Intelligent imputation based on missing %
- **Encoding**: Auto-selects encoding strategy by cardinality
  - Low cardinality (≤10): One-hot encoding
  - Medium (≤50): Label encoding
  - High (>50): Target encoding
- **Scaling**: Auto-selects scaler based on distribution
  - Outliers present: RobustScaler
  - Bounded [0,1]: MinMaxScaler
  - Default: StandardScaler

### 3. Feature Engineering (Optional)
- Interaction features
- Polynomial features
- Time-based features
- Text features (TF-IDF)
- Feature selection

### 4. Model Training
- **Primary**: AutoGluon (state-of-the-art AutoML)
- Multi-layer stacking
- Bagging for stability
- Automatic hyperparameter tuning

### 5. Outputs
- Competition submission file (ready to upload)
- Trained models (saved for later use)
- Recipe file (showing exactly what was done)
- Execution logs

## 📊 Output Files

After running, check `outputs/<run_id>/`:

```
outputs/20250114_143022/
├── submissions/
│   └── submission_20250114_143022.csv    # Ready to upload!
├── ag_models/                            # Trained models
├── recipe.txt                            # What was done
└── logs/
    └── competeml_20250114_143022.log    # Detailed logs
```

## 🏅 Competition Workflow

**Standard Competition Workflow:**

```bash
# 1. Quick test (5 min) - verify everything works
python main.py run --train train.csv --test test.csv --preset quick

# 2. Default run (1 hour) - get baseline
python main.py run --train train.csv --test test.csv

# 3. Competition run (2+ hours) - maximize performance
python main.py run --train train.csv --test test.csv --preset competition

# 4. Submit outputs/latest/submissions/submission_*.csv to competition
```

## 🔬 Understanding Your Results

### Recipe File
Shows exactly what the system did:
```
================================================================================
COMPETEML PIPELINE RECIPE
================================================================================
Run ID: 20250114_143022
Date: 2025-01-14 14:30:22

STEPS PERFORMED:
--------------------------------------------------------------------------------
1. Loaded data and detected problem type
2. Validated data quality
3. Preprocessed data (handled missing, encoded, scaled)
4. Trained models using autogluon
5. Evaluated models and generated predictions
6. Created submission file: submission_20250114_143022.csv
================================================================================
```

### Logs
Check detailed logs in `outputs/<run_id>/logs/` for:
- Data statistics
- Preprocessing decisions
- Model performance
- Feature importance
- Warnings and errors

## 🎓 Learning Mode

Want to learn what works? CompeteML shows you:

1. **Recipe files**: Exactly what was done
2. **Logs**: Why decisions were made
3. **Model leaderboard**: Which models performed best
4. **Feature importance**: Which features matter
5. **Code templates**: Manual implementation templates
6. **Deployment package**: Ready-to-deploy model + inference script

### Manual Mode Templates

Found in `templates/code/`:
- `preprocessing_template.py` - Replicate preprocessing manually
- `feature_engineering_template.py` - Create features manually
- `training_template.py` - Train models manually

Each template shows what auto mode did and how to replicate it.

### Utility Scripts

Found in `scripts/`:
- `setup.bat` - One-time setup
- `quick_test.bat <data>` - 5-minute validation
- `competition_run.bat <train> <test>` - Full competition run

### Deployment Package

After training, find in `outputs/<run_id>/model_<run_id>/`:
- `model.pkl` - Trained model
- `preprocessor.pkl` - Preprocessing pipeline
- `feature_engineer.pkl` - Feature engineering pipeline
- `inference.py` - Ready-to-use prediction script
- `metadata.json` - Model info and metrics

Use for deployment:
```python
from outputs.model_20250114_143022.inference import ModelPredictor

predictor = ModelPredictor('outputs/model_20250114_143022')
predictions = predictor.predict(new_data)
```

Review these to understand the automated process, then switch to manual mode when needed.

## 🛠️ Requirements

### Core
- Python 3.8+
- pandas, numpy, scikit-learn
- AutoGluon (primary AutoML)

### Optional
- Optuna (hyperparameter tuning)
- PyCaret (backup AutoML)
- ydata-profiling (EDA reports)

Install all:
```bash
pip install -r requirements.txt
```

## 📖 Configuration Options

Key configuration options (edit `configs/*.yaml`):

```yaml
# Time
time_limit: 3600  # seconds

# Preprocessing
handle_missing: true
handle_outliers: true
scaling_strategy: auto  # auto, standard, minmax, robust, none
encoding_strategy: auto  # auto, onehot, target, ordinal

# Feature Engineering
auto_features: true
interaction_features: true
polynomial_features: false

# Modeling
automl_framework: autogluon
ag_preset: medium_quality  # best_quality, high_quality, medium_quality
ag_num_bag_folds: 5
ag_num_stack_levels: 1

# Output
generate_submission: true
generate_recipe: true
```

## 🚀 Tips for Winning

1. **Start with quick test**: Verify everything works (5 min)
2. **Run default mode**: Get baseline (1 hour)
3. **Check recipe & logs**: Understand what worked
4. **Run competition mode**: Maximize performance (2+ hours)
5. **Submit and iterate**: Use feedback to improve

## 📝 License

MIT License - feel free to use for competitions and learning!

## 🤝 Contributing

Contributions welcome! This is a learning project designed to help people win ML competitions.

## 📧 Support

- Check logs in `outputs/<run_id>/logs/`
- Review recipe in `outputs/<run_id>/recipe.txt`
- Open an issue with your question

---

**Built for the ModelX dementia risk prediction competition.** 🏆
