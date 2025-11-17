# Quick Use Guide - CompeteML

## 🚀 How to Use (Simple Steps)

### STEP 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### STEP 2: Put Your Data
Place your CSV files in `data/raw/` folder:
```
data/raw/
├── train.csv    ← Your training data
└── test.csv     ← Your test data (optional)
```

### STEP 3: Run the System
```bash
python main.py run --train data/raw/train.csv --test data/raw/test.csv
```

---

## 🎯 Target Variable - AUTO or MANUAL

### Option 1: AUTOMATIC (Recommended)
System will **automatically detect** the target column if:
- Column is named: `target`, `label`, `y`, or similar
- It's the last column in your CSV

Just run:
```bash
python main.py run --train data/raw/train.csv --test data/raw/test.csv
```

### Option 2: MANUAL (If auto-detection fails)
Specify target column name:
```bash
python main.py run --train data/raw/train.csv --test data/raw/test.csv --target "column_name"
```

Example:
```bash
python main.py run --train data/raw/train.csv --test data/raw/test.csv --target "price"
```

---

## 🔧 Feature Engineering

**YES - 100% AUTOMATIC!**

The system automatically:
- ✅ Creates interaction features (col1 × col2, col1 / col2)
- ✅ Creates polynomial features (x², x³, √x)
- ✅ Creates statistical features (mean, std, min, max)
- ✅ Applies competition tricks (target encoding, frequency encoding)
- ✅ Removes redundant features

**You don't need to do anything - it's all automatic!**

---

## 📁 Outputs - Where to Find Results

After running, check: `outputs/YYYYMMDD_HHMMSS/`

### Main Files:

```
outputs/20250115_143022/          ← Your run folder
│
├── submission.csv                ← UPLOAD THIS TO COMPETITION! ⭐
│
├── recipe.txt                    ← What the system did (read this!)
│
├── model_20250115_143022/        ← Deployment package
│   ├── model.pkl
│   ├── inference.py              ← Use for new predictions
│   └── metadata.json
│
└── logs/
    └── competeml_*.log          ← Detailed logs
```

### Key Outputs:

| File | What It Is | What To Do |
|------|------------|------------|
| **submission.csv** | Competition predictions | **Upload to Kaggle/competition** |
| **recipe.txt** | What system did | Read to understand pipeline |
| **model_*/inference.py** | Prediction script | Use for deployment |
| **logs/** | Execution logs | Check if errors |

---

## 📊 What You'll See (Example Output)

```
================================================================================
CompeteML Pipeline Started
================================================================================
Run ID: 20250115_143022
Mode: auto
Time limit: 3600s

[1/7] Loading data... (elapsed: 0.5s)
✓ Complete (2.3s)
Train shape: (10000, 25)
Target column: price
Problem type: regression

[2/7] Validating data... (elapsed: 2.8s)
✓ Complete (1.1s)
No critical issues found

[3/7] Preprocessing data... (elapsed: 3.9s)
✓ Complete (15.2s)
4 → 23 features (encoded categoricals)

[4/7] Engineering features... (elapsed: 19.1s)
✓ Complete (45.3s)
23 → 47 features (created interactions, polynomials)
Applied competition tricks: +4 features

[5/7] Training models... (elapsed: 64.4s)
✓ Complete (850.2s)
Best model: WeightedEnsemble_L2
CV Score: 0.8542

[6/7] Generating outputs... (elapsed: 914.6s)
✓ Complete (2.1s)
Submission saved: submission.csv
Model package saved: model_20250115_143022

================================================================================
Pipeline Completed Successfully
================================================================================
Total time: 916.7s (15.3 min)
All 6 steps completed!

✓ All outputs saved to: outputs/20250115_143022
```

---

## 🎯 Quick Modes

### 1. Quick Test (5 minutes)
```bash
python main.py run --train data/raw/train.csv --preset quick
```
Fast validation - check if data loads correctly

### 2. Default Run (1 hour)
```bash
python main.py run --train data/raw/train.csv --test data/raw/test.csv
```
Balanced run - good baseline

### 3. Competition Mode (2 hours)
```bash
python main.py run --train data/raw/train.csv --test data/raw/test.csv --preset competition
```
Maximum performance - best for competitions

---

## ❓ Common Questions

### Q: Do I need to specify target column?
**A:** Usually NO - system auto-detects. Only specify if auto-detection fails.

### Q: Do I need to do feature engineering?
**A:** NO - 100% automatic! System creates and selects best features.

### Q: Where is my submission file?
**A:** `outputs/<run_id>/submission.csv` - just upload this to competition!

### Q: How do I know what the system did?
**A:** Read `outputs/<run_id>/recipe.txt` - shows all steps performed.

### Q: Can I use the model for new predictions?
**A:** YES! Use `outputs/<run_id>/model_*/inference.py`

---

## 🔥 Fastest Way to Start

```bash
# 1. Put your data
data/raw/train.csv
data/raw/test.csv

# 2. Run this ONE command
python main.py run --train data/raw/train.csv --test data/raw/test.csv --preset competition

# 3. Upload submission
outputs/<run_id>/submission.csv
```

**That's it!** 🎯

---

## 📋 Full Example

```bash
# Your data
data/raw/house_prices_train.csv
data/raw/house_prices_test.csv

# Target column is "SalePrice"

# Run competition mode
python main.py run \
  --train data/raw/house_prices_train.csv \
  --test data/raw/house_prices_test.csv \
  --target SalePrice \
  --preset competition

# Wait 1-2 hours...

# Get submission
outputs/20250115_143022/submission.csv

# Upload to Kaggle!
```

---

**That's all you need to know to start!** 🚀
