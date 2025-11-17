# CompeteML Quick Start Guide

Get started with CompeteML in 5 minutes.

## Installation

```bash
# Navigate to project directory
cd CompeteML

# Install dependencies
pip install -r requirements.txt
```

## Your First Run

### 1. Prepare Your Data

Place your competition data in the `data/raw/` folder:
```
data/
└── raw/
    ├── train.csv
    └── test.csv
```

### 2. Run the Pipeline

**Option A: Quick Test (5 minutes)**
```bash
python main.py run \
  --train data/raw/train.csv \
  --test data/raw/test.csv \
  --preset quick
```

**Option B: Full Run (1 hour)**
```bash
python main.py run \
  --train data/raw/train.csv \
  --test data/raw/test.csv
```

**Option C: Competition Mode (2 hours)**
```bash
python main.py run \
  --train data/raw/train.csv \
  --test data/raw/test.csv \
  --preset competition
```

### 3. Get Your Results

After running, check `outputs/<run_id>/`:

```
outputs/20250114_143022/
├── submissions/
│   └── submission_20250114_143022.csv    ← Upload this to competition!
├── recipe.txt                            ← See what was done
└── logs/
    └── competeml_20250114_143022.log    ← Detailed information
```

## Common Scenarios

### Specify Target and ID Columns

```bash
python main.py run \
  --train data/raw/train.csv \
  --test data/raw/test.csv \
  --target price \
  --id-col id
```

### Custom Time Limit

```bash
python main.py run \
  --train data/raw/train.csv \
  --test data/raw/test.csv \
  --time-limit 7200  # 2 hours in seconds
```

### Explore Dataset First

```bash
python main.py explore --train data/raw/train.csv
```

This shows:
- Dataset shape
- Column types
- Missing values
- First few rows

## Understanding Output

### Submission File

The submission file is ready to upload to your competition:
```csv
id,target
1,0.532
2,0.891
3,0.234
...
```

### Recipe File

Shows exactly what CompeteML did:
```
STEPS PERFORMED:
1. Loaded data and detected problem type
2. Validated data quality
3. Preprocessed data (handled missing, encoded, scaled)
4. Trained models using autogluon
5. Evaluated models and generated predictions
6. Created submission file
```

### Log File

Contains detailed information:
- Data statistics
- Preprocessing decisions
- Model performance
- Feature importance
- Warnings/errors

## Configuration Presets

### Quick (`--preset quick`)
- **Time**: 5 minutes
- **Use Case**: Test that everything works
- **Quality**: Low
- **Features**: Basic preprocessing only

### Default (`--preset default`)
- **Time**: 1 hour
- **Use Case**: Baseline performance
- **Quality**: Medium
- **Features**: Full preprocessing + auto features

### Competition (`--preset competition`)
- **Time**: 2 hours
- **Use Case**: Maximum performance
- **Quality**: Best
- **Features**: Everything enabled + advanced ensembling

## Typical Workflow

```bash
# 1. Explore data (1 minute)
python main.py explore --train data/raw/train.csv

# 2. Quick test (5 minutes)
python main.py run \
  --train data/raw/train.csv \
  --test data/raw/test.csv \
  --preset quick

# 3. Check outputs work
cat outputs/latest/recipe.txt

# 4. Full run (1 hour)
python main.py run \
  --train data/raw/train.csv \
  --test data/raw/test.csv

# 5. Submit!
# Upload: outputs/<run_id>/submissions/submission_*.csv

# 6. Competition run for best score (2+ hours)
python main.py run \
  --train data/raw/train.csv \
  --test data/raw/test.csv \
  --preset competition
```

## Troubleshooting

### "Module not found" Error
```bash
# Reinstall requirements
pip install -r requirements.txt
```

### "AutoGluon installation failed"
```bash
# Install AutoGluon separately (may take 10-20 minutes)
pip install autogluon
```

### Pipeline Failed
1. Check logs: `outputs/<run_id>/logs/*.log`
2. Check recipe: `outputs/<run_id>/recipe.txt`
3. Look for error messages

### Out of Memory
- Use `--preset quick` for less memory
- Reduce `time_limit`
- Run on machine with more RAM

## Next Steps

1. **Read the full README.md** for advanced usage
2. **Check configuration files** in `configs/` to customize behavior
3. **Review logs and recipes** to understand what works
4. **Experiment with settings** to optimize for your competition

## Quick Reference

```bash
# Run pipeline
python main.py run --train <train> --test <test> [OPTIONS]

# Explore data
python main.py explore --train <train>

# System info
python main.py info

# Help
python main.py --help
python main.py run --help
```

## Example: Titanic Competition

```bash
# Download Titanic data from Kaggle first

# Run CompeteML
python main.py run \
  --train data/raw/titanic_train.csv \
  --test data/raw/titanic_test.csv \
  --target Survived \
  --id-col PassengerId \
  --preset competition

# Submit: outputs/<run_id>/submissions/submission_*.csv to Kaggle
```

---

## Competition Mode

**For actual competitions, use:**
```bash
python main.py run \
  --train competition_train.csv \
  --test competition_test.csv \
  --preset competition \
  --time-limit 7200  # 2 hours
```

**What competition mode does:**
- ✓ Creates interaction & polynomial features
- ✓ Uses best quality AutoGluon preset
- ✓ 8-fold bagging + 2-level stacking
- ✓ Maximizes performance

**See COMPETITION_GUIDE.md for full competition workflow.**

---

**That's it! You're ready to compete.** 🚀
