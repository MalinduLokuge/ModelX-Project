# CompeteML Competition Guide

**Win ML competitions fast with CompeteML.**

---

## Quick Competition Workflow

### 1. Before Competition (Practice)

```bash
# Test on sample data
python main.py run --train sample.csv --preset quick

# Verify system works
cat outputs/*/recipe.txt
```

### 2. During Competition

**First 10 Minutes:**
```bash
# Quick test to verify data loads
python main.py explore --train train.csv

# Check: target column, problem type, data quality
```

**Next 2-4 Hours:**
```bash
# Run competition mode
python main.py run \
  --train train.csv \
  --test test.csv \
  --preset competition \
  --target <target_col> \
  --id-col <id_col>

# Let it run. Go get coffee ☕
```

**Output Location:**
```
outputs/<run_id>/submissions/submission_*.csv ← UPLOAD THIS!
```

### 3. Review Results

**Check recipe:**
```bash
cat outputs/*/recipe.txt
```

Shows what worked:
- Features created
- Models tried
- Best model

**Check logs:**
```bash
cat outputs/*/logs/*.log
```

Details on decisions made.

---

## Configuration Presets

| Preset | Time | Use When | Features |
|--------|------|----------|----------|
| `quick` | 5 min | Testing | Basic only |
| `default` | 1 hour | Learning | Balanced |
| `competition` | 2 hours | **Winning** | **All features** |

**For competitions, always use:** `--preset competition`

---

## Time Management

**4-Hour Competition:**
- 0:00-0:10: Quick explore & verify
- 0:10-2:10: Run competition preset (2 hours)
- 2:10-3:00: Review, analyze results
- 3:00-3:45: Improve based on insights (manual tweaks)
- 3:45-4:00: Final submission

**1-Hour Competition:**
- Use `--preset default` instead
- Skip analysis, trust automation

---

## Common Issues

**Issue**: "Target column not found"
```bash
# Specify manually
python main.py run --train train.csv --target my_target
```

**Issue**: "Out of memory"
```bash
# Use quick preset to test first
python main.py run --train train.csv --preset quick
```

**Issue**: "Taking too long"
```bash
# Reduce time limit
python main.py run --train train.csv --time-limit 1800
```

---

## What Gets Created

**Competition-Winning Features:**
- Interaction features (A × B, A / B, A + B)
- Polynomial features (A², A³, √A)
- Statistical features (row mean, std, etc.)

**Best Models:**
- AutoGluon trains 10-20 models automatically
- Creates ensemble of best performers
- Uses stacking for extra performance

**Submission File:**
- `outputs/<run_id>/submissions/submission_*.csv`
- Ready to upload immediately
- ID column + predictions

---

## Advanced Tips

**1. Review Feature Importance:**
Check logs to see which features helped most.

**2. Learn From Recipe:**
See `recipe.txt` to understand what worked.

**3. Manual Improvements:**
If time allows, use insights to create custom features.

**4. Multiple Runs:**
Try different seeds:
```bash
# Run 1
python main.py run ... --seed 42

# Run 2
python main.py run ... --seed 123

# Ensemble predictions
```

---

## Competition Checklist

- [ ] Test system works (`--preset quick`)
- [ ] Verify data loads correctly (`explore`)
- [ ] Run competition preset
- [ ] Check submission file exists
- [ ] Upload before deadline
- [ ] Review recipe for learning

---

**Remember**: CompeteML handles:
✓ Feature engineering
✓ Model selection
✓ Hyperparameter tuning
✓ Ensembling

**You focus on**: Understanding results & strategy.

Good luck! 🏆
