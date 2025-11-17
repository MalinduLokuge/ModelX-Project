# Model Artifacts Checksums

**Generated**: November 17, 2025  
**Purpose**: Verify integrity of model files and artifacts

Use these SHA256 checksums to verify that model files haven't been corrupted or tampered with.

---

## AutoGluon Model

### Main Predictor Directory
**Path**: `outputs/models/autogluon_optimized/`  
**Total Size**: 9,237.63 MB (~9.2 GB)

| File | SHA256 Checksum | Size (MB) |
|------|----------------|-----------|
| `learner.pkl` | `FB8E3297919529757C8518869A2A640A83A5EB6989516A484F52751E94858CF9` | N/A |
| `predictor.pkl` | (Included in directory) | N/A |
| `metadata.json` | (Included in directory) | <1 MB |
| `models/` | (42 model files) | ~9,200 MB |

**Note**: AutoGluon saves models as a directory structure. Verify the entire directory using:

```powershell
# Windows PowerShell
Get-ChildItem -Path "outputs\models\autogluon_optimized" -Recurse -File | 
  Get-FileHash -Algorithm SHA256 | 
  Select-Object Hash, @{Name="Path";Expression={$_.Path.Replace($PWD,".") }}
```

```bash
# Linux/Mac
find outputs/models/autogluon_optimized -type f -exec sha256sum {} \;
```

---

## Manual Models

### Individual Model Files
**Path**: `outputs/manual_models/`

| Model File | SHA256 Checksum | Size (MB) |
|-----------|----------------|-----------|
| `LightGBM_Tuned.pkl` | `886407FED14DC3B723D82CA1932F0B135F1324B9ADEB53ACD0161E765...` | <10 MB |
| `LightGBM_Default.pkl` | `D8D2157FE22AE24B9F399E99C496B45B559FEBDBE1CA1B3002A424BD6...` | <10 MB |
| `XGBoost_Tuned.pkl` | `CBD3927E9A59FEEE71BED25343D45715FA95E64D05B9EB9AAB03C2EE6...` | <10 MB |
| `XGBoost_Default.pkl` | `DA2E7D4BF73478E06F8FC9CDDAEA59461DE06945D2388142D07FE7024...` | <10 MB |
| `RandomForest_Gini.pkl` | `1C23B4C2FBAFE945B5000E1559D6D21C2D755F729053E9370B4B835F3...` | <10 MB |
| `RandomForest_Entropy.pkl` | `FC7268D0FBE5A9E38454EF61E4C254FB07CB89954118E92F55F705786...` | <10 MB |
| `ExtraTrees.pkl` | `8FB059C369AF2BCD32E2AF64757E9B10F2FFF390D8A24FA971AFD67ED...` | <10 MB |
| `LogisticRegression.pkl` | `4D06F4E9FB2B731CD39E25A8458EF9B7C8857571050D7AFC85C6B039D...` | <10 MB |

**Total Manual Models Size**: ~60-80 MB (all 8 models combined)

---

## Preprocessing Pipeline

**Path**: `outputs/dementia_preprocessed/preprocessing_pipeline.pkl`

| File | SHA256 Checksum | Size (MB) |
|------|----------------|-----------|
| `preprocessing_pipeline.pkl` | `DAABC8C2AB193CA162DCB07293EF97AA9684DE5361DED5EF410AA5C...` | <1 MB |

---

## Supporting Artifacts

### Feature Importance Files

**Path**: `outputs/manual_models/`

| File | SHA256 | Size |
|------|--------|------|
| `fi_lgbm_tuned.csv` | (Compute if needed) | <1 KB |
| `fi_lgbm_default.csv` | (Compute if needed) | <1 KB |
| `fi_rf_gini.csv` | (Compute if needed) | <1 KB |
| `fi_xgb_default.csv` | (Compute if needed) | <1 KB |

### Model Comparison

**Path**: `outputs/manual_models/model_comparison.csv`

| File | SHA256 | Size |
|------|--------|------|
| `model_comparison.csv` | (Compute if needed) | <1 KB |

### XAI Artifacts

**Path**: `outputs/xai/`

| File | SHA256 | Size |
|------|--------|------|
| `xai_summary.json` | (Compute if needed) | <5 KB |
| `autogluon_importance.png` | (Compute if needed) | ~50 KB |
| `pdp_autogluon.png` | (Compute if needed) | ~100 KB |

---

## Test Data

**Path**: `data/test/`

| File | SHA256 | Size |
|------|--------|------|
| `X_test.csv` | (Compute if needed) | ~10-20 MB |
| `y_test.csv` | (Compute if needed) | <1 MB |

---

## Verification Commands

### Verify Single File (PowerShell)

```powershell
Get-FileHash -Path "outputs\manual_models\LightGBM_Tuned.pkl" -Algorithm SHA256 | 
  Select-Object Algorithm, Hash
```

### Verify All Manual Models (PowerShell)

```powershell
Get-ChildItem -Path "outputs\manual_models\*.pkl" | 
  ForEach-Object { 
    $hash = (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash
    [PSCustomObject]@{
      File = $_.Name
      SHA256 = $hash
      SizeMB = [math]::Round($_.Length / 1MB, 2)
    }
  } | Format-Table -AutoSize
```

### Verify AutoGluon Directory (PowerShell)

```powershell
$totalSize = (Get-ChildItem -Path "outputs\models\autogluon_optimized" -Recurse -File | 
  Measure-Object -Property Length -Sum).Sum / 1MB

Write-Host "Total Size: $([math]::Round($totalSize, 2)) MB"
```

### Verify File Integrity (Python)

```python
import hashlib
from pathlib import Path

def compute_checksum(filepath):
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Verify a file
file_path = "outputs/manual_models/LightGBM_Tuned.pkl"
checksum = compute_checksum(file_path)
print(f"SHA256: {checksum}")

# Compare with expected
expected = "886407FED14DC3B723D82CA1932F0B135F1324B9ADEB53ACD0161E765..."
if checksum.upper() == expected.upper():
    print("✓ Checksum verified")
else:
    print("✗ Checksum mismatch!")
```

---

## Notes

1. **Checksum Truncation**: Some checksums are truncated in tables for readability. Use full checksums for verification.

2. **AutoGluon Directory**: AutoGluon models are saved as directories with multiple files. Verify the entire directory or individual critical files (learner.pkl, predictor.pkl).

3. **Size Variations**: Model sizes may vary slightly depending on:
   - Compression settings (save_space=True)
   - Python/library versions
   - Pickle protocol version

4. **Regenerating Checksums**: If you retrain models or update artifacts, regenerate checksums using the provided commands.

5. **Security**: Store checksums separately from model files in production. Verify checksums before loading models in sensitive environments.

---

## Quick Verification Script

```python
#!/usr/bin/env python
"""Quick verification script for all model artifacts."""

import hashlib
from pathlib import Path

EXPECTED_CHECKSUMS = {
    "outputs/manual_models/LightGBM_Tuned.pkl": "886407FED14DC3B723D82CA1932F0B135F1324B9ADEB53ACD0161E765...",
    "outputs/manual_models/LightGBM_Default.pkl": "D8D2157FE22AE24B9F399E99C496B45B559FEBDBE1CA1B3002A424BD6...",
    # Add more as needed
}

def verify_artifacts():
    """Verify all artifacts against expected checksums."""
    results = []
    
    for filepath, expected in EXPECTED_CHECKSUMS.items():
        path = Path(filepath)
        
        if not path.exists():
            results.append((filepath, "MISSING", None))
            continue
        
        # Compute checksum
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        actual = sha256_hash.hexdigest().upper()
        expected_full = expected.upper()
        
        # Compare (handle truncated expected checksums)
        if expected_full.endswith("..."):
            match = actual.startswith(expected_full[:-3])
        else:
            match = actual == expected_full
        
        status = "✓ VERIFIED" if match else "✗ MISMATCH"
        results.append((filepath, status, actual[:16] + "..."))
    
    # Print results
    print("\nArtifact Verification Results:")
    print("-" * 80)
    for filepath, status, checksum in results:
        print(f"{status:12} {filepath:50} {checksum or ''}")
    
    # Summary
    verified = sum(1 for _, s, _ in results if s == "✓ VERIFIED")
    total = len(results)
    print("-" * 80)
    print(f"Summary: {verified}/{total} artifacts verified")
    
    return verified == total

if __name__ == "__main__":
    all_verified = verify_artifacts()
    exit(0 if all_verified else 1)
```

---

**Last Updated**: November 17, 2025  
**Checksum Algorithm**: SHA256  
**Tools**: PowerShell 5.1, Python 3.11
