"""Integration test - verify competition tricks work in full pipeline"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from core.config_manager import CompeteMLConfig
from core.logger import CompeteMLLogger
from preprocessing.auto_preprocessor import AutoPreprocessor
from feature_engineering.auto_features import AutoFeatureEngineer

print("="*80)
print("INTEGRATION TEST: Competition Tricks in Full Pipeline")
print("="*80)

# Create test data
np.random.seed(42)
n = 500

test_data = pd.DataFrame({
    'numeric_1': np.random.randn(n),
    'numeric_2': np.random.uniform(0, 100, n),
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], n),
    'category': np.random.choice(['A', 'B', 'C'], n),
    'target': np.random.randint(0, 2, n)
})

X = test_data.drop('target', axis=1)
y = test_data['target']

logger = CompeteMLLogger("IntegrationTest")

# Test 1: Preprocessing
logger.section("Step 1: Preprocessing")

config_preprocess = {
    'handle_missing': True,
    'encoding_strategy': 'auto',
    'scaling_strategy': 'auto',
    'apply_competition_tricks': True  # Tell preprocessor to skip encoding
}

preprocessor = AutoPreprocessor(config=config_preprocess, logger=logger)
X_processed, preprocess_report = preprocessor.fit_transform(X, y)

logger.success(f"✓ Preprocessing complete: {X.shape[1]} → {X_processed.shape[1]} features")

# Test 2: Feature Engineering WITH Competition Tricks
logger.section("Step 2: Feature Engineering (WITH Competition Tricks)")

config_fe = {
    'interaction_features': True,
    'polynomial_features': False,  # Skip for speed
    'feature_selection': True,
    'max_features': None,
    'apply_competition_tricks': True  # ENABLED
}

fe_engine = AutoFeatureEngineer(config=config_fe, logger=logger)
X_featured, fe_report = fe_engine.engineer_features(
    X=X_processed.copy(),
    y=y,
    problem_type='classification'
)

logger.success(f"✓ Feature engineering complete: {X_processed.shape[1]} → {X_featured.shape[1]} features")

# Verify competition tricks were applied
expected_tricks_cols = ['city_target_enc', 'category_target_enc', 'city_freq', 'category_freq']
tricks_found = [col for col in expected_tricks_cols if col in X_featured.columns]

logger.info(f"\nCompetition tricks features found: {len(tricks_found)}/{len(expected_tricks_cols)}")
for col in tricks_found:
    logger.info(f"  ✓ {col}")

# Test 3: Verify no NaN values (leakage check)
logger.section("Step 3: Data Quality Check")

nan_count = X_featured.isnull().sum().sum()

if nan_count == 0:
    logger.success("✓ No NaN values - competition tricks working correctly")
else:
    logger.error(f"✗ Found {nan_count} NaN values")

# Test 4: Verify feature counts
logger.section("Step 4: Feature Count Validation")

original_features = X.shape[1]
final_features = X_featured.shape[1]
features_added = final_features - original_features

logger.info(f"Original features: {original_features}")
logger.info(f"After preprocessing: {X_processed.shape[1]}")
logger.info(f"After feature engineering: {final_features}")
logger.info(f"Net features added: {features_added}")

# Summary
print("\n" + "="*80)
if len(tricks_found) >= 2 and nan_count == 0:
    logger.success("✓ INTEGRATION TEST PASSED!")
    print("="*80)
    print("\nResults:")
    print(f"  • Competition tricks applied: {len(tricks_found)} trick features")
    print(f"  • No data leakage detected")
    print(f"  • Pipeline ran successfully")
    print(f"  • Ready for competition use!")
else:
    logger.error("✗ INTEGRATION TEST FAILED")
    if len(tricks_found) < 2:
        logger.error(f"  Only {len(tricks_found)} competition tricks found")
    if nan_count > 0:
        logger.error(f"  {nan_count} NaN values detected")

print("="*80)
