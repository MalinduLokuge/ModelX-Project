"""Basic pipeline test without AutoGluon"""
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np

# Test imports
print("Testing imports...")
try:
    from core.logger import CompeteMLLogger
    print("✓ Logger import successful")
except Exception as e:
    print(f"✗ Logger import failed: {e}")
    sys.exit(1)

try:
    from core.config_manager import CompeteMLConfig, ConfigManager
    print("✓ Config Manager import successful")
except Exception as e:
    print(f"✗ Config Manager import failed: {e}")
    sys.exit(1)

try:
    from core.data_loader import DataLoader
    print("✓ Data Loader import successful")
except Exception as e:
    print(f"✗ Data Loader import failed: {e}")
    sys.exit(1)

try:
    from core.data_validator import DataValidator
    print("✓ Data Validator import successful")
except Exception as e:
    print(f"✗ Data Validator import failed: {e}")
    sys.exit(1)

try:
    from preprocessing.missing_handler import MissingValueHandler
    from preprocessing.encoder import CategoricalEncoder
    from preprocessing.scaler import FeatureScaler
    from preprocessing.auto_preprocessor import AutoPreprocessor
    print("✓ Preprocessing modules import successful")
except Exception as e:
    print(f"✗ Preprocessing import failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("All imports successful!")
print("="*80)

# Test logger
print("\nTesting Logger...")
logger = CompeteMLLogger("Test")
logger.info("Logger test message")
logger.section("TEST SECTION")
print("✓ Logger works")

# Test config
print("\nTesting Config Manager...")
config = CompeteMLConfig()
print(f"Default config mode: {config.mode}")
print(f"Default time_limit: {config.time_limit}")
print("✓ Config Manager works")

# Create test data
print("\nCreating test dataset...")
np.random.seed(42)
n = 100

test_data = pd.DataFrame({
    'id': range(1, n + 1),
    'numeric_1': np.random.randn(n),
    'numeric_2': np.random.uniform(0, 100, n),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n),
    'categorical_2': np.random.choice(['Low', 'Medium', 'High'], n),
    'target': np.random.choice([0, 1], n)
})

# Add missing values
test_data.loc[np.random.choice(n, 5), 'numeric_2'] = np.nan
test_data.loc[np.random.choice(n, 5), 'categorical_1'] = np.nan

print(f"Test data shape: {test_data.shape}")
print(f"Missing values: {test_data.isnull().sum().sum()}")

# Save test data
output_dir = Path(__file__).parent.parent / "data" / "sample"
output_dir.mkdir(parents=True, exist_ok=True)
test_data.to_csv(output_dir / "test_train.csv", index=False)
print(f"✓ Test data saved to: {output_dir / 'test_train.csv'}")

# Test data loader
print("\nTesting Data Loader...")
loader = DataLoader(logger)
train_df, test_df = loader.load(
    train_path=str(output_dir / "test_train.csv"),
    target_column='target',
    id_column='id'
)
print(f"✓ Data loaded: {train_df.shape}")
print(f"  Target column: {loader.target_column}")
print(f"  Problem type: {loader.problem_type}")
print(f"  ID column: {loader.id_column}")

# Test data validator
print("\nTesting Data Validator...")
validator = DataValidator(logger)
validation_results = validator.validate(train_df, target_column='target')
print(f"✓ Validation complete: Valid={validation_results['is_valid']}")

# Test preprocessing
print("\nTesting Preprocessing Pipeline...")
X, y = loader.get_X_y(train_df)
print(f"X shape: {X.shape}, y shape: {y.shape}")

config_dict = {
    'handle_missing': True,
    'encoding_strategy': 'auto',
    'scaling_strategy': 'auto'
}

preprocessor = AutoPreprocessor(config=config_dict, logger=logger)
X_processed, report = preprocessor.fit_transform(X, y)
print(f"✓ Preprocessing complete: {X.shape} -> {X_processed.shape}")
print(f"  Steps: {preprocessor.get_steps()}")

print("\n" + "="*80)
print("✓ ALL TESTS PASSED!")
print("="*80)
print("\nCore system is functional. Ready for full pipeline testing.")
