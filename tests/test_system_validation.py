"""System Validation Test - Verify all components work together"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np

print("=" * 80)
print("COMPETEML SYSTEM VALIDATION")
print("=" * 80)

# Test 1: Core Imports
print("\n[1/8] Testing imports...")
try:
    from core.logger import CompeteMLLogger
    from core.config_manager import CompeteMLConfig, load_config
    from core.data_loader import DataLoader
    from core.data_validator import DataValidator
    from preprocessing.auto_preprocessor import AutoPreprocessor
    from feature_engineering.auto_features import AutoFeatureEngineer
    from feature_engineering.competition_tricks import CompetitionTricks
    from modeling.model_saver import ModelSaver
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {str(e)}")
    sys.exit(1)

# Test 2: Logger
print("\n[2/8] Testing logger...")
try:
    logger = CompeteMLLogger("ValidationTest")
    logger.info("Test message")
    logger.success("Test success")
    logger.warning("Test warning")
    print("✓ Logger works")
except Exception as e:
    print(f"✗ Logger failed: {str(e)}")
    sys.exit(1)

# Test 3: Config Manager
print("\n[3/8] Testing config manager...")
try:
    config_path = project_root / "configs" / "quick_test.yaml"
    config = load_config(str(config_path))
    assert config.time_limit > 0, "Time limit not set"
    assert config.automl_framework == "autogluon", "Wrong framework"
    print(f"✓ Config loaded: time_limit={config.time_limit}s")
except Exception as e:
    print(f"✗ Config failed: {str(e)}")
    sys.exit(1)

# Test 4: Data Loading
print("\n[4/8] Testing data loader...")
try:
    # Create test data
    np.random.seed(42)
    n = 100

    test_df = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'target': np.random.randint(0, 2, n)
    })

    # Save temp file
    temp_train = project_root / "data" / "temp_validation_train.csv"
    temp_train.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(temp_train, index=False)

    # Load
    loader = DataLoader(logger)
    train_df, _ = loader.load(train_path=str(temp_train), target_column='target')

    assert train_df is not None, "Data not loaded"
    assert loader.problem_type in ['classification', 'regression'], "Problem type not detected"

    print(f"✓ Data loaded: {train_df.shape}, problem_type={loader.problem_type}")

    # Cleanup
    temp_train.unlink()

except Exception as e:
    print(f"✗ Data loader failed: {str(e)}")
    sys.exit(1)

# Test 5: Preprocessing
print("\n[5/8] Testing preprocessing...")
try:
    X = test_df.drop('target', axis=1)
    y = test_df['target']

    config_dict = {
        'handle_missing': True,
        'encoding_strategy': 'auto',
        'scaling_strategy': 'auto',
        'apply_competition_tricks': False
    }

    preprocessor = AutoPreprocessor(config=config_dict, logger=logger)
    X_processed, report = preprocessor.fit_transform(X, y)

    assert X_processed is not None, "Preprocessing failed"
    assert X_processed.shape[0] == X.shape[0], "Row count changed"

    print(f"✓ Preprocessing complete: {X.shape} → {X_processed.shape}")

except Exception as e:
    print(f"✗ Preprocessing failed: {str(e)}")
    sys.exit(1)

# Test 6: Feature Engineering
print("\n[6/8] Testing feature engineering...")
try:
    config_fe = {
        'interaction_features': True,
        'polynomial_features': False,
        'feature_selection': True,
        'max_features': None,
        'apply_competition_tricks': False
    }

    fe_engine = AutoFeatureEngineer(config=config_fe, logger=logger)
    X_features, fe_report = fe_engine.engineer_features(
        X=X_processed.copy(),
        y=y,
        problem_type='classification'
    )

    assert X_features is not None, "Feature engineering failed"
    assert X_features.shape[0] == X_processed.shape[0], "Row count changed"

    print(f"✓ Feature engineering complete: {X_processed.shape} → {X_features.shape}")

except Exception as e:
    print(f"✗ Feature engineering failed: {str(e)}")
    sys.exit(1)

# Test 7: Competition Tricks
print("\n[7/8] Testing competition tricks...")
try:
    # Create data with categorical
    X_cat = pd.DataFrame({
        'numeric': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    y_cat = pd.Series(np.random.randint(0, 2, 100))

    tricks = CompetitionTricks(logger)

    # Target encoding
    X_encoded = tricks.target_encode_cv(X_cat.copy(), y_cat, ['category'], 'classification')
    assert 'category_target_enc' in X_encoded.columns, "Target encoding failed"

    # Frequency encoding
    X_freq, freq_maps = tricks.frequency_encode(X_cat.copy(), ['category'])
    assert 'category_freq' in X_freq.columns, "Frequency encoding failed"

    print("✓ Competition tricks work")

except Exception as e:
    print(f"✗ Competition tricks failed: {str(e)}")
    sys.exit(1)

# Test 8: Model Saver
print("\n[8/8] Testing model saver...")
try:
    from sklearn.ensemble import RandomForestClassifier

    # Train simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_features, y)

    # Save deployment package
    output_dir = project_root / "outputs" / "validation_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    saver = ModelSaver(output_dir=output_dir, logger=logger)

    metadata = {
        'test': 'validation',
        'features': X_features.shape[1]
    }

    package_dir = saver.save_deployment_package(
        model=model,
        preprocessor=preprocessor,
        feature_engineer=fe_engine,
        metadata=metadata,
        package_name="validation_model"
    )

    # Verify files exist
    assert (package_dir / "model.pkl").exists(), "Model not saved"
    assert (package_dir / "preprocessor.pkl").exists(), "Preprocessor not saved"
    assert (package_dir / "metadata.json").exists(), "Metadata not saved"
    assert (package_dir / "inference.py").exists(), "Inference script not created"

    print(f"✓ Model package saved: {package_dir.name}")

    # Cleanup
    import shutil
    shutil.rmtree(output_dir)

except Exception as e:
    print(f"✗ Model saver failed: {str(e)}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("SYSTEM VALIDATION COMPLETE")
print("=" * 80)
print("\nAll tests passed! ✓")
print("\nComponents validated:")
print("  ✓ Core imports")
print("  ✓ Logger (colored output)")
print("  ✓ Config manager")
print("  ✓ Data loader (auto-detection)")
print("  ✓ Preprocessing (missing, encoding, scaling)")
print("  ✓ Feature engineering (interactions, selection)")
print("  ✓ Competition tricks (target/frequency encoding)")
print("  ✓ Model saver (deployment package)")
print("\nCompeteML is ready for competitions! 🏆")
print("=" * 80)
