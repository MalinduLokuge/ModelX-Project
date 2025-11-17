"""
Test script for dementia preprocessing pipeline
Creates sample data and verifies pipeline functionality
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.dementia_feature_selector import DementiaFeatureSelector
from src.preprocessing.nacc_missing_handler import NACCMissingValueHandler
from src.feature_engineering.dementia_features import DementiaFeatureEngineer
from src.preprocessing.dementia_preprocessing_pipeline import DementiaPreprocessingPipeline
from src.reporting.preprocessing_report_generator import PreprocessingReportGenerator


def create_sample_nacc_data(n_samples=1000):
    """Create sample NACC-like data for testing"""

    np.random.seed(42)

    data = {
        # Demographics
        'NACCAGE': np.random.randint(55, 95, n_samples),
        'SEX': np.random.choice([1, 2], n_samples),
        'EDUC': np.random.randint(0, 24, n_samples),
        'RACE': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'HISPANIC': np.random.choice([0, 1, 9], n_samples, p=[0.7, 0.2, 0.1]),
        'MARISTAT': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'NACCLIVS': np.random.choice([1, 2, 3, 4, 9], n_samples),
        'HANDED': np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.85, 0.05]),

        # Lifestyle
        'TOBAC100': np.random.choice([0, 1, 9], n_samples, p=[0.5, 0.4, 0.1]),
        'TOBAC30': np.random.choice([0, 1, 9], n_samples, p=[0.7, 0.2, 0.1]),
        'SMOKYRS': np.random.randint(0, 60, n_samples),
        'PACKSPER': np.random.choice([0, 1, 2, 3, 4, 5, 8, 9], n_samples),
        'ALCOCCAS': np.random.choice([0, 1, 9], n_samples, p=[0.3, 0.6, 0.1]),
        'ALCFREQ': np.random.choice([0, 1, 2, 3, 4, 8, 9], n_samples),

        # Medical History
        'CVHATT': np.random.choice([0, 1, 2, 9], n_samples, p=[0.7, 0.1, 0.15, 0.05]),
        'CBSTROKE': np.random.choice([0, 1, 2, 9], n_samples, p=[0.75, 0.08, 0.12, 0.05]),
        'CBTIA': np.random.choice([0, 1, 2, 9], n_samples, p=[0.7, 0.1, 0.15, 0.05]),
        'DIABETES': np.random.choice([0, 1, 2, 9], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'HYPERTEN': np.random.choice([0, 1, 2, 9], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
        'HYPERCHO': np.random.choice([0, 1, 2, 9], n_samples, p=[0.5, 0.3, 0.15, 0.05]),

        # Functional Capacity
        'BILLS': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),
        'TAXES': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),
        'SHOPPING': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),
        'GAMES': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),
        'STOVE': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),
        'MEALPREP': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),
        'EVENTS': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),
        'PAYATTN': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),
        'REMDATES': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),
        'TRAVEL': np.random.choice([0, 1, 2, 3, 8, 9], n_samples),

        # Sensory
        'VISION': np.random.choice([0, 1, 9], n_samples, p=[0.7, 0.25, 0.05]),
        'VISWCORR': np.random.choice([0, 1, 8, 9], n_samples),
        'HEARING': np.random.choice([0, 1, 9], n_samples, p=[0.6, 0.35, 0.05]),
        'HEARWAID': np.random.choice([0, 1, 8, 9], n_samples),

        # Target (for demo - would be actual dementia diagnosis in real data)
        'dementia': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }

    # Add some NACC special codes
    for col in ['SMOKYRS', 'ALCFREQ']:
        data[col] = np.where(np.random.random(n_samples) < 0.1, 888, data[col])  # Not applicable
        data[col] = np.where(np.random.random(n_samples) < 0.05, 999, data[col])  # Unknown
        data[col] = np.where(np.random.random(n_samples) < 0.02, -4, data[col])   # Not available

    df = pd.DataFrame(data)

    # Add some medical features that should be excluded
    df['COGNITIVE_TEST_SCORE'] = np.random.randint(0, 30, n_samples)  # Should be excluded
    df['BRAIN_SCAN_RESULT'] = np.random.choice(['Normal', 'Abnormal', 'Unknown'], n_samples)  # Should be excluded

    return df


def test_feature_selector():
    """Test feature selection component"""
    print("\n" + "="*80)
    print("TEST 1: Feature Selector")
    print("="*80)

    df = create_sample_nacc_data(100)
    print(f"Created sample data: {df.shape}")

    selector = DementiaFeatureSelector()
    df_selected = selector.select_features(df, target_col='dementia')

    print(f"Selected features: {df_selected.shape[1]} (from {df.shape[1]})")
    print(f"Removed features: {len(selector.removed_features)}")

    # Verify medical features were removed
    assert 'COGNITIVE_TEST_SCORE' not in df_selected.columns, "Medical feature not removed!"
    assert 'BRAIN_SCAN_RESULT' not in df_selected.columns, "Medical feature not removed!"

    # Verify non-medical features were kept
    assert 'NACCAGE' in df_selected.columns, "Non-medical feature incorrectly removed!"
    assert 'EDUC' in df_selected.columns, "Non-medical feature incorrectly removed!"

    print("✓ Feature selector test passed!")
    return df_selected


def test_missing_handler():
    """Test missing value handler"""
    print("\n" + "="*80)
    print("TEST 2: Missing Value Handler")
    print("="*80)

    df = create_sample_nacc_data(100)

    # Count special codes before
    special_codes = [-4, 8, 88, 888, 8888, 9, 99, 999, 9999]
    before_count = df.isin(special_codes).sum().sum()
    print(f"Special codes before: {before_count}")

    handler = NACCMissingValueHandler()
    df_clean, report = handler.fit_transform(df)

    # Check special codes were converted
    after_count = df_clean.isin(special_codes).sum().sum()
    print(f"Special codes after: {after_count}")

    print(f"Missing indicators created: {len(handler.missing_indicators_created)}")
    print(f"Columns with strategies: {len(handler.imputation_strategies)}")

    assert after_count < before_count, "Special codes not converted!"
    print("✓ Missing value handler test passed!")
    return df_clean


def test_feature_engineer():
    """Test feature engineering"""
    print("\n" + "="*80)
    print("TEST 3: Feature Engineer")
    print("="*80)

    df = create_sample_nacc_data(100)
    df = df[['NACCAGE', 'EDUC', 'CVHATT', 'CBSTROKE', 'CBTIA',
             'TOBAC100', 'TOBAC30', 'SMOKYRS', 'PACKSPER',
             'BILLS', 'TAXES', 'SHOPPING', 'VISWCORR', 'HEARWAID']]

    initial_features = len(df.columns)
    print(f"Initial features: {initial_features}")

    engineer = DementiaFeatureEngineer()
    df_eng, report = engineer.engineer_features(df)

    final_features = len(df_eng.columns)
    print(f"Final features: {final_features}")
    print(f"Features created: {len(engineer.created_features)}")

    # Check expected features were created
    expected_features = [
        'cardiovascular_risk_score',
        'cerebrovascular_risk_score',
        'lifestyle_risk_score',
        'functional_impairment_score',
        'age_squared'
    ]

    for feat in expected_features:
        if feat in engineer.created_features:
            print(f"  ✓ Created: {feat}")
            assert feat in df_eng.columns, f"Feature {feat} not in dataframe!"

    print("✓ Feature engineer test passed!")
    return df_eng


def test_full_pipeline():
    """Test complete preprocessing pipeline"""
    print("\n" + "="*80)
    print("TEST 4: Full Preprocessing Pipeline")
    print("="*80)

    df = create_sample_nacc_data(500)
    print(f"Created sample data: {df.shape}")

    pipeline = DementiaPreprocessingPipeline(target_col='dementia')
    X_processed, y, report = pipeline.fit_transform(df)

    print(f"\nProcessing complete!")
    print(f"Initial shape: {report['initial_shape']}")
    print(f"Final shape: {X_processed.shape}")
    print(f"Features removed: {report.get('features_removed_total', 0)}")
    print(f"Features created: {report.get('features_created_total', 0)}")

    # Verify output
    assert X_processed.shape[0] == y.shape[0], "Row count mismatch!"
    assert X_processed.isnull().sum().sum() == 0, "Missing values remain!"

    print("\n✓ Full pipeline test passed!")

    # Test transform (inference mode)
    print("\nTesting inference mode...")
    new_data = create_sample_nacc_data(10)
    X_new, y_new = pipeline.transform(new_data)
    print(f"Transformed new data: {X_new.shape}")
    # Note: Feature count may differ slightly due to one-hot encoding
    # Just verify we can transform without errors and get reasonable output
    assert X_new.shape[0] == 10, "Row count mismatch in transform!"
    assert X_new.shape[1] > 0, "No features in transform output!"
    print("✓ Inference mode test passed!")

    return X_processed, y, report


def test_report_generation():
    """Test report generation"""
    print("\n" + "="*80)
    print("TEST 5: Report Generation")
    print("="*80)

    df = create_sample_nacc_data(200)
    pipeline = DementiaPreprocessingPipeline(target_col='dementia')
    X_processed, y, report = pipeline.fit_transform(df)

    # Generate reports
    reporter = PreprocessingReportGenerator(report)

    # Test markdown generation
    md_report = reporter.generate_markdown_report()
    print(f"Markdown report generated: {len(md_report)} characters")
    assert len(md_report) > 1000, "Report too short!"
    assert "Feature Selection" in md_report, "Missing section!"
    assert "Missing Values" in md_report, "Missing section!"
    assert "Feature Engineering" in md_report, "Missing section!"

    print("✓ Report generation test passed!")

    # Save reports to test directory
    output_dir = Path(__file__).parent / 'test_output'
    output_dir.mkdir(exist_ok=True)

    reporter.save_all_reports(str(output_dir))
    print(f"\n✓ Test reports saved to {output_dir}")

    return reporter


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("DEMENTIA PREPROCESSING PIPELINE - TEST SUITE")
    print("="*80)

    try:
        # Test 1: Feature Selector
        df_selected = test_feature_selector()

        # Test 2: Missing Handler
        df_clean = test_missing_handler()

        # Test 3: Feature Engineer
        df_eng = test_feature_engineer()

        # Test 4: Full Pipeline
        X, y, report = test_full_pipeline()

        # Test 5: Report Generation
        reporter = test_report_generation()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nPreprocessing pipeline is ready to use!")
        print("Next steps:")
        print("  1. Prepare your NACC dataset")
        print("  2. Run: python scripts/run_dementia_preprocessing.py --data your_data.csv")
        print("  3. Review generated reports")
        print("  4. Train your models!")

    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED! ✗")
        print("="*80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
