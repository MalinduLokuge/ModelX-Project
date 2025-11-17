"""Create sample datasets for testing"""
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed
np.random.seed(42)

# Create output directory
output_dir = Path(__file__).parent.parent / "data" / "sample"
output_dir.mkdir(parents=True, exist_ok=True)

def create_classification_dataset():
    """Create simple binary classification dataset"""
    n_train = 1000
    n_test = 300

    # Training data
    train_data = {
        'id': range(1, n_train + 1),
        'feature_1': np.random.randn(n_train),
        'feature_2': np.random.randn(n_train),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_train),
        'feature_4': np.random.uniform(0, 100, n_train),
        'feature_5': np.random.choice(['Low', 'Medium', 'High'], n_train),
    }

    # Create target (simple rule)
    train_data['target'] = (
        (train_data['feature_1'] > 0).astype(int) &
        (train_data['feature_4'] > 50).astype(int)
    )

    # Add some noise
    noise_idx = np.random.choice(n_train, size=int(0.1 * n_train), replace=False)
    train_data['target'][noise_idx] = 1 - train_data['target'][noise_idx]

    # Add some missing values
    for col in ['feature_2', 'feature_4', 'feature_5']:
        missing_idx = np.random.choice(n_train, size=int(0.05 * n_train), replace=False)
        train_data[col][missing_idx] = np.nan

    train_df = pd.DataFrame(train_data)

    # Test data (similar distribution, no target)
    test_data = {
        'id': range(n_train + 1, n_train + n_test + 1),
        'feature_1': np.random.randn(n_test),
        'feature_2': np.random.randn(n_test),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_test),
        'feature_4': np.random.uniform(0, 100, n_test),
        'feature_5': np.random.choice(['Low', 'Medium', 'High'], n_test),
    }

    test_df = pd.DataFrame(test_data)

    # Save
    train_df.to_csv(output_dir / "classification_train.csv", index=False)
    test_df.to_csv(output_dir / "classification_test.csv", index=False)

    print(f"✓ Created classification dataset")
    print(f"  Train: {train_df.shape}")
    print(f"  Test: {test_df.shape}")
    print(f"  Saved to: {output_dir}")

def create_regression_dataset():
    """Create simple regression dataset"""
    n_train = 1000
    n_test = 300

    # Training data
    train_data = {
        'id': range(1, n_train + 1),
        'feature_1': np.random.randn(n_train),
        'feature_2': np.random.randn(n_train),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_train),
        'feature_4': np.random.uniform(0, 100, n_train),
    }

    # Create target (linear combination with noise)
    train_data['target'] = (
        2.5 * train_data['feature_1'] +
        1.5 * train_data['feature_2'] +
        0.01 * train_data['feature_4'] +
        np.random.randn(n_train) * 2  # noise
    )

    train_df = pd.DataFrame(train_data)

    # Test data
    test_data = {
        'id': range(n_train + 1, n_train + n_test + 1),
        'feature_1': np.random.randn(n_test),
        'feature_2': np.random.randn(n_test),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_test),
        'feature_4': np.random.uniform(0, 100, n_test),
    }

    test_df = pd.DataFrame(test_data)

    # Save
    train_df.to_csv(output_dir / "regression_train.csv", index=False)
    test_df.to_csv(output_dir / "regression_test.csv", index=False)

    print(f"✓ Created regression dataset")
    print(f"  Train: {train_df.shape}")
    print(f"  Test: {test_df.shape}")
    print(f"  Saved to: {output_dir}")

if __name__ == "__main__":
    print("Creating sample datasets...")
    create_classification_dataset()
    create_regression_dataset()
    print("\n✓ All sample datasets created successfully!")
