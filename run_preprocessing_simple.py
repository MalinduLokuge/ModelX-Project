"""
Simple script to run dementia preprocessing pipeline
No complex dependencies - just run it!
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.dementia_preprocessing_pipeline import DementiaPreprocessingPipeline
from src.reporting.preprocessing_report_generator import PreprocessingReportGenerator

# Simple logger class
class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def section(self, msg):
        print(f"\n{'='*80}\n{msg}\n{'='*80}")

def main():
    print("="*80)
    print("DEMENTIA RISK PREDICTION - PREPROCESSING PIPELINE")
    print("="*80)

    # Configuration
    data_path = "data/raw/Dementia Prediction Dataset.csv"
    target_col = "DEMENTED"
    output_dir = "outputs/dementia_preprocessed"

    print(f"\nInput data: {data_path}")
    print(f"Target column: {target_col}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✓ Loaded {df.shape[0]:,} samples with {df.shape[1]:,} features")

    # Initialize preprocessing pipeline
    print("\n[2/4] Running preprocessing pipeline...")
    logger = SimpleLogger()
    pipeline = DementiaPreprocessingPipeline(target_col=target_col, logger=logger)

    # Run preprocessing
    X_processed, y, report = pipeline.fit_transform(df, create_report=True)

    # Save processed data
    print("\n[3/4] Saving processed data...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_processed.to_csv(output_path / 'X_processed.csv', index=False)
    y.to_csv(output_path / 'y.csv', index=False)
    print(f"✓ Saved to {output_dir}/")

    # Save pipeline
    pipeline.save_pipeline(str(output_dir))
    print(f"✓ Pipeline saved")

    # Generate reports
    print("\n[4/4] Generating reports...")
    report_gen = PreprocessingReportGenerator(report)
    report_gen.save_all_reports(str(output_dir))

    # Summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"Initial shape: {report['initial_shape']}")
    print(f"Final shape: {X_processed.shape}")
    print(f"Features removed: {report.get('features_removed_total', 0)}")
    print(f"Features created: {report.get('features_created_total', 0)}")
    print(f"\nOutput files in {output_dir}/:")
    print("  ✓ X_processed.csv - Clean features")
    print("  ✓ y.csv - Target variable")
    print("  ✓ PREPROCESSING_REPORT.md - Full documentation")
    print("  ✓ preprocessing_report.json - Machine-readable report")
    print("  ✓ preprocessing_pipeline.pkl - Saved pipeline")
    print("\n✓ All preprocessing complete!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
