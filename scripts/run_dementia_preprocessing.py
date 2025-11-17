"""
Example Script: Dementia Risk Prediction - Complete Preprocessing Pipeline

This script demonstrates how to use the comprehensive preprocessing pipeline
for dementia risk prediction using non-medical features.

Usage:
    python scripts/run_dementia_preprocessing.py --data data/nacc_dataset.csv --output outputs/preprocessed/
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logger import setup_logger
from src.preprocessing.dementia_preprocessing_pipeline import DementiaPreprocessingPipeline
from src.reporting.preprocessing_report_generator import PreprocessingReportGenerator


def main():
    parser = argparse.ArgumentParser(description='Run dementia risk preprocessing pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--target', type=str, default='dementia', help='Target column name')
    parser.add_argument('--output', type=str, default='outputs/preprocessed/', help='Output directory')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('dementia_preprocessing')

    logger.info("=" * 80)
    logger.info("DEMENTIA RISK PREDICTION - PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input data: {args.data}")
    logger.info(f"Target column: {args.target}")
    logger.info(f"Output directory: {args.output}")

    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")

    # Initialize preprocessing pipeline
    logger.info("\nInitializing preprocessing pipeline...")
    pipeline = DementiaPreprocessingPipeline(target_col=args.target, logger=logger)

    # Run preprocessing
    logger.info("\nRunning preprocessing pipeline...")
    X_processed, y, report = pipeline.fit_transform(df, create_report=True)

    # Save processed data
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features and target
    X_processed.to_csv(output_dir / 'X_processed.csv', index=False)
    if y is not None:
        y.to_csv(output_dir / 'y.csv', index=False)
    logger.info(f"\n✓ Processed data saved to {output_dir}")

    # Save pipeline
    pipeline.save_pipeline(str(output_dir))
    logger.info(f"✓ Pipeline saved to {output_dir}")

    # Generate comprehensive reports
    logger.info("\nGenerating preprocessing reports...")
    report_generator = PreprocessingReportGenerator(report)
    report_generator.save_all_reports(str(output_dir))
    logger.info(f"✓ Reports generated")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Initial shape: {report['initial_shape']}")
    logger.info(f"Final shape: {X_processed.shape}")
    logger.info(f"Features removed: {report.get('features_removed_total', 0)}")
    logger.info(f"Features created: {report.get('features_created_total', 0)}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - {output_dir / 'X_processed.csv'}")
    logger.info(f"  - {output_dir / 'y.csv'}")
    logger.info(f"  - {output_dir / 'PREPROCESSING_REPORT.md'}")
    logger.info(f"  - {output_dir / 'preprocessing_report.xlsx'}")
    logger.info(f"  - {output_dir / 'preprocessing_pipeline.pkl'}")
    logger.info(f"  - {output_dir / 'preprocessing_report.json'}")
    logger.info("\n✓ All preprocessing complete!")


if __name__ == '__main__':
    main()
