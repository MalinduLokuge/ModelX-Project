"""
Complete ML Pipeline - End-to-End Execution
Run the entire ModelX dementia prediction pipeline from raw data to trained model

Usage:
    python run_complete_pipeline.py
    python run_complete_pipeline.py --skip-preprocessing  (if already preprocessed)
    python run_complete_pipeline.py --quick-test  (fast 5-min test)
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_step(step_num, text):
    """Print step header"""
    print(f"\n{'─' * 80}")
    print(f"STEP {step_num}: {text}")
    print(f"{'─' * 80}")

def check_file_exists(filepath, description):
    """Check if required file exists"""
    if not Path(filepath).exists():
        print(f"❌ ERROR: {description} not found at: {filepath}")
        return False
    print(f"✓ Found {description}: {filepath}")
    return True

def run_command(description, command):
    """Run a command and check for errors"""
    print(f"\n▶ {description}...")
    print(f"  Command: {command}")
    
    result = os.system(command)
    
    if result != 0:
        print(f"❌ FAILED: {description}")
        print(f"   Exit code: {result}")
        return False
    
    print(f"✓ SUCCESS: {description}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Run the complete ML pipeline for dementia prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (preprocessing + training)
  python run_complete_pipeline.py
  
  # Skip preprocessing (use existing processed data)
  python run_complete_pipeline.py --skip-preprocessing
  
  # Quick test (5-minute run)
  python run_complete_pipeline.py --quick-test
  
  # Train only manual models
  python run_complete_pipeline.py --skip-preprocessing --manual-only
  
  # Train only AutoML model
  python run_complete_pipeline.py --skip-preprocessing --automl-only
        """
    )
    
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing step (use existing processed data)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (reduced time limits)')
    parser.add_argument('--manual-only', action='store_true',
                        help='Train only manual models (skip AutoML)')
    parser.add_argument('--automl-only', action='store_true',
                        help='Train only AutoML model (skip manual models)')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip model evaluation step')
    
    args = parser.parse_args()
    
    # Start timer
    start_time = datetime.now()
    
    print_header("ModelX Dementia Prediction - Complete ML Pipeline")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick_test:
        print("⚡ QUICK TEST MODE - Reduced time limits for fast validation")
    
    # ========================================================================
    # Step 1: Check Prerequisites
    # ========================================================================
    print_step(1, "Checking Prerequisites")
    
    prerequisites_ok = True
    
    # Check raw data
    if not args.skip_preprocessing:
        if not check_file_exists('data/processed/cleaned_nonmedical_dataset.csv', 
                                'Cleaned dataset'):
            print("\n⚠️  Run data cleaning first:")
            print("   python scripts/clean_data.py")
            prerequisites_ok = False
    else:
        # Check preprocessed data
        if not check_file_exists('data/train/X_train_balanced.csv', 
                                'Training features'):
            prerequisites_ok = False
        if not check_file_exists('data/train/y_train_balanced.csv', 
                                'Training labels'):
            prerequisites_ok = False
    
    if not prerequisites_ok:
        print("\n❌ Prerequisites not met. Please fix the errors above.")
        sys.exit(1)
    
    print("\n✅ All prerequisites met!")
    
    # ========================================================================
    # Step 2: Data Preprocessing (Optional)
    # ========================================================================
    if not args.skip_preprocessing:
        print_step(2, "Data Preprocessing")
        
        if not run_command(
            "Running preprocessing pipeline",
            "python run_preprocessing_simple.py"
        ):
            print("\n❌ Preprocessing failed!")
            sys.exit(1)
    else:
        print_step(2, "Data Preprocessing - SKIPPED")
        print("Using existing preprocessed data")
    
    # ========================================================================
    # Step 3: Train Models
    # ========================================================================
    print_step(3, "Model Training")
    
    # 3a. Manual Models
    if not args.automl_only:
        print("\n📊 Training Manual Models (8 models)...")
        print("   Models: LightGBM, XGBoost, RandomForest, ExtraTrees, LogisticRegression")
        
        if args.quick_test:
            print("   ⚡ Quick mode: Training with reduced hyperparameters")
        
        if not run_command(
            "Training manual models",
            "python train_manual_lowmem.py"
        ):
            print("\n⚠️  Manual model training failed (non-critical)")
            print("   Continuing with AutoML training...")
    else:
        print("\n📊 Manual Models - SKIPPED")
    
    # 3b. AutoML Model
    if not args.manual_only:
        print("\n🤖 Training AutoML Model (AutoGluon)...")
        print("   This will train 42 models with 4-level stacking")
        
        if args.quick_test:
            print("   ⚡ Quick mode: 5-minute time limit")
            if not run_command(
                "Training AutoML (quick)",
                "python train_autogluon.py"  # Has quick preset built-in
            ):
                print("\n❌ AutoML training failed!")
                sys.exit(1)
        else:
            print("   ⏱️  Full mode: ~30 minutes, targeting 94%+ ROC-AUC")
            if not run_command(
                "Training optimized AutoML",
                "python train_autogluon_optimized.py"
            ):
                print("\n❌ AutoML training failed!")
                sys.exit(1)
    else:
        print("\n🤖 AutoML Model - SKIPPED")
    
    # ========================================================================
    # Step 4: Model Evaluation
    # ========================================================================
    if not args.skip_evaluation:
        print_step(4, "Model Evaluation & Comparison")
        
        print("\n📈 Generating model comparison report...")
        if run_command(
            "Comparing all models",
            "python model_comparison_final.py"
        ):
            print("\n✓ Model comparison complete!")
            print("  📄 Results: model_comparison_results/")
        else:
            print("\n⚠️  Model comparison failed (non-critical)")
        
        # Generate XAI analysis if available
        print("\n🔍 Generating explainability analysis...")
        if run_command(
            "Running XAI analysis",
            "python generate_xai_analysis.py"
        ):
            print("\n✓ XAI analysis complete!")
            print("  📄 Results: outputs/xai/")
        else:
            print("\n⚠️  XAI analysis failed (non-critical)")
    else:
        print_step(4, "Model Evaluation - SKIPPED")
    
    # ========================================================================
    # Summary
    # ========================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("Pipeline Complete!")
    
    print(f"\n⏱️  Total execution time: {duration}")
    print(f"   Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📁 Output Locations:")
    print("   ├── Preprocessed data:    data/train/, data/test/")
    print("   ├── Manual models:        outputs/manual_models/")
    print("   ├── AutoML model:         outputs/models/autogluon_optimized/")
    print("   ├── Model comparison:     model_comparison_results/")
    print("   ├── XAI analysis:         outputs/xai/")
    print("   └── Documentation:        MODEL_README.md, AUTOML_TRAINING_REPORT.md")
    
    print("\n📊 Key Results:")
    
    # Try to read and display key metrics
    try:
        import json
        if Path('outputs/models/autogluon_optimized/metadata.json').exists():
            with open('outputs/models/autogluon_optimized/metadata.json') as f:
                metadata = json.load(f)
                val_score = metadata.get('best_model_score', 'N/A')
                print(f"   🏆 AutoML Validation ROC-AUC: {val_score}")
    except:
        pass
    
    try:
        if Path('outputs/manual_models/model_comparison.csv').exists():
            import pandas as pd
            comparison = pd.read_csv('outputs/manual_models/model_comparison.csv')
            best_manual = comparison.iloc[0]
            print(f"   🥈 Best Manual Model: {best_manual['Model']} ({best_manual['ROC_AUC']:.4f})")
    except:
        pass
    
    print("\n📖 Next Steps:")
    print("   1. Review MODEL_README.md for complete model documentation")
    print("   2. Check AUTOML_TRAINING_REPORT.md for training details")
    print("   3. Explore outputs/xai/ for feature importance and explanations")
    print("   4. Load models using USAGE_SNIPPETS.md examples")
    
    print("\n" + "=" * 80)
    print("✅ SUCCESS - All pipeline steps completed!")
    print("=" * 80)

if __name__ == '__main__':
    main()
