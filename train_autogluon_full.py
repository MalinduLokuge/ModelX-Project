"""
Train Full Production AutoGluon Model (36+ models, high performance)
Target: 97.09% ROC-AUC with bagging and stacking
"""

from autogluon.tabular import TabularPredictor
import pandas as pd
import os

def train_production_model():
    """Train full production AutoGluon model with all optimizations"""
    
    print("=" * 80)
    print("Training FULL PRODUCTION AutoGluon Model")
    print("Target: 97.09% ROC-AUC | 36+ models | Bagging + Stacking")
    print("=" * 80)
    
    # Load training data
    print("\n1. Loading training data...")
    X_train = pd.read_csv("data/train/X_train.csv")
    y_train = pd.read_csv("data/train/y_train.csv")
    
    # Combine X and y for AutoGluon
    train_data = X_train.copy()
    train_data["dementia_status"] = y_train.values
    
    print(f"   Training samples: {len(train_data):,}")
    print(f"   Features: {len(X_train.columns)}")
    print(f"   Target distribution:")
    print(f"     - No Dementia: {(y_train == 0).sum().values[0]:,}")
    print(f"     - Dementia: {(y_train == 1).sum().values[0]:,}")
    
    # Create output directory
    model_path = "outputs/models/autogluon_production_full"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize predictor
    print("\n2. Initializing AutoGluon predictor...")
    predictor = TabularPredictor(
        label="dementia_status",
        eval_metric="roc_auc",
        path=model_path
    )
    
    # Train model - FULL PRODUCTION CONFIGURATION
    print("\n3. Training PRODUCTION model (this will take ~30 minutes)...")
    print("   Configuration:")
    print("     - Preset: medium_quality")
    print("     - Time limit: 1800 seconds (30 minutes)")
    print("     - Bagging: 3 folds (ensemble multiple models)")
    print("     - Stacking: 1 level (meta-model on top)")
    print("     - Models: All model types enabled")
    print("     - Expected: 30-40 models total")
    
    predictor.fit(
        train_data=train_data,
        time_limit=1800,  # 30 minutes
        presets="medium_quality",
        num_bag_folds=3,  # 3-fold bagging for robust ensembles
        num_stack_levels=1,  # 1 stack level (L1 base + L2 stacked)
        ag_args_fit={
            'num_cpus': 8,  # Use more CPUs
            'num_gpus': 0,  # CPU only
        },
        # Include all model types (no exclusions)
        hyperparameters={
            'GBM': {},  # LightGBM with default hyperparameters
            'CAT': {},  # CatBoost with default hyperparameters
            'XGB': {},  # XGBoost with default hyperparameters
            'RF': {'n_estimators': 300},  # Random Forest
            'XT': {'n_estimators': 300},  # Extra Trees
        },
        verbosity=2
    )
    
    # Show leaderboard
    print("\n4. Training complete! Model leaderboard:")
    leaderboard = predictor.leaderboard()
    print(leaderboard[["model", "score_val", "pred_time_val", "fit_time"]].head(15))
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"   Best model: {predictor.model_best}")
    print(f"   Total models: {len(predictor.model_names())}")
    
    # Evaluate on test data
    print("\n5. Evaluating on test set...")
    X_test = pd.read_csv("data/test/X_test.csv")
    y_test = pd.read_csv("data/test/y_test.csv")
    
    test_data = X_test.copy()
    test_data["dementia_status"] = y_test.values
    
    performance = predictor.evaluate(test_data)
    print(f"\n   Test Performance:")
    print(f"     ROC-AUC: {performance['roc_auc']:.4f}")
    print(f"     Accuracy: {performance['accuracy']:.4f}")
    
    # Test predictions
    print("\n6. Testing predictions on first 5 samples...")
    test_sample = X_test.head(5)
    predictions = predictor.predict(test_sample)
    probabilities = predictor.predict_proba(test_sample)
    
    print("   Predictions:")
    positive_class_col = probabilities.columns[1] if len(probabilities.columns) > 1 else probabilities.columns[0]
    for i, (pred, prob) in enumerate(zip(predictions, probabilities[positive_class_col])):
        print(f"     Sample {i+1}: Class {pred} (probability: {prob:.2%})")
    
    print("\n" + "=" * 80)
    print("PRODUCTION MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel Summary:")
    print(f"  Location: {model_path}")
    print(f"  Models trained: {len(predictor.model_names())}")
    print(f"  Best model: {predictor.model_best}")
    print(f"  Test ROC-AUC: {performance['roc_auc']:.4f}")
    print(f"\nTo use the model:")
    print(f'  predictor = TabularPredictor.load("{model_path}")')
    print(f'  predictions = predictor.predict(new_data)')

if __name__ == "__main__":
    train_production_model()
