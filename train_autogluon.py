"""
Quick script to train AutoGluon model
"""

from autogluon.tabular import TabularPredictor
import pandas as pd
import os

def train_autogluon_model():
    """Train AutoGluon model on dementia dataset"""
    
    print("=" * 80)
    print("Training AutoGluon Model")
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
    model_path = "outputs/models/autogluon_production_lowmem"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize predictor
    print("\n2. Initializing AutoGluon predictor...")
    predictor = TabularPredictor(
        label="dementia_status",
        eval_metric="roc_auc",
        path=model_path
    )
    
    # Train model (LOW MEMORY MODE)
    print("\n3. Training model (this will take ~30 minutes)...")
    print("   Preset: medium_quality (LOW MEMORY)")
    print("   Time limit: 1800 seconds (30 minutes)")
    print("   Memory optimization: enabled")
    
    predictor.fit(
        train_data=train_data,
        time_limit=1800,  # 30 minutes
        presets="medium_quality",
        num_bag_folds=0,  # No bagging to save memory
        num_stack_levels=0,  # No stacking to save memory
        ag_args_fit={
            'num_cpus': 2,  # Limit CPU usage
            'num_gpus': 0,  # CPU only
            'ag.max_memory_usage_ratio': 2.0,  # Allow up to 200% estimated memory (risky but necessary)
        },
        excluded_model_types=['NN_TORCH', 'FASTAI', 'CAT'],  # Exclude neural networks and CatBoost (most memory intensive)
        hyperparameters={
            'GBM': {'num_boost_round': 50, 'num_leaves': 15},  # Smaller trees
            'XGB': {'n_estimators': 50, 'max_depth': 3},  # Smaller trees
        },
        verbosity=2
    )
    
    # Show leaderboard
    print("\n4. Training complete! Model leaderboard:")
    leaderboard = predictor.leaderboard()
    print(leaderboard[["model", "score_val", "pred_time_val", "fit_time"]].head(10))
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"   Best model: {predictor.model_best}")
    print(f"   Total models: {len(predictor.model_names())}")
    
    # Test prediction
    print("\n5. Testing model on first 5 samples...")
    test_sample = X_train.head(5)
    predictions = predictor.predict(test_sample)
    probabilities = predictor.predict_proba(test_sample)
    
    print("   Predictions:")
    # Get the positive class column (either 1 or 'Dementia')
    positive_class_col = probabilities.columns[1] if len(probabilities.columns) > 1 else probabilities.columns[0]
    for i, (pred, prob) in enumerate(zip(predictions, probabilities[positive_class_col])):
        risk_pct = prob * 100 if isinstance(pred, (int, float)) else prob
        print(f"     Sample {i+1}: Class {pred} (probability: {risk_pct:.2f}%)")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nTo use the model:")
    print(f'  predictor = TabularPredictor.load("{model_path}")')
    print(f'  predictions = predictor.predict(new_data)')

if __name__ == "__main__":
    train_autogluon_model()
