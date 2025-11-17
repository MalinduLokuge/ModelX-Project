"""Manual Training Template
Train models manually based on what AutoML discovered.

Usage:
1. Run CompeteML in auto mode first
2. Review outputs/YOUR_RUN_ID/recipe.txt
3. Check model_results section for best model and hyperparameters
4. Customize this template to replicate the best model
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import joblib


def train_manual(X: pd.DataFrame, y: pd.Series, problem_type: str = 'classification',
                 recipe_insights: dict = None) -> object:
    """
    Train model manually based on recipe

    Recipe typically shows:
    - Best model type (LightGBM, RandomForest, etc.)
    - Best hyperparameters discovered
    - Cross-validation scores

    Args:
        X: Feature matrix
        y: Target variable
        problem_type: 'classification' or 'regression'
        recipe_insights: Dict from recipe.json (optional)

    Returns:
        Trained model
    """

    print(f"Training {problem_type} model...")

    # ============================================================================
    # STEP 1: CHOOSE MODEL
    # (Check recipe.txt to see which model worked best)
    # ============================================================================

    if problem_type == 'classification':
        # LightGBM typically performs well in competitions
        model = LGBMClassifier(
            learning_rate=0.05,      # Lower = slower but often better
            n_estimators=500,        # More trees usually better
            max_depth=7,             # Prevent overfitting
            num_leaves=31,
            subsample=0.8,           # Row sampling
            colsample_bytree=0.8,    # Column sampling
            random_state=42,
            verbose=-1
        )

        # Alternative: RandomForest (more robust, slower)
        # model = RandomForestClassifier(
        #     n_estimators=300,
        #     max_depth=10,
        #     min_samples_split=5,
        #     min_samples_leaf=2,
        #     random_state=42,
        #     n_jobs=-1
        # )

    else:  # regression
        model = LGBMRegressor(
            learning_rate=0.05,
            n_estimators=500,
            max_depth=7,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

    # ============================================================================
    # STEP 2: CROSS-VALIDATION
    # (Validate model performance before final training)
    # ============================================================================

    print("Running cross-validation...")

    if problem_type == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'roc_auc'  # Or 'accuracy', 'f1', etc.
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'neg_mean_squared_error'  # Or 'r2', 'neg_mean_absolute_error'

    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ============================================================================
    # STEP 3: TRAIN FINAL MODEL
    # (Train on full dataset)
    # ============================================================================

    print("Training final model on full dataset...")
    model.fit(X, y)

    print("✓ Training complete")

    return model


def save_model(model: object, output_path: str = "outputs/models/manual_model.pkl"):
    """Save trained model"""
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


def make_predictions(model: object, X_test: pd.DataFrame,
                     output_path: str = "outputs/submissions/manual_predictions.csv"):
    """Make predictions on test set"""
    predictions = model.predict(X_test)

    # Save predictions
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'prediction': predictions
    })
    submission.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    return predictions


# ============================================================================
# EXAMPLE USAGE - FULL PIPELINE
# ============================================================================

if __name__ == "__main__":
    # 1. Load data
    train_df = pd.read_csv("data/processed/manual_features.csv")
    test_df = pd.read_csv("data/raw/test.csv")  # Raw test set

    # 2. Split features and target
    target_col = 'target'  # Replace with your target column
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]

    # 3. Train model
    model = train_manual(X_train, y_train, problem_type='classification')

    # 4. Save model
    save_model(model, "outputs/models/manual_model.pkl")

    # 5. Preprocess test set (use same steps as training)
    # (Import your preprocessing and feature engineering functions)
    # X_test_processed = preprocess_manual(test_df)
    # X_test_features = engineer_features_manual(X_test_processed)

    # 6. Make predictions
    # predictions = make_predictions(model, X_test_features)

    print("\n" + "="*80)
    print("Manual training complete!")
    print("="*80)
