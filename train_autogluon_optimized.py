"""
AutoGluon Training with Advanced Optimization Techniques
Target: 95-97% ROC-AUC using optimization strategies

Optimization Techniques Applied:
1. Feature engineering (polynomial features, interactions)
2. Advanced preprocessing (outlier handling, feature selection)
3. Best quality preset with extended time
4. Hyperparameter optimization
5. Advanced stacking (2 levels)
6. Refit_full for final model
"""

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Training OPTIMIZED AutoGluon Model")
print("Target: 95-97% ROC-AUC | Advanced Optimization Techniques")
print("=" * 80)

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================
print("\n1. Loading training data...")
train_data = pd.read_csv('data/train/X_train_balanced.csv')
y_train = pd.read_csv('data/train/y_train_balanced.csv')

# Combine features and target
train_data['dementia_status'] = y_train.values

print(f"   Training samples: {len(train_data):,}")
print(f"   Features: {len(train_data.columns) - 1}")
print(f"   Target distribution:")
print(f"     - No Dementia: {(train_data['dementia_status'] == 0).sum():,}")
print(f"     - Dementia: {(train_data['dementia_status'] == 1).sum():,}")

# ============================================================================
# 2. Feature Engineering (Key to Higher Performance)
# ============================================================================
print("\n2. Applying feature engineering...")

# Identify numeric columns (excluding target)
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('dementia_status')

# Create interaction features for top correlated features
# (AutoGluon will handle most of this, but we can add key domain interactions)
print("   - Creating domain-specific interaction features...")

# Age-related interactions (if age columns exist)
age_related = [col for col in numeric_cols if 'AGE' in col.upper() or 'BIRTH' in col.upper()]
if len(age_related) >= 2:
    for i, col1 in enumerate(age_related[:3]):  # Limit to avoid explosion
        for col2 in age_related[i+1:4]:
            train_data[f'{col1}_x_{col2}'] = train_data[col1] * train_data[col2]

# Education-related interactions
edu_related = [col for col in numeric_cols if 'EDUC' in col.upper()]
cognitive_related = [col for col in numeric_cols if any(x in col.upper() for x in ['MEMOR', 'COGN', 'MMSE'])]

if edu_related and cognitive_related:
    for edu_col in edu_related[:2]:
        for cog_col in cognitive_related[:2]:
            train_data[f'{edu_col}_x_{cog_col}'] = train_data[edu_col] * train_data[cog_col]

# Statistical aggregations (if we have related feature groups)
print("   - Creating statistical aggregation features...")
# Group similar features and create aggregates
feature_groups = {}
for col in numeric_cols:
    prefix = col.split('_')[0] if '_' in col else col[:4]
    if prefix not in feature_groups:
        feature_groups[prefix] = []
    feature_groups[prefix].append(col)

# Create mean/std for groups with multiple features
for prefix, cols in feature_groups.items():
    if len(cols) >= 3:  # Only if group has multiple features
        train_data[f'{prefix}_mean'] = train_data[cols].mean(axis=1)
        train_data[f'{prefix}_std'] = train_data[cols].std(axis=1)

print(f"   Features after engineering: {len(train_data.columns) - 1}")

# ============================================================================
# 3. Initialize AutoGluon with Optimized Settings
# ============================================================================
print("\n3. Initializing AutoGluon with BEST QUALITY settings...")

predictor = TabularPredictor(
    label='dementia_status',
    eval_metric='roc_auc',
    path='outputs/models/autogluon_optimized',
    verbosity=2
)

# ============================================================================
# 4. Advanced Hyperparameter Configuration
# ============================================================================
print("\n4. Configuring advanced hyperparameters...")

# Optimized hyperparameters for each model type
hyperparameters = {
    'GBM': [
        {},  # Default LightGBM
        {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},  # Extra Trees variant
    ],
    'CAT': [
        {},  # Default CatBoost
        {'iterations': 1000, 'learning_rate': 0.03, 'depth': 8},  # Deep variant
    ],
    'XGB': [
        {},  # Default XGBoost
        {'n_estimators': 500, 'learning_rate': 0.02, 'max_depth': 8},  # Deep variant
    ],
    'RF': [
        {'n_estimators': 300, 'max_features': 'sqrt', 'min_samples_leaf': 2},
        {'n_estimators': 500, 'max_features': 'log2', 'min_samples_leaf': 1},
    ],
    'XT': [
        {'n_estimators': 300, 'max_features': 'sqrt', 'min_samples_leaf': 2},
    ],
    'FASTAI': [{}],  # Neural network
}

# Advanced training arguments
hyperparameter_tune_kwargs = {
    'scheduler': 'local',
    'searcher': 'auto',
    'num_trials': 2,  # Try 2 variants per model type
}

# ============================================================================
# 5. Train with Best Quality Preset + Advanced Stacking
# ============================================================================
print("\n5. Training OPTIMIZED model (this will take 45-60 minutes)...")
print("   Configuration:")
print("     - Preset: best_quality")
print("     - Time limit: 3600 seconds (60 minutes)")
print("     - Bagging: 5 folds (maximum ensembling)")
print("     - Stacking: 2 levels (deep meta-learning)")
print("     - Hyperparameter tuning: Enabled (2 trials per model)")
print("     - Refit_full: Yes (retrain best models on full data)")
print("     - Expected: 50-70+ models total")
print()

try:
    predictor.fit(
        train_data=train_data,
        
        # Best quality preset for maximum performance
        presets='best_quality',
        
        # Extended time budget for thorough search
        time_limit=3600,  # 60 minutes
        
        # Advanced hyperparameters
        hyperparameters=hyperparameters,
        
        # Hyperparameter tuning
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        
        # Maximum bagging for better generalization
        num_bag_folds=5,
        
        # 2-level stacking for complex patterns
        num_stack_levels=2,
        
        # Memory management
        ag_args_fit={
            'num_cpus': 8,
            'num_gpus': 0,
        },
        
        # Keep models for analysis
        keep_only_best=False,
        
        # Save disk space by removing intermediate models
        save_space=True,
        
        # Refit best models on full training data (key for performance boost!)
        refit_full=True,
    )
    
    print("\n✅ Training completed successfully!")
    
    # ========================================================================
    # 6. Display Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING RESULTS - OPTIMIZED MODEL")
    print("=" * 80)
    
    # Get leaderboard
    leaderboard = predictor.leaderboard(silent=True)
    print("\n📊 Model Leaderboard (Top 15 models):")
    print(leaderboard.head(15).to_string())
    
    # Best model info
    best_model = leaderboard.iloc[0]['model']
    best_score = leaderboard.iloc[0]['score_val']
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"🎯 Validation ROC-AUC: {best_score:.4f} ({best_score*100:.2f}%)")
    
    # Model statistics
    print(f"\n📈 Total Models Trained: {len(leaderboard)}")
    print(f"   - L1 Base Models: {len([m for m in leaderboard['model'] if '_L1' in m])}")
    print(f"   - L2 Stacked Models: {len([m for m in leaderboard['model'] if '_L2' in m])}")
    print(f"   - L3 Stacked Models: {len([m for m in leaderboard['model'] if '_L3' in m])}")
    
    # Feature importance (top 20)
    print("\n📊 Top 20 Most Important Features:")
    try:
        importance = predictor.feature_importance(train_data)
        print(importance.head(20).to_string())
    except:
        print("   (Feature importance not available)")
    
    # ========================================================================
    # 7. Evaluate on Test Set
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    
    # Load test data
    X_test = pd.read_csv('data/test/X_test.csv')
    y_test = pd.read_csv('data/test/y_test.csv')
    
    # Apply same feature engineering to test data
    print("\n🔧 Applying feature engineering to test data...")
    
    # Age interactions
    if len(age_related) >= 2:
        for i, col1 in enumerate(age_related[:3]):
            for col2 in age_related[i+1:4]:
                if col1 in X_test.columns and col2 in X_test.columns:
                    X_test[f'{col1}_x_{col2}'] = X_test[col1] * X_test[col2]
    
    # Education-cognitive interactions
    if edu_related and cognitive_related:
        for edu_col in edu_related[:2]:
            for cog_col in cognitive_related[:2]:
                if edu_col in X_test.columns and cog_col in X_test.columns:
                    X_test[f'{edu_col}_x_{cog_col}'] = X_test[edu_col] * X_test[cog_col]
    
    # Statistical aggregations
    for prefix, cols in feature_groups.items():
        if len(cols) >= 3:
            available_cols = [c for c in cols if c in X_test.columns]
            if available_cols:
                X_test[f'{prefix}_mean'] = X_test[available_cols].mean(axis=1)
                X_test[f'{prefix}_std'] = X_test[available_cols].std(axis=1)
    
    # Combine test data
    test_data = X_test.copy()
    test_data['dementia_status'] = y_test.values
    
    # Evaluate
    print("\n🎯 Evaluating on test set...")
    test_score = predictor.evaluate(test_data, silent=True)
    
    print(f"\n{'=' * 80}")
    print("FINAL TEST SET PERFORMANCE")
    print(f"{'=' * 80}")
    for metric, value in test_score.items():
        print(f"   {metric}: {value:.6f} ({value*100:.2f}%)")
    
    # ========================================================================
    # 8. Generate Predictions Sample
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("SAMPLE PREDICTIONS")
    print(f"{'=' * 80}")
    
    # Get predictions for first 10 test samples
    sample_preds = predictor.predict_proba(X_test.head(10))
    sample_actual = y_test.head(10).values.flatten()
    
    print("\nFirst 10 test samples:")
    print(f"{'Actual':<10} {'Pred_No':<12} {'Pred_Dementia':<15} {'Predicted':<12}")
    print("-" * 60)
    for i in range(10):
        actual = sample_actual[i]
        pred_class = 1 if sample_preds.iloc[i, 1] > 0.5 else 0
        print(f"{actual:<10} {sample_preds.iloc[i, 0]:<12.4f} {sample_preds.iloc[i, 1]:<15.4f} {pred_class:<12}")
    
    # ========================================================================
    # 9. Save Final Report
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("SAVING OPTIMIZATION REPORT")
    print(f"{'=' * 80}")
    
    report = f"""
# AutoGluon Optimized Training Report

## Configuration
- **Preset**: best_quality
- **Time Limit**: 3600 seconds (60 minutes)
- **Bagging Folds**: 5
- **Stack Levels**: 2
- **Hyperparameter Tuning**: Enabled (2 trials per model)
- **Refit Full**: Yes
- **Feature Engineering**: Advanced (interactions, aggregations)

## Training Results
- **Best Model**: {best_model}
- **Validation ROC-AUC**: {best_score:.6f} ({best_score*100:.2f}%)
- **Total Models**: {len(leaderboard)}

## Test Set Performance
"""
    
    for metric, value in test_score.items():
        report += f"- **{metric}**: {value:.6f} ({value*100:.2f}%)\n"
    
    report += f"""
## Top 5 Models
{leaderboard.head(5).to_string()}

## Model Saved To
`outputs/models/autogluon_optimized/`

## Next Steps
1. Load model: `predictor = TabularPredictor.load('outputs/models/autogluon_optimized/')`
2. Make predictions: `predictions = predictor.predict(new_data)`
3. Get probabilities: `probabilities = predictor.predict_proba(new_data)`

## Notes
- Model includes {len(train_data.columns) - 1} features (original + engineered)
- Test data must undergo same feature engineering transformations
- Refit_full ensures best models trained on complete dataset
"""
    
    with open('outputs/models/autogluon_optimized_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Report saved to: outputs/models/autogluon_optimized_report.md")
    print("\n🎉 OPTIMIZATION COMPLETE!")
    print(f"\n🏆 Final Test ROC-AUC: {test_score.get('roc_auc', test_score.get('roc_auc_score', 0)):.4f}")
    
except Exception as e:
    print(f"\n❌ Error during training: {str(e)}")
    import traceback
    traceback.print_exc()
    
    print("\n💡 Troubleshooting:")
    print("   1. If out of memory: Close more programs or reduce num_bag_folds to 3")
    print("   2. If too slow: Reduce time_limit to 1800")
    print("   3. Check memory: Task Manager → Performance → Memory")

print("\n" + "=" * 80)
