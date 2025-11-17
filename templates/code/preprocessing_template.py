"""Manual Preprocessing Template
Based on what AutoML discovered works best for your data.

Usage:
1. Run CompeteML in auto mode first
2. Review outputs/YOUR_RUN_ID/recipe.txt
3. Copy the preprocessing steps that worked
4. Customize this template based on recipe
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def preprocess_manual(df: pd.DataFrame, recipe_insights: dict = None) -> pd.DataFrame:
    """
    Manual preprocessing based on auto mode insights

    Recipe typically shows what worked:
    - Missing value strategy (median/mean/mode)
    - Scaling strategy (standard/robust/minmax)
    - Encoding strategy (onehot/label/target)

    Args:
        df: Raw dataframe
        recipe_insights: Dict from recipe.json (optional)

    Returns:
        Preprocessed dataframe
    """

    df_processed = df.copy()

    # ============================================================================
    # STEP 1: HANDLE MISSING VALUES
    # (Check recipe.txt to see what auto mode did)
    # ============================================================================

    # Example: Median for numeric with <30% missing
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_pct = df_processed[col].isnull().sum() / len(df_processed)
        if missing_pct > 0:
            if missing_pct < 0.3:
                # Low missing: Use median
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            else:
                # High missing: Create indicator + median
                df_processed[f'{col}_missing'] = df_processed[col].isnull().astype(int)
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    # Example: Mode for categorical
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    # ============================================================================
    # STEP 2: ENCODE CATEGORICAL VARIABLES
    # (Check recipe.txt for encoding strategy used)
    # ============================================================================

    # Example: One-hot encoding for low cardinality
    for col in categorical_cols:
        cardinality = df_processed[col].nunique()

        if cardinality <= 10:
            # Low cardinality: One-hot encode
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed.drop(col, axis=1, inplace=True)
        else:
            # High cardinality: Label encode
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))

    # ============================================================================
    # STEP 3: SCALE FEATURES
    # (Check recipe.txt for scaling strategy used)
    # ============================================================================

    # Example: StandardScaler for normal distributions
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns

    scaler = StandardScaler()  # Or RobustScaler() if recipe says outliers present
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    print(f"✓ Preprocessing complete: {df.shape} → {df_processed.shape}")

    return df_processed


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("data/raw/your_data.csv")

    # Preprocess
    df_processed = preprocess_manual(df)

    # Save
    df_processed.to_csv("data/processed/manual_preprocessed.csv", index=False)
    print("Saved to data/processed/manual_preprocessed.csv")
