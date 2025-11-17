"""Manual Feature Engineering Template
Create features manually based on what AutoML discovered.

Usage:
1. Run CompeteML in auto mode first
2. Review outputs/YOUR_RUN_ID/recipe.txt
3. Check feature_engineering section for what features were created
4. Customize this template to create similar features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def engineer_features_manual(df: pd.DataFrame, recipe_insights: dict = None) -> pd.DataFrame:
    """
    Create features manually based on recipe

    Recipe typically shows important features created:
    - Interaction features (col1 * col2, col1 / col2)
    - Polynomial features (col^2, col^3, sqrt(col))
    - Statistical features (row mean, std, min, max)
    - Competition tricks (target encoding, frequency encoding)

    Args:
        df: Preprocessed dataframe
        recipe_insights: Dict from recipe.json (optional)

    Returns:
        Dataframe with engineered features
    """

    df_features = df.copy()
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

    # ============================================================================
    # STEP 1: INTERACTION FEATURES
    # (Check recipe.txt to see which interactions were created)
    # ============================================================================

    # Example: Create interactions between top features
    # Replace with actual important features from recipe
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]  # Replace with actual important cols

        # Multiplication
        df_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]

        # Division (avoid divide by zero)
        df_features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-10)

        # Addition
        df_features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]

        print(f"  Created interactions: {col1} x {col2}")

    # ============================================================================
    # STEP 2: POLYNOMIAL FEATURES
    # (Check recipe.txt if polynomial features were enabled)
    # ============================================================================

    # Example: Create polynomial features for important columns
    # Replace with actual important features from recipe
    if len(numeric_cols) >= 1:
        important_col = numeric_cols[0]  # Replace with actual important col

        # Squared
        df_features[f'{important_col}_squared'] = df[important_col] ** 2

        # Cubed
        df_features[f'{important_col}_cubed'] = df[important_col] ** 3

        # Square root (if positive)
        if df[important_col].min() >= 0:
            df_features[f'{important_col}_sqrt'] = np.sqrt(df[important_col])

        print(f"  Created polynomials: {important_col}")

    # ============================================================================
    # STEP 3: STATISTICAL FEATURES
    # (Row-wise aggregations - usually helpful)
    # ============================================================================

    if len(numeric_cols) >= 3:
        # Row-wise statistics
        df_features['row_mean'] = df[numeric_cols].mean(axis=1)
        df_features['row_std'] = df[numeric_cols].std(axis=1)
        df_features['row_min'] = df[numeric_cols].min(axis=1)
        df_features['row_max'] = df[numeric_cols].max(axis=1)
        df_features['row_median'] = df[numeric_cols].median(axis=1)

        print(f"  Created 5 statistical features")

    # ============================================================================
    # STEP 4: DOMAIN-SPECIFIC FEATURES
    # (Add your own domain knowledge here)
    # ============================================================================

    # Example: Time-based features
    # if 'date' in df.columns:
    #     df_features['date'] = pd.to_datetime(df['date'])
    #     df_features['year'] = df_features['date'].dt.year
    #     df_features['month'] = df_features['date'].dt.month
    #     df_features['day_of_week'] = df_features['date'].dt.dayofweek
    #     df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)

    # Example: Business logic features
    # if 'price' in df.columns and 'quantity' in df.columns:
    #     df_features['total_value'] = df['price'] * df['quantity']

    print(f"✓ Feature engineering complete: {df.shape[1]} → {df_features.shape[1]} features")

    return df_features


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("data/processed/manual_preprocessed.csv")

    # Engineer features
    df_features = engineer_features_manual(df)

    # Save
    df_features.to_csv("data/processed/manual_features.csv", index=False)
    print("Saved to data/processed/manual_features.csv")
