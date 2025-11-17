"""Competition-winning techniques and tricks"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import KFold, StratifiedKFold


class CompetitionTricks:
    """Collection of competition-winning feature engineering techniques"""

    def __init__(self, logger=None):
        self.logger = logger
        self.encoders = {}

    def target_encode_cv(self, X: pd.DataFrame, y: pd.Series, categorical_cols: list,
                         problem_type: str = 'classification', n_folds: int = 5) -> pd.DataFrame:
        """
        Target encoding with cross-validation to prevent leakage

        CRITICAL: Must use CV to avoid leakage!
        This is a common competition technique that often improves scores.
        """

        if self.logger:
            self.logger.info(f"Applying CV target encoding to {len(categorical_cols)} columns...")

        X_encoded = X.copy()

        # Setup CV
        if problem_type == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for col in categorical_cols:
            if col not in X.columns:
                continue

            # Create new column for target encoding
            new_col = f'{col}_target_enc'
            X_encoded[new_col] = 0.0

            # Calculate global mean as fallback
            global_mean = y.mean()

            # CV target encoding
            for train_idx, val_idx in kf.split(X, y):
                # Calculate mean target for each category on training fold
                means = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()

                # Apply to validation fold (prevents leakage)
                X_encoded.loc[X.index[val_idx], new_col] = X[col].iloc[val_idx].map(means).fillna(global_mean)

            # Store encoding for test set (use full training data)
            self.encoders[col] = y.groupby(X[col]).mean().to_dict()
            self.encoders[col]['__global_mean__'] = global_mean

        if self.logger:
            self.logger.info(f"  Created {len(categorical_cols)} target-encoded features")

        return X_encoded

    def target_encode_test(self, X_test: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """Apply learned target encoding to test set"""

        X_encoded = X_test.copy()

        for col in categorical_cols:
            if col not in X_test.columns or col not in self.encoders:
                continue

            new_col = f'{col}_target_enc'
            encoding_map = self.encoders[col]
            global_mean = encoding_map['__global_mean__']

            # Apply encoding, use global mean for unseen categories
            X_encoded[new_col] = X_test[col].map(encoding_map).fillna(global_mean)

        return X_encoded

    def frequency_encode(self, X: pd.DataFrame, categorical_cols: list) -> Tuple[pd.DataFrame, Dict]:
        """
        Frequency encoding: Replace category with its frequency

        Simple but often effective for high-cardinality categoricals
        """

        if self.logger:
            self.logger.info(f"Applying frequency encoding to {len(categorical_cols)} columns...")

        X_encoded = X.copy()
        freq_maps = {}

        for col in categorical_cols:
            if col not in X.columns:
                continue

            # Calculate frequencies
            freq = X[col].value_counts(normalize=True).to_dict()
            freq_maps[col] = freq

            # Create new feature
            new_col = f'{col}_freq'
            X_encoded[new_col] = X[col].map(freq).fillna(0)

        if self.logger:
            self.logger.info(f"  Created {len(categorical_cols)} frequency-encoded features")

        return X_encoded, freq_maps

    def create_feature_combinations(self, X: pd.DataFrame, col_pairs: list) -> pd.DataFrame:
        """
        Create combined features from pairs of categorical columns

        Example: city + store_type → 'NYC_supermarket'
        Often captures interactions better than one-hot encoding
        """

        if self.logger:
            self.logger.info(f"Creating {len(col_pairs)} feature combinations...")

        X_combined = X.copy()

        for col1, col2 in col_pairs:
            if col1 not in X.columns or col2 not in X.columns:
                continue

            # Combine as string
            new_col = f'{col1}_{col2}_combo'
            X_combined[new_col] = X[col1].astype(str) + '_' + X[col2].astype(str)

        if self.logger:
            self.logger.info(f"  Created {len(col_pairs)} combined features")

        return X_combined

    def add_noise_features(self, X: pd.DataFrame, n_features: int = 3, seed: int = 42) -> pd.DataFrame:
        """
        Add random noise features to detect overfitting

        If model uses these features significantly, it's overfitting!
        Competition trick: helps identify when model is fitting noise
        """

        if self.logger:
            self.logger.info(f"Adding {n_features} noise features for overfitting detection...")

        X_noise = X.copy()
        np.random.seed(seed)

        for i in range(n_features):
            X_noise[f'noise_{i}'] = np.random.randn(len(X))

        return X_noise


def apply_competition_tricks(X: pd.DataFrame, y: pd.Series, problem_type: str,
                             config: Dict, logger=None) -> pd.DataFrame:
    """
    Apply collection of competition tricks

    This function applies various competition-winning techniques that
    often improve model performance by 1-3%.
    """

    if not config.get('apply_competition_tricks', False):
        return X

    if logger:
        logger.info("Applying competition tricks...")

    tricks = CompetitionTricks(logger)
    X_enhanced = X.copy()

    # 1. Target encoding with CV (for high-cardinality categoricals)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality_cols = [col for col in categorical_cols if X[col].nunique() > 10]

    if high_cardinality_cols:
        X_enhanced = tricks.target_encode_cv(X_enhanced, y, high_cardinality_cols, problem_type)

    # 2. Frequency encoding (for all categoricals)
    if categorical_cols:
        X_enhanced, _ = tricks.frequency_encode(X_enhanced, categorical_cols)

    # 3. Feature combinations (top categorical pairs)
    if len(categorical_cols) >= 2:
        # Combine top 2 categoricals
        pairs = [(categorical_cols[0], categorical_cols[1])]
        X_enhanced = tricks.create_feature_combinations(X_enhanced, pairs)

    if logger:
        logger.info(f"  Competition tricks applied: {X.shape[1]} → {X_enhanced.shape[1]} features")

    return X_enhanced
