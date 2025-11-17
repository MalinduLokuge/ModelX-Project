"""Automated feature engineering for competitions"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression

from .competition_tricks import apply_competition_tricks


class AutoFeatureEngineer:
    """Automatically create and select features"""

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.created_features = []
        self.removed_features = []
        self.feature_scores = {}

    def engineer_features(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main feature engineering pipeline"""

        if self.logger:
            self.logger.section("Feature Engineering")

        X_featured = X.copy()
        original_cols = set(X.columns)

        report = {
            'original_features': len(X.columns),
            'steps': []
        }

        # 0. Apply competition tricks FIRST (before encoding categoricals)
        # This ensures categorical columns are still available for target/frequency encoding
        if self.config.get('apply_competition_tricks', False):
            before_tricks = X_featured.shape[1]
            X_featured = apply_competition_tricks(X_featured, y, problem_type, self.config, self.logger)
            tricks_created = X_featured.shape[1] - before_tricks
            if tricks_created > 0:
                report['steps'].append(f"Applied competition tricks: +{tricks_created} features")

        # Now encode any remaining categorical columns
        X_featured = self._encode_remaining_categoricals(X_featured)

        # 1. Interaction features
        if self.config.get('interaction_features', True):
            X_featured = self._create_interactions(X_featured, y, problem_type)
            report['steps'].append(f"Created {len(self.created_features)} interaction features")

        # 2. Polynomial features (top features only)
        if self.config.get('polynomial_features', False):
            X_featured = self._create_polynomials(X_featured, y, problem_type)
            report['steps'].append("Created polynomial features")

        # 3. Statistical features (row-wise)
        X_featured = self._create_statistical_features(X_featured)
        report['steps'].append("Created statistical features")

        # 4. Feature selection
        if self.config.get('feature_selection', True):
            X_featured = self._select_features(X_featured, y, problem_type)
            report['steps'].append(f"Selected best features, removed {len(self.removed_features)}")

        # Calculate final stats
        new_cols = set(X_featured.columns) - original_cols
        report['features_created'] = list(new_cols)
        report['features_removed'] = self.removed_features
        report['final_features'] = len(X_featured.columns)

        if self.logger:
            self.logger.info(f"✓ Feature engineering complete: {len(X.columns)} → {len(X_featured.columns)} features")

        return X_featured, report

    def _encode_remaining_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode any remaining categorical columns using label encoding"""
        from sklearn.preprocessing import LabelEncoder

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            return X

        if self.logger:
            self.logger.info(f"Encoding {len(categorical_cols)} remaining categorical columns...")

        X_encoded = X.copy()

        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))

        return X_encoded

    def _create_interactions(self, X: pd.DataFrame, y: pd.Series, problem_type: str, top_n: int = 5) -> pd.DataFrame:
        """Create interaction features from top numeric columns"""

        if self.logger:
            self.logger.info("Creating interaction features...")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return X

        # Get top N most important features
        if len(numeric_cols) > top_n:
            # Quick importance scoring
            scores = self._get_feature_importance(X[numeric_cols], y, problem_type)
            top_cols = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_n]
        else:
            top_cols = numeric_cols

        X_new = X.copy()
        interactions_created = 0

        # Create pairwise interactions
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                col1, col2 = top_cols[i], top_cols[j]

                # Multiplication
                new_col = f"{col1}_x_{col2}"
                X_new[new_col] = X[col1] * X[col2]
                self.created_features.append(new_col)
                interactions_created += 1

                # Division (avoid divide by zero)
                new_col = f"{col1}_div_{col2}"
                X_new[new_col] = X[col1] / (X[col2] + 1e-10)
                self.created_features.append(new_col)
                interactions_created += 1

                # Addition
                new_col = f"{col1}_plus_{col2}"
                X_new[new_col] = X[col1] + X[col2]
                self.created_features.append(new_col)
                interactions_created += 1

        if self.logger:
            self.logger.info(f"  Created {interactions_created} interaction features")

        return X_new

    def _create_polynomials(self, X: pd.DataFrame, y: pd.Series, problem_type: str, top_n: int = 5) -> pd.DataFrame:
        """Create polynomial features for top columns"""

        if self.logger:
            self.logger.info("Creating polynomial features...")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            return X

        # Get top features
        scores = self._get_feature_importance(X[numeric_cols], y, problem_type)
        top_cols = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:min(top_n, len(numeric_cols))]

        X_new = X.copy()

        for col in top_cols:
            # Squared
            new_col = f"{col}_squared"
            X_new[new_col] = X[col] ** 2
            self.created_features.append(new_col)

            # Cubed
            new_col = f"{col}_cubed"
            X_new[new_col] = X[col] ** 3
            self.created_features.append(new_col)

            # Square root (for positive values)
            if X[col].min() >= 0:
                new_col = f"{col}_sqrt"
                X_new[new_col] = np.sqrt(X[col])
                self.created_features.append(new_col)

        if self.logger:
            self.logger.info(f"  Created polynomial features for {len(top_cols)} columns")

        return X_new

    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create row-wise statistical features"""

        if self.logger:
            self.logger.info("Creating statistical features...")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 3:
            return X

        X_new = X.copy()

        # Row-wise statistics
        X_new['row_mean'] = X[numeric_cols].mean(axis=1)
        X_new['row_std'] = X[numeric_cols].std(axis=1)
        X_new['row_min'] = X[numeric_cols].min(axis=1)
        X_new['row_max'] = X[numeric_cols].max(axis=1)
        X_new['row_median'] = X[numeric_cols].median(axis=1)

        self.created_features.extend(['row_mean', 'row_std', 'row_min', 'row_max', 'row_median'])

        if self.logger:
            self.logger.info(f"  Created 5 row-wise statistical features")

        return X_new

    def _select_features(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> pd.DataFrame:
        """Select best features using various methods"""

        if self.logger:
            self.logger.info("Selecting best features...")

        # Remove low variance features
        X_filtered = self._remove_low_variance(X)

        # Remove highly correlated features
        X_filtered = self._remove_high_correlation(X_filtered)

        # Limit total features if too many
        max_features = self.config.get('max_features', None)
        if max_features and len(X_filtered.columns) > max_features:
            X_filtered = self._select_k_best(X_filtered, y, problem_type, k=max_features)

        return X_filtered

    def _remove_low_variance(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with very low variance"""

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        low_var_cols = []

        for col in numeric_cols:
            if X[col].std() / (X[col].mean() + 1e-10) < threshold:
                low_var_cols.append(col)
                self.removed_features.append(col)

        if low_var_cols:
            X = X.drop(columns=low_var_cols)
            if self.logger:
                self.logger.info(f"  Removed {len(low_var_cols)} low variance features")

        return X

    def _remove_high_correlation(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features"""

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return X

        corr_matrix = X[numeric_cols].corr().abs()

        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

        # Remove one from each pair
        to_remove = set()
        for col1, col2 in high_corr_pairs:
            to_remove.add(col2)  # Keep first, remove second

        if to_remove:
            X = X.drop(columns=list(to_remove))
            self.removed_features.extend(list(to_remove))
            if self.logger:
                self.logger.info(f"  Removed {len(to_remove)} highly correlated features")

        return X

    def _select_k_best(self, X: pd.DataFrame, y: pd.Series, problem_type: str, k: int) -> pd.DataFrame:
        """Select K best features using statistical tests"""

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) <= k:
            return X

        # Select scoring function
        if problem_type == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression

        # Select K best
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X[numeric_cols], y)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = numeric_cols[selected_mask].tolist()

        # Keep selected numeric + all non-numeric
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        final_cols = selected_features + non_numeric_cols

        removed = set(X.columns) - set(final_cols)
        self.removed_features.extend(list(removed))

        if self.logger:
            self.logger.info(f"  Selected {k} best features from {len(numeric_cols)}")

        return X[final_cols]

    def _get_feature_importance(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> Dict[str, float]:
        """Quick feature importance scoring"""

        # Fill NaN values temporarily for importance calculation
        X_filled = X.fillna(X.mean())

        if problem_type == 'classification':
            try:
                scores = mutual_info_classif(X_filled, y, random_state=42)
            except:
                try:
                    scores = f_classif(X_filled, y)[0]
                except:
                    # If both fail, return uniform scores
                    scores = np.ones(len(X.columns))
        else:
            try:
                scores = mutual_info_regression(X_filled, y, random_state=42)
            except:
                try:
                    scores = f_regression(X_filled, y)[0]
                except:
                    scores = np.ones(len(X.columns))

        return {col: float(score) for col, score in zip(X.columns, scores)}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using same feature engineering"""
        # This would apply the same transformations learned during fit
        # For now, return as is (full implementation would store transformers)
        return X
