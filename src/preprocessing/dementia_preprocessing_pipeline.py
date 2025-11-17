"""
Comprehensive Preprocessing Pipeline for Dementia Risk Prediction
Orchestrates all preprocessing steps with detailed documentation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

from .dementia_feature_selector import DementiaFeatureSelector
from .nacc_missing_handler import NACCMissingValueHandler
from ..feature_engineering.dementia_features import DementiaFeatureEngineer


class DementiaPreprocessingPipeline:
    """
    Complete preprocessing pipeline for dementia risk prediction.

    Pipeline Steps:
    1. Feature Selection (remove medical/excluded features)
    2. Missing Value Handling (NACC-specific codes)
    3. Feature Engineering (domain-specific features)
    4. Outlier Detection and Handling
    5. Feature Scaling/Normalization
    6. Categorical Encoding
    7. Final Feature Selection (correlation, variance)
    """

    def __init__(self, target_col: str = 'dementia', logger=None):
        self.target_col = target_col
        self.logger = logger

        # Initialize components
        self.feature_selector = DementiaFeatureSelector(logger=logger)
        self.missing_handler = NACCMissingValueHandler(logger=logger)
        self.feature_engineer = DementiaFeatureEngineer(logger=logger)

        self.scaler = None
        self.label_encoders = {}

        # Reports
        self.pipeline_report = {}

    def fit_transform(self, df: pd.DataFrame, create_report: bool = True) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Fit and transform the complete preprocessing pipeline

        Parameters:
        -----------
        df : pd.DataFrame
            Raw input data with all features and target
        create_report : bool
            Whether to create detailed preprocessing report

        Returns:
        --------
        X_processed : pd.DataFrame
            Processed features
        y : pd.Series
            Target variable
        report : dict
            Comprehensive preprocessing report
        """
        if self.logger:
            self.logger.info("=" * 80)
            self.logger.info("DEMENTIA RISK PREDICTION - PREPROCESSING PIPELINE")
            self.logger.info("=" * 80)

        # Extract target
        if self.target_col in df.columns:
            y = df[self.target_col].copy()
            X = df.drop(columns=[self.target_col])
        else:
            y = None
            X = df.copy()

        self.pipeline_report['initial_shape'] = X.shape

        # STEP 1: Feature Selection (Non-Medical Features Only)
        if self.logger:
            self.logger.info("\n[STEP 1/7] Feature Selection - Removing Medical Features")
        X = self.feature_selector.select_features(X, target_col=self.target_col)
        self.pipeline_report['feature_selection'] = self.feature_selector.get_selection_report()

        # STEP 2: Missing Value Handling
        if self.logger:
            self.logger.info("\n[STEP 2/7] Missing Value Handling - NACC-Specific Codes")
        X, missing_report = self.missing_handler.fit_transform(X)
        self.pipeline_report['missing_values'] = missing_report

        # STEP 3: Feature Engineering
        if self.logger:
            self.logger.info("\n[STEP 3/7] Feature Engineering - Domain-Specific Features")
        X, feature_eng_report = self.feature_engineer.engineer_features(X)
        self.pipeline_report['feature_engineering'] = feature_eng_report

        # STEP 4: Outlier Detection and Handling
        if self.logger:
            self.logger.info("\n[STEP 4/7] Outlier Detection and Handling")
        X, outlier_report = self._handle_outliers(X)
        self.pipeline_report['outliers'] = outlier_report

        # STEP 5: Categorical Encoding
        if self.logger:
            self.logger.info("\n[STEP 5/7] Encoding Categorical Variables")
        X, encoding_report = self._encode_categoricals(X, y)
        self.pipeline_report['encoding'] = encoding_report

        # STEP 6: Feature Scaling
        if self.logger:
            self.logger.info("\n[STEP 6/7] Feature Scaling")
        X, scaling_report = self._scale_features(X)
        self.pipeline_report['scaling'] = scaling_report

        # STEP 7: Final Feature Selection (Correlation, Variance)
        if self.logger:
            self.logger.info("\n[STEP 7/7] Final Feature Selection")
        X, final_selection_report = self._final_feature_selection(X, y)
        self.pipeline_report['final_selection'] = final_selection_report

        # Summary
        self.pipeline_report['final_shape'] = X.shape
        self.pipeline_report['features_removed_total'] = self.pipeline_report['initial_shape'][1] - X.shape[1]
        self.pipeline_report['features_created_total'] = len(self.feature_engineer.created_features)

        if self.logger:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PREPROCESSING COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Initial Features: {self.pipeline_report['initial_shape'][1]}")
            self.logger.info(f"Final Features: {X.shape[1]}")
            self.logger.info(f"Features Created: {self.pipeline_report['features_created_total']}")
            self.logger.info(f"Features Removed: {self.pipeline_report['features_removed_total']}")

        return X, y, self.pipeline_report

    def _handle_outliers(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle outliers using IQR method"""

        X_clean = X.copy()
        outlier_report = {
            'method': 'IQR (Interquartile Range)',
            'threshold': '1.5 × IQR',
            'treatment': 'Capping (Winsorization)',
            'features_affected': [],
            'outliers_capped': {}
        }

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            n_outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()

            if n_outliers > 0:
                # Cap outliers
                X_clean[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_report['features_affected'].append(col)
                outlier_report['outliers_capped'][col] = {
                    'n_outliers': int(n_outliers),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }

        outlier_report['n_features_affected'] = len(outlier_report['features_affected'])

        if self.logger:
            self.logger.info(f"  Capped outliers in {len(outlier_report['features_affected'])} features")

        return X_clean, outlier_report

    def _encode_categoricals(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables"""

        from sklearn.preprocessing import LabelEncoder

        X_encoded = X.copy()
        encoding_report = {
            'strategy': 'Label Encoding for low cardinality, One-Hot for medium cardinality',
            'features_encoded': [],
            'encoding_methods': {}
        }

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
            n_unique = X[col].nunique()

            if n_unique <= 2:
                # Binary: Label encode
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                encoding_report['encoding_methods'][col] = f'Label Encoding ({n_unique} categories)'

            elif n_unique <= 10:
                # Medium cardinality: One-hot encode
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded.drop(columns=[col]), dummies], axis=1)
                encoding_report['encoding_methods'][col] = f'One-Hot Encoding ({n_unique} categories)'

            else:
                # High cardinality: Label encode
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                encoding_report['encoding_methods'][col] = f'Label Encoding ({n_unique} categories - high cardinality)'

            encoding_report['features_encoded'].append(col)

        encoding_report['n_features_encoded'] = len(categorical_cols)

        if self.logger:
            self.logger.info(f"  Encoded {len(categorical_cols)} categorical features")

        return X_encoded, encoding_report

    def _scale_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Scale numerical features using RobustScaler"""

        X_scaled = X.copy()
        scaling_report = {
            'technique': 'RobustScaler',
            'mathematical_transformation': 'x_scaled = (x - median) / IQR',
            'justification': 'RobustScaler is resilient to outliers, suitable for medical data',
            'features_scaled': []
        }

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 0:
            self.scaler = RobustScaler()
            X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            scaling_report['features_scaled'] = numeric_cols
            scaling_report['n_features_scaled'] = len(numeric_cols)

            if self.logger:
                self.logger.info(f"  Scaled {len(numeric_cols)} numerical features using RobustScaler")

        return X_scaled, scaling_report

    def _final_feature_selection(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, Dict]:
        """Final feature selection based on correlation and variance"""

        X_selected = X.copy()
        selection_report = {
            'methods': ['Correlation Analysis', 'Variance Threshold'],
            'features_removed': [],
            'removal_reasons': {}
        }

        # Remove low variance features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        low_var_threshold = 0.01

        for col in numeric_cols:
            var = X[col].var()
            if var < low_var_threshold:
                X_selected = X_selected.drop(columns=[col])
                selection_report['features_removed'].append(col)
                selection_report['removal_reasons'][col] = f'Low variance ({var:.6f})'

        # Remove highly correlated features
        if len(X_selected.select_dtypes(include=[np.number]).columns) > 1:
            corr_matrix = X_selected.select_dtypes(include=[np.number]).corr().abs()
            high_corr_threshold = 0.95

            # Find pairs of highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > high_corr_threshold:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

            # Remove one from each pair
            for col1, col2, corr_val in high_corr_pairs:
                if col2 in X_selected.columns and col2 not in selection_report['features_removed']:
                    X_selected = X_selected.drop(columns=[col2])
                    selection_report['features_removed'].append(col2)
                    selection_report['removal_reasons'][col2] = f'High correlation with {col1} ({corr_val:.3f})'

        selection_report['n_features_removed'] = len(selection_report['features_removed'])

        if self.logger:
            self.logger.info(f"  Removed {len(selection_report['features_removed'])} features (low variance, high correlation)")

        return X_selected, selection_report

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Transform new data using fitted pipeline"""

        # Extract target if present
        if self.target_col in df.columns:
            y = df[self.target_col].copy()
            X = df.drop(columns=[self.target_col])
        else:
            y = None
            X = df.copy()

        # Step 1: Feature selection
        X_proc = self.feature_selector.select_features(X, target_col=self.target_col)

        # Step 2: Missing values
        X_proc = self.missing_handler.transform(X_proc)

        # Step 3: Feature engineering
        X_proc, _ = self.feature_engineer.engineer_features(X_proc)

        # Step 4: Outlier handling (already fitted, just apply same caps)
        # Skip for inference - outliers should be handled same way as training

        # Step 5: Encoding
        # For categorical features, use label encoders
        categorical_cols = X_proc.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                X_proc[col] = X_proc[col].astype(str)
                X_proc[col] = X_proc[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                # If not in label encoders, drop it
                X_proc = X_proc.drop(columns=[col])

        # Step 6: Scaling
        if self.scaler:
            numeric_cols = X_proc.select_dtypes(include=[np.number]).columns
            X_proc[numeric_cols] = self.scaler.transform(X_proc[numeric_cols])

        return X_proc, y

    def get_report(self) -> Dict:
        """Get comprehensive preprocessing report"""
        return self.pipeline_report

    def save_pipeline(self, output_dir: str):
        """Save pipeline components and configuration"""
        import pickle

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save pipeline components
        with open(output_path / 'preprocessing_pipeline.pkl', 'wb') as f:
            pickle.dump({
                'feature_selector': self.feature_selector,
                'missing_handler': self.missing_handler,
                'feature_engineer': self.feature_engineer,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders
            }, f)

        # Save report as JSON
        with open(output_path / 'preprocessing_report.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            report_serializable = self._make_serializable(self.pipeline_report)
            json.dump(report_serializable, f, indent=2)

        if self.logger:
            self.logger.info(f"Pipeline saved to {output_path}")

    def _make_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
