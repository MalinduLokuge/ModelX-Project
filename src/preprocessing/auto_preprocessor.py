"""Automated preprocessing pipeline"""
import pandas as pd
from typing import Dict, Any, Tuple

from .missing_handler import MissingValueHandler
from .encoder import CategoricalEncoder
from .scaler import FeatureScaler


class AutoPreprocessor:
    """Automated data preprocessing"""

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

        # Initialize components
        self.missing_handler = MissingValueHandler(logger=logger)
        self.encoder = CategoricalEncoder(
            strategy=config.get('encoding_strategy', 'auto'),
            logger=logger
        )
        self.scaler = FeatureScaler(
            strategy=config.get('scaling_strategy', 'auto'),
            logger=logger
        )

        self.is_fitted = False
        self.preprocessing_steps = []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit and transform the data"""

        if self.logger:
            self.logger.info("Starting preprocessing pipeline")

        X_processed = X.copy()
        report = {'steps': []}

        # Step 1: Handle missing values
        if self.config.get('handle_missing', True):
            if self.logger:
                self.logger.info("Step 1/3: Handling missing values")

            X_processed = self.missing_handler.fit_transform(X_processed)
            self.preprocessing_steps.append("Handled missing values")
            report['steps'].append({
                'name': 'missing_values',
                'details': self.missing_handler.get_report()
            })

        # Step 2: Encode categorical variables (skip if competition tricks enabled)
        skip_encoding = self.config.get('apply_competition_tricks', False)
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) > 0 and not skip_encoding:
            if self.logger:
                self.logger.info(f"Step 2/3: Encoding {len(categorical_cols)} categorical features")

            X_processed = self.encoder.fit_transform(X_processed, y)
            self.preprocessing_steps.append("Encoded categorical variables")
            report['steps'].append({
                'name': 'categorical_encoding',
                'details': self.encoder.get_report()
            })
        else:
            if self.logger:
                if skip_encoding and len(categorical_cols) > 0:
                    self.logger.info(f"Step 2/3: Skipping encoding ({len(categorical_cols)} categoricals preserved for competition tricks)")
                else:
                    self.logger.info("Step 2/3: No categorical features to encode")

        # Step 3: Scale features
        if self.config.get('scaling_strategy', 'auto') != 'none':
            if self.logger:
                self.logger.info("Step 3/3: Scaling features")

            X_processed = self.scaler.fit_transform(X_processed)
            self.preprocessing_steps.append("Scaled features")
            report['steps'].append({
                'name': 'feature_scaling',
                'details': self.scaler.get_report()
            })
        else:
            if self.logger:
                self.logger.info("Step 3/3: Skipping feature scaling")

        self.is_fitted = True

        if self.logger:
            self.logger.info(f"Preprocessing complete: {X.shape} -> {X_processed.shape}")

        report['input_shape'] = X.shape
        report['output_shape'] = X_processed.shape
        report['n_features_added'] = X_processed.shape[1] - X.shape[1]

        return X_processed, report

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""

        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X_processed = X.copy()

        # Apply transformations in same order
        if self.config.get('handle_missing', True):
            X_processed = self.missing_handler.transform(X_processed)

        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X_processed = self.encoder.transform(X_processed)

        if self.config.get('scaling_strategy', 'auto') != 'none':
            X_processed = self.scaler.transform(X_processed)

        return X_processed

    def get_steps(self):
        """Get list of preprocessing steps performed"""
        return self.preprocessing_steps
