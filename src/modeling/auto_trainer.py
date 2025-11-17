"""Automated model training orchestrator"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from .autogluon_wrapper import AutoGluonWrapper


class AutoTrainer:
    """Automated model training"""

    def __init__(self, config: Dict[str, Any], output_dir: Path, logger=None):
        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        self.framework = config.get('automl_framework', 'autogluon')
        self.model_wrapper = None
        self.training_results = {}

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Train models using selected AutoML framework"""

        if self.logger:
            self.logger.info(f"Training with {self.framework}")

        # Initialize the appropriate wrapper
        if self.framework == 'autogluon':
            self.model_wrapper = AutoGluonWrapper(
                config=self.config,
                output_dir=self.output_dir,
                logger=self.logger
            )
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        # Prepare training data
        # For AutoGluon, we can pass the full dataframe with target
        if val_data is not None:
            train_data_combined = pd.concat([train_data, val_data], axis=0, ignore_index=True)
        else:
            train_data_combined = train_data

        # Get time limit
        time_limit = self.config.get('time_limit', 3600)

        # Quick test mode
        if self.config.get('quick_test', False):
            time_limit = 300  # 5 minutes
            if self.logger:
                self.logger.info("Quick test mode: time limit set to 5 minutes")

        # Get preset
        preset = self.config.get('ag_preset', 'medium_quality')

        # Train
        results = self.model_wrapper.train(
            train_data=train_data_combined,
            time_limit=time_limit,
            preset=preset
        )

        self.training_results = results

        return results

    def predict(self, test_data: pd.DataFrame) -> pd.Series:
        """Make predictions"""
        if self.model_wrapper is None:
            raise ValueError("Model not trained yet")

        return self.model_wrapper.predict(test_data)

    def predict_proba(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities"""
        if self.model_wrapper is None:
            raise ValueError("Model not trained yet")

        return self.model_wrapper.predict_proba(test_data)

    def get_results(self) -> Dict[str, Any]:
        """Get training results"""
        return self.training_results

    def get_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard"""
        if self.model_wrapper is None:
            raise ValueError("Model not trained yet")

        return self.model_wrapper.get_leaderboard()

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance"""
        if self.model_wrapper is None:
            raise ValueError("Model not trained yet")

        return self.model_wrapper.get_feature_importance()

    def load_model(self, path: str):
        """Load a saved model"""
        if self.framework == 'autogluon':
            self.model_wrapper = AutoGluonWrapper(
                config=self.config,
                output_dir=self.output_dir,
                logger=self.logger
            )
            self.model_wrapper.load_model(path)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
