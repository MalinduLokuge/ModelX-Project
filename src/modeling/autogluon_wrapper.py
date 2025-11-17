"""AutoGluon integration wrapper"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from autogluon.tabular import TabularPredictor


class AutoGluonWrapper:
    """Wrapper for AutoGluon TabularPredictor"""

    def __init__(self, config: Dict[str, Any], output_dir: Path, logger=None):
        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        self.predictor = None
        self.problem_type = config.get('problem_type', None)
        self.target_column = config.get('target_column')

    def train(self, train_data: pd.DataFrame, time_limit: int = None,
              preset: str = "medium_quality") -> Dict[str, Any]:
        """Train AutoGluon model"""

        if self.logger:
            self.logger.info(f"Training AutoGluon with preset: {preset}")
            self.logger.info(f"Time limit: {time_limit}s")

        # Prepare save path
        save_path = self.output_dir / "ag_models"
        save_path.mkdir(parents=True, exist_ok=True)

        # Determine eval metric
        eval_metric = self._get_eval_metric()

        # Train predictor
        try:
            self.predictor = TabularPredictor(
                label=self.target_column,
                problem_type=self.problem_type,
                eval_metric=eval_metric,
                path=str(save_path),
                verbosity=2 if self.config.get('verbose', 2) >= 2 else 0
            )

            self.predictor.fit(
                train_data=train_data,
                time_limit=time_limit,
                presets=preset,
                num_bag_folds=self.config.get('ag_num_bag_folds', 5),
                num_stack_levels=self.config.get('ag_num_stack_levels', 1),
            )

            if self.logger:
                self.logger.info("✓ AutoGluon training completed")

            # Get results
            results = self._get_training_results()

            return results

        except Exception as e:
            if self.logger:
                self.logger.error(f"AutoGluon training failed: {str(e)}")
            raise

    def predict(self, test_data: pd.DataFrame) -> pd.Series:
        """Make predictions on test data"""
        if self.predictor is None:
            raise ValueError("Model not trained yet")

        return self.predictor.predict(test_data)

    def predict_proba(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities (classification only)"""
        if self.predictor is None:
            raise ValueError("Model not trained yet")

        if self.problem_type != 'classification':
            raise ValueError("predict_proba only available for classification")

        return self.predictor.predict_proba(test_data)

    def _get_eval_metric(self) -> Optional[str]:
        """Determine evaluation metric"""
        eval_metric = self.config.get('eval_metric', None)

        if eval_metric:
            return eval_metric

        # Auto-select based on problem type
        if self.problem_type == 'classification':
            return 'roc_auc'  # Default for binary classification
        elif self.problem_type == 'regression':
            return 'rmse'

        return None

    def _get_training_results(self) -> Dict[str, Any]:
        """Extract training results and model info"""

        results = {
            'framework': 'autogluon',
            'best_model': self.predictor.model_best,
            'leaderboard': None,
            'feature_importance': None,
            'model_path': str(self.predictor.path)
        }

        # Get leaderboard
        try:
            leaderboard = self.predictor.leaderboard(silent=True)
            results['leaderboard'] = leaderboard.to_dict()

            if self.logger:
                self.logger.info(f"Best model: {results['best_model']}")
                self.logger.info(f"Score: {leaderboard.iloc[0]['score_val']:.4f}")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not get leaderboard: {e}")

        # Get feature importance
        try:
            feature_importance = self.predictor.feature_importance(data=self.predictor.load_data_internal()[0])
            results['feature_importance'] = feature_importance.to_dict()

            if self.logger:
                top_features = feature_importance.head(10)
                self.logger.info(f"Top 10 features: {list(top_features.index)}")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not get feature importance: {e}")

        return results

    def load_model(self, path: str):
        """Load a saved model"""
        self.predictor = TabularPredictor.load(path)

        if self.logger:
            self.logger.info(f"Model loaded from: {path}")

    def get_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard"""
        if self.predictor is None:
            raise ValueError("Model not trained yet")

        return self.predictor.leaderboard(silent=True)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance"""
        if self.predictor is None:
            raise ValueError("Model not trained yet")

        return self.predictor.feature_importance(data=self.predictor.load_data_internal()[0])
