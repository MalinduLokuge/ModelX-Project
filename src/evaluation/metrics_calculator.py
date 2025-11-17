"""Calculate evaluation metrics"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, log_loss
)


class MetricsCalculator:
    """Calculate various evaluation metrics"""

    def __init__(self, problem_type: str, logger=None):
        self.problem_type = problem_type
        self.logger = logger

    def calculate(self, y_true: pd.Series, y_pred: pd.Series,
                  y_pred_proba: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate appropriate metrics based on problem type"""

        if self.problem_type == 'classification':
            return self._calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        elif self.problem_type == 'regression':
            return self._calculate_regression_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")

    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: pd.Series,
                                           y_pred_proba: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate classification metrics"""

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Handle binary vs multiclass
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            # Binary classification
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

            # ROC-AUC if probabilities available
            if y_pred_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba.iloc[:, 1])
                except:
                    pass

            # Log loss if probabilities available
            if y_pred_proba is not None:
                try:
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                except:
                    pass

        else:
            # Multiclass
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            # ROC-AUC if probabilities available
            if y_pred_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                except:
                    pass

        if self.logger:
            self.logger.info("Classification Metrics:")
            for metric, value in metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate regression metrics"""

        metrics = {}

        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)

        # MAPE (if no zeros in y_true)
        if not (y_true == 0).any():
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        if self.logger:
            self.logger.info("Regression Metrics:")
            for metric, value in metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def get_primary_metric(self, metrics: Dict[str, float]) -> tuple:
        """Get the primary metric for the problem type"""

        if self.problem_type == 'classification':
            if 'roc_auc' in metrics:
                return 'roc_auc', metrics['roc_auc']
            else:
                return 'accuracy', metrics['accuracy']

        elif self.problem_type == 'regression':
            return 'rmse', metrics['rmse']

        return None, None
