"""Feature scaling strategies"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureScaler:
    """Smart feature scaling"""

    def __init__(self, strategy: str = "auto", logger=None):
        self.strategy = strategy
        self.logger = logger
        self.scalers = {}
        self.scaled_columns = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform numeric features"""
        df_scaled = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return df_scaled

        if self.logger:
            self.logger.info(f"Scaling {len(numeric_cols)} numeric columns with {self.strategy} strategy")

        for col in numeric_cols:
            df_scaled[col] = self._scale_column(df_scaled[col], col)

        self.scaled_columns = numeric_cols

        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted scalers"""
        df_scaled = df.copy()

        for col, scaler in self.scalers.items():
            if col in df.columns:
                df_scaled[col] = scaler.transform(df[[col]])

        return df_scaled

    def _scale_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """Scale a single column"""

        # Determine scaling strategy
        if self.strategy == "auto":
            scaler = self._choose_scaler(series)
        elif self.strategy == "standard":
            scaler = StandardScaler()
        elif self.strategy == "minmax":
            scaler = MinMaxScaler()
        elif self.strategy == "robust":
            scaler = RobustScaler()
        else:
            return series  # No scaling

        # Fit and transform
        scaled_values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        self.scalers[col_name] = scaler

        return pd.Series(scaled_values, index=series.index, name=col_name)

    def _choose_scaler(self, series: pd.Series) -> Any:
        """Choose scaler based on data distribution"""

        # Check for outliers using IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
        outlier_pct = outlier_mask.sum() / len(series)

        # If many outliers, use RobustScaler
        if outlier_pct > 0.05:
            return RobustScaler()

        # Check if bounded (0-1 or positive only)
        if series.min() >= 0 and series.max() <= 1:
            return MinMaxScaler()

        # Default to StandardScaler
        return StandardScaler()

    def get_report(self) -> Dict[str, Any]:
        """Get scaling report"""
        scaler_types = {}
        for col, scaler in self.scalers.items():
            scaler_types[col] = type(scaler).__name__

        return {
            'strategy': self.strategy,
            'columns_scaled': self.scaled_columns,
            'scaler_types': scaler_types,
            'n_columns': len(self.scaled_columns)
        }
