"""Handle missing values intelligently"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.impute import SimpleImputer, KNNImputer


class MissingValueHandler:
    """Smart missing value imputation"""

    def __init__(self, strategy: str = "auto", logger=None):
        self.strategy = strategy
        self.logger = logger
        self.imputers = {}
        self.fill_values = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        df_filled = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Handle numeric columns
        if len(numeric_cols) > 0:
            df_filled[numeric_cols] = self._handle_numeric(df[numeric_cols])

        # Handle categorical columns
        if len(categorical_cols) > 0:
            df_filled[categorical_cols] = self._handle_categorical(df[categorical_cols])

        if self.logger:
            n_filled = df.isnull().sum().sum() - df_filled.isnull().sum().sum()
            self.logger.info(f"Filled {n_filled} missing values")

        return df_filled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted imputers"""
        df_filled = df.copy()

        for col, imputer in self.imputers.items():
            if col in df.columns:
                df_filled[col] = imputer.transform(df[[col]])

        for col, value in self.fill_values.items():
            if col in df.columns:
                df_filled[col].fillna(value, inplace=True)

        return df_filled

    def _handle_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle numeric missing values"""
        df_filled = df.copy()

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            missing_pct = df[col].isnull().sum() / len(df)

            if missing_pct < 0.05:
                # Low missing: use median
                fill_value = df[col].median()
                df_filled[col] = df_filled[col].fillna(fill_value)
                self.fill_values[col] = fill_value

            elif missing_pct < 0.30:
                # Medium missing: use mean
                fill_value = df[col].mean()
                df_filled[col] = df_filled[col].fillna(fill_value)
                self.fill_values[col] = fill_value

            else:
                # High missing: create missing indicator and fill with median
                df_filled[f'{col}_missing'] = df[col].isnull().astype(int)
                fill_value = df[col].median()
                df_filled[col] = df_filled[col].fillna(fill_value)
                self.fill_values[col] = fill_value

        return df_filled

    def _handle_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical missing values"""
        df_filled = df.copy()

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            missing_pct = df[col].isnull().sum() / len(df)

            if missing_pct < 0.05:
                # Low missing: use mode
                fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df_filled[col] = df_filled[col].fillna(fill_value)
                self.fill_values[col] = fill_value

            else:
                # Medium/High missing: fill with 'Missing'
                df_filled[col] = df_filled[col].fillna('Missing')
                self.fill_values[col] = 'Missing'

        return df_filled

    def get_report(self) -> Dict[str, Any]:
        """Get report of missing value handling"""
        return {
            'strategy': self.strategy,
            'columns_handled': list(set(list(self.imputers.keys()) + list(self.fill_values.keys()))),
            'n_columns': len(set(list(self.imputers.keys()) + list(self.fill_values.keys())))
        }
