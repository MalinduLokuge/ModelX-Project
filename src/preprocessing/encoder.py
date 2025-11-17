"""Categorical encoding strategies"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from category_encoders import TargetEncoder


class CategoricalEncoder:
    """Smart categorical encoding"""

    def __init__(self, strategy: str = "auto", target_col: str = None, logger=None):
        self.strategy = strategy
        self.target_col = target_col
        self.logger = logger
        self.encoders = {}
        self.encoding_map = {}

    def fit_transform(self, df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Fit and transform categorical variables"""
        df_encoded = df.copy()

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            return df_encoded

        if self.logger:
            self.logger.info(f"Encoding {len(categorical_cols)} categorical columns")

        for col in categorical_cols:
            df_encoded = self._encode_column(df_encoded, col, target)

        return df_encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted encoders"""
        df_encoded = df.copy()

        for col, encoder_info in self.encoding_map.items():
            if col not in df.columns:
                continue

            method = encoder_info['method']

            if method == 'label':
                encoder = self.encoders[col]
                # Handle unseen categories
                df_encoded[col] = df[col].map(lambda x: encoder.transform([x])[0]
                                               if x in encoder.classes_ else -1)

            elif method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                # Align columns with training
                for dummy_col in encoder_info['columns']:
                    if dummy_col not in dummies.columns:
                        dummies[dummy_col] = 0
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies[encoder_info['columns']]], axis=1)

            elif method == 'target':
                encoder = self.encoders[col]
                df_encoded[col] = encoder.transform(df[[col]])

        return df_encoded

    def _encode_column(self, df: pd.DataFrame, col: str, target: pd.Series = None) -> pd.DataFrame:
        """Encode a single column based on cardinality"""
        n_unique = df[col].nunique()

        # Low cardinality: One-hot encode
        if n_unique <= 10:
            return self._onehot_encode(df, col)

        # Medium cardinality: Label encode
        elif n_unique <= 50:
            return self._label_encode(df, col)

        # High cardinality: Target encode if target available, else label encode
        else:
            if target is not None and len(target) == len(df):
                return self._target_encode(df, col, target)
            else:
                return self._label_encode(df, col)

    def _onehot_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """One-hot encoding"""
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df_encoded = pd.concat([df.drop(col, axis=1), dummies], axis=1)

        self.encoding_map[col] = {
            'method': 'onehot',
            'columns': dummies.columns.tolist()
        }

        return df_encoded

    def _label_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Label encoding"""
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

        self.encoders[col] = encoder
        self.encoding_map[col] = {
            'method': 'label',
            'n_classes': len(encoder.classes_)
        }

        return df

    def _target_encode(self, df: pd.DataFrame, col: str, target: pd.Series) -> pd.DataFrame:
        """Target encoding"""
        encoder = TargetEncoder(cols=[col])

        df_encoded = df.copy()
        df_encoded[col] = encoder.fit_transform(df[[col]], target)

        self.encoders[col] = encoder
        self.encoding_map[col] = {
            'method': 'target',
            'original_cardinality': df[col].nunique()
        }

        return df_encoded

    def get_report(self) -> Dict[str, Any]:
        """Get encoding report"""
        return {
            'strategy': self.strategy,
            'columns_encoded': list(self.encoding_map.keys()),
            'encoding_methods': {col: info['method'] for col, info in self.encoding_map.items()},
            'n_columns': len(self.encoding_map)
        }
