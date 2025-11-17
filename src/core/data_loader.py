"""Smart data loading and detection"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split


class DataLoader:
    """Intelligent data loading with auto-detection"""

    def __init__(self, logger=None):
        self.logger = logger
        self.train_df = None
        self.test_df = None
        self.target_column = None
        self.id_column = None
        self.problem_type = None
        self.metadata = {}

    def load(self, train_path: str, test_path: Optional[str] = None,
             target_column: Optional[str] = None, id_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load training and test data"""

        if self.logger:
            self.logger.info(f"Loading data from: {train_path}")

        # Load train
        self.train_df = self._read_file(train_path)
        if self.logger:
            self.logger.info(f"Train shape: {self.train_df.shape}")

        # Load test if provided
        if test_path:
            self.test_df = self._read_file(test_path)
            if self.logger:
                self.logger.info(f"Test shape: {self.test_df.shape}")

        # Detect target column
        self.target_column = self._detect_target(target_column)

        # Detect ID column
        self.id_column = self._detect_id_column(id_column)

        # Detect problem type
        self.problem_type = self._detect_problem_type()

        # Generate metadata
        self._generate_metadata()

        if self.logger:
            self.logger.info(f"Target column: {self.target_column}")
            self.logger.info(f"ID column: {self.id_column}")
            self.logger.info(f"Problem type: {self.problem_type}")

        return self.train_df, self.test_df

    def _read_file(self, path: str) -> pd.DataFrame:
        """Read file based on extension with auto-detection"""
        path = Path(path)
        ext = path.suffix.lower()

        if ext == '.csv':
            # Try different encodings and delimiters
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    # Try to detect delimiter
                    with open(path, 'r', encoding=encoding) as f:
                        first_line = f.readline()

                    delimiter = self._detect_delimiter(first_line)
                    df = pd.read_csv(path, encoding=encoding, delimiter=delimiter)

                    if self.logger:
                        self.logger.debug(f"CSV loaded: encoding={encoding}, delimiter='{delimiter}'")

                    return df
                except Exception:
                    continue

            # Fallback
            return pd.read_csv(path)

        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif ext == '.parquet':
            return pd.read_parquet(path)
        elif ext == '.json':
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _detect_delimiter(self, line: str) -> str:
        """Detect CSV delimiter"""
        delimiters = [',', ';', '\t', '|']
        counts = {delim: line.count(delim) for delim in delimiters}
        return max(counts, key=counts.get) if max(counts.values()) > 0 else ','

    def _detect_target(self, target_column: Optional[str] = None) -> str:
        """Detect or validate target column"""
        if target_column:
            if target_column in self.train_df.columns:
                return target_column
            else:
                raise ValueError(f"Target column '{target_column}' not found in data")

        # Common target column names
        common_targets = ['target', 'label', 'y', 'class', 'outcome']
        for col in common_targets:
            if col in self.train_df.columns:
                return col

        # If test data exists, find column not in test
        if self.test_df is not None:
            test_cols = set(self.test_df.columns)
            train_cols = set(self.train_df.columns)
            unique_cols = train_cols - test_cols

            # Exclude likely ID columns
            id_patterns = ['id', 'index', 'key']
            candidates = [col for col in unique_cols
                          if not any(pattern in col.lower() for pattern in id_patterns)]

            if len(candidates) == 1:
                return candidates[0]

        # Default to last column
        return self.train_df.columns[-1]

    def _detect_id_column(self, id_column: Optional[str] = None) -> Optional[str]:
        """Detect ID column"""
        if id_column:
            return id_column if id_column in self.train_df.columns else None

        # Common ID column patterns
        id_patterns = ['id', 'index', 'key', 'idx']

        for col in self.train_df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in id_patterns):
                # Check if values are unique
                if self.train_df[col].nunique() == len(self.train_df):
                    return col

        return None

    def _detect_problem_type(self) -> str:
        """Detect if problem is classification or regression"""
        if self.target_column is None:
            return "unknown"

        target = self.train_df[self.target_column]

        # Check if numeric
        if not pd.api.types.is_numeric_dtype(target):
            return "classification"

        # Check unique values ratio
        n_unique = target.nunique()
        n_samples = len(target)
        unique_ratio = n_unique / n_samples

        # If < 5% unique values and < 50 unique values, likely classification
        if unique_ratio < 0.05 and n_unique < 50:
            return "classification"

        # Check if all integers and small range
        if target.dtype in [np.int32, np.int64]:
            if n_unique <= 20:
                return "classification"

        return "regression"

    def _generate_metadata(self):
        """Generate dataset metadata with advanced column type detection"""

        # Detect datetime columns
        datetime_cols = list(self.train_df.select_dtypes(include=['datetime']).columns)

        # Try to parse object columns as datetime
        for col in self.train_df.select_dtypes(include=['object']).columns:
            if col == self.target_column or col == self.id_column:
                continue
            try:
                pd.to_datetime(self.train_df[col].dropna().head(100))
                datetime_cols.append(col)
            except:
                pass

        # Separate text from categorical
        text_cols = []
        categorical_cols = []

        for col in self.train_df.select_dtypes(include=['object', 'category']).columns:
            if col in datetime_cols or col == self.target_column or col == self.id_column:
                continue

            # Calculate average string length
            sample = self.train_df[col].dropna().astype(str).head(100)
            avg_length = sample.str.len().mean() if len(sample) > 0 else 0

            if avg_length > 50:  # Long text
                text_cols.append(col)
            else:
                categorical_cols.append(col)

        # Binary columns (exactly 2 unique values)
        binary_cols = []
        for col in self.train_df.columns:
            if col == self.target_column or col == self.id_column:
                continue
            if self.train_df[col].nunique() == 2:
                binary_cols.append(col)

        self.metadata = {
            'n_samples': len(self.train_df),
            'n_features': len(self.train_df.columns) - 1 if self.target_column else len(self.train_df.columns),
            'target_column': self.target_column,
            'id_column': self.id_column,
            'problem_type': self.problem_type,
            'numeric_features': self.train_df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': categorical_cols,
            'text_features': text_cols,
            'datetime_features': datetime_cols,
            'binary_features': binary_cols,
            'missing_values': self.train_df.isnull().sum().sum(),
            'missing_percentage': (self.train_df.isnull().sum().sum() / (len(self.train_df) * len(self.train_df.columns)) * 100),
            'duplicate_rows': self.train_df.duplicated().sum(),
            'memory_usage_mb': self.train_df.memory_usage(deep=True).sum() / 1024**2,
            'has_test_set': self.test_df is not None
        }

    def split_train_val(self, val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split training data into train and validation sets"""
        if self.target_column is None:
            raise ValueError("Target column not set")

        stratify = None
        if self.problem_type == "classification":
            stratify = self.train_df[self.target_column]

        train, val = train_test_split(
            self.train_df,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify
        )

        return train, val

    def get_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataframe into features and target"""
        if self.target_column is None:
            raise ValueError("Target column not set")

        # Get columns to exclude
        exclude_cols = [self.target_column]
        if self.id_column:
            exclude_cols.append(self.id_column)

        X = df.drop(columns=exclude_cols)
        y = df[self.target_column]

        return X, y

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return self.metadata
