"""
Specialized Missing Value Handler for NACC Dataset
Handles domain-specific missing value codes and patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.impute import SimpleImputer, KNNImputer


class NACCMissingValueHandler:
    """
    Handles missing values in NACC dementia dataset with domain-specific logic.

    Special Codes in NACC Dataset:
    - -4: Not available (UDS form didn't collect this data)
    - 8, 88, 888, 8888: Not applicable
    - 9, 99, 999, 9999: Unknown
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.missing_value_codes = {
            'not_available': [-4],
            'not_applicable': [8, 88, 888, 8888],
            'unknown': [9, 99, 999, 9999]
        }
        self.imputation_strategies = {}
        self.missing_indicators_created = []
        self.imputation_report = {}

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fit and transform data, handling NACC-specific missing codes

        Returns:
        --------
        df_processed : pd.DataFrame
            Processed dataframe with missing values handled
        report : dict
            Detailed report of missing value handling
        """
        if self.logger:
            self.logger.info("Handling NACC dataset missing values")

        df_processed = df.copy()

        # Step 1: Convert special codes to NaN
        df_processed = self._convert_special_codes_to_nan(df_processed)

        # Step 2: Analyze missing patterns
        missing_analysis = self._analyze_missing_patterns(df_processed)

        # Step 3: Handle missing values by feature type
        df_processed = self._handle_missing_values(df_processed, missing_analysis)

        # Step 4: Generate report
        report = self._generate_report(df, df_processed, missing_analysis)

        if self.logger:
            self.logger.info(f"Missing value handling complete")
            self.logger.info(f"Created {len(self.missing_indicators_created)} missing indicators")

        return df_processed, report

    def _convert_special_codes_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert NACC special codes to NaN"""
        df_clean = df.copy()

        all_special_codes = []
        for code_list in self.missing_value_codes.values():
            all_special_codes.extend(code_list)

        # Replace special codes with NaN
        df_clean = df_clean.replace(all_special_codes, np.nan)

        if self.logger:
            n_replaced = (df.isin(all_special_codes)).sum().sum()
            self.logger.info(f"Converted {n_replaced} special codes to NaN")

        return df_clean

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns in the dataset"""

        analysis = {
            'columns': {},
            'total_missing': df.isnull().sum().sum(),
            'total_values': df.shape[0] * df.shape[1],
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }

        for col in df.columns:
            n_missing = df[col].isnull().sum()
            pct_missing = (n_missing / len(df)) * 100

            analysis['columns'][col] = {
                'n_missing': n_missing,
                'pct_missing': pct_missing,
                'dtype': df[col].dtype,
                'n_unique': df[col].nunique(),
                'recommended_strategy': self._recommend_strategy(df[col], pct_missing)
            }

        return analysis

    def _recommend_strategy(self, series: pd.Series, pct_missing: float) -> str:
        """Recommend imputation strategy based on feature characteristics"""

        if pct_missing == 0:
            return 'none'
        elif pct_missing > 80:
            return 'drop_column'
        elif pct_missing > 50:
            return 'median_with_indicator'
        elif pct_missing > 20:
            return 'mode_or_median_with_indicator'
        elif pct_missing > 5:
            return 'mode_or_median'
        else:
            return 'mode_or_median'

    def _handle_missing_values(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """Handle missing values using appropriate strategies"""

        df_handled = df.copy()

        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle numeric features
        for col in numeric_cols:
            if col not in df_handled.columns:
                continue

            strategy = analysis['columns'][col]['recommended_strategy']
            pct_missing = analysis['columns'][col]['pct_missing']

            if strategy == 'drop_column':
                if self.logger:
                    self.logger.info(f"  Dropping {col} ({pct_missing:.1f}% missing)")
                df_handled = df_handled.drop(columns=[col])
                self.imputation_strategies[col] = 'dropped'

            elif 'indicator' in strategy:
                # Create missing indicator
                indicator_col = f'{col}_missing'
                df_handled[indicator_col] = df_handled[col].isnull().astype(int)
                self.missing_indicators_created.append(indicator_col)

                # Impute with median
                median_val = df_handled[col].median()
                df_handled[col] = df_handled[col].fillna(median_val)
                self.imputation_strategies[col] = f'median ({median_val:.2f}) + indicator'

            else:
                # Simple median imputation
                median_val = df_handled[col].median()
                df_handled[col] = df_handled[col].fillna(median_val)
                self.imputation_strategies[col] = f'median ({median_val:.2f})'

        # Handle categorical features
        for col in categorical_cols:
            if col not in df_handled.columns:
                continue

            strategy = analysis['columns'][col]['recommended_strategy']
            pct_missing = analysis['columns'][col]['pct_missing']

            if strategy == 'drop_column':
                if self.logger:
                    self.logger.info(f"  Dropping {col} ({pct_missing:.1f}% missing)")
                df_handled = df_handled.drop(columns=[col])
                self.imputation_strategies[col] = 'dropped'

            elif pct_missing > 20:
                # High missing: create explicit 'Missing' category
                df_handled[col] = df_handled[col].fillna('Missing')
                self.imputation_strategies[col] = 'explicit_missing_category'

            else:
                # Low missing: use mode
                mode_val = df_handled[col].mode()
                if len(mode_val) > 0:
                    mode_val = mode_val[0]
                    df_handled[col] = df_handled[col].fillna(mode_val)
                    self.imputation_strategies[col] = f'mode ({mode_val})'
                else:
                    df_handled[col] = df_handled[col].fillna('Unknown')
                    self.imputation_strategies[col] = 'unknown'

        return df_handled

    def _generate_report(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                        analysis: Dict) -> Dict[str, Any]:
        """Generate comprehensive missing value handling report"""

        report = {
            'method_used': 'NACC-specific handling with domain knowledge',
            'special_codes_handled': self.missing_value_codes,
            'before': {
                'shape': df_before.shape,
                'total_missing': df_before.isnull().sum().sum(),
                'missing_percentage': (df_before.isnull().sum().sum() /
                                      (df_before.shape[0] * df_before.shape[1])) * 100
            },
            'after': {
                'shape': df_after.shape,
                'total_missing': df_after.isnull().sum().sum(),
                'missing_percentage': (df_after.isnull().sum().sum() /
                                      (df_after.shape[0] * df_after.shape[1])) * 100
            },
            'columns_dropped': [col for col, strategy in self.imputation_strategies.items()
                               if strategy == 'dropped'],
            'missing_indicators_created': self.missing_indicators_created,
            'imputation_strategies': self.imputation_strategies,
            'features_affected': list(self.imputation_strategies.keys())
        }

        return report

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted strategies"""

        df_processed = df.copy()

        # Convert special codes
        all_special_codes = []
        for code_list in self.missing_value_codes.values():
            all_special_codes.extend(code_list)
        df_processed = df_processed.replace(all_special_codes, np.nan)

        # Apply saved strategies
        for col, strategy in self.imputation_strategies.items():
            if col not in df_processed.columns:
                continue

            if strategy == 'dropped':
                df_processed = df_processed.drop(columns=[col])
            elif 'indicator' in strategy:
                # Create indicator
                indicator_col = f'{col}_missing'
                df_processed[indicator_col] = df_processed[col].isnull().astype(int)
                # Extract median value from strategy string
                median_val = float(strategy.split('(')[1].split(')')[0])
                df_processed[col] = df_processed[col].fillna(median_val)
            elif strategy.startswith('median'):
                median_val = float(strategy.split('(')[1].split(')')[0])
                df_processed[col] = df_processed[col].fillna(median_val)
            elif strategy.startswith('mode'):
                mode_val = strategy.split('(')[1].split(')')[0]
                df_processed[col] = df_processed[col].fillna(mode_val)
            elif strategy == 'explicit_missing_category':
                df_processed[col] = df_processed[col].fillna('Missing')

        return df_processed

    def get_missing_summary_table(self) -> pd.DataFrame:
        """Create a summary table of missing value handling"""

        summary_data = []
        for col, strategy in self.imputation_strategies.items():
            summary_data.append({
                'Feature': col,
                'Strategy': strategy,
                'Has_Indicator': f'{col}_missing' in self.missing_indicators_created
            })

        return pd.DataFrame(summary_data)
