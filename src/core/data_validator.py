"""Enhanced data validation with structured issue levels"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class DataValidator:
    """Validates data quality with critical/warning/info structure"""

    def __init__(self, logger=None):
        self.logger = logger
        self.critical_issues = []
        self.warnings = []
        self.info = []
        self.recommendations = []

    def validate(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Run comprehensive validation with structured levels"""
        self.critical_issues = []
        self.warnings = []
        self.info = []
        self.recommendations = []

        results = {
            'is_valid': True,
            'critical_issues': [],
            'warnings': [],
            'info': [],
            'recommendations': [],
            'stats': {}
        }

        # Critical checks (must fix)
        self._check_critical(df, target_column)

        # Warning checks (should fix)
        self._check_warnings(df, target_column)

        # Info checks (good to know)
        self._check_info(df, target_column)

        # Generate recommendations
        self._generate_recommendations()

        # Set results
        results['critical_issues'] = self.critical_issues
        results['warnings'] = self.warnings
        results['info'] = self.info
        results['recommendations'] = self.recommendations

        if self.critical_issues:
            results['is_valid'] = False

        # Log results
        if self.logger:
            if results['is_valid']:
                self.logger.success("✓ Data validation passed")
            else:
                self.logger.error("✗ Data validation failed")

            for issue in self.critical_issues:
                self.logger.error(f"  CRITICAL: {issue}")

            for warning in self.warnings:
                self.logger.warning(f"  WARNING: {warning}")

            if self.logger.logger.level <= 10:  # DEBUG
                for info_msg in self.info:
                    self.logger.info(f"  INFO: {info_msg}")

        return results

    def _check_critical(self, df: pd.DataFrame, target_column: str = None):
        """Critical issues that prevent training"""

        # Empty dataset
        if df.empty or len(df) == 0:
            self.critical_issues.append("Dataset is empty")
            return

        # Check target column if specified
        if target_column:
            if target_column not in df.columns:
                self.critical_issues.append(f"Target column '{target_column}' not found")
                return

            target = df[target_column]

            # All target values missing
            if target.isnull().all():
                self.critical_issues.append("All target values are missing")

            # Only one class in classification
            n_unique = target.nunique()
            if n_unique == 1:
                self.critical_issues.append("Target has only one unique value (cannot train)")

            # Extreme class imbalance (>99:1)
            if n_unique < 50:  # Likely classification
                value_counts = target.value_counts()
                if len(value_counts) > 1:
                    imbalance_ratio = value_counts.max() / value_counts.min()
                    if imbalance_ratio > 99:
                        self.critical_issues.append(
                            f"Extreme class imbalance ({imbalance_ratio:.0f}:1) - may need special handling"
                        )

        # All features are constant
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            all_constant = all(df[col].nunique() <= 1 for col in numeric_cols)
            if all_constant:
                self.critical_issues.append("All numeric features have zero variance")

    def _check_warnings(self, df: pd.DataFrame, target_column: str = None):
        """Warning-level issues that should be addressed"""

        # High missing values (>50%)
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing_cols = missing_pct[missing_pct > 50].index.tolist()

        if high_missing_cols:
            self.warnings.append(
                f"High missing values (>50%): {len(high_missing_cols)} columns - {high_missing_cols[:5]}"
            )

        # High cardinality categoricals
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col == target_column:
                continue

            n_unique = df[col].nunique()
            if n_unique > 100 and n_unique < len(df) * 0.95:
                self.warnings.append(
                    f"High cardinality categorical: {col} ({n_unique} unique values)"
                )

        # Duplicated columns
        duplicated_cols = []
        cols = df.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if df[cols[i]].equals(df[cols[j]]):
                    duplicated_cols.append((cols[i], cols[j]))

        if duplicated_cols:
            self.warnings.append(f"Duplicated columns found: {duplicated_cols[:3]}")

        # Moderate class imbalance (80:20 to 95:5)
        if target_column and target_column in df.columns:
            target = df[target_column]
            n_unique = target.nunique()

            if 2 <= n_unique < 50:  # Classification
                value_counts = target.value_counts()
                if len(value_counts) > 1:
                    imbalance_ratio = value_counts.max() / value_counts.min()
                    if 4 < imbalance_ratio <= 20:  # Between 80:20 and 95:5
                        self.warnings.append(
                            f"Moderate class imbalance ({imbalance_ratio:.1f}:1) - consider resampling"
                        )

        # Many outliers in numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == target_column:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
            outlier_pct = outliers / len(df) * 100

            if outlier_pct > 5:
                self.warnings.append(
                    f"Many outliers in {col}: {outlier_pct:.1f}% of values"
                )

    def _check_info(self, df: pd.DataFrame, target_column: str = None):
        """Informational observations"""

        # Moderate missing values (10-50%)
        missing_pct = (df.isnull().sum() / len(df) * 100)
        moderate_missing_cols = missing_pct[(missing_pct > 10) & (missing_pct <= 50)].index.tolist()

        if moderate_missing_cols:
            self.info.append(
                f"Moderate missing values (10-50%): {len(moderate_missing_cols)} columns"
            )

        # Duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            dup_pct = n_duplicates / len(df) * 100
            self.info.append(f"Duplicate rows: {n_duplicates} ({dup_pct:.1f}%)")

        # Skewed distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewed_cols = []

        for col in numeric_cols:
            if col == target_column:
                continue

            try:
                skewness = df[col].skew()
                if abs(skewness) > 2:
                    skewed_cols.append(col)
            except:
                pass

        if skewed_cols:
            self.info.append(f"Highly skewed features: {len(skewed_cols)} columns")

        # Low variance features
        for col in numeric_cols:
            if col == target_column:
                continue

            if df[col].std() / (df[col].mean() + 1e-10) < 0.01:
                self.info.append(f"Low variance feature: {col}")

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        if memory_mb > 1000:  # > 1GB
            self.info.append(f"Large dataset: {memory_mb:.1f} MB in memory")

    def _generate_recommendations(self):
        """Generate actionable recommendations based on issues"""

        # Based on critical issues
        if any('empty' in str(issue).lower() for issue in self.critical_issues):
            self.recommendations.append("Load the correct dataset file")

        if any('target' in str(issue).lower() for issue in self.critical_issues):
            self.recommendations.append("Verify target column name or handle missing targets")

        if any('one unique value' in str(issue).lower() for issue in self.critical_issues):
            self.recommendations.append("Check if this is the correct target variable")

        # Based on warnings
        if any('missing values' in str(warning).lower() for warning in self.warnings):
            self.recommendations.append("Use missing value imputation or drop columns with >70% missing")

        if any('high cardinality' in str(warning).lower() for warning in self.warnings):
            self.recommendations.append("Consider target encoding for high cardinality categoricals")

        if any('imbalance' in str(warning).lower() for warning in self.warnings):
            self.recommendations.append("Consider SMOTE or class weighting for imbalanced data")

        if any('outliers' in str(warning).lower() for warning in self.warnings):
            self.recommendations.append("Use robust scaling or outlier removal")

        if any('duplicated columns' in str(warning).lower() for warning in self.warnings):
            self.recommendations.append("Remove duplicate columns to reduce redundancy")

        # Based on info
        if any('skewed' in str(info).lower() for info in self.info):
            self.recommendations.append("Apply log transformation to skewed features")

        if any('duplicate rows' in str(info).lower() for info in self.info):
            self.recommendations.append("Consider removing duplicate rows")
