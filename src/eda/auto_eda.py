"""Lightweight automated EDA - no heavy dependencies"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional


class AutoEDA:
    """Lightweight automated exploratory data analysis"""

    def __init__(self, output_dir: Path, logger=None):
        self.output_dir = output_dir / "eda"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def analyze(self, df: pd.DataFrame, target_column: str = None, problem_type: str = None) -> Dict[str, Any]:
        """Run lightweight EDA"""

        if self.logger:
            self.logger.section("Exploratory Data Analysis")

        results = {
            'statistics': {},
            'plots': [],
            'insights': []
        }

        # 1. Basic statistics
        results['statistics'] = self._compute_statistics(df, target_column)

        # 2. Generate plots
        if self.logger:
            self.logger.info("Generating visualizations...")

        try:
            # Target distribution
            if target_column and target_column in df.columns:
                plot_path = self._plot_target_distribution(df, target_column, problem_type)
                results['plots'].append(plot_path)

            # Correlation heatmap
            plot_path = self._plot_correlation_heatmap(df)
            if plot_path:
                results['plots'].append(plot_path)

            # Missing values
            plot_path = self._plot_missing_values(df)
            if plot_path:
                results['plots'].append(plot_path)

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not generate some plots: {e}")

        # 3. Generate insights
        results['insights'] = self._generate_insights(df, results['statistics'], target_column)

        if self.logger:
            self.logger.success(f"✓ EDA complete: {len(results['plots'])} plots, {len(results['insights'])} insights")

        return results

    def _compute_statistics(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Compute basic statistics"""

        stats = {
            'shape': df.shape,
            'columns': len(df.columns),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing': df.isnull().sum().sum(),
            'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'duplicates': df.duplicated().sum(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }

        # Numeric stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric'] = df[numeric_cols].describe().to_dict()

        # Categorical stats
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            stats['categorical'] = {}
            for col in categorical_cols[:10]:  # Limit to 10
                stats['categorical'][col] = {
                    'unique': df[col].nunique(),
                    'top': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                    'freq': df[col].value_counts().iloc[0] if len(df) > 0 else 0
                }

        # Target stats
        if target_column and target_column in df.columns:
            target = df[target_column]
            stats['target'] = {
                'name': target_column,
                'dtype': str(target.dtype),
                'unique': target.nunique(),
                'missing': target.isnull().sum()
            }

            if pd.api.types.is_numeric_dtype(target):
                stats['target']['mean'] = float(target.mean())
                stats['target']['std'] = float(target.std())
                stats['target']['min'] = float(target.min())
                stats['target']['max'] = float(target.max())
            else:
                stats['target']['value_counts'] = target.value_counts().head(10).to_dict()

        return stats

    def _plot_target_distribution(self, df: pd.DataFrame, target_column: str, problem_type: str) -> Path:
        """Plot target variable distribution"""

        fig, ax = plt.subplots(figsize=(10, 6))

        if problem_type == 'classification':
            df[target_column].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Target Distribution: {target_column}')
            ax.set_xlabel(target_column)
            ax.set_ylabel('Count')
        else:
            df[target_column].hist(bins=50, ax=ax)
            ax.set_title(f'Target Distribution: {target_column}')
            ax.set_xlabel(target_column)
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        plot_path = self.output_dir / 'target_distribution.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_correlation_heatmap(self, df: pd.DataFrame) -> Optional[Path]:
        """Plot correlation heatmap for numeric features"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return None

        # Limit to top 20 features
        if len(numeric_cols) > 20:
            numeric_cols = numeric_cols[:20]

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax,
                    square=True, linewidths=0.5)
        ax.set_title('Feature Correlation Heatmap')
        plt.tight_layout()

        plot_path = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_missing_values(self, df: pd.DataFrame) -> Optional[Path]:
        """Plot missing values"""

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, max(6, len(missing) * 0.3)))
        missing.plot(kind='barh', ax=ax)
        ax.set_title('Missing Values by Column')
        ax.set_xlabel('Count')
        plt.tight_layout()

        plot_path = self.output_dir / 'missing_values.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        return plot_path

    def _generate_insights(self, df: pd.DataFrame, stats: Dict, target_column: str = None) -> list:
        """Generate simple insights from statistics"""

        insights = []

        # Missing values insight
        if stats['missing_pct'] > 10:
            insights.append(f"Dataset has {stats['missing_pct']:.1f}% missing values - consider imputation")

        # Duplicates insight
        if stats['duplicates'] > 0:
            dup_pct = stats['duplicates'] / len(df) * 100
            insights.append(f"Found {stats['duplicates']} duplicate rows ({dup_pct:.1f}%) - consider removing")

        # Imbalance insight (for classification)
        if target_column and 'target' in stats and 'value_counts' in stats['target']:
            value_counts = stats['target']['value_counts']
            if len(value_counts) > 1:
                values = list(value_counts.values())
                ratio = max(values) / min(values)
                if ratio > 5:
                    insights.append(f"Target is imbalanced ({ratio:.1f}:1) - consider resampling or class weighting")

        # High cardinality insight
        if 'categorical' in stats:
            high_card_cols = [col for col, info in stats['categorical'].items() if info['unique'] > 100]
            if high_card_cols:
                insights.append(f"{len(high_card_cols)} categorical columns have high cardinality (>100) - consider target encoding")

        # Memory insight
        if stats['memory_mb'] > 1000:
            insights.append(f"Large dataset ({stats['memory_mb']:.0f} MB) - consider data type optimization")

        return insights
