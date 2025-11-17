"""
Preprocessing Report Generator
Creates comprehensive documentation of preprocessing pipeline
"""

import pandas as pd
from typing import Dict, Any
from pathlib import Path


class PreprocessingReportGenerator:
    """
    Generates comprehensive preprocessing documentation following
    competition requirements and best practices.
    """

    def __init__(self, pipeline_report: Dict[str, Any]):
        self.report = pipeline_report

    def generate_markdown_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive preprocessing report in Markdown format

        Returns:
        --------
        str : Markdown formatted report
        """

        markdown = []

        # Header
        markdown.append("# Data Preprocessing Report")
        markdown.append("\n## Dementia Risk Prediction - Non-Medical Features")
        markdown.append("\n---\n")

        # 1. Feature Selection
        markdown.append(self._generate_feature_selection_section())

        # 2. Missing Values
        markdown.append(self._generate_missing_values_section())

        # 3. Feature Engineering
        markdown.append(self._generate_feature_engineering_section())

        # 4. Outlier Handling
        markdown.append(self._generate_outlier_section())

        # 5. Categorical Encoding
        markdown.append(self._generate_encoding_section())

        # 6. Feature Scaling
        markdown.append(self._generate_scaling_section())

        # 7. Final Feature Selection
        markdown.append(self._generate_final_selection_section())

        # 8. Summary
        markdown.append(self._generate_summary_section())

        report_text = "\n".join(markdown)

        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)

        return report_text

    def _generate_feature_selection_section(self) -> str:
        """Generate feature selection section"""

        section = []
        section.append("\n## 1. Feature Selection - Non-Medical Variables Only\n")
        section.append("### Approach")
        section.append("Selected only **non-medical features** that people typically know about themselves.")
        section.append("Excluded all medical/diagnostic variables (cognitive tests, scans, lab results).\n")

        if 'feature_selection' in self.report:
            fs = self.report['feature_selection']

            section.append("### Features Selected by Category\n")

            if 'categories' in fs:
                for category, info in fs['categories'].items():
                    section.append(f"#### {category} ({info['count']} features)")
                    for feat in info['features'][:10]:  # Show first 10
                        section.append(f"- `{feat}`")
                    if len(info['features']) > 10:
                        section.append(f"- ... and {len(info['features']) - 10} more")
                    section.append("")

            section.append("### Summary")
            section.append(f"- **Total Features Selected**: {fs.get('total_selected', 'N/A')}")
            section.append(f"- **Total Features Removed**: {fs.get('total_removed', 'N/A')}")
            section.append(f"- **Justification**: Only non-medical information that people know about themselves")

        return "\n".join(section)

    def _generate_missing_values_section(self) -> str:
        """Generate missing values handling section"""

        section = []
        section.append("\n## 2. Handling Missing Values\n")

        if 'missing_values' in self.report:
            mv = self.report['missing_values']

            section.append("### Method Used")
            section.append(f"**{mv.get('method_used', 'Domain-specific imputation')}**\n")

            section.append("### Special Codes Handled (NACC Dataset)")
            if 'special_codes_handled' in mv:
                codes = mv['special_codes_handled']
                section.append(f"- **Not Available**: {codes.get('not_available', [])} - Form didn't collect this data")
                section.append(f"- **Not Applicable**: {codes.get('not_applicable', [])} - Question doesn't apply")
                section.append(f"- **Unknown**: {codes.get('unknown', [])} - Information not known\n")

            section.append("### Imputation Strategies")
            section.append("| Missing % | Strategy |")
            section.append("|-----------|----------|")
            section.append("| 0-5% | Median (numeric) / Mode (categorical) |")
            section.append("| 5-20% | Median/Mode |")
            section.append("| 20-50% | Median/Mode + Missing Indicator |")
            section.append("| 50-80% | Median/Mode + Missing Indicator |")
            section.append("| >80% | Drop Column |\n")

            section.append("### Impact")
            before = mv.get('before', {})
            after = mv.get('after', {})
            section.append(f"- **Before**: {before.get('total_missing', 'N/A')} missing values ({before.get('missing_percentage', 0):.2f}%)")
            section.append(f"- **After**: {after.get('total_missing', 'N/A')} missing values ({after.get('missing_percentage', 0):.2f}%)")
            section.append(f"- **Columns Dropped**: {len(mv.get('columns_dropped', []))}")
            section.append(f"- **Missing Indicators Created**: {len(mv.get('missing_indicators_created', []))}")

            if mv.get('missing_indicators_created'):
                section.append("\n### Missing Indicators Created")
                for indicator in mv['missing_indicators_created'][:10]:
                    section.append(f"- `{indicator}`")
                if len(mv.get('missing_indicators_created', [])) > 10:
                    section.append(f"- ... and {len(mv['missing_indicators_created']) - 10} more")

            section.append("\n### Justification")
            section.append("- Domain-specific handling of NACC dataset special codes")
            section.append("- Missing indicators preserve information about missingness patterns")
            section.append("- Median/Mode imputation is robust and interpretable for medical data")
            section.append("- Dropping columns with >80% missing preserves data quality")

        return "\n".join(section)

    def _generate_feature_engineering_section(self) -> str:
        """Generate feature engineering section"""

        section = []
        section.append("\n## 3. Feature Engineering\n")

        if 'feature_engineering' in self.report:
            fe = self.report['feature_engineering']

            section.append("### Features Created")
            section.append(f"**Total**: {fe.get('features_created', 0)} domain-specific features\n")

            if 'feature_definitions' in fe:
                defs = fe['feature_definitions']

                section.append("### Created Features (Detailed)\n")

                # Group by category if available
                if 'feature_categories' in fe:
                    for i, feat_name in enumerate(fe.get('new_feature_names', [])[:15], 1):
                        if feat_name in defs:
                            feat_def = defs[feat_name]
                            section.append(f"#### {i}. `{feat_name}`")
                            section.append(f"- **Formula**: {feat_def.get('formula', 'N/A')}")
                            section.append(f"- **Rationale**: {feat_def.get('rationale', 'N/A')}")
                            section.append(f"- **Expected Impact**: {feat_def.get('expected_impact', 'N/A')}\n")
                else:
                    for i, feat_name in enumerate(fe.get('new_feature_names', [])[:15], 1):
                        section.append(f"{i}. `{feat_name}`")

                if len(fe.get('new_feature_names', [])) > 15:
                    section.append(f"\n... and {len(fe['new_feature_names']) - 15} more features")

            section.append("\n### Justification")
            section.append("- Features based on established dementia risk factors from medical literature")
            section.append("- Composite scores capture complex multi-factorial risk patterns")
            section.append("- Domain knowledge encoded into interpretable features")

        return "\n".join(section)

    def _generate_outlier_section(self) -> str:
        """Generate outlier handling section"""

        section = []
        section.append("\n## 4. Handling Outliers\n")

        if 'outliers' in self.report:
            outliers = self.report['outliers']

            section.append("### Detection Method")
            section.append(f"**{outliers.get('method', 'IQR')}**")
            section.append(f"- **Threshold**: {outliers.get('threshold', 'N/A')}")
            section.append(f"- **Treatment**: {outliers.get('treatment', 'Capping (Winsorization)')}\n")

            section.append("### Justification")
            section.append("- IQR method is robust to extreme values")
            section.append("- Capping preserves data distribution while limiting extreme values")
            section.append("- Medical data often contains legitimate extreme values that shouldn't be removed\n")

            section.append(f"### Impact")
            section.append(f"- **Features Affected**: {outliers.get('n_features_affected', 0)}")

            if outliers.get('outliers_capped'):
                section.append("\n### Examples of Outliers Capped")
                section.append("| Feature | N Outliers | Lower Bound | Upper Bound |")
                section.append("|---------|------------|-------------|-------------|")

                for feat, info in list(outliers['outliers_capped'].items())[:10]:
                    section.append(f"| `{feat}` | {info['n_outliers']} | {info['lower_bound']:.2f} | {info['upper_bound']:.2f} |")

        return "\n".join(section)

    def _generate_encoding_section(self) -> str:
        """Generate categorical encoding section"""

        section = []
        section.append("\n## 5. Encoding Categorical Variables\n")

        if 'encoding' in self.report:
            enc = self.report['encoding']

            section.append("### Strategy")
            section.append(f"**{enc.get('strategy', 'Mixed encoding approach')}**\n")

            section.append("### Encoding Methods")
            section.append("| Cardinality | Method |")
            section.append("|-------------|--------|")
            section.append("| Binary (2 categories) | Label Encoding |")
            section.append("| Medium (3-10 categories) | One-Hot Encoding |")
            section.append("| High (>10 categories) | Label Encoding |\n")

            if enc.get('encoding_methods'):
                section.append("### Features Encoded")
                section.append("| Feature | Method |")
                section.append("|---------|--------|")

                for feat, method in list(enc['encoding_methods'].items())[:15]:
                    section.append(f"| `{feat}` | {method} |")

                if len(enc['encoding_methods']) > 15:
                    section.append(f"\n... and {len(enc['encoding_methods']) - 15} more")

            section.append(f"\n### Summary")
            section.append(f"- **Total Features Encoded**: {enc.get('n_features_encoded', 0)}")

            section.append("\n### Justification")
            section.append("- Label encoding for binary and high-cardinality features (ordinal relationship)")
            section.append("- One-hot encoding for medium-cardinality features (no ordinal relationship)")
            section.append("- Prevents arbitrary numerical ordering for nominal categories")

        return "\n".join(section)

    def _generate_scaling_section(self) -> str:
        """Generate feature scaling section"""

        section = []
        section.append("\n## 6. Feature Scaling/Normalization\n")

        if 'scaling' in self.report:
            scale = self.report['scaling']

            section.append("### Technique Used")
            section.append(f"**{scale.get('technique', 'RobustScaler')}**\n")

            section.append("### Mathematical Transformation")
            section.append(f"```")
            section.append(f"{scale.get('mathematical_transformation', 'x_scaled = (x - median) / IQR')}")
            section.append(f"```\n")

            section.append("### Features Scaled")
            section.append(f"- **Count**: {scale.get('n_features_scaled', 0)} numerical features")
            section.append(f"- **Method**: All numerical features scaled using RobustScaler\n")

            section.append("### Justification")
            section.append(f"- {scale.get('justification', 'RobustScaler is resilient to outliers')}")
            section.append("- Uses median and IQR instead of mean and standard deviation")
            section.append("- Essential for algorithms sensitive to feature scale (e.g., logistic regression, neural networks)")
            section.append("- Preserves interpretability better than other scaling methods")

        return "\n".join(section)

    def _generate_final_selection_section(self) -> str:
        """Generate final feature selection section"""

        section = []
        section.append("\n## 7. Feature Reduction - Final Selection\n")

        if 'final_selection' in self.report:
            fs = self.report['final_selection']

            section.append("### Techniques Used")
            if 'methods' in fs:
                for method in fs['methods']:
                    section.append(f"- **{method}**")
            section.append("")

            section.append("### Correlation Analysis")
            section.append("- **Threshold**: 0.95")
            section.append("- **Action**: Removed one feature from each highly correlated pair")
            section.append("- **Rationale**: Redundant features don't improve model performance\n")

            section.append("### Variance Threshold")
            section.append("- **Threshold**: 0.01")
            section.append("- **Action**: Removed features with very low variance")
            section.append("- **Rationale**: Low-variance features provide little information\n")

            section.append("### Features Removed")
            section.append(f"**Total**: {fs.get('n_features_removed', 0)} features\n")

            if fs.get('removal_reasons'):
                section.append("| Feature | Reason |")
                section.append("|---------|--------|")

                for feat, reason in list(fs['removal_reasons'].items())[:10]:
                    section.append(f"| `{feat}` | {reason} |")

                if len(fs['removal_reasons']) > 10:
                    section.append(f"\n... and {len(fs['removal_reasons']) - 10} more")

        return "\n".join(section)

    def _generate_summary_section(self) -> str:
        """Generate overall summary section"""

        section = []
        section.append("\n## 8. Finalized Feature Set - Summary\n")

        section.append("### Pipeline Overview")
        section.append(f"- **Initial Features**: {self.report.get('initial_shape', (0, 0))[1]}")
        section.append(f"- **Features Removed**: {self.report.get('features_removed_total', 0)}")
        section.append(f"- **Features Created**: {self.report.get('features_created_total', 0)}")
        section.append(f"- **Final Features**: {self.report.get('final_shape', (0, 0))[1]}\n")

        section.append("### Data Quality")
        section.append(f"- **Initial Samples**: {self.report.get('initial_shape', (0, 0))[0]}")
        section.append(f"- **Final Samples**: {self.report.get('final_shape', (0, 0))[0]}")
        section.append(f"- **Sample Retention**: {self.report.get('final_shape', (0, 0))[0] / max(self.report.get('initial_shape', (1, 1))[0], 1) * 100:.1f}%\n")

        section.append("### Preprocessing Steps Summary")
        section.append("1. ✓ Feature Selection (Non-medical only)")
        section.append("2. ✓ Missing Value Handling (NACC-specific codes)")
        section.append("3. ✓ Feature Engineering (Domain-specific features)")
        section.append("4. ✓ Outlier Detection and Handling (IQR + Capping)")
        section.append("5. ✓ Categorical Encoding (Label + One-Hot)")
        section.append("6. ✓ Feature Scaling (RobustScaler)")
        section.append("7. ✓ Final Feature Selection (Correlation + Variance)\n")

        section.append("### Ready for Modeling")
        section.append("The preprocessed dataset is now ready for training machine learning models.")
        section.append("All features are numerical, properly scaled, and free of missing values.")

        return "\n".join(section)

    def generate_excel_report(self, output_path: str):
        """Generate Excel report with multiple sheets"""

        try:
            import openpyxl
        except ImportError:
            print("Warning: openpyxl not installed. Skipping Excel report generation.")
            print("Install with: pip install openpyxl")
            return

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # Sheet 1: Overview
            overview_data = {
                'Metric': [
                    'Initial Features',
                    'Features Removed',
                    'Features Created',
                    'Final Features',
                    'Initial Samples',
                    'Final Samples',
                    'Sample Retention %'
                ],
                'Value': [
                    self.report.get('initial_shape', (0, 0))[1],
                    self.report.get('features_removed_total', 0),
                    self.report.get('features_created_total', 0),
                    self.report.get('final_shape', (0, 0))[1],
                    self.report.get('initial_shape', (0, 0))[0],
                    self.report.get('final_shape', (0, 0))[0],
                    f"{self.report.get('final_shape', (0, 0))[0] / max(self.report.get('initial_shape', (1, 1))[0], 1) * 100:.1f}%"
                ]
            }
            pd.DataFrame(overview_data).to_excel(writer, sheet_name='Overview', index=False)

            # Sheet 2: Feature Engineering
            if 'feature_engineering' in self.report and 'feature_definitions' in self.report['feature_engineering']:
                fe_data = []
                for feat_name, definition in self.report['feature_engineering']['feature_definitions'].items():
                    fe_data.append({
                        'Feature Name': feat_name,
                        'Formula': definition.get('formula', 'N/A'),
                        'Rationale': definition.get('rationale', 'N/A'),
                        'Expected Impact': definition.get('expected_impact', 'N/A')
                    })
                if fe_data:
                    pd.DataFrame(fe_data).to_excel(writer, sheet_name='Feature Engineering', index=False)

            # Sheet 3: Missing Values
            if 'missing_values' in self.report and 'imputation_strategies' in self.report['missing_values']:
                mv_data = []
                for feat, strategy in self.report['missing_values']['imputation_strategies'].items():
                    mv_data.append({
                        'Feature': feat,
                        'Strategy': strategy
                    })
                if mv_data:
                    pd.DataFrame(mv_data).to_excel(writer, sheet_name='Missing Values', index=False)

    def save_all_reports(self, output_dir: str):
        """Save both Markdown and Excel reports"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate and save markdown
        md_path = output_path / 'PREPROCESSING_REPORT.md'
        self.generate_markdown_report(str(md_path))

        # Generate and save Excel
        excel_path = output_path / 'preprocessing_report.xlsx'
        self.generate_excel_report(str(excel_path))

        print(f"✓ Reports saved to {output_dir}")
        print(f"  - {md_path.name}")
        print(f"  - {excel_path.name}")
