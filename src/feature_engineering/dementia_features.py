"""
Domain-Specific Feature Engineering for Dementia Risk Prediction
Creates meaningful features based on medical research and domain knowledge
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


class DementiaFeatureEngineer:
    """
    Creates domain-specific features for dementia risk prediction.

    Based on research, key risk factors include:
    - Age and education
    - Cardiovascular health
    - Lifestyle factors (smoking, alcohol)
    - Functional decline
    - Social factors
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.created_features = []
        self.feature_definitions = {}

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create domain-specific features for dementia prediction

        Returns:
        --------
        df_engineered : pd.DataFrame
            DataFrame with new features
        report : dict
            Report of feature engineering process
        """
        if self.logger:
            self.logger.info("Creating dementia-specific features")

        df_eng = df.copy()
        initial_features = len(df_eng.columns)

        # 1. Cardiovascular Risk Score
        df_eng = self._create_cardiovascular_risk_score(df_eng)

        # 2. Cerebrovascular Risk Score
        df_eng = self._create_cerebrovascular_risk_score(df_eng)

        # 3. Lifestyle Risk Score
        df_eng = self._create_lifestyle_risk_score(df_eng)

        # 4. Functional Impairment Score
        df_eng = self._create_functional_impairment_score(df_eng)

        # 5. Age-related features
        df_eng = self._create_age_features(df_eng)

        # 6. Education-related features
        df_eng = self._create_education_features(df_eng)

        # 7. Social isolation indicators
        df_eng = self._create_social_features(df_eng)

        # 8. Smoking pack-years
        df_eng = self._create_smoking_features(df_eng)

        # 9. Comorbidity count
        df_eng = self._create_comorbidity_features(df_eng)

        # 10. Sensory impairment features
        df_eng = self._create_sensory_features(df_eng)

        final_features = len(df_eng.columns)

        report = {
            'initial_features': initial_features,
            'features_created': len(self.created_features),
            'final_features': final_features,
            'new_feature_names': self.created_features,
            'feature_definitions': self.feature_definitions
        }

        if self.logger:
            self.logger.info(f"Created {len(self.created_features)} domain-specific features")

        return df_eng, report

    def _create_cardiovascular_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cardiovascular risk score from heart-related conditions.

        Higher score = more CV risk factors
        """
        cv_features = ['CVHATT', 'CVAFIB', 'CVCHF', 'CVANGINA', 'HYPERTEN',
                      'HYPERCHO', 'DIABETES']

        if not any(f in df.columns for f in cv_features):
            return df

        df_new = df.copy()
        cv_score = 0

        for feat in cv_features:
            if feat in df.columns:
                # CVHATT, CVAFIB, etc: 0=Absent, 1=Recent/Active, 2=Remote/Inactive
                # Count both active (1) and remote (2) as risk factors, with active weighted higher
                cv_score = cv_score + df[feat].apply(
                    lambda x: 2 if x == 1 else (1 if x == 2 else 0)
                )

        df_new['cardiovascular_risk_score'] = cv_score
        self.created_features.append('cardiovascular_risk_score')
        self.feature_definitions['cardiovascular_risk_score'] = {
            'formula': 'Sum of cardiovascular conditions (active=2, remote=1, absent=0)',
            'rationale': 'CV disease is a major dementia risk factor',
            'expected_impact': 'Higher score indicates higher dementia risk'
        }

        return df_new

    def _create_cerebrovascular_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cerebrovascular risk score (stroke, TIA).

        Stroke/TIA are strong dementia risk factors.
        """
        cerebro_features = ['CBSTROKE', 'CBTIA', 'STROKMUL', 'TIAMULT']

        if not any(f in df.columns for f in cerebro_features):
            return df

        df_new = df.copy()
        cerebro_score = 0

        if 'CBSTROKE' in df.columns:
            cerebro_score = cerebro_score + df['CBSTROKE'].apply(
                lambda x: 3 if x == 1 else (2 if x == 2 else 0)
            )

        if 'STROKMUL' in df.columns:
            # Multiple strokes = additional risk
            cerebro_score = cerebro_score + df['STROKMUL'].apply(
                lambda x: 2 if x == 1 else 0
            )

        if 'CBTIA' in df.columns:
            cerebro_score = cerebro_score + df['CBTIA'].apply(
                lambda x: 2 if x == 1 else (1 if x == 2 else 0)
            )

        if 'TIAMULT' in df.columns:
            cerebro_score = cerebro_score + df['TIAMULT'].apply(
                lambda x: 1 if x == 1 else 0
            )

        df_new['cerebrovascular_risk_score'] = cerebro_score
        self.created_features.append('cerebrovascular_risk_score')
        self.feature_definitions['cerebrovascular_risk_score'] = {
            'formula': 'Weighted sum: stroke (3 active, 2 remote) + multiple strokes (2) + TIA (2 active, 1 remote) + multiple TIA (1)',
            'rationale': 'Stroke and TIA strongly increase dementia risk',
            'expected_impact': 'Higher score indicates higher dementia risk'
        }

        return df_new

    def _create_lifestyle_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lifestyle risk score (smoking, alcohol abuse)"""

        df_new = df.copy()
        lifestyle_score = 0

        # Smoking
        if 'TOBAC100' in df.columns:
            lifestyle_score = lifestyle_score + df['TOBAC100'].apply(
                lambda x: 1 if x == 1 else 0
            )

        if 'TOBAC30' in df.columns:
            # Current smoking = higher risk
            lifestyle_score = lifestyle_score + df['TOBAC30'].apply(
                lambda x: 2 if x == 1 else 0
            )

        # Alcohol abuse
        if 'ALCOHOL' in df.columns:
            lifestyle_score = lifestyle_score + df['ALCOHOL'].apply(
                lambda x: 2 if x == 1 else (1 if x == 2 else 0)
            )

        # Other substance abuse
        if 'ABUSOTHR' in df.columns:
            lifestyle_score = lifestyle_score + df['ABUSOTHR'].apply(
                lambda x: 2 if x == 1 else (1 if x == 2 else 0)
            )

        df_new['lifestyle_risk_score'] = lifestyle_score
        self.created_features.append('lifestyle_risk_score')
        self.feature_definitions['lifestyle_risk_score'] = {
            'formula': 'Current smoking (2) + ever smoked 100+ cigarettes (1) + alcohol abuse (2 active, 1 remote) + other substance abuse (2 active, 1 remote)',
            'rationale': 'Smoking and substance abuse increase dementia risk',
            'expected_impact': 'Higher score indicates higher dementia risk'
        }

        return df_new

    def _create_functional_impairment_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create functional impairment score from ADL/IADL difficulties.

        Functional decline is early dementia sign.
        """
        functional_features = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
                              'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']

        if not any(f in df.columns for f in functional_features):
            return df

        df_new = df.copy()
        impairment_score = 0

        for feat in functional_features:
            if feat in df.columns:
                # 0=Normal, 1=Has difficulty, 2=Requires assistance, 3=Dependent
                impairment_score = impairment_score + df[feat].apply(
                    lambda x: 0 if x == 0 else (1 if x == 1 else (2 if x == 2 else (3 if x == 3 else 0)))
                )

        df_new['functional_impairment_score'] = impairment_score
        self.created_features.append('functional_impairment_score')
        self.feature_definitions['functional_impairment_score'] = {
            'formula': 'Sum of difficulties across 10 activities: Normal=0, Difficulty=1, Assistance=2, Dependent=3',
            'rationale': 'Functional decline is early indicator of cognitive impairment',
            'expected_impact': 'Higher score indicates higher dementia risk'
        }

        # Also count number of impaired functions
        n_impaired = 0
        for feat in functional_features:
            if feat in df.columns:
                n_impaired = n_impaired + df[feat].apply(lambda x: 1 if x >= 1 else 0)

        df_new['functional_domains_impaired'] = n_impaired
        self.created_features.append('functional_domains_impaired')
        self.feature_definitions['functional_domains_impaired'] = {
            'formula': 'Count of functional domains with any impairment (score >= 1)',
            'rationale': 'Number of impaired domains indicates breadth of decline',
            'expected_impact': 'More impaired domains indicates higher dementia risk'
        }

        return df_new

    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-related features"""

        df_new = df.copy()

        if 'NACCAGE' in df.columns:
            # Age is exponential risk factor for dementia
            df_new['age_squared'] = df['NACCAGE'] ** 2
            self.created_features.append('age_squared')
            self.feature_definitions['age_squared'] = {
                'formula': 'NACCAGE^2',
                'rationale': 'Dementia risk increases exponentially with age',
                'expected_impact': 'Captures non-linear age effect'
            }

            # Age bins
            df_new['age_65plus'] = (df['NACCAGE'] >= 65).astype(int)
            df_new['age_75plus'] = (df['NACCAGE'] >= 75).astype(int)
            df_new['age_85plus'] = (df['NACCAGE'] >= 85).astype(int)
            self.created_features.extend(['age_65plus', 'age_75plus', 'age_85plus'])
            self.feature_definitions['age_bins'] = {
                'formula': 'Binary indicators for age >= 65, 75, 85',
                'rationale': 'Risk increases sharply at certain age thresholds',
                'expected_impact': 'Captures age-related risk increases'
            }

        return df_new

    def _create_education_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create education-related features"""

        df_new = df.copy()

        if 'EDUC' in df.columns:
            # Low education is risk factor (< 12 years)
            df_new['low_education'] = (df['EDUC'] < 12).astype(int)
            self.created_features.append('low_education')
            self.feature_definitions['low_education'] = {
                'formula': 'Education < 12 years',
                'rationale': 'Low education is established dementia risk factor (cognitive reserve)',
                'expected_impact': 'Low education increases dementia risk'
            }

            # High education is protective (>= 16 years)
            df_new['high_education'] = (df['EDUC'] >= 16).astype(int)
            self.created_features.append('high_education')
            self.feature_definitions['high_education'] = {
                'formula': 'Education >= 16 years (college degree)',
                'rationale': 'Higher education provides cognitive reserve',
                'expected_impact': 'High education decreases dementia risk'
            }

        return df_new

    def _create_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create social isolation indicators"""

        df_new = df.copy()

        # Living alone
        if 'NACCLIVS' in df.columns:
            df_new['lives_alone'] = (df['NACCLIVS'] == 1).astype(int)
            self.created_features.append('lives_alone')
            self.feature_definitions['lives_alone'] = {
                'formula': 'NACCLIVS == 1 (lives alone)',
                'rationale': 'Social isolation is dementia risk factor',
                'expected_impact': 'Living alone may increase dementia risk'
            }

        # Never married
        if 'MARISTAT' in df.columns:
            df_new['never_married'] = (df['MARISTAT'] == 5).astype(int)
            self.created_features.append('never_married')

        # Widowed (potential social isolation)
        if 'MARISTAT' in df.columns:
            df_new['widowed'] = (df['MARISTAT'] == 2).astype(int)
            self.created_features.append('widowed')

        return df_new

    def _create_smoking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create smoking pack-years and related features"""

        df_new = df.copy()

        # Pack-years calculation
        if 'SMOKYRS' in df.columns and 'PACKSPER' in df.columns:
            # PACKSPER: 0=No use, 1=<0.5, 2=0.5-<1, 3=1-1.5, 4=1.5-2, 5=>2
            # Convert to average packs per day
            packs_mapping = {0: 0, 1: 0.25, 2: 0.75, 3: 1.25, 4: 1.75, 5: 2.5}
            avg_packs = df['PACKSPER'].map(packs_mapping).fillna(0)
            df_new['pack_years'] = df['SMOKYRS'] * avg_packs
            self.created_features.append('pack_years')
            self.feature_definitions['pack_years'] = {
                'formula': 'SMOKYRS × average packs per day',
                'rationale': 'Pack-years is standard measure of smoking exposure',
                'expected_impact': 'Higher pack-years increases dementia risk'
            }

        # Years since quit smoking
        if 'QUITSMOK' in df.columns and 'NACCAGE' in df.columns:
            df_new['years_since_quit'] = df['NACCAGE'] - df['QUITSMOK']
            df_new['years_since_quit'] = df_new['years_since_quit'].apply(
                lambda x: x if x >= 0 else 0
            )
            self.created_features.append('years_since_quit')
            self.feature_definitions['years_since_quit'] = {
                'formula': 'Current age - age when quit smoking',
                'rationale': 'Time since quitting affects risk reduction',
                'expected_impact': 'More years since quitting reduces risk'
            }

        return df_new

    def _create_comorbidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comorbidity count and patterns"""

        df_new = df.copy()

        # Count total number of medical conditions
        medical_conditions = [
            'CVHATT', 'CVAFIB', 'CVCHF', 'CVANGINA', 'CBSTROKE', 'CBTIA',
            'PD', 'SEIZURES', 'TBI', 'DIABETES', 'HYPERTEN', 'HYPERCHO',
            'B12DEF', 'THYROID', 'ARTHRIT', 'APNEA', 'RBD', 'INSOMN'
        ]

        comorbidity_count = 0
        for cond in medical_conditions:
            if cond in df.columns:
                # Count as comorbidity if active (1) or remote (2)
                comorbidity_count = comorbidity_count + df[cond].apply(
                    lambda x: 1 if x in [1, 2] else 0
                )

        df_new['total_comorbidities'] = comorbidity_count
        self.created_features.append('total_comorbidities')
        self.feature_definitions['total_comorbidities'] = {
            'formula': 'Count of all medical conditions (active or remote)',
            'rationale': 'Higher comorbidity burden increases dementia risk',
            'expected_impact': 'More comorbidities indicates higher dementia risk'
        }

        return df_new

    def _create_sensory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sensory impairment indicators"""

        df_new = df.copy()

        # Vision impairment
        if 'VISWCORR' in df.columns:
            df_new['vision_impaired'] = (df['VISWCORR'] == 1).astype(int)
            self.created_features.append('vision_impaired')
            self.feature_definitions['vision_impaired'] = {
                'formula': 'Vision problems even with corrective lenses',
                'rationale': 'Sensory impairment linked to cognitive decline',
                'expected_impact': 'Vision impairment may increase dementia risk'
            }

        # Hearing impairment
        if 'HEARWAID' in df.columns:
            df_new['hearing_impaired'] = (df['HEARWAID'] == 1).astype(int)
            self.created_features.append('hearing_impaired')
            self.feature_definitions['hearing_impaired'] = {
                'formula': 'Hearing problems even with hearing aids',
                'rationale': 'Hearing loss is established dementia risk factor',
                'expected_impact': 'Hearing impairment increases dementia risk'
            }

        # Dual sensory impairment (both vision and hearing)
        if 'vision_impaired' in df_new.columns and 'hearing_impaired' in df_new.columns:
            df_new['dual_sensory_impairment'] = (
                (df_new['vision_impaired'] == 1) & (df_new['hearing_impaired'] == 1)
            ).astype(int)
            self.created_features.append('dual_sensory_impairment')
            self.feature_definitions['dual_sensory_impairment'] = {
                'formula': 'Both vision and hearing impaired',
                'rationale': 'Dual sensory impairment compounds dementia risk',
                'expected_impact': 'Dual impairment significantly increases risk'
            }

        return df_new

    def get_feature_engineering_report(self) -> Dict[str, Any]:
        """Generate comprehensive feature engineering report"""

        report = {
            'total_features_created': len(self.created_features),
            'feature_names': self.created_features,
            'feature_definitions': self.feature_definitions,
            'feature_categories': {
                'Cardiovascular Risk': ['cardiovascular_risk_score'],
                'Cerebrovascular Risk': ['cerebrovascular_risk_score'],
                'Lifestyle Risk': ['lifestyle_risk_score', 'pack_years', 'years_since_quit'],
                'Functional Status': ['functional_impairment_score', 'functional_domains_impaired'],
                'Age Features': ['age_squared', 'age_65plus', 'age_75plus', 'age_85plus'],
                'Education Features': ['low_education', 'high_education'],
                'Social Features': ['lives_alone', 'never_married', 'widowed'],
                'Comorbidities': ['total_comorbidities'],
                'Sensory': ['vision_impaired', 'hearing_impaired', 'dual_sensory_impairment']
            }
        }

        return report
