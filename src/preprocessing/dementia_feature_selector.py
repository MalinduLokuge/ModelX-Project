"""
Domain-Specific Feature Selection for Dementia Risk Prediction
Filters features to only include non-medical variables as per competition rules
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set


class DementiaFeatureSelector:
    """
    Selects non-medical features for dementia risk prediction.

    Medical data = things only doctors/medical staff would measure or interpret
    (e.g., detailed cognitive test scores, scans, lab results)

    Allowed = demographics, lifestyle, simple known diagnoses, functional capacity
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.selected_features = []
        self.removed_features = []
        self.feature_categories = {}

        # Define allowed non-medical feature groups
        self._define_feature_groups()

    def _define_feature_groups(self):
        """Define categories of allowed non-medical features"""

        # 1. DEMOGRAPHICS (ALWAYS ALLOWED)
        self.demographics = [
            'NACCAGE', 'NACCAGEB', 'SEX', 'EDUC',
            'BIRTHMO', 'BIRTHYR', 'VISITMO', 'VISITDAY', 'VISITYR',
            'HISPANIC', 'HISPOR', 'HISPORX',
            'RACE', 'RACEX', 'RACESEC', 'RACESECX', 'RACETER', 'RACETERX',
            'NACCNIHR', 'PRIMLANG', 'PRIMLANX',
            'MARISTAT', 'NACCLIVS', 'RESIDENC', 'HANDED'
        ]

        # 2. LIFESTYLE FACTORS (ALLOWED)
        self.lifestyle = [
            # Smoking
            'TOBAC30', 'TOBAC100', 'SMOKYRS', 'PACKSPER', 'QUITSMOK',
            # Alcohol
            'ALCOCCAS', 'ALCFREQ', 'ALCOHOL', 'ABUSOTHR', 'ABUSX'
        ]

        # 3. SIMPLE KNOWN DIAGNOSES (ALLOWED - people know these about themselves)
        self.medical_history = [
            # Cardiovascular
            'CVHATT', 'HATTMULT', 'HATTYEAR', 'CVAFIB', 'CVANGIO',
            'CVBYPASS', 'CVPACDEF', 'CVPACE', 'CVCHF', 'CVANGINA',
            'CVHVALVE', 'CVOTHR', 'CVOTHRX',
            # Cerebrovascular
            'CBSTROKE', 'STROKMUL', 'NACCSTYR', 'CBTIA', 'TIAMULT', 'NACCTIYR',
            # Neurological (simple diagnoses)
            'PD', 'PDYR', 'PDOTHR', 'PDOTHRYR', 'SEIZURES',
            'TBI', 'TBIBRIEF', 'TRAUMBRF', 'TBIEXTEN', 'TRAUMEXT',
            'TBIWOLOS', 'TRAUMCHR', 'TBIYEAR', 'NCOTHR', 'NCOTHRX',
            # Common conditions
            'DIABETES', 'DIABTYPE', 'HYPERTEN', 'HYPERCHO',
            'B12DEF', 'THYROID', 'ARTHRIT', 'ARTHTYPE', 'ARTHTYPX',
            'ARTHUPEX', 'ARTHLOEX', 'ARTHSPIN', 'ARTHUNK',
            # Other health
            'INCONTU', 'INCONTF', 'APNEA', 'RBD', 'INSOMN', 'OTHSLEEP', 'OTHSLEEX'
        ]

        # 4. FUNCTIONAL CAPACITY (ALLOWED - observable daily activities)
        self.functional = [
            'INDEPEND', 'BILLS', 'TAXES', 'SHOPPING', 'GAMES',
            'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL'
        ]

        # 5. SENSORY (ALLOWED - subjective self-assessment)
        self.sensory = [
            'VISION', 'VISCORR', 'VISWCORR',
            'HEARING', 'HEARAID', 'HEARWAID'
        ]

        # 6. CO-PARTICIPANT INFO (ALLOWED - demographics and relationship)
        self.coparticipant = [
            'INBIRMO', 'INBIRYR', 'INSEX', 'INHISP', 'INHISPOR', 'INHISPOX',
            'INRACE', 'INRACEX', 'INRASEC', 'INRASECX', 'INRATER', 'INRATERX',
            'INEDUC', 'INRELTO', 'INRELTOX', 'INKNOWN', 'INLIVWTH',
            'INVISITS', 'INCALLS', 'INRELY', 'NACCNINR'
        ]

        # 7. LANGUAGE PROFICIENCY (ALLOWED - self-reported)
        self.language = [
            'APREFLAN', 'AYRSPAN', 'AYRENGL', 'APCSPAN', 'APCENGL',
            'ASPKSPAN', 'AREASPAN', 'AWRISPAN', 'AUNDSPAN',
            'ASPKENGL', 'AREAENGL', 'AWRIENGL', 'AUNDENGL',
            'NACCSPNL', 'NACCENGL'
        ]

        # Combine all allowed features
        self.all_allowed_features = (
            self.demographics + self.lifestyle + self.medical_history +
            self.functional + self.sensory + self.coparticipant + self.language
        )

        # Create category mapping
        for feat in self.demographics:
            self.feature_categories[feat] = 'Demographics'
        for feat in self.lifestyle:
            self.feature_categories[feat] = 'Lifestyle'
        for feat in self.medical_history:
            self.feature_categories[feat] = 'Medical History (Simple Diagnoses)'
        for feat in self.functional:
            self.feature_categories[feat] = 'Functional Capacity'
        for feat in self.sensory:
            self.feature_categories[feat] = 'Sensory Assessment'
        for feat in self.coparticipant:
            self.feature_categories[feat] = 'Co-participant Information'
        for feat in self.language:
            self.feature_categories[feat] = 'Language Proficiency'

    def select_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Select only non-medical features from the dataset

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset with all features
        target_col : str, optional
            Name of target column to preserve

        Returns:
        --------
        pd.DataFrame
            Dataset with only non-medical features
        """
        if self.logger:
            self.logger.info(f"Selecting non-medical features from {len(df.columns)} total columns")

        # Get available columns
        available_cols = df.columns.tolist()

        # Select features that are in our allowed list and present in the data
        selected = []
        for col in available_cols:
            if col in self.all_allowed_features:
                selected.append(col)
                self.selected_features.append(col)
            elif target_col and col == target_col:
                # Always keep target column
                selected.append(col)
            else:
                self.removed_features.append(col)

        if self.logger:
            self.logger.info(f"Selected {len(selected)} non-medical features")
            self.logger.info(f"Removed {len(self.removed_features)} medical/excluded features")

        return df[selected]

    def get_feature_categories_summary(self) -> Dict[str, List[str]]:
        """Get summary of selected features by category"""
        summary = {}
        for feature in self.selected_features:
            category = self.feature_categories.get(feature, 'Unknown')
            if category not in summary:
                summary[category] = []
            summary[category].append(feature)
        return summary

    def get_selection_report(self) -> Dict:
        """Generate detailed selection report"""
        category_summary = self.get_feature_categories_summary()

        report = {
            'total_selected': len(self.selected_features),
            'total_removed': len(self.removed_features),
            'categories': {},
            'removed_features': self.removed_features
        }

        for category, features in category_summary.items():
            report['categories'][category] = {
                'count': len(features),
                'features': features
            }

        return report

    def print_selection_summary(self):
        """Print human-readable selection summary"""
        report = self.get_selection_report()

        print("=" * 80)
        print("FEATURE SELECTION SUMMARY")
        print("=" * 80)
        print(f"\nTotal Features Selected: {report['total_selected']}")
        print(f"Total Features Removed: {report['total_removed']}")

        print("\n" + "-" * 80)
        print("SELECTED FEATURES BY CATEGORY:")
        print("-" * 80)

        for category, info in report['categories'].items():
            print(f"\n{category}: {info['count']} features")
            for feat in info['features']:
                print(f"  • {feat}")

        if report['removed_features']:
            print("\n" + "-" * 80)
            print("REMOVED FEATURES (Medical/Not Allowed):")
            print("-" * 80)
            for feat in report['removed_features'][:20]:  # Show first 20
                print(f"  • {feat}")
            if len(report['removed_features']) > 20:
                print(f"  ... and {len(report['removed_features']) - 20} more")
