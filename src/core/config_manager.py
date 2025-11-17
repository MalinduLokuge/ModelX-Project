"""Configuration management for CompeteML"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class CompeteMLConfig:
    """Main configuration class"""

    # Mode
    mode: str = "auto"  # auto, guided, manual

    # Data
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    target_column: Optional[str] = None
    id_column: Optional[str] = None
    problem_type: Optional[str] = None  # auto-detect, classification, regression

    # Time constraints
    time_limit: int = 3600  # seconds (1 hour default)
    quick_test: bool = False  # 5-minute test run

    # EDA
    run_eda: bool = True
    generate_profile: bool = True
    eda_visualizations: bool = True

    # Preprocessing
    handle_missing: bool = True
    handle_outliers: bool = True
    scaling_strategy: str = "auto"  # auto, standard, minmax, robust, none
    encoding_strategy: str = "auto"  # auto, onehot, target, ordinal

    # Feature Engineering
    auto_features: bool = True
    interaction_features: bool = True
    polynomial_features: bool = False
    time_features: bool = True
    text_features: bool = True
    feature_selection: bool = True
    max_features: Optional[int] = None
    apply_competition_tricks: bool = False  # Target encoding, frequency encoding, etc.

    # Modeling
    automl_framework: str = "autogluon"  # autogluon, pycaret, optuna
    cv_folds: int = 5
    ensemble: bool = True

    # AutoGluon specific
    ag_preset: str = "medium_quality"  # best_quality, high_quality, medium_quality, optimize_for_deployment
    ag_num_bag_folds: int = 5
    ag_num_stack_levels: int = 1

    # Evaluation
    eval_metric: Optional[str] = None  # auto-detect or specify

    # Output
    output_dir: str = "outputs"
    save_predictions: bool = True
    save_models: bool = True
    generate_report: bool = True
    generate_recipe: bool = True
    generate_submission: bool = True

    # Logging
    verbose: int = 2  # 0=silent, 1=basic, 2=detailed, 3=debug

    # Advanced
    random_seed: int = 42
    n_jobs: int = -1

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages configuration loading and merging"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config = CompeteMLConfig()

        if config_path and config_path.exists():
            self.load_from_file(config_path)

    def load_from_file(self, path: Path) -> CompeteMLConfig:
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Update config with loaded values
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        return self.config

    def update(self, **kwargs) -> CompeteMLConfig:
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        return self.config

    def save(self, path: Path):
        """Save configuration to YAML file"""
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_')
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def get(self) -> CompeteMLConfig:
        """Get current configuration"""
        return self.config


def load_config(config_path: Optional[str] = None, **kwargs) -> CompeteMLConfig:
    """Convenience function to load and update config"""
    manager = ConfigManager(Path(config_path) if config_path else None)
    if kwargs:
        manager.update(**kwargs)
    return manager.get()
