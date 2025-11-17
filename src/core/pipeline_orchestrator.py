"""Main pipeline orchestrator - the brain of CompeteML"""
from pathlib import Path
from typing import Optional, Dict, Any
import time
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config_manager import CompeteMLConfig
from .logger import CompeteMLLogger
from .data_loader import DataLoader
from .data_validator import DataValidator
from preprocessing.auto_preprocessor import AutoPreprocessor
from modeling.auto_trainer import AutoTrainer
from evaluation.metrics_calculator import MetricsCalculator
from reporting.submission_creator import SubmissionCreator
from eda.auto_eda import AutoEDA
from feature_engineering.auto_features import AutoFeatureEngineer
from modeling.model_saver import ModelSaver


class PipelineOrchestrator:
    """Orchestrates the entire ML pipeline"""

    def __init__(self, config: CompeteMLConfig):
        self.config = config
        self.logger = CompeteMLLogger("CompeteML")
        self.start_time = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Components
        self.data_loader = DataLoader(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.preprocessor = None
        self.trainer = None
        self.metrics_calculator = None
        self.submission_creator = None
        self.eda_engine = None
        self.feature_engineer = None
        self.model_saver = None

        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.test_ids = None

        # Results storage
        self.results = {
            'run_id': self.run_id,
            'config': config,
            'data_info': {},
            'eda_results': {},
            'preprocessing_steps': [],
            'feature_engineering_steps': [],
            'model_results': {},
            'best_model': None,
            'predictions': None,
            'recipe': []
        }

        # Output directory
        self.output_dir = Path(self.config.output_dir) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline"""
        self.start_time = time.time()
        self.logger.section("CompeteML Pipeline Started")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Mode: {self.config.mode}")
        self.logger.info(f"Time limit: {self.config.time_limit}s")

        # Calculate total steps
        total_steps = 5  # load, validate, preprocess, train, output
        if self.config.run_eda:
            total_steps += 1
        if self.config.auto_features:
            total_steps += 1
        current_step = 0

        try:
            # 1. Load data
            current_step += 1
            self._log_progress(current_step, total_steps, "Loading data")
            step_start = time.time()
            self._load_data()
            self._log_step_complete(step_start)

            # 2. Validate data
            current_step += 1
            self._log_progress(current_step, total_steps, "Validating data")
            step_start = time.time()
            self._validate_data()
            self._log_step_complete(step_start)

            # 3. EDA (if enabled)
            if self.config.run_eda:
                current_step += 1
                self._log_progress(current_step, total_steps, "Performing EDA")
                step_start = time.time()
                self._run_eda()
                self._log_step_complete(step_start)

            # 4. Preprocessing
            current_step += 1
            self._log_progress(current_step, total_steps, "Preprocessing data")
            step_start = time.time()
            self._preprocess()
            self._log_step_complete(step_start)

            # 5. Feature engineering
            if self.config.auto_features:
                current_step += 1
                self._log_progress(current_step, total_steps, "Engineering features")
                step_start = time.time()
                self._engineer_features()
                self._log_step_complete(step_start)

            # 6. Train models
            current_step += 1
            self._log_progress(current_step, total_steps, "Training models")
            step_start = time.time()
            self._train_models()
            self._log_step_complete(step_start)

            # 7. Evaluate
            self._evaluate()

            # 8. Generate outputs
            current_step += 1
            self._log_progress(current_step, total_steps, "Generating outputs")
            step_start = time.time()
            self._generate_outputs()
            self._log_step_complete(step_start)

            elapsed = time.time() - self.start_time
            self.logger.section("Pipeline Completed Successfully")
            self.logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            self.logger.success(f"All {total_steps} steps completed!")

            return self.results

        except MemoryError:
            self.logger.error("Out of memory!")
            self.logger.error("Suggestions:")
            self.logger.error("  1. Reduce max_features in config")
            self.logger.error("  2. Disable polynomial_features")
            self.logger.error("  3. Use smaller ag_num_bag_folds")
            self._save_error_report("MemoryError", "Out of memory during pipeline execution")
            raise

        except TimeoutError:
            self.logger.error("Time limit exceeded!")
            self.logger.info("Saving best model found so far...")
            if self.trainer and hasattr(self.trainer, 'model'):
                self._save_error_report("TimeoutError", "Time limit exceeded, partial results saved")
            raise

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")

            # Provide helpful suggestions based on error type
            if "autogluon" in str(e).lower():
                self.logger.error("AutoGluon failed. Suggestions:")
                self.logger.error("  1. Check data types are correct")
                self.logger.error("  2. Ensure target column exists")
                self.logger.error("  3. Try with quick_test config first")
            elif "disk" in str(e).lower() or "space" in str(e).lower():
                self.logger.error("Disk space issue. Suggestions:")
                self.logger.error("  1. Free up disk space")
                self.logger.error("  2. Change output_dir to different drive")
                self.logger.error("  3. Set save_models=False in config")

            self._save_error_report(type(e).__name__, str(e))
            raise

    def _load_data(self):
        """Load and prepare data"""
        self.logger.section("Loading Data")

        train_df, test_df = self.data_loader.load(
            train_path=self.config.train_path,
            test_path=self.config.test_path,
            target_column=self.config.target_column,
            id_column=self.config.id_column
        )

        # Update config with detected values
        self.config.target_column = self.data_loader.target_column
        self.config.id_column = self.data_loader.id_column

        if not self.config.problem_type or self.config.problem_type == "auto":
            self.config.problem_type = self.data_loader.problem_type

        self.results['data_info'] = self.data_loader.get_metadata()
        self.results['recipe'].append("Loaded data and detected problem type")

    def _validate_data(self):
        """Validate data quality"""
        self.logger.section("Validating Data")

        validation_results = self.data_validator.validate(
            self.data_loader.train_df,
            target_column=self.config.target_column
        )

        self.results['data_info']['validation'] = validation_results

        if not validation_results['is_valid']:
            raise ValueError("Data validation failed. Check issues in validation results.")

        self.results['recipe'].append("Validated data quality")

    def _run_eda(self):
        """Run exploratory data analysis"""
        self.eda_engine = AutoEDA(output_dir=self.output_dir, logger=self.logger)

        eda_results = self.eda_engine.analyze(
            df=self.data_loader.train_df,
            target_column=self.config.target_column,
            problem_type=self.config.problem_type
        )

        self.results['eda_results'] = eda_results

        # Log key insights
        if eda_results.get('insights'):
            self.logger.info("\nKey Insights:")
            for insight in eda_results['insights']:
                self.logger.info(f"  • {insight}")

        self.results['recipe'].append(f"Ran EDA: {len(eda_results.get('plots', []))} plots, {len(eda_results.get('insights', []))} insights")

    def _preprocess(self):
        """Preprocess data"""
        self.logger.section("Preprocessing Data")

        # Split features and target
        self.X_train, self.y_train = self.data_loader.get_X_y(self.data_loader.train_df)

        # Initialize preprocessor
        config_dict = {
            'handle_missing': self.config.handle_missing,
            'encoding_strategy': self.config.encoding_strategy,
            'scaling_strategy': self.config.scaling_strategy,
            'apply_competition_tricks': self.config.apply_competition_tricks
        }
        self.preprocessor = AutoPreprocessor(config=config_dict, logger=self.logger)

        # Fit and transform training data
        self.X_train, preprocess_report = self.preprocessor.fit_transform(self.X_train, self.y_train)

        self.results['preprocessing_steps'] = self.preprocessor.get_steps()
        self.results['preprocessing_report'] = preprocess_report

        # Preprocess test data if available
        if self.data_loader.test_df is not None:
            # Store test IDs if available
            if self.config.id_column and self.config.id_column in self.data_loader.test_df.columns:
                self.test_ids = self.data_loader.test_df[self.config.id_column]

            # Remove ID column if present
            test_cols_to_drop = []
            if self.config.id_column and self.config.id_column in self.data_loader.test_df.columns:
                test_cols_to_drop.append(self.config.id_column)

            self.X_test = self.data_loader.test_df.drop(columns=test_cols_to_drop, errors='ignore')
            self.X_test = self.preprocessor.transform(self.X_test)

            self.logger.info(f"Test data preprocessed: {self.X_test.shape}")

        self.results['recipe'].append("Preprocessed data (handled missing, encoded, scaled)")

    def _engineer_features(self):
        """Engineer features"""
        self.logger.section("Feature Engineering")

        # Initialize feature engineer
        config_dict = {
            'interaction_features': self.config.interaction_features,
            'polynomial_features': self.config.polynomial_features,
            'feature_selection': self.config.feature_selection,
            'max_features': self.config.max_features,
            'apply_competition_tricks': self.config.apply_competition_tricks
        }

        self.feature_engineer = AutoFeatureEngineer(config=config_dict, logger=self.logger)

        # Engineer features
        self.X_train, fe_report = self.feature_engineer.engineer_features(
            X=self.X_train,
            y=self.y_train,
            problem_type=self.config.problem_type
        )

        self.results['feature_engineering_steps'] = fe_report

        # Also apply to test if available
        if self.X_test is not None:
            self.X_test = self.feature_engineer.transform(self.X_test)
            self.logger.info(f"Applied feature engineering to test set: {self.X_test.shape}")

        self.results['recipe'].append(f"Engineered features: {fe_report['original_features']} → {fe_report['final_features']} features")

    def _train_models(self):
        """Train models"""
        self.logger.section("Training Models")

        # Prepare training data (combine X and y back for AutoGluon)
        import pandas as pd
        train_data = pd.concat([self.X_train, self.y_train], axis=1)

        # Initialize trainer
        config_dict = {
            'automl_framework': self.config.automl_framework,
            'target_column': self.config.target_column,
            'problem_type': self.config.problem_type,
            'time_limit': self.config.time_limit,
            'quick_test': self.config.quick_test,
            'ag_preset': self.config.ag_preset,
            'ag_num_bag_folds': self.config.ag_num_bag_folds,
            'ag_num_stack_levels': self.config.ag_num_stack_levels,
            'eval_metric': self.config.eval_metric,
            'verbose': self.config.verbose
        }

        self.trainer = AutoTrainer(config=config_dict, output_dir=self.output_dir, logger=self.logger)

        # Train
        training_results = self.trainer.train(train_data=train_data)

        self.results['model_results'] = training_results
        self.results['best_model'] = training_results.get('best_model')

        self.results['recipe'].append(f"Trained models using {self.config.automl_framework}")

    def _evaluate(self):
        """Evaluate models"""
        self.logger.section("Evaluating Models")

        # AutoGluon already evaluates during training
        # We just need to get predictions for test set if available

        if self.X_test is not None:
            self.logger.info("Generating predictions for test set")

            predictions = self.trainer.predict(self.X_test)
            self.results['predictions'] = predictions

            self.logger.info(f"Predictions generated: {len(predictions)} samples")

            # Try to get probabilities for classification
            if self.config.problem_type == 'classification':
                try:
                    predictions_proba = self.trainer.predict_proba(self.X_test)
                    self.results['predictions_proba'] = predictions_proba
                    self.logger.info("Probability predictions generated")
                except:
                    self.logger.warning("Could not generate probability predictions")

        self.results['recipe'].append("Evaluated models and generated predictions")

    def _generate_outputs(self):
        """Generate all outputs"""
        self.logger.section("Generating Outputs")

        # Generate submission file
        if self.config.generate_submission and self.results['predictions'] is not None:
            self.submission_creator = SubmissionCreator(output_dir=self.output_dir, logger=self.logger)

            submission_path = self.submission_creator.create_submission(
                predictions=self.results['predictions'],
                id_column=self.test_ids,
                id_name=self.config.id_column if self.config.id_column else 'id',
                target_name=self.config.target_column
            )

            self.results['submission_path'] = str(submission_path)
            self.results['recipe'].append(f"Created submission file: {submission_path.name}")

        # Save deployment package
        if self.config.save_models and self.trainer:
            self.model_saver = ModelSaver(output_dir=self.output_dir, logger=self.logger)

            metadata = {
                'run_id': self.run_id,
                'problem_type': self.config.problem_type,
                'model_results': self.results.get('model_results', {}),
                'final_features': self.X_train.shape[1] if self.X_train is not None else 0
            }

            package_dir = self.model_saver.save_deployment_package(
                model=self.trainer.model if hasattr(self.trainer, 'model') else self.trainer.predictor,
                preprocessor=self.preprocessor,
                feature_engineer=self.feature_engineer,
                metadata=metadata,
                package_name=f"model_{self.run_id}"
            )

            self.results['model_package_path'] = str(package_dir)
            self.results['recipe'].append(f"Saved deployment package: {package_dir.name}")

        # Save recipe
        if self.config.generate_recipe:
            self._save_recipe()

        # Report generation (placeholder)
        if self.config.generate_report:
            self.logger.info("Detailed report generation will be implemented")

        self.logger.info(f"✓ All outputs saved to: {self.output_dir}")

    def _save_recipe(self):
        """Save recipe file showing what was done"""
        recipe_path = self.output_dir / "recipe.txt"

        with open(recipe_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPETEML PIPELINE RECIPE\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("STEPS PERFORMED:\n")
            f.write("-" * 80 + "\n")
            for i, step in enumerate(self.results['recipe'], 1):
                f.write(f"{i}. {step}\n")

            f.write("\n" + "=" * 80 + "\n")

        self.logger.info(f"Recipe saved: {recipe_path}")

    def _save_error_report(self, error_type: str, error_message: str):
        """Save error report for debugging"""
        try:
            error_path = self.output_dir / "error_report.txt"

            with open(error_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("COMPETEML ERROR REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Run ID: {self.run_id}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error Type: {error_type}\n\n")

                f.write("ERROR MESSAGE:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{error_message}\n\n")

                f.write("PIPELINE STATE:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Data loaded: {self.X_train is not None}\n")
                f.write(f"Preprocessing complete: {self.preprocessor is not None}\n")
                f.write(f"Feature engineering complete: {self.feature_engineer is not None}\n")
                f.write(f"Training complete: {self.trainer is not None}\n\n")

                if self.results['recipe']:
                    f.write("STEPS COMPLETED:\n")
                    f.write("-" * 80 + "\n")
                    for i, step in enumerate(self.results['recipe'], 1):
                        f.write(f"{i}. {step}\n")

                f.write("\n" + "=" * 80 + "\n")

            self.logger.info(f"Error report saved: {error_path}")

        except Exception as e:
            self.logger.warning(f"Could not save error report: {str(e)}")

    def _log_progress(self, current_step: int, total_steps: int, task: str):
        """Log progress with step counter"""
        elapsed = time.time() - self.start_time
        self.logger.info(f"\n[{current_step}/{total_steps}] {task}... (elapsed: {elapsed:.1f}s)")

    def _log_step_complete(self, step_start: float):
        """Log step completion time"""
        step_time = time.time() - step_start
        self.logger.success(f"✓ Complete ({step_time:.1f}s)")
