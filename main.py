"""
CompeteML - Competition-Ready Automated ML System
Main CLI Entry Point
"""
import click
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config_manager import load_config
from src.core.pipeline_orchestrator import PipelineOrchestrator


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    CompeteML - Automated ML for Competitions

    Win ML competitions fast with automated pipelines.
    """
    pass


@cli.command()
@click.option('--train', '-t', required=True, type=click.Path(exists=True),
              help='Path to training data (CSV, Excel, Parquet, JSON)')
@click.option('--test', '-T', type=click.Path(exists=True), default=None,
              help='Path to test data (optional)')
@click.option('--target', '-y', default=None,
              help='Target column name (auto-detected if not provided)')
@click.option('--id-col', '-i', default=None,
              help='ID column name (auto-detected if not provided)')
@click.option('--config', '-c', type=click.Path(exists=True), default=None,
              help='Path to config YAML file')
@click.option('--preset', type=click.Choice(['quick', 'default', 'competition']),
              default='default',
              help='Configuration preset')
@click.option('--time-limit', '-l', type=int, default=None,
              help='Time limit in seconds')
@click.option('--output-dir', '-o', default='outputs',
              help='Output directory')
def run(train, test, target, id_col, config, preset, time_limit, output_dir):
    """
    Run the complete ML pipeline on your dataset.

    Example:
        competeml run --train data/train.csv --test data/test.csv
    """
    click.echo("=" * 80)
    click.echo("CompeteML - Automated ML Pipeline")
    click.echo("=" * 80)

    # Load configuration
    if config:
        cfg = load_config(config)
    else:
        # Use preset
        preset_configs = {
            'quick': 'configs/quick_test.yaml',
            'default': 'configs/default.yaml',
            'competition': 'configs/competition.yaml'
        }
        config_path = Path(__file__).parent / preset_configs[preset]
        if config_path.exists():
            cfg = load_config(str(config_path))
        else:
            cfg = load_config()

    # Override with CLI arguments
    cfg.train_path = train
    cfg.test_path = test
    if target:
        cfg.target_column = target
    if id_col:
        cfg.id_column = id_col
    if time_limit:
        cfg.time_limit = time_limit
    cfg.output_dir = output_dir

    # Create and run pipeline
    try:
        orchestrator = PipelineOrchestrator(cfg)
        results = orchestrator.run()

        # Print summary
        click.echo("\n" + "=" * 80)
        click.echo("PIPELINE SUMMARY")
        click.echo("=" * 80)
        click.echo(f"Run ID: {results['run_id']}")
        click.echo(f"Problem Type: {cfg.problem_type}")
        click.echo(f"Best Model: {results.get('best_model', 'N/A')}")

        if 'submission_path' in results:
            click.echo(f"Submission: {results['submission_path']}")

        click.echo("\n✓ Pipeline completed successfully!")

    except Exception as e:
        click.echo(f"\n✗ Pipeline failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--train', '-t', required=True, type=click.Path(exists=True),
              help='Path to training data')
def explore(train):
    """
    Quickly explore and analyze your dataset.

    Example:
        competeml explore --train data/train.csv
    """
    click.echo("Exploring dataset...")
    click.echo(f"Loading: {train}")

    # Quick EDA
    import pandas as pd
    df = pd.read_csv(train)

    click.echo(f"\nShape: {df.shape}")
    click.echo(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        click.echo(f"  - {col}: {df[col].dtype}")

    click.echo(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            pct = count / len(df) * 100
            click.echo(f"  - {col}: {count} ({pct:.1f}%)")
    else:
        click.echo("  No missing values")

    click.echo(f"\nFirst few rows:")
    click.echo(df.head())


@cli.command()
def info():
    """Display system information and configuration."""
    click.echo("CompeteML v1.0.0")
    click.echo("\nInstalled AutoML Frameworks:")

    try:
        import autogluon
        click.echo(f"  ✓ AutoGluon {autogluon.__version__}")
    except ImportError:
        click.echo("  ✗ AutoGluon (not installed)")

    try:
        import optuna
        click.echo(f"  ✓ Optuna {optuna.__version__}")
    except ImportError:
        click.echo("  ✗ Optuna (not installed)")

    click.echo("\nConfiguration Presets:")
    click.echo("  - quick: 5-minute test run")
    click.echo("  - default: 1-hour balanced run")
    click.echo("  - competition: 2-hour high-performance run")


if __name__ == '__main__':
    cli()
