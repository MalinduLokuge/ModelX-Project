"""Create competition submission files"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


class SubmissionCreator:
    """Generate competition submission files"""

    def __init__(self, output_dir: Path, logger=None):
        self.output_dir = output_dir / "submissions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def create_submission(self, predictions: pd.Series, id_column: pd.Series = None,
                          id_name: str = "id", target_name: str = "target",
                          filename: Optional[str] = None) -> Path:
        """Create submission CSV file"""

        # Create submission dataframe
        if id_column is not None:
            submission = pd.DataFrame({
                id_name: id_column,
                target_name: predictions
            })
        else:
            # Use index as ID
            submission = pd.DataFrame({
                id_name: range(len(predictions)),
                target_name: predictions
            })

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{timestamp}.csv"

        # Save submission
        submission_path = self.output_dir / filename
        submission.to_csv(submission_path, index=False)

        if self.logger:
            self.logger.info(f"✓ Submission saved: {submission_path}")
            self.logger.info(f"  Shape: {submission.shape}")
            self.logger.info(f"  Sample: {submission.head(3).to_dict()}")

        return submission_path

    def create_proba_submission(self, proba_df: pd.DataFrame, id_column: pd.Series = None,
                                id_name: str = "id", filename: Optional[str] = None) -> Path:
        """Create submission with probability predictions"""

        # Add ID column
        if id_column is not None:
            submission = pd.DataFrame({id_name: id_column})
        else:
            submission = pd.DataFrame({id_name: range(len(proba_df))})

        # Add probability columns
        for col in proba_df.columns:
            submission[f"class_{col}"] = proba_df[col]

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_proba_{timestamp}.csv"

        # Save submission
        submission_path = self.output_dir / filename
        submission.to_csv(submission_path, index=False)

        if self.logger:
            self.logger.info(f"✓ Probability submission saved: {submission_path}")

        return submission_path
