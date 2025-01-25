"""
Main Marketing Engagement Validator

Orchestrates all marketing engagement validations and provides the main entry point
for running validations.
"""

import os

print("PYTHONPATH:", os.environ.get("PYTHONPATH"))


import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import json
import sys

# Change to relative imports
from validation.marketing.campaign_validator import CampaignValidator
from validation.marketing.engagement_validator import EngagementValidator
from validation.marketing.loyalty_validator import LoyaltyValidator
from utils.encoders import NumpyEncoder
from validation.marketing.base_validator import BaseValidator


class MarketingEngagementValidator:
    """Master validator for marketing engagement data only."""

    def __init__(
        self,
        campaigns: pd.DataFrame,
        engagement: pd.DataFrame,
        loyalty: pd.DataFrame,
    ):
        """Initialize with only marketing engagement data."""
        self.validators = {
            "campaign": CampaignValidator(campaigns),
            "engagement": EngagementValidator(engagement),
            "loyalty": LoyaltyValidator(loyalty),
        }

    def run_validation(self) -> Dict:
        """Run all validations and return combined results."""
        # Store results as instance attribute
        self.results = {
            "campaign": self.validators["campaign"].validate(),
            "engagement": self.validators["engagement"].validate(),
            "loyalty": self.validators["loyalty"].validate(),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "validator_version": "1.0",
            },
        }

        # Calculate overall metrics
        total_checks = sum(
            len(v)
            for k, v in self.results.items()
            if isinstance(v, dict) and k != "metadata"
        )
        passing_checks = sum(
            len(
                [
                    r
                    for r in v.values()
                    if isinstance(r, dict) and r.get("within_tolerance", False)
                ]
            )
            for k, v in self.results.items()
            if isinstance(v, dict) and k != "metadata"
        )

        self.results["metadata"].update(
            {
                "total_metrics_checked": total_checks,
                "passing_metrics": passing_checks,
                "overall_score": (
                    passing_checks / total_checks if total_checks > 0 else 0.0
                ),
            }
        )

        return self.results

    def save_results(self, output_path: Path) -> None:
        """Save validation results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, cls=NumpyEncoder, indent=2)


def setup_logging(log_dir: Path) -> None:
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"validation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main execution function."""
    # Set up directories - fix path resolution
    script_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = script_dir / "data"
    marketing_dir = data_dir / "marketing_engagement"
    output_dir = data_dir / "validation_results"
    log_dir = script_dir / "logs"

    # Create directories
    for directory in [data_dir, marketing_dir, output_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    try:
        # Find marketing engagement data files with version support
        campaign_files = sorted(marketing_dir.glob("**/campaigns*.csv"))
        engagement_files = sorted(marketing_dir.glob("**/engagement*.csv"))
        loyalty_files = sorted(marketing_dir.glob("**/loyalty*.csv"))

        # Validate file sets
        if not (campaign_files and engagement_files and loyalty_files):
            raise FileNotFoundError(
                "Required marketing engagement data files not found"
            )

        if len(campaign_files) != len(engagement_files) or len(engagement_files) != len(
            loyalty_files
        ):
            raise ValueError("Mismatched number of data files")

        # Process each version set
        for c_file, e_file, l_file in zip(
            campaign_files, engagement_files, loyalty_files
        ):
            version = c_file.parent.name
            logger.info(f"\nProcessing version: {version}")
            logger.info(
                f"Validating:\n- {c_file.name}\n- {e_file.name}\n- {l_file.name}"
            )

            # Load data
            campaign_data = pd.read_csv(c_file)
            engagement_data = pd.read_csv(e_file)
            loyalty_data = pd.read_csv(l_file)

            # Run validation
            validator = MarketingEngagementValidator(
                campaigns=campaign_data,
                engagement=engagement_data,
                loyalty=loyalty_data,
            )
            results = validator.run_validation()

            # Save results with version info
            output_path = output_dir / version / f"validation_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            validator.save_results(output_path)

            logger.info(
                f"Validation completed with score: {results['metadata']['overall_score']:.2%}"
            )
            logger.info(f"Results saved to: {output_path}")

        logger.info("\nMarketing engagement validation completed successfully")

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
