"""Main demographic validator orchestrator"""

import pandas as pd
from typing import Dict, Any
from datetime import datetime
import json

from utils.encoders import NumpyEncoder
from validation.demographic import (
    AgeValidator,
    IncomeValidator,
    LocationValidator,
    EducationValidator,
)
from utils.logging import setup_logging

logger = setup_logging(
    log_dir=None, log_to_file=False, module_name="demographic_validator"
)


class DemographicValidator:
    """Master validator for demographic data."""

    def __init__(self, state_data: Dict[str, pd.DataFrame]):
        """Initialize with demographic data dictionary."""
        # Validate state data
        if not state_data:
            raise ValueError("No state data provided")

        # Combine all state DataFrames into one
        self.df = pd.concat(state_data.values(), ignore_index=True)

        # Log available columns
        logger.info(
            f"Available columns in combined dataset: {', '.join(self.df.columns)}"
        )

        # Check for required columns
        required_columns = {
            "PWGTP": "Person weight",
            "AGEP": "Age",
            "PINCP": "Personal income",
            "ADJINC": "Income adjustment factor",
            "STATE": "State code",
            "SCHL": "Educational attainment",
        }

        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            logger.warning("Missing required columns:")
            for col in missing_columns:
                logger.warning(f"  - {col} ({required_columns[col]})")

        self.validators = {
            "age": AgeValidator(self.df),
            "income": IncomeValidator(self.df),
            "location": LocationValidator(self.df),
            "education": EducationValidator(self.df),
        }

    def validate_all(self) -> Dict[str, Any]:
        """Run all validations and return combined results."""
        results = {
            "age": self.validators["age"].validate(),
            "income": self.validators["income"].validate(),
            "location": self.validators["location"].validate(),
            "education": self.validators["education"].validate(),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "validator_version": "1.0",
            },
        }

        # Calculate overall metrics - This is wrong
        total_checks = sum(
            len(v)
            for k, v in results.items()
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
            for k, v in results.items()
            if isinstance(v, dict) and k != "metadata"
        )

        # We need to count the actual validations and their results
        total_metrics = 0
        passing_metrics = 0

        for validator_name, validator_results in results.items():
            if validator_name == "metadata":
                continue

            for metric_name, metric_data in validator_results.items():
                if isinstance(metric_data, dict):
                    if "details" in metric_data:
                        # Count distribution validations
                        for category, details in metric_data["details"].items():
                            total_metrics += 1
                            if details.get("within_tolerance", False):
                                passing_metrics += 1
                    elif "actual_value" in metric_data:
                        # Count correlation validations
                        total_metrics += 1
                        if metric_data.get("within_tolerance", False):
                            passing_metrics += 1

        results["metadata"].update(
            {
                "total_metrics_checked": total_metrics,
                "passing_metrics": passing_metrics,
                "overall_score": (
                    passing_metrics / total_metrics if total_metrics > 0 else 0.0
                ),
            }
        )

        return results

    def to_json(self) -> str:
        """Convert validation results to JSON string."""
        return json.dumps(self.validate_all(), cls=NumpyEncoder, indent=2)
