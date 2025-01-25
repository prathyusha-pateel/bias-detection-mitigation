"""
Base Validator Class for Demographics

Provides common validation functionality and utilities for all demographic validators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime
import json
from utils.encoders import NumpyEncoder
from utils.logging import setup_logging

logger = setup_logging(log_dir=None, log_to_file=False, module_name="base_validator")


class BaseValidator(ABC):
    """Base validator for demographic data validation."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with demographic data."""
        self.df = df
        self.required_columns = {
            "PWGTP": "Person weight",
            "AGEP": "Age",
            "PINCP": "Personal income",
            "ADJINC": "Income adjustment factor",
            "STATE": "State code",
            "SCHL": "Educational attainment",
        }
        self._validate_required_columns()

    def _validate_required_columns(self) -> None:
        """Check for required columns and log warnings for missing ones."""
        missing_columns = [
            col for col in self.required_columns if col not in self.df.columns
        ]
        if missing_columns:
            for col in missing_columns:
                logger.warning(
                    f"Missing required column: {col} ({self.required_columns[col]})"
                )

    def _validate_distribution(
        self, actual: pd.Series, expected: Dict[str, float], tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """Validate distribution against expected values."""
        results = {"details": {}}

        # Only log weighted calculation message once per validator instance
        if not hasattr(self, "_weighted_logged"):
            if "PWGTP" in self.df.columns:
                logger.debug("Using weighted calculations for population estimates")
            else:
                logger.warning(
                    "Using unweighted calculations - results represent sample only"
                )
            self._weighted_logged = True

        for category, expected_value in expected.items():
            if category in actual.index:
                actual_value = actual[category]
                within_tolerance = abs(actual_value - expected_value) <= tolerance
                results["details"][category] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "within_tolerance": within_tolerance,
                }
            else:
                # Only log missing categories for multi-state validations
                if len(self.df["STATE"].unique()) > 1:
                    logger.debug(
                        f"Category {category} not found in actual distribution"
                    )
                results["details"][category] = {
                    "expected": expected_value,
                    "actual": 0.0,
                    "within_tolerance": False,
                }

        return results

    def validate(self) -> Dict[str, Any]:
        """Run validation and return results."""
        try:
            self._validate_required_columns()
            results = self._perform_validation()
            return results
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise ValidationError(f"Validation failed: {str(e)}") from e

    @abstractmethod
    def _perform_validation(self) -> Dict[str, Any]:
        """Perform specific validation checks."""
        pass

    def to_json(self) -> str:
        """Convert validation results to JSON string."""
        return json.dumps(self.validate(), cls=NumpyEncoder, indent=2)

    def _count_metrics(self, results: Dict) -> int:
        """Count total number of metrics checked."""
        return len(
            [
                metric
                for category in results.values()
                if isinstance(category, dict)
                for metric in category.values()
                if isinstance(metric, dict) and "within_tolerance" in metric
            ]
        )

    def _count_passing_metrics(self, results: Dict) -> int:
        """Count number of passing metrics."""
        return len(
            [
                metric
                for category in results.values()
                if isinstance(category, dict)
                for metric in category.values()
                if isinstance(metric, dict) and metric.get("within_tolerance", False)
            ]
        )


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass
