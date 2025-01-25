"""Base Validator Class for Consumer Preferences."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache

from constants import VALIDATION_THRESHOLDS
from utils.logging import (
    log_validation_step,
    log_validation_error,
    log_metric_validation,
    log_validation_summary,
)


class BaseValidator(ABC):
    """Base validator for consumer preference data validation."""

    def __init__(
        self,
        preferences: pd.DataFrame,
        demographic_data: Optional[pd.DataFrame] = None,
    ):
        """Initialize with consumer preference data."""
        self.preferences = preferences
        self.demographic_data = demographic_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache = {}

    def validate(self) -> Dict[str, Any]:
        """Validate consumer preference data against benchmarks."""
        try:
            log_validation_step(
                self.logger, f"Starting {self.__class__.__name__} validation"
            )

            # Basic data quality checks
            self._validate_data_structure()

            # Run specific validations
            results = self._perform_validation()

            # Validate data distributions
            distribution_results = self._validate_distributions()
            results["distributions"] = distribution_results

            # Calculate summary metrics
            summary = self._calculate_summary_metrics(results)
            results["summary"] = summary

            # Log validation results
            self._log_validation_results(results)

            return results

        except KeyError as e:
            # Handle missing columns more gracefully
            column = str(e).strip("'")
            log_validation_error(
                self.logger,
                "Data structure validation",
                e,
                {
                    "missing_column": column,
                    "available_columns": list(self.preferences.columns),
                },
            )
            return {
                "error": f"Missing required column: {column}",
                "available_columns": list(self.preferences.columns),
            }
        except Exception as e:
            log_validation_error(
                self.logger, f"{self.__class__.__name__} validation", e
            )
            raise ValidationError(f"Validation failed: {str(e)}") from e

    def _validate_data_structure(self) -> None:
        """Validate basic data structure requirements."""
        log_validation_step(self.logger, "Data structure validation")

        # Required columns for consumer preference data
        required_columns = {
            "consumer_id",
            "age_group",
            "online_shopping_rate",
            "loyalty_memberships",
        }

        missing_cols = required_columns - set(self.preferences.columns)
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")

        # Validate age_group values
        if "age_group" in self.preferences.columns:
            valid_age_groups = {"18-34", "35-54", "55+"}
            invalid_groups = (
                set(self.preferences["age_group"].unique()) - valid_age_groups
            )
            if invalid_groups:
                self.logger.warning(f"Found invalid age groups: {invalid_groups}")

    @lru_cache(maxsize=32)
    def _validate_distribution(
        self,
        actual_dist: pd.Series,
        expected_dist: Dict[str, float],
        tolerance: float = 0.05,
        min_support: int = 100,
    ) -> Dict[str, Dict]:
        """Validate distribution against expected values."""
        results = {}

        # Ensure we have enough data
        if len(actual_dist) < min_support:
            self.logger.warning(
                f"Insufficient data for distribution validation: {len(actual_dist)} < {min_support}"
            )
            return {"error": "Insufficient data"}

        for category, expected_share in expected_dist.items():
            actual_share = actual_dist.get(category, 0.0)
            difference = abs(actual_share - expected_share)
            within_tolerance = difference <= tolerance

            results[category] = {
                "expected_value": float(expected_share),
                "actual_value": float(actual_share),
                "difference": float(difference),
                "within_tolerance": within_tolerance,
            }

            log_metric_validation(
                self.logger,
                f"{category} distribution",
                expected_share,
                actual_share,
                difference,
                within_tolerance,
            )

        return results

    def _validate_metric(
        self,
        actual: float,
        expected: float,
        metric_type: str,
        additional_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Validate metric against expected value."""
        difference = abs(actual - expected)
        tolerance = VALIDATION_THRESHOLDS.get(metric_type, {}).get("tolerance", 0.1)
        within_tolerance = difference <= tolerance

        result = {
            "expected_value": float(expected),
            "actual_value": float(actual),
            "difference": float(difference),
            "within_tolerance": within_tolerance,
            "metric_type": metric_type,
            "tolerance": tolerance,
        }

        if additional_info:
            result.update(additional_info)

        log_metric_validation(
            self.logger,
            metric_type,
            expected,
            actual,
            difference,
            within_tolerance,
            additional_info,
        )

        return result

    def _calculate_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics for validation results."""
        total_metrics = self._count_metrics(results)
        passing_metrics = self._count_passing_metrics(results)

        return {
            "total_metrics": total_metrics,
            "passing_metrics": passing_metrics,
            "success_rate": (
                passing_metrics / total_metrics if total_metrics > 0 else 0.0
            ),
            "timestamp": datetime.now().isoformat(),
        }

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

    def _validate_distributions(self) -> Dict[str, Any]:
        """Validate basic data quality metrics."""
        return {
            "missing_values": self._check_missing_values(),
            "duplicates": self._check_duplicates(),
            "outliers": self._check_outliers(),
            "value_ranges": self._check_value_ranges(),
        }

    def _check_missing_values(self) -> Dict[str, float]:
        """Check for missing values in each column."""
        missing_rates = self.preferences.isnull().mean()
        return {col: float(rate) for col, rate in missing_rates.items()}

    def _check_duplicates(self) -> Dict[str, int]:
        """Check for duplicate records."""
        return {"duplicate_records": int(self.preferences.duplicated().sum())}

    def _check_outliers(self) -> Dict[str, Dict[str, int]]:
        """Check for outliers in numerical columns."""
        outliers = {}
        for col in self.preferences.select_dtypes(include=[np.number]).columns:
            q1 = self.preferences[col].quantile(0.25)
            q3 = self.preferences[col].quantile(0.75)
            iqr = q3 - q1
            outliers[col] = {
                "below": int(sum(self.preferences[col] < (q1 - 1.5 * iqr))),
                "above": int(sum(self.preferences[col] > (q3 + 1.5 * iqr))),
            }
        return outliers

    def _check_value_ranges(self) -> Dict[str, Dict[str, float]]:
        """Check value ranges for numerical columns."""
        ranges = {}
        for col in self.preferences.select_dtypes(include=[np.number]).columns:
            ranges[col] = {
                "min": float(self.preferences[col].min()),
                "max": float(self.preferences[col].max()),
                "mean": float(self.preferences[col].mean()),
                "std": float(self.preferences[col].std()),
            }
        return ranges

    @abstractmethod
    def _perform_validation(self) -> Dict[str, Any]:
        """Perform specific validation checks."""
        pass


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass
