"""Age distribution validator"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from validation.demographic.base_validator import BaseValidator
from validation.demographic.demographics_benchmarks import (
    AGE_DISTRIBUTION,
    VALIDATION_THRESHOLDS,
)
from utils.logging import setup_logging

# Get logger instance
logger = setup_logging(log_dir=None, log_to_file=False, module_name="age_validator")


class AgeValidator(BaseValidator):
    """Validator for age-related demographic metrics."""

    def _perform_validation(self) -> Dict[str, Any]:
        results = {
            "age_distribution": self._validate_age_distribution(),
            "age_correlations": self._validate_age_correlations(),
            "generational_balance": self._validate_generational_balance(),
        }

        return results

    def _validate_age_distribution(self) -> Dict[str, Any]:
        """Validate age distribution against census benchmarks."""
        bins = [0, 18, 25, 35, 50, 65, 100]  # Aligned with AGE_DISTRIBUTION
        labels = list(AGE_DISTRIBUTION["age_groups"].keys())

        age_dist = pd.cut(
            self.df["AGEP"], bins=bins, labels=labels, include_lowest=True
        )

        # Check if PWGTP exists, if not use simple count
        if "PWGTP" in self.df.columns:
            distribution = (
                self.df.groupby(age_dist, observed=True)["PWGTP"]
                .sum()
                .div(self.df["PWGTP"].sum())
                .to_dict()
            )
        else:
            logger.warning("PWGTP column not found, using unweighted counts")
            distribution = age_dist.value_counts(normalize=True).to_dict()

        return self._validate_distribution(
            pd.Series(distribution),
            AGE_DISTRIBUTION["age_groups"],
            tolerance=VALIDATION_THRESHOLDS["distribution"]["tolerance"],
        )

    def _validate_age_correlations(self) -> Dict[str, Any]:
        """Validate age correlations with other variables."""
        results = {}

        # Age-Income correlation
        if all(col in self.df.columns for col in ["AGEP", "PINCP"]):
            # Adjust income for inflation
            adjusted_income = self.df["PINCP"] * (self.df["ADJINC"] / 1_000_000)
            correlation = self.df["AGEP"].corr(adjusted_income)
            results["age_income"] = {
                "actual_correlation": correlation,
                "expected_correlation": 0.35,  # From benchmarks
                "within_tolerance": abs(correlation - 0.35)
                < VALIDATION_THRESHOLDS["correlation"]["tolerance"],
            }

        # Age-Education correlation
        if all(col in self.df.columns for col in ["AGEP", "SCHL"]):
            correlation = self.df["AGEP"].corr(self.df["SCHL"])
            results["age_education"] = {
                "actual_correlation": correlation,
                "expected_correlation": 0.15,  # Weak positive correlation expected
                "within_tolerance": abs(correlation - 0.15)
                < VALIDATION_THRESHOLDS["correlation"]["tolerance"],
            }

        return results

    def _validate_generational_balance(self) -> Dict[str, Any]:
        """Validate generational representation."""
        gen_bins = [0, 25, 41, 57, 76, 100]
        gen_labels = list(AGE_DISTRIBUTION["generations"].keys())

        generations = pd.cut(
            self.df["AGEP"],
            bins=gen_bins,
            labels=gen_labels,
            include_lowest=True,
            right=True,
        )

        # Use PWGTP for weighted counts
        if "PWGTP" in self.df.columns:
            actual_dist = (
                self.df.groupby(generations, observed=True)["PWGTP"]
                .sum()
                .div(self.df["PWGTP"].sum())
                .to_dict()
            )
        else:
            logger.warning(
                "PWGTP column not found, using unweighted counts for generations"
            )
            actual_dist = generations.value_counts(normalize=True).to_dict()

        return self._validate_distribution(
            pd.Series(actual_dist),
            AGE_DISTRIBUTION["generations"],
            tolerance=VALIDATION_THRESHOLDS["distribution"]["tolerance"],
        )
