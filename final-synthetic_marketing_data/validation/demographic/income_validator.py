"""Income distribution validator"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from validation.demographic.base_validator import BaseValidator
from validation.demographic.demographics_benchmarks import (
    INCOME_DISTRIBUTION,
    VALIDATION_THRESHOLDS,
)


class IncomeValidator(BaseValidator):
    """Validator for income-related demographic metrics."""

    def _perform_validation(self) -> Dict[str, Any]:
        results = {
            "income_distribution": self._validate_income_distribution(),
            "correlations": self._validate_correlations(),
        }
        return results

    def _validate_income_distribution(self) -> Dict[str, Any]:
        """Validate income distribution."""
        # Adjust income using ADJINC factor
        adjusted_income = self.df["PINCP"] * (self.df["ADJINC"] / 1_000_000)

        # Create bins based on INCOME_DISTRIBUTION brackets
        bins = [0, 25000, 50000, 75000, 100000, 150000, float("inf")]
        labels = list(INCOME_DISTRIBUTION["income_brackets"].keys())

        income_dist = pd.cut(adjusted_income, bins=bins, labels=labels)
        weighted_dist = (
            self.df.groupby(income_dist, observed=True)["PWGTP"]
            .sum()
            .div(self.df["PWGTP"].sum())
            .to_dict()
        )

        return self._validate_distribution(
            pd.Series(weighted_dist),
            INCOME_DISTRIBUTION["income_brackets"],
            tolerance=VALIDATION_THRESHOLDS["distribution"]["tolerance"],
        )

    def _validate_correlations(self) -> Dict[str, Any]:
        """Validate correlations with income."""
        results = {}

        # Adjust income for inflation
        adjusted_income = self.df["PINCP"] * (self.df["ADJINC"] / 1_000_000)

        # Income-Age correlation
        age_corr = adjusted_income.corr(self.df["AGEP"])
        results["income_age"] = {
            "actual_value": age_corr,
            "expected_value": 0.35,
            "within_tolerance": abs(age_corr - 0.35) < 0.1,
        }

        # Income-Education correlation
        edu_corr = adjusted_income.corr(self.df["SCHL"])
        results["income_education"] = {
            "actual_value": edu_corr,
            "expected_value": 0.42,
            "within_tolerance": abs(edu_corr - 0.42) < 0.1,
        }

        return results
