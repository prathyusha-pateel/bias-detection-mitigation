"""Transaction product patterns validator"""

import pandas as pd
from typing import Dict, Any
import logging
from validation.transaction.base_validator import BaseTransactionValidator
from validation.transaction.transaction_benchmarks import (
    CATEGORY_BENCHMARKS,
    REGIONAL_BENCHMARKS,
    RESEARCH_SOURCES,
    VALIDATION_THRESHOLDS,
)
from utils.logging import log_validation_step, log_metric_validation


class ProductValidator(BaseTransactionValidator):
    """Validates transaction product patterns."""

    def _perform_validation(self) -> Dict[str, Any]:
        """Validate product category patterns."""
        if self.detail_data is None:
            self.logger.warning(
                "No transaction detail data provided for product validation"
            )
            return {"error": "Missing transaction detail data"}

        log_validation_step(self.logger, "Starting product pattern validation")

        results = {
            "category_distribution": self._validate_category_distribution(),
            "price_patterns": self._validate_price_patterns(),
            "regional_preferences": self._validate_regional_preferences(),
        }

        return results

    def _validate_category_distribution(self) -> Dict[str, Any]:
        """Validate product category distribution."""
        results = {}
        actual_dist = self.detail_data["category"].value_counts(normalize=True)
        expected_dist = CATEGORY_BENCHMARKS["distribution"]

        for category, expected_share in expected_dist.items():
            actual_share = actual_dist.get(category, 0.0)
            results[category] = {
                "expected": expected_share,
                "actual": actual_share,
                "difference": abs(actual_share - expected_share),
                "within_tolerance": abs(actual_share - expected_share) <= 0.05,
                "source": RESEARCH_SOURCES["category_benchmarks"],
            }

        return results

    def _validate_price_patterns(self) -> Dict[str, Any]:
        """Validate product price patterns."""
        results = {}
        expected_prices = CATEGORY_BENCHMARKS["average_price"]

        for category, expected_price in expected_prices.items():
            category_data = self.detail_data[self.detail_data["category"] == category]
            if len(category_data) > 0:
                actual_price = category_data["unit_price"].mean()
                results[category] = {
                    "expected": expected_price,
                    "actual": actual_price,
                    "difference": abs(actual_price - expected_price),
                    "within_tolerance": abs(actual_price - expected_price) <= 1.00,
                    "sample_size": len(category_data),
                }

        return results

    def _validate_regional_preferences(self) -> Dict[str, Any]:
        """Validate regional product preferences."""
        results = {}

        # Merge transaction and detail data to get region information
        if (
            self.transaction_data is not None
            and "region" in self.transaction_data.columns
        ):
            merged_data = pd.merge(
                self.detail_data,
                self.transaction_data[["transaction_id", "region"]],
                on="transaction_id",
            )

            for region, preferences in REGIONAL_BENCHMARKS.items():
                region_data = merged_data[merged_data["region"] == region]
                if len(region_data) > 0:
                    results[region] = {}
                    region_dist = region_data["category"].value_counts(normalize=True)

                    for category, expected_share in preferences.items():
                        actual_share = region_dist.get(category, 0.0)
                        results[region][category] = {
                            "expected": expected_share,
                            "actual": actual_share,
                            "difference": abs(actual_share - expected_share),
                            "within_tolerance": abs(actual_share - expected_share)
                            <= 0.05,
                            "sample_size": len(region_data),
                        }

        return results
