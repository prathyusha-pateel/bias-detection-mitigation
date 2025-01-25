"""Transaction value patterns validator"""

import pandas as pd
from typing import Dict, Any
import logging
from validation.transaction.base_validator import BaseTransactionValidator
from validation.transaction.transaction_benchmarks import (
    TRANSACTION_VALUE_BENCHMARKS,
    RESEARCH_SOURCES,
    VALIDATION_THRESHOLDS,
)
from utils.logging import log_validation_step, log_metric_validation


class TransactionValueValidator(BaseTransactionValidator):
    """Validates transaction value patterns."""

    def _perform_validation(self) -> Dict[str, Any]:
        """Validate transaction value patterns against benchmarks."""
        log_validation_step(self.logger, "Starting transaction value validation")

        results = {
            "overall_metrics": self._validate_overall_metrics(),
            "regional_patterns": self._validate_regional_patterns(),
            "channel_patterns": self._validate_channel_patterns(),
        }

        return results

    def _validate_overall_metrics(self) -> Dict[str, Any]:
        """Validate overall transaction metrics."""
        benchmarks = TRANSACTION_VALUE_BENCHMARKS["average_order_value"]
        actual_avg = self.transaction_data["transaction_value"].mean()
        tolerance = VALIDATION_THRESHOLDS["transaction"]["value_tolerance"]

        result = {
            "average_order_value": {
                "expected": benchmarks["overall"],
                "actual": actual_avg,
                "difference": abs(actual_avg - benchmarks["overall"]),
                "within_tolerance": abs(actual_avg - benchmarks["overall"])
                <= tolerance,
                "source": RESEARCH_SOURCES["transaction_values"],
            }
        }

        # Add basket size validation
        basket_benchmarks = TRANSACTION_VALUE_BENCHMARKS["basket_size"]
        actual_basket_mean = self.transaction_data["num_items"].mean()
        actual_basket_std = self.transaction_data["num_items"].std()

        result["basket_size"] = {
            "mean": {
                "expected": basket_benchmarks["mean"],
                "actual": actual_basket_mean,
                "difference": abs(actual_basket_mean - basket_benchmarks["mean"]),
                "within_tolerance": abs(actual_basket_mean - basket_benchmarks["mean"])
                <= VALIDATION_THRESHOLDS["transaction"]["basket_size_tolerance"],
            },
            "std_dev": {
                "expected": basket_benchmarks["std_dev"],
                "actual": actual_basket_std,
                "difference": abs(actual_basket_std - basket_benchmarks["std_dev"]),
                "within_tolerance": abs(
                    actual_basket_std - basket_benchmarks["std_dev"]
                )
                <= VALIDATION_THRESHOLDS["numerical"]["std_difference"],
            },
        }

        return result

    def _validate_regional_patterns(self) -> Dict[str, Any]:
        """Validate transaction patterns by region."""
        results = {}
        regional_benchmarks = TRANSACTION_VALUE_BENCHMARKS["average_order_value"][
            "by_region"
        ]

        for region, expected_value in regional_benchmarks.items():
            region_data = self.transaction_data[
                self.transaction_data["region"] == region
            ]
            if len(region_data) > 0:
                actual_avg = region_data["transaction_value"].mean()
                results[region] = {
                    "expected": expected_value,
                    "actual": actual_avg,
                    "difference": abs(actual_avg - expected_value),
                    "within_tolerance": abs(actual_avg - expected_value)
                    <= TRANSACTION_VALUE_BENCHMARKS["average_order_value"]["tolerance"],
                    "sample_size": len(region_data),
                }

        return results

    def _validate_channel_patterns(self) -> Dict[str, Any]:
        """Validate transaction patterns by channel."""
        results = {}
        channel_benchmarks = TRANSACTION_VALUE_BENCHMARKS["basket_size"]["by_channel"]

        for channel, expected_size in channel_benchmarks.items():
            channel_data = self.transaction_data[
                self.transaction_data["channel"] == channel
            ]
            if len(channel_data) > 0:
                actual_size = channel_data["num_items"].mean()
                results[channel] = {
                    "expected": expected_size,
                    "actual": actual_size,
                    "difference": abs(actual_size - expected_size),
                    "within_tolerance": abs(actual_size - expected_size) <= 0.5,
                    "sample_size": len(channel_data),
                }

        return results
