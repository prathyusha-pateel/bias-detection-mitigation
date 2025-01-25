"""Transaction channel patterns validator"""

import pandas as pd
from typing import Dict, Any
import logging
from validation.transaction.base_validator import BaseTransactionValidator
from validation.transaction.transaction_benchmarks import (
    CHANNEL_DISTRIBUTION,
    RESEARCH_SOURCES,
    VALIDATION_THRESHOLDS,
)
from utils.logging import log_validation_step, log_metric_validation


class ChannelValidator(BaseTransactionValidator):
    """Validates transaction channel patterns."""

    def _perform_validation(self) -> Dict[str, Any]:
        """Validate channel distribution patterns."""
        log_validation_step(self.logger, "Starting channel distribution validation")

        results = {
            "overall_distribution": self._validate_overall_distribution(),
            "age_group_distribution": self._validate_age_group_distribution(),
        }

        return results

    def _validate_overall_distribution(self) -> Dict[str, Any]:
        """Validate overall channel distribution."""
        results = {}
        actual_dist = self.transaction_data["channel"].value_counts(normalize=True)

        for channel, expected_share in {
            k: v for k, v in CHANNEL_DISTRIBUTION.items() if k not in ["by_age_group"]
        }.items():
            actual_share = actual_dist.get(channel, 0.0)
            results[channel] = {
                "expected": expected_share,
                "actual": actual_share,
                "difference": abs(actual_share - expected_share),
                "within_tolerance": abs(actual_share - expected_share) <= 0.05,
                "source": RESEARCH_SOURCES["channel_distribution"],
            }

        return results

    def _validate_age_group_distribution(self) -> Dict[str, Any]:
        """Validate channel distribution by age group."""
        results = {}
        age_benchmarks = CHANNEL_DISTRIBUTION["by_age_group"]

        for age_group, channel_dist in age_benchmarks.items():
            age_data = self.transaction_data[
                self.transaction_data["age_group"] == age_group
            ]
            if len(age_data) > 0:
                actual_dist = age_data["channel"].value_counts(normalize=True)
                results[age_group] = {}

                for channel, expected_share in channel_dist.items():
                    actual_share = actual_dist.get(channel, 0.0)
                    results[age_group][channel] = {
                        "expected": expected_share,
                        "actual": actual_share,
                        "difference": abs(actual_share - expected_share),
                        "within_tolerance": abs(actual_share - expected_share) <= 0.05,
                        "sample_size": len(age_data),
                    }

        return results
