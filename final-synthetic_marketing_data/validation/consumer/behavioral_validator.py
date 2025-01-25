"""Behavioral patterns validator"""

import pandas as pd
from typing import Dict, Any
import logging
from validation.consumer.base_validator import BaseValidator
from validation.consumer.consumer_benchmarks import (
    ONLINE_ADOPTION,
    LOYALTY_METRICS,
    INCOME_CORRELATIONS,
    RESEARCH_SOURCES,
    BRAND_AFFINITY,
)
from constants import VALIDATION_THRESHOLDS
from utils.logging import log_validation_step, log_metric_validation


class BehavioralValidator(BaseValidator):
    """Validates consumer behavioral patterns."""

    def _perform_validation(self) -> Dict[str, Any]:
        """Validate behavioral patterns against benchmarks from research."""
        log_validation_step(self.logger, "Starting behavioral pattern validation")

        results = {}

        # Online Shopping Adoption
        log_validation_step(self.logger, "Online Shopping Adoption validation")
        results["online_adoption"] = self._validate_online_adoption()

        # Brand Affinity Distribution
        log_validation_step(self.logger, "Brand Affinity validation")
        results["brand_affinity"] = self._validate_brand_affinity()

        # Income-Based Correlations
        log_validation_step(self.logger, "Income Correlations validation")
        results["income_correlations"] = self._validate_income_correlations()

        return results

    def _validate_online_adoption(self) -> Dict[str, Dict]:
        """Validate online shopping adoption rates by age group."""
        results = {}

        for age_group, expected in ONLINE_ADOPTION.items():
            age_group_data = self.preferences[
                self.preferences["age_group"] == age_group
            ]
            if len(age_group_data) > 0:
                # Using monthly_purchases > 0 as proxy for digital adoption
                actual = (age_group_data["monthly_purchases"] > 0).mean()

                metric_result = self._validate_metric(actual, expected, "adoption")

                results[age_group] = metric_result

                # Log online adoption validation
                log_metric_validation(
                    self.logger,
                    f"Online Adoption - {age_group}",
                    expected,
                    actual,
                    metric_result["difference"],
                    metric_result["within_tolerance"],
                    {
                        "age_group": age_group,
                        "source": RESEARCH_SOURCES.get("online_adoption"),
                    },
                )

        return results

    def _validate_brand_affinity(self) -> Dict[str, Dict]:
        """Validate brand affinity distribution."""
        results = {}

        actual_affinity_dist = self.preferences["brand_affinity"].value_counts(
            normalize=True
        )

        for affinity, expected in BRAND_AFFINITY.items():
            actual = actual_affinity_dist.get(affinity, 0.0)

            metric_result = self._validate_metric(actual, expected, "affinity")

            results[affinity] = metric_result

            # Log brand affinity validation
            log_metric_validation(
                self.logger,
                f"Brand Affinity - {affinity}",
                expected,
                actual,
                metric_result["difference"],
                metric_result["within_tolerance"],
                {"affinity_type": affinity, "source": RESEARCH_SOURCES.get("loyalty")},
            )

        return results

    def _validate_income_correlations(self) -> Dict[str, Dict]:
        """Validate income-based correlations."""
        results = {}

        # Map our columns to the correlation metrics
        correlation_mapping = {
            "multi_channel": "social_media_preference",
            "digital_channel": "email_preference",
            "digital_service": "newsletter_preference",
            "mobile_app": "social_media_interaction_preference",
        }

        for metric, column in correlation_mapping.items():
            if column in self.preferences.columns and self.demographic_data is not None:
                actual = self.preferences[column].corr(self.demographic_data["PINCP"])

                metric_result = self._validate_metric(
                    actual, INCOME_CORRELATIONS[metric], "correlation"
                )

                results[metric] = metric_result

                # Log income correlation validation
                log_metric_validation(
                    self.logger,
                    f"Income Correlation - {metric}",
                    INCOME_CORRELATIONS[metric],
                    actual,
                    metric_result["difference"],
                    metric_result["within_tolerance"],
                    {
                        "correlation_type": metric,
                        "source": RESEARCH_SOURCES.get("income_correlations"),
                    },
                )

        return results

    def _validate_metric(
        self, actual: float, expected: float, metric_type: str
    ) -> Dict[str, Any]:
        """Validate metric with appropriate tolerance."""
        difference = abs(actual - expected)
        tolerance = VALIDATION_THRESHOLDS.get(metric_type, 0.05)
        within_tolerance = difference <= tolerance

        return {
            "expected_value": float(expected),
            "actual_value": float(actual),
            "difference": float(difference),
            "within_tolerance": within_tolerance,
            "metric_type": metric_type,
            "tolerance": tolerance,
            "source": RESEARCH_SOURCES.get(metric_type, "Unknown"),
        }
