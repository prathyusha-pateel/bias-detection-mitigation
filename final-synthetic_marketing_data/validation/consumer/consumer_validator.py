"""Consumer preference patterns validator"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
from validation.consumer.base_validator import BaseValidator
from validation.consumer.consumer_benchmarks import (
    SOCIAL_MEDIA_ADOPTION,
    PLATFORM_ENGAGEMENT,
    DEVICE_PATTERNS,
    PRODUCT_PREFERENCES,
    ONLINE_ADOPTION,
    LOYALTY_METRICS,
    INCOME_CORRELATIONS,
    BRAND_AFFINITY,
    RESEARCH_SOURCES,
)
from constants import VALIDATION_THRESHOLDS
from utils.logging import log_validation_step, log_metric_validation


class PreferenceValidator(BaseValidator):
    """Validates consumer product and channel preferences."""

    def _perform_validation(self) -> Dict[str, Any]:
        """Validate preference patterns against benchmarks."""
        log_validation_step(self.logger, "Starting preference pattern validation")

        results = {
            "age_distribution": self._validate_age_distribution(),
            "online_adoption": self._validate_online_adoption(),
            "loyalty_metrics": self._validate_loyalty_metrics(),
            "product_preferences": self._validate_product_preferences(),
            "social_media": self._validate_social_media(),
            "device_usage": self._validate_device_usage(),
            "income_correlations": self._validate_income_correlations(),
        }

        return results

    def _validate_age_distribution(self) -> Dict[str, Any]:
        """Validate age group distribution."""
        results = {}
        age_dist = self.preferences["age_group"].value_counts(normalize=True)

        target_dist = {"18-34": 0.33, "35-54": 0.33, "55+": 0.34}

        for age_group, target in target_dist.items():
            actual = age_dist.get(age_group, 0)
            difference = abs(actual - target)
            within_tolerance = (
                difference <= VALIDATION_THRESHOLDS["distribution"]["tolerance"]
            )

            results[age_group] = {
                "expected_value": target,
                "actual_value": actual,
                "difference": difference,
                "within_tolerance": within_tolerance,
                "source": "Demographics Benchmark 2023",
            }

        return results

    def _validate_online_adoption(self) -> Dict[str, Any]:
        """Validate online shopping adoption rates."""
        results = {}

        for age_group, expected_rate in ONLINE_ADOPTION.items():
            mask = self.preferences["age_group"] == age_group
            if not mask.any():
                continue

            actual_rate = self.preferences.loc[mask, "online_shopping_rate"].mean()
            difference = abs(actual_rate - expected_rate)
            within_tolerance = (
                difference <= VALIDATION_THRESHOLDS["online_adoption"]["tolerance"]
            )

            results[age_group] = {
                "expected_value": expected_rate,
                "actual_value": actual_rate,
                "difference": difference,
                "within_tolerance": within_tolerance,
                "source": RESEARCH_SOURCES["online_adoption"],
            }

        return results

    def _validate_loyalty_metrics(self) -> Dict[str, Any]:
        """Validate loyalty program metrics."""
        results = {}

        # Average memberships
        avg_memberships = self.preferences["loyalty_memberships"].mean()
        expected_memberships = LOYALTY_METRICS["average_memberships"]
        difference = abs(avg_memberships - expected_memberships)
        within_tolerance = difference <= VALIDATION_THRESHOLDS["loyalty"]["tolerance"]

        results["average_memberships"] = {
            "expected_value": expected_memberships,
            "actual_value": avg_memberships,
            "difference": difference,
            "within_tolerance": within_tolerance,
            "source": RESEARCH_SOURCES["loyalty"],
        }

        # Redemption rates by age group
        for age_group, expected_rate in LOYALTY_METRICS["redemption_rates"].items():
            mask = self.preferences["age_group"] == age_group
            if not mask.any():
                continue

            actual_rate = self.preferences.loc[mask, "redemption_rate"].mean()
            difference = abs(actual_rate - expected_rate)
            within_tolerance = (
                difference <= VALIDATION_THRESHOLDS["loyalty"]["tolerance"]
            )

            results[f"redemption_rate_{age_group}"] = {
                "expected_value": expected_rate,
                "actual_value": actual_rate,
                "difference": difference,
                "within_tolerance": within_tolerance,
                "source": RESEARCH_SOURCES["loyalty"],
            }

        return results

    def _validate_product_preferences(self) -> Dict[str, Any]:
        """Validate product preferences by age group."""
        results = {}

        for age_group, prefs in PRODUCT_PREFERENCES.items():
            mask = self.preferences["age_group"] == age_group
            if not mask.any():
                continue

            for product, expected_rate in prefs.items():
                col_name = f"{product}_preference"
                if col_name not in self.preferences.columns:
                    continue

                actual_rate = self.preferences.loc[mask, col_name].mean()
                difference = abs(actual_rate - expected_rate)
                within_tolerance = (
                    difference <= VALIDATION_THRESHOLDS["preference"]["tolerance"]
                )

                results[f"{age_group}_{product}"] = {
                    "expected_value": expected_rate,
                    "actual_value": actual_rate,
                    "difference": difference,
                    "within_tolerance": within_tolerance,
                    "source": RESEARCH_SOURCES["product_preferences"],
                }

        return results

    def _validate_social_media(self) -> Dict[str, Any]:
        """Validate social media adoption and engagement."""
        results = {}

        # Adoption rates
        for age_group, expected_rate in SOCIAL_MEDIA_ADOPTION.items():
            mask = self.preferences["age_group"] == age_group
            if not mask.any():
                continue

            actual_rate = self.preferences.loc[
                mask, "social_media_engagement_rate"
            ].mean()
            difference = abs(actual_rate - expected_rate)
            within_tolerance = (
                difference <= VALIDATION_THRESHOLDS["engagement"]["tolerance"]
            )

            results[f"adoption_{age_group}"] = {
                "expected_value": expected_rate,
                "actual_value": actual_rate,
                "difference": difference,
                "within_tolerance": within_tolerance,
                "source": RESEARCH_SOURCES["social_media"],
            }

        # Platform engagement
        for platform, expected_rate in PLATFORM_ENGAGEMENT.items():
            col_name = f"{platform}_engagement"
            if col_name not in self.preferences.columns:
                continue

            actual_rate = self.preferences[col_name].mean()
            difference = abs(actual_rate - expected_rate)
            within_tolerance = (
                difference <= VALIDATION_THRESHOLDS["engagement"]["tolerance"]
            )

            results[f"engagement_{platform}"] = {
                "expected_value": expected_rate,
                "actual_value": actual_rate,
                "difference": difference,
                "within_tolerance": within_tolerance,
                "source": RESEARCH_SOURCES["platform_engagement"],
            }

        return results

    def _validate_device_usage(self) -> Dict[str, Any]:
        """Validate device usage patterns."""
        results = {}

        for device, expected_rate in DEVICE_PATTERNS.items():
            col_name = f"{device}_usage"
            if col_name not in self.preferences.columns:
                continue

            actual_rate = self.preferences[col_name].mean()
            difference = abs(actual_rate - expected_rate)
            within_tolerance = (
                difference <= VALIDATION_THRESHOLDS["preference"]["tolerance"]
            )

            results[device] = {
                "expected_value": expected_rate,
                "actual_value": actual_rate,
                "difference": difference,
                "within_tolerance": within_tolerance,
                "source": RESEARCH_SOURCES["device_usage"],
            }

        return results

    def _validate_income_correlations(self) -> Dict[str, Any]:
        """Validate income-based correlations."""
        results = {}

        if self.demographic_data is None:
            return results

        income = pd.to_numeric(self.demographic_data["PINCP"], errors="coerce")

        for metric, expected_corr in INCOME_CORRELATIONS.items():
            col_name = f"{metric}_score"
            if col_name not in self.preferences.columns:
                continue

            actual_corr = self.preferences[col_name].corr(income)
            difference = abs(actual_corr - expected_corr)
            within_tolerance = (
                difference <= VALIDATION_THRESHOLDS["correlation"]["tolerance"]
            )

            results[metric] = {
                "expected_value": expected_corr,
                "actual_value": actual_corr,
                "difference": difference,
                "within_tolerance": within_tolerance,
                "source": RESEARCH_SOURCES["income_correlations"],
            }

        return results

    def log_validation_summary(self, results: Dict[str, Any]) -> None:
        """Log comprehensive validation summary."""
        total_checks = 0
        passing_checks = 0

        for category, metrics in results.items():
            if isinstance(metrics, dict):
                category_checks = len(
                    [m for m in metrics.values() if isinstance(m, dict)]
                )
                category_passing = len(
                    [
                        m
                        for m in metrics.values()
                        if isinstance(m, dict) and m.get("within_tolerance", False)
                    ]
                )

                total_checks += category_checks
                passing_checks += category_passing

                if category_checks > 0:
                    pass_rate = (category_passing / category_checks) * 100
                    self.logger.info(
                        f"{category}: {category_passing}/{category_checks} checks passed ({pass_rate:.1f}%)"
                    )

        if total_checks > 0:
            overall_rate = (passing_checks / total_checks) * 100
            self.logger.info(
                f"\nOverall: {passing_checks}/{total_checks} checks passed ({overall_rate:.1f}%)"
            )
