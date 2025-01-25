"""Preference patterns validator"""

from typing import Dict, Any
from validation.consumer.base_validator import BaseValidator
from validation.consumer.consumer_benchmarks import (
    PRODUCT_PREFERENCES,
    DEVICE_PATTERNS,
    RESEARCH_SOURCES,
)
from constants import VALIDATION_THRESHOLDS
from utils.logging import log_validation_step, log_metric_validation


class PreferenceValidator(BaseValidator):
    """Validates consumer product and channel preferences."""

    def _perform_validation(self) -> Dict[str, Any]:
        """Validate preference patterns against benchmarks from research."""
        log_validation_step(self.logger, "Starting preference pattern validation")

        results = {}

        # Product Category Preferences (Deloitte, 2023)
        log_validation_step(self.logger, "Product Category Preferences validation")
        results["product_preferences"] = self._validate_product_preferences()

        # Device-Specific Patterns (DataReportal, 2023)
        log_validation_step(self.logger, "Device Usage Patterns validation")
        results["device_usage"] = self._validate_device_usage()

        return results

    def _validate_product_preferences(self) -> Dict[str, Dict]:
        """Validate product preferences for each age group."""
        results = {}

        for age_group, prefs in PRODUCT_PREFERENCES.items():
            age_group_data = self.preferences[
                self.preferences["age_group"] == age_group
            ]
            if len(age_group_data) > 0:
                results[age_group] = {}
                for product, expected_rate in prefs.items():
                    col_name = f"{product}_preference"
                    if col_name in age_group_data.columns:
                        actual_rate = age_group_data[col_name].mean()

                        metric_result = self._validate_metric(
                            actual_rate, expected_rate, "product_preference"
                        )

                        results[age_group][product] = metric_result

                        # Log individual preference validation
                        log_metric_validation(
                            self.logger,
                            f"{age_group} - {product}",
                            expected_rate,
                            actual_rate,
                            metric_result["difference"],
                            metric_result["within_tolerance"],
                            {
                                "age_group": age_group,
                                "product_type": product,
                                "source": RESEARCH_SOURCES.get("product_preferences"),
                            },
                        )

        return results

    def _validate_device_usage(self) -> Dict[str, Dict]:
        """Validate device usage patterns."""
        results = {}

        for device, expected in DEVICE_PATTERNS.items():
            if f"{device}_usage" in self.preferences.columns:
                actual = self.preferences[f"{device}_usage"].mean()

                metric_result = self._validate_metric(actual, expected, "device_usage")

                results[device] = metric_result

                # Log device usage validation
                log_metric_validation(
                    self.logger,
                    f"Device - {device}",
                    expected,
                    actual,
                    metric_result["difference"],
                    metric_result["within_tolerance"],
                    {
                        "device_type": device,
                        "source": RESEARCH_SOURCES.get("device_usage"),
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
