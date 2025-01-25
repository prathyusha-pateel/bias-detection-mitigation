"""Social media engagement validator"""

from typing import Dict, Any
from validation.consumer.base_validator import BaseValidator
from validation.consumer.consumer_benchmarks import (
    SOCIAL_MEDIA_ADOPTION,
    PLATFORM_ENGAGEMENT,
    DEVICE_PATTERNS,
    RESEARCH_SOURCES,
)
from constants import VALIDATION_THRESHOLDS
from utils.logging import log_validation_step, log_metric_validation


class SocialMediaValidator(BaseValidator):
    """Validates social media engagement patterns."""

    def _perform_validation(self) -> Dict[str, Any]:
        """Validate social media engagement patterns against benchmarks."""
        log_validation_step(self.logger, "Starting social media engagement validation")

        results = {}

        # Age-based Adoption Rates
        log_validation_step(self.logger, "Age-based Adoption Rates validation")
        results["age_adoption"] = self._validate_age_adoption()

        # Platform-Specific Engagement
        log_validation_step(self.logger, "Platform-Specific Engagement validation")
        results["platform_engagement"] = self._validate_platform_engagement()

        # Device-Based Social Media Usage
        log_validation_step(self.logger, "Device-Based Social Media Usage validation")
        results["social_device_usage"] = self._validate_device_usage()

        return results

    def _validate_age_adoption(self) -> Dict[str, Dict]:
        """Validate social media adoption rates by age group."""
        results = {}

        for age_group, expected_rate in SOCIAL_MEDIA_ADOPTION.items():
            age_group_data = self.preferences[
                self.preferences["age_group"] == age_group
            ]
            if len(age_group_data) > 0:
                actual_rate = age_group_data["social_media_engagement_rate"].mean()

                metric_result = self._validate_metric(
                    actual_rate, expected_rate, "adoption"
                )

                results[age_group] = metric_result

                # Log age-based adoption validation
                log_metric_validation(
                    self.logger,
                    f"Social Media Adoption - {age_group}",
                    expected_rate,
                    actual_rate,
                    metric_result["difference"],
                    metric_result["within_tolerance"],
                    {
                        "age_group": age_group,
                        "source": RESEARCH_SOURCES.get("social_media"),
                    },
                )

        return results

    def _validate_platform_engagement(self) -> Dict[str, Dict]:
        """Validate platform-specific engagement rates."""
        results = {}

        for platform, expected in PLATFORM_ENGAGEMENT.items():
            if f"{platform}_engagement" in self.preferences.columns:
                actual = self.preferences[f"{platform}_engagement"].mean()

                metric_result = self._validate_metric(actual, expected, "platform")

                results[platform] = metric_result

                # Log platform engagement validation
                log_metric_validation(
                    self.logger,
                    f"Platform Engagement - {platform}",
                    expected,
                    actual,
                    metric_result["difference"],
                    metric_result["within_tolerance"],
                    {
                        "platform": platform,
                        "source": RESEARCH_SOURCES.get("platform_engagement"),
                    },
                )

        return results

    def _validate_device_usage(self) -> Dict[str, Dict]:
        """Validate device-based social media usage."""
        results = {}

        for device, expected in DEVICE_PATTERNS.items():
            if f"social_{device}_usage" in self.preferences.columns:
                actual = self.preferences[f"social_{device}_usage"].mean()

                metric_result = self._validate_metric(actual, expected, "device")

                results[device] = metric_result

                # Log device usage validation
                log_metric_validation(
                    self.logger,
                    f"Social Media Device Usage - {device}",
                    expected,
                    actual,
                    metric_result["difference"],
                    metric_result["within_tolerance"],
                    {"device": device, "source": RESEARCH_SOURCES.get("device_usage")},
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
