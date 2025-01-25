"""Engagement data validator"""

import logging
import pandas as pd
import numpy as np
from typing import Dict
from validation.marketing.base_validator import BaseValidator
from validation.marketing.marketing_benchmarks import (
    ENGAGEMENT_METRICS,
    VALIDATION_THRESHOLDS,
    DIGITAL_ADOPTION,
    PRODUCT_PREFERENCES,
)


class EngagementValidator(BaseValidator):
    def __init__(self, engagement: pd.DataFrame):
        """Initialize with only engagement data"""
        self.engagement = engagement
        self.logger = logging.getLogger(self.__class__.__name__)

        # Add validation of required columns
        required_columns = ["engagement_rate", "conversion_rate", "clicks"]
        missing_columns = [
            col for col in required_columns if col not in engagement.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. Available columns: {engagement.columns.tolist()}"
            )

    def validate(self) -> Dict[str, Dict[str, bool]]:
        """Validate engagement metrics against benchmarks"""
        self.logger.info("\nValidating Engagement Data")
        self.logger.info("=" * 50)

        results = {}

        # Overall engagement metrics
        self.logger.info("\nOverall Engagement Metrics:")
        overall_results = {}

        # Engagement rate validation - use Instagram as baseline
        avg_engagement = self.engagement["engagement_rate"].mean()
        overall_results["engagement_rate"] = self._validate_metric(
            avg_engagement,
            ENGAGEMENT_METRICS["social_media"][
                "instagram"
            ],  # Use Instagram as baseline
            "engagement",
        )

        # Click-through rate validation
        click_rate = (self.engagement["clicks"] / self.engagement["impressions"]).mean()
        overall_results["click_rate"] = self._validate_metric(
            click_rate,
            ENGAGEMENT_METRICS["email"][
                "click_rate"
            ],  # Use email click rate as baseline
            "engagement",
        )

        # Conversion rate validation (if available)
        if "conversion_rate" in self.engagement.columns:
            avg_conversion = self.engagement["conversion_rate"].mean()
            overall_results["conversion_rate"] = self._validate_metric(
                avg_conversion,
                0.01,  # 1% industry standard conversion rate
                "engagement",
            )

        results["overall"] = overall_results
        self.log_validation_results(overall_results, "Overall Metrics")

        # Time-based analysis
        time_results = self._analyze_time_patterns()
        results["temporal"] = time_results
        self.log_validation_results(time_results, "Temporal Patterns")

        # Log summary statistics
        self.logger.info("\nEngagement Summary Statistics:")
        self.logger.info(f"Total Engagement Records: {len(self.engagement):,}")
        self.logger.info(
            f"Average Engagement Rate: {self.engagement['engagement_rate'].mean():.2%}"
        )
        self.logger.info(
            f"Average Conversion Rate: {self.engagement['conversion_rate'].mean():.2%}"
        )
        self.logger.info(f"Total Clicks: {self.engagement['clicks'].sum():,}")

        # Digital adoption validation
        digital_adoption_results = self._validate_digital_adoption()
        results["digital_adoption"] = digital_adoption_results
        self.log_validation_results(digital_adoption_results, "Digital Adoption")

        # Product preferences validation
        product_preferences_results = self._validate_product_preferences()
        results["product_preferences"] = product_preferences_results
        self.log_validation_results(product_preferences_results, "Product Preferences")

        return results

    def _analyze_time_patterns(self) -> Dict:
        """Analyze temporal patterns in engagement metrics"""
        results = {}

        # Day of week analysis
        if "day_of_week" in self.engagement.columns:
            dow_engagement = self.engagement.groupby("day_of_week")[
                "engagement_rate"
            ].mean()
            expected_dow_pattern = ENGAGEMENT_METRICS["temporal"]["day_multipliers"]

            for day, actual_rate in dow_engagement.items():
                expected_rate = (
                    expected_dow_pattern.get(day, 1.0)
                    * ENGAGEMENT_METRICS["social_media"]["instagram"]
                )
                results[f"day_{day}"] = self._validate_metric(
                    actual_rate, expected_rate, "engagement"
                )

        # Campaign progress analysis
        if "campaign_progress" in self.engagement.columns:
            # Define expected_rates before try block so it's available in except block
            base_rate = ENGAGEMENT_METRICS["social_media"]["instagram"]
            expected_rates = {
                "early": base_rate * 1.2,  # 20% higher at start
                "middle": base_rate,  # baseline in middle
                "late": base_rate * 0.8,  # 20% lower at end
            }

            try:
                # Handle potential duplicate bin edges
                progress_bins = pd.qcut(
                    self.engagement["campaign_progress"],
                    q=3,
                    labels=["early", "middle", "late"],
                    duplicates="drop",
                )

                progress_engagement = self.engagement.groupby(progress_bins)[
                    "engagement_rate"
                ].mean()

                for stage in ["early", "middle", "late"]:
                    if stage in progress_engagement:
                        results[f"progress_{stage}"] = self._validate_metric(
                            progress_engagement[stage],
                            expected_rates[stage],
                            "engagement",
                        )
                    else:
                        self.logger.warning(
                            f"No data for campaign progress stage: {stage}"
                        )

            except ValueError as e:
                self.logger.warning(f"Could not analyze campaign progress: {str(e)}")
                # Fallback to simple tercile analysis
                progress_values = self.engagement["campaign_progress"].values
                terciles = np.percentile(progress_values, [33.33, 66.67])

                for stage, (start, end) in zip(
                    ["early", "middle", "late"],
                    [(0, terciles[0]), (terciles[0], terciles[1]), (terciles[1], 1)],
                ):
                    stage_data = self.engagement[
                        (self.engagement["campaign_progress"] >= start)
                        & (self.engagement["campaign_progress"] <= end)
                    ]
                    if len(stage_data) > 0:
                        results[f"progress_{stage}"] = self._validate_metric(
                            stage_data["engagement_rate"].mean(),
                            expected_rates[stage],
                            "engagement",
                        )

        return results

    def _validate_device_patterns(self) -> Dict:
        """Validate device-specific engagement patterns"""
        results = {}

        if "device" not in self.engagement.columns:
            self.logger.warning("Device column not found - skipping device validation")
            return results

        device_dist = self.engagement.groupby("device")["engagement_rate"].mean()
        expected_dist = ENGAGEMENT_METRICS["device"]

        results = self._validate_distribution(
            device_dist,
            expected_dist,
            tolerance=VALIDATION_THRESHOLDS["distribution"]["tolerance"],
        )

        return results

    def _validate_temporal_patterns(self) -> Dict:
        """Validate generation-specific temporal patterns"""
        results = {}

        if not all(col in self.engagement.columns for col in ["hour", "generation"]):
            self.logger.warning(
                "Missing hour or generation columns - skipping temporal validation"
            )
            return results

        for gen, peak_hours in ENGAGEMENT_METRICS["temporal"].items():
            gen_data = self.engagement[self.engagement["generation"] == gen]
            if len(gen_data) == 0:
                continue

            peak_engagement = gen_data[gen_data["hour"].isin(peak_hours)][
                "engagement_rate"
            ].mean()
            off_peak_engagement = gen_data[~gen_data["hour"].isin(peak_hours)][
                "engagement_rate"
            ].mean()

            # Peak hours should have higher engagement
            results[f"{gen}_peak_ratio"] = self._validate_metric(
                peak_engagement / off_peak_engagement,
                1.5,  # Expected 50% higher engagement during peak hours
                "temporal",
            )

        return results

    def _validate_digital_adoption(self) -> Dict:
        """Validate digital adoption patterns by age group"""
        results = {}

        if "age_group" not in self.engagement.columns:
            self.logger.warning(
                "Age group column not found - skipping digital adoption validation"
            )
            return results

        for age_group, expected_rate in DIGITAL_ADOPTION["age_groups"].items():
            group_data = self.engagement[self.engagement["age_group"] == age_group]
            if len(group_data) == 0:
                continue

            actual_rate = len(
                group_data[
                    group_data["channel"].str.contains(
                        "digital|online|mobile", case=False
                    )
                ]
            ) / len(group_data)

            results[f"{age_group}_adoption"] = self._validate_metric(
                actual_rate, expected_rate, "adoption"
            )

        return results

    def _validate_product_preferences(self) -> Dict:
        """Validate product preferences by age group"""
        results = {}

        required_cols = ["age_group", "product_category"]
        if not all(col in self.engagement.columns for col in required_cols):
            self.logger.warning(
                "Missing required columns for product preference validation"
            )
            return results

        for age_group, preferences in PRODUCT_PREFERENCES.items():
            group_data = self.engagement[self.engagement["age_group"] == age_group]
            if len(group_data) == 0:
                continue

            category_dist = group_data["product_category"].value_counts(normalize=True)

            results[age_group] = self._validate_distribution(
                category_dist,
                preferences,
                tolerance=VALIDATION_THRESHOLDS["distribution"]["tolerance"],
            )

        return results
