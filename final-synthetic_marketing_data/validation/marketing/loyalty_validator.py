import logging
from validation.marketing.marketing_benchmarks import (
    LOYALTY_METRICS,
    VALIDATION_THRESHOLDS,
)
import pandas as pd
from typing import Dict
from validation.marketing.base_validator import BaseValidator

"""Loyalty data validator"""


class LoyaltyValidator(BaseValidator):
    def __init__(self, loyalty: pd.DataFrame):
        """Initialize with only loyalty data"""
        self.loyalty = loyalty
        self.logger = logging.getLogger(self.__class__.__name__)

        # Add validation of required columns
        required_columns = ["status", "points_balance"]
        missing_columns = [
            col for col in required_columns if col not in loyalty.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. Available columns: {loyalty.columns.tolist()}"
            )

    def validate(self) -> Dict:
        """Validate loyalty metrics against benchmarks"""
        self.logger.info("\nValidating Loyalty Data")
        self.logger.info("=" * 50)

        results = {}

        # Program enrollment
        self.logger.info("\nProgram Enrollment Distribution:")
        status_dist = self.loyalty["status"].value_counts(normalize=True)

        # Use _validate_distribution with a default tolerance if not specified
        results["enrollment"] = self._validate_distribution(
            status_dist,
            LOYALTY_METRICS["program_enrollment"],
            tolerance=VALIDATION_THRESHOLDS.get("loyalty", {}).get(
                "tolerance", 0.05
            ),  # Default 5% tolerance
        )
        self.log_validation_results(results["enrollment"], "Enrollment")

        # Points-based tier validation
        if "points_balance" in self.loyalty.columns:
            tier_results = {}
            for tier, threshold in LOYALTY_METRICS["points"]["tiers"].items():
                actual_rate = (self.loyalty["points_balance"] >= threshold).mean()
                expected_rate = (
                    1.0 if tier == "bronze" else (0.4 if tier == "silver" else 0.1)
                )  # Example tier distribution
                tier_results[tier] = self._validate_metric(
                    actual_rate, expected_rate, "loyalty"
                )
            results["tiers"] = tier_results
            self.log_validation_results(tier_results, "Tier Distribution")

        # Log summary statistics
        self._log_summary_statistics()

        return results

    def _analyze_points_distribution(self) -> Dict:
        """Analyze points balance distribution"""
        results = {}

        try:
            # Use custom bins to avoid duplicate edges
            points_bins = [
                0,
                LOYALTY_METRICS["points"]["tiers"]["bronze"],
                LOYALTY_METRICS["points"]["tiers"]["silver"],
                LOYALTY_METRICS["points"]["tiers"]["gold"],
                float("inf"),
            ]

            # Ensure bins are unique and sorted
            points_bins = sorted(list(set(points_bins)))

            labels = ["bronze", "silver", "gold", "platinum"][: len(points_bins) - 1]
            points_categories = pd.cut(
                self.loyalty["points_balance"],
                bins=points_bins,
                labels=labels,
                include_lowest=True,
            )

            tier_dist = points_categories.value_counts(normalize=True)

            # Validate against expected distributions
            for tier, actual_rate in tier_dist.items():
                expected_rate = {
                    "bronze": 0.50,  # 50% in bronze tier
                    "silver": 0.30,  # 30% in silver tier
                    "gold": 0.15,  # 15% in gold tier
                    "platinum": 0.05,  # 5% in platinum tier
                }.get(tier, 0.0)

                results[tier] = self._validate_metric(
                    actual_rate, expected_rate, "loyalty"
                )

        except ValueError as e:
            self.logger.warning(f"Could not analyze points distribution: {str(e)}")
            results["error"] = str(e)

        return results

    def _log_summary_statistics(self):
        """Log summary statistics for loyalty data"""
        self.logger.info("\nLoyalty Summary Statistics:")
        self.logger.info(f"Total Members: {len(self.loyalty):,}")
        self.logger.info(
            f"Average Points Balance: {self.loyalty['points_balance'].mean():,.0f}"
        )
        self.logger.info(
            f"Median Points Balance: {self.loyalty['points_balance'].median():,.0f}"
        )

        # Log status distribution
        status_counts = self.loyalty["status"].value_counts()
        for status, count in status_counts.items():
            self.logger.info(f"{status.title()} Members: {count:,}")
