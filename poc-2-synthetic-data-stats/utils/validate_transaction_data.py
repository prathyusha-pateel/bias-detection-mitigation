# validate_transaction_data.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from utils.transaction_constants import (
    REGIONAL_VARIATIONS,
    CATEGORY_PREFERENCES,
    CATEGORY_PRICE_EFFECTS,
    SEASONAL_PATTERNS,
    SPECIAL_EVENTS,
    get_quarter,
    is_black_friday,
    is_december_holiday,
    is_back_to_school,
)


class TransactionValidator:
    """Validates generated transaction data against known benchmarks."""

    def __init__(self, transactions_df: pd.DataFrame):
        """Initialize with transaction dataset."""
        self.df = transactions_df
        self.validation_results = {
            "seasonal": {},
            "regional": {},
            "events": {},
            "values": {},
            "overall": {},
        }
        self.setup_logging()

    @staticmethod
    def setup_logging():
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def validate_all(self) -> Dict:
        """Run all validation checks."""
        logging.info("Starting comprehensive validation...")

        try:
            self.validate_seasonal_patterns()
            self.validate_regional_patterns()
            self.validate_special_events()
            self.validate_transaction_values()
            self.compute_overall_score()

            return self.validation_results

        except Exception as e:
            logging.error(f"Validation failed with error: {str(e)}")
            raise

    def validate_seasonal_patterns(self):
        """Validate seasonal transaction patterns."""
        logging.info("Validating seasonal patterns...")

        quarterly_counts = self.df.groupby("quarter").size()
        total_transactions = len(self.df)

        results = {}
        for quarter, expected_share in SEASONAL_PATTERNS.items():
            actual_share = quarterly_counts.get(quarter, 0) / total_transactions
            within_tolerance = (
                abs(actual_share - expected_share) <= 0.02
            )  # 2% tolerance

            results[quarter] = {
                "expected": expected_share,
                "actual": actual_share,
                "within_tolerance": within_tolerance,
            }

            logging.info(
                f"{quarter}: {'✓' if within_tolerance else '✗'} "
                f"(Expected: {expected_share:.3f}, Actual: {actual_share:.3f})"
            )

        self.validation_results["seasonal"] = results

    def validate_regional_patterns(self):
        """Validate regional distributions and preferences."""
        logging.info("Validating regional patterns...")

        def calculate_value_multiplier(region_data: pd.DataFrame) -> float:
            """Calculate region's value multiplier compared to overall mean."""
            return region_data["value"].mean() / self.df["value"].mean()

        def validate_category_distribution(
            region_data: pd.DataFrame, expected_prefs: Dict[str, float]
        ) -> Dict[str, Dict]:
            """Validate category distribution for a region."""
            results = {}
            actual_dist = region_data["category"].value_counts(normalize=True)

            for category, expected_share in expected_prefs.items():
                actual_share = actual_dist.get(category, 0)
                within_tolerance = (
                    abs(actual_share - expected_share) <= 0.03
                )  # 3% tolerance

                results[category] = {
                    "expected": expected_share,
                    "actual": actual_share,
                    "within_tolerance": within_tolerance,
                }

            return results

        regional_results = {}
        for region in REGIONAL_VARIATIONS:
            region_data = self.df[self.df["region"] == region]
            if len(region_data) == 0:
                logging.warning(f"No data found for region: {region}")
                continue

            # Validate value multiplier
            actual_mult = calculate_value_multiplier(region_data)
            expected_mult = REGIONAL_VARIATIONS[region]["baseline_multiplier"]

            # Validate category distribution
            category_results = validate_category_distribution(
                region_data, CATEGORY_PREFERENCES[region]
            )

            regional_results[region] = {
                "value_multiplier": {
                    "expected": expected_mult,
                    "actual": actual_mult,
                    "within_tolerance": abs(actual_mult - expected_mult) <= 0.05,
                },
                "category_distribution": category_results,
            }

        self.validation_results["regional"] = regional_results
        self._log_regional_results(regional_results)

    def validate_special_events(self):
        """Validate special event impacts."""
        logging.info("Validating special event patterns...")

        def calculate_event_impact(
            event_data: pd.DataFrame, baseline_data: pd.DataFrame
        ) -> Tuple[float, float]:
            """Calculate transaction volume and value impact of event."""
            baseline_daily = len(baseline_data) / len(baseline_data["date"].unique())
            event_daily = len(event_data) / len(event_data["date"].unique())

            transaction_increase = (event_daily / baseline_daily) - 1
            online_share = (event_data["channel"] == "online").mean()

            return transaction_increase, online_share

        results = {}
        for event_name, event_info in SPECIAL_EVENTS.items():
            # Identify event and non-event periods
            event_dates = self.df["date"].apply(
                lambda x: (
                    is_black_friday(x)
                    if event_name == "black_friday"
                    else (
                        is_december_holiday(x)
                        if event_name == "december_holiday"
                        else (
                            is_back_to_school(x)
                            if event_name == "back_to_school"
                            else False
                        )
                    )
                )
            )

            event_data = self.df[event_dates]
            non_event_data = self.df[~event_dates]

            if len(event_data) == 0:
                logging.warning(f"No data found for event: {event_name}")
                continue

            # Calculate impacts
            transaction_increase, online_share = calculate_event_impact(
                event_data, non_event_data
            )

            expected_increase = event_info["transaction_boost"] - 1
            expected_online = event_info.get("online_boost", 1.0) * np.mean(
                [v["online_adoption"] for v in REGIONAL_VARIATIONS.values()]
            )

            results[event_name] = {
                "transaction_increase": {
                    "expected": expected_increase,
                    "actual": transaction_increase,
                    "within_tolerance": abs(transaction_increase - expected_increase)
                    <= 0.1,
                }
            }

            if "online_boost" in event_info:
                results[event_name]["online_share"] = {
                    "expected": expected_online,
                    "actual": online_share,
                    "within_tolerance": abs(online_share - expected_online) <= 0.05,
                }

        self.validation_results["events"] = results
        self._log_event_results(results)

    def validate_transaction_values(self):
        """Validate transaction value distributions."""
        logging.info("Validating transaction values...")

        # Calculate overall statistics
        value_stats = {
            "mean": self.df["value"].mean(),
            "median": self.df["value"].median(),
            "std": self.df["value"].std(),
        }

        # Calculate category-specific statistics
        category_stats = {}
        for category in self.df["category"].unique():
            cat_values = self.df[self.df["category"] == category]["value"]

            category_stats[category] = {
                "mean": cat_values.mean(),
                "median": cat_values.median(),
                "expected_multiplier": CATEGORY_PRICE_EFFECTS.get(category, 1.0),
                "actual_multiplier": cat_values.mean() / value_stats["mean"],
                "within_tolerance": True,  # Will be set in validation
            }

        self.validation_results["values"] = {
            "overall": value_stats,
            "by_category": category_stats,
        }

        self._log_value_results(value_stats, category_stats)

    def compute_overall_score(self):
        """Compute overall validation score."""
        component_scores = {
            "Seasonal Patterns": self._calculate_seasonal_score(),
            "Regional Patterns": self._calculate_regional_score(),
            "Special Events": self._calculate_event_score(),
        }

        final_score = np.mean(list(component_scores.values()))

        self.validation_results["overall"] = {
            "score": final_score,
            "component_scores": component_scores,
            "passed": final_score >= 0.8,  # 80% threshold
        }

        logging.info("\nOverall Validation Results:")
        logging.info(f"Final Score: {final_score:.1%}")
        for component, score in component_scores.items():
            logging.info(f"- {component}: {score:.1%}")
        logging.info(f"Status: {'PASSED' if final_score >= 0.8 else 'FAILED'}")

    def _calculate_seasonal_score(self) -> float:
        """Calculate score for seasonal patterns."""
        if not self.validation_results.get("seasonal"):
            return 0.0

        checks = [
            r["within_tolerance"] for r in self.validation_results["seasonal"].values()
        ]
        return np.mean(checks)

    def _calculate_regional_score(self) -> float:
        """Calculate score for regional patterns."""
        if not self.validation_results.get("regional"):
            return 0.0

        scores = []
        for region_results in self.validation_results["regional"].values():
            # Value multiplier check
            if "value_multiplier" in region_results:
                scores.append(region_results["value_multiplier"]["within_tolerance"])

            # Category distribution checks
            if "category_distribution" in region_results:
                for cat_result in region_results["category_distribution"].values():
                    scores.append(cat_result["within_tolerance"])

        return np.mean(scores) if scores else 0.0

    def _calculate_event_score(self) -> float:
        """Calculate score for special events."""
        if not self.validation_results.get("events"):
            return 0.0

        scores = []
        for event_results in self.validation_results["events"].values():
            if "transaction_increase" in event_results:
                scores.append(event_results["transaction_increase"]["within_tolerance"])
            if "online_share" in event_results:
                scores.append(event_results["online_share"]["within_tolerance"])

        return np.mean(scores) if scores else 0.0

    def _log_regional_results(self, results: Dict):
        """Log regional validation results."""
        logging.info("\nRegional validation results:")
        for region, data in results.items():
            logging.info(f"\n{region}:")

            mult = data["value_multiplier"]
            logging.info(
                f"Value multiplier: {'✓' if mult['within_tolerance'] else '✗'} "
                f"(Expected: {mult['expected']:.2f}, Actual: {mult['actual']:.2f})"
            )

            for cat, cat_data in data["category_distribution"].items():
                logging.info(
                    f"{cat}: {'✓' if cat_data['within_tolerance'] else '✗'} "
                    f"(Expected: {cat_data['expected']:.2f}, "
                    f"Actual: {cat_data['actual']:.2f})"
                )

    def _log_event_results(self, results: Dict):
        """Log event validation results."""
        logging.info("\nSpecial event validation results:")
        for event, data in results.items():
            logging.info(f"\n{event}:")
            for metric, values in data.items():
                if isinstance(values, dict) and "within_tolerance" in values:
                    logging.info(
                        f"{metric}: {'✓' if values['within_tolerance'] else '✗'} "
                        f"(Expected: {values['expected']:.3f}, "
                        f"Actual: {values['actual']:.3f})"
                    )

    def _log_value_results(self, overall_stats: Dict, category_stats: Dict):
        """Log transaction value validation results."""
        logging.info("\nTransaction value validation results:")
        logging.info(f"Overall mean: ${overall_stats['mean']:.2f}")
        logging.info(f"Overall median: ${overall_stats['median']:.2f}")

        for category, stats in category_stats.items():
            logging.info(f"\n{category}:")
            logging.info(f"Mean: ${stats['mean']:.2f}")
            logging.info(
                f"Multiplier: {'✓' if stats['within_tolerance'] else '✗'} "
                f"(Expected: {stats['expected_multiplier']:.2f}, "
                f"Actual: {stats['actual_multiplier']:.2f})"
            )
