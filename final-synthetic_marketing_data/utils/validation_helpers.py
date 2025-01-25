"""Helper functions for validation details across modules."""

import pandas as pd
from typing import Dict, Any


def _get_numerical_validation_details(
    synthetic_data: pd.DataFrame, original_data: pd.DataFrame, config: Dict
) -> Dict[str, Any]:
    """Get validation details for numerical columns."""
    details = {}
    for col in config["numerical_columns"]:
        if col in synthetic_data.columns:
            real_val = original_data[col].mean()
            synth_val = synthetic_data[col].mean()
            within_tolerance = abs(synth_val - real_val) <= config.get(
                "tolerance", 0.05
            )
            details[col] = {
                "expected": real_val,
                "actual": synth_val,
                "within_tolerance": within_tolerance,
            }
    return details


def _get_categorical_validation_details(
    synthetic_data: pd.DataFrame, original_data: pd.DataFrame, config: Dict
) -> Dict[str, Any]:
    """Get validation details for categorical columns."""
    details = {}
    for col in config["categorical_columns"]:
        if col in synthetic_data.columns:
            real_dist = original_data[col].value_counts(normalize=True)
            synth_dist = synthetic_data[col].value_counts(normalize=True)

            for category in sorted(set(real_dist.index) | set(synth_dist.index)):
                real_val = real_dist.get(category, 0)
                synth_val = synth_dist.get(category, 0)
                within_tolerance = abs(synth_val - real_val) <= config.get(
                    "tolerance", 0.05
                )
                details[f"{col}_{category}"] = {
                    "expected": real_val,
                    "actual": synth_val,
                    "within_tolerance": within_tolerance,
                }
    return details


def _get_temporal_validation_details(
    synthetic_data: pd.DataFrame, original_data: pd.DataFrame
) -> Dict[str, Any]:
    """Get validation details for temporal patterns."""
    details = {}
    for day in range(7):
        real_rate = original_data[original_data["day_of_week"] == day][
            "engagement_rate"
        ].mean()
        synth_rate = synthetic_data[synthetic_data["day_of_week"] == day][
            "engagement_rate"
        ].mean()
        within_tolerance = abs(synth_rate - real_rate) <= 0.05
        details[f"day_{day}"] = {
            "expected": real_rate,
            "actual": synth_rate,
            "within_tolerance": within_tolerance,
        }
    return details


def _get_transaction_validation_details(
    synthetic_data: pd.DataFrame,
    original_data: pd.DataFrame,
) -> Dict[str, Any]:
    """Get validation details for transaction metrics."""
    details = {}

    # Validate transaction values
    real_avg = original_data["transaction_value"].mean()
    synth_avg = synthetic_data["transaction_value"].mean()
    within_tolerance = abs(synth_avg - real_avg) / real_avg <= 0.05
    details["average_transaction_value"] = {
        "expected": real_avg,
        "actual": synth_avg,
        "within_tolerance": within_tolerance,
    }

    # Validate items per transaction
    real_items = original_data["num_items"].mean()
    synth_items = synthetic_data["num_items"].mean()
    within_tolerance = abs(synth_items - real_items) / real_items <= 0.05
    details["items_per_transaction"] = {
        "expected": real_items,
        "actual": synth_items,
        "within_tolerance": within_tolerance,
    }

    # Validate channel distribution
    for channel in original_data["channel"].unique():
        real_dist = (original_data["channel"] == channel).mean()
        synth_dist = (synthetic_data["channel"] == channel).mean()
        within_tolerance = abs(synth_dist - real_dist) <= 0.05
        details[f"channel_{channel}"] = {
            "expected": real_dist,
            "actual": synth_dist,
            "within_tolerance": within_tolerance,
        }

    return details


def _get_regional_validation_details(
    synthetic_data: pd.DataFrame,
    original_data: pd.DataFrame,
) -> Dict[str, Any]:
    """Get validation details for regional metrics."""
    details = {}

    # Validate regional distribution
    for region in original_data["region"].unique():
        real_dist = (original_data["region"] == region).mean()
        synth_dist = (synthetic_data["region"] == region).mean()
        within_tolerance = abs(synth_dist - real_dist) <= 0.05
        details[f"region_{region}"] = {
            "expected": real_dist,
            "actual": synth_dist,
            "within_tolerance": within_tolerance,
        }

        # Validate regional average transaction values
        real_avg = original_data[original_data["region"] == region][
            "transaction_value"
        ].mean()
        synth_avg = synthetic_data[synthetic_data["region"] == region][
            "transaction_value"
        ].mean()
        within_tolerance = abs(synth_avg - real_avg) / real_avg <= 0.05
        details[f"avg_value_{region}"] = {
            "expected": real_avg,
            "actual": synth_avg,
            "within_tolerance": within_tolerance,
        }

    return details


def _calculate_overall_score(
    synthetic_data: pd.DataFrame,
    original_data: pd.DataFrame,
) -> float:
    """Calculate overall validation score."""
    transaction_details = _get_transaction_validation_details(
        synthetic_data, original_data
    )
    regional_details = _get_regional_validation_details(synthetic_data, original_data)

    total_metrics = len(transaction_details) + len(regional_details)
    passing_metrics = sum(
        1 for d in transaction_details.values() if d["within_tolerance"]
    )
    passing_metrics += sum(
        1 for d in regional_details.values() if d["within_tolerance"]
    )

    return passing_metrics / total_metrics if total_metrics > 0 else 0.0


def _count_total_metrics() -> int:
    """Count total number of metrics being validated."""
    # Base transaction metrics
    count = 2  # average_transaction_value and items_per_transaction
    count += 3  # Standard channels (mobile, desktop, in_store)
    count += 8  # 4 regions * 2 metrics each (distribution and avg value)
    return count


def _count_passing_metrics(results: Dict) -> int:
    """Count number of passing metrics."""
    passing = 0

    # Helper function to count passing metrics in a details dictionary
    def count_passing(details: Dict) -> int:
        return sum(
            1 for metric in details.values() if metric.get("within_tolerance", False)
        )

    # Count passing metrics from all sections
    for section in results.values():
        if isinstance(section, dict):
            for subsection in section.values():
                if isinstance(subsection, dict) and "details" in subsection:
                    passing += count_passing(subsection["details"])

    return passing
