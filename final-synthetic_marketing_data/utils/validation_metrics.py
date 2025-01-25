"""Core validation utilities for synthetic data validation."""

from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime


def get_numerical_validation_details(
    synthetic_data: pd.DataFrame, original_data: pd.DataFrame, config: Dict
) -> Dict[str, Any]:
    """Get validation details for numerical columns."""
    details = {}
    for col in config.get("numerical_columns", []):
        if col in synthetic_data.columns and col in original_data.columns:
            real_val = original_data[col].mean()
            synth_val = synthetic_data[col].mean()
            within_tolerance = abs(synth_val - real_val) <= config.get("tolerance", 0.1)

            details[col] = {
                "expected": real_val,
                "actual": synth_val,
                "within_tolerance": within_tolerance,
            }
    return details


def get_categorical_validation_details(
    synthetic_data: pd.DataFrame, original_data: pd.DataFrame, config: Dict
) -> Dict[str, Any]:
    """Get validation details for categorical columns."""
    details = {}
    for col in config.get("categorical_columns", []):
        if col in synthetic_data.columns and col in original_data.columns:
            real_dist = original_data[col].value_counts(normalize=True)
            synth_dist = synthetic_data[col].value_counts(normalize=True)

            # Calculate distribution difference
            categories = set(real_dist.index) | set(synth_dist.index)
            max_diff = max(
                abs(real_dist.get(cat, 0) - synth_dist.get(cat, 0))
                for cat in categories
            )

            within_tolerance = max_diff <= config.get("cat_tolerance", 0.1)

            details[col] = {
                "expected": real_dist.to_dict(),
                "actual": synth_dist.to_dict(),
                "max_difference": max_diff,
                "within_tolerance": within_tolerance,
            }
    return details


def get_temporal_validation_details(
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


def get_transaction_validation_details(
    synthetic_data: pd.DataFrame, original_data: pd.DataFrame, tolerance: float = 0.1
) -> Dict[str, Any]:
    """Get validation details for transaction metrics."""
    details = {}

    # Validate transaction values
    real_avg = original_data["transaction_value"].mean()
    synth_avg = synthetic_data["transaction_value"].mean()
    within_tolerance = abs(synth_avg - real_avg) / real_avg <= tolerance
    details["average_transaction_value"] = {
        "expected": real_avg,
        "actual": synth_avg,
        "within_tolerance": within_tolerance,
    }

    # Validate items per transaction
    real_items = original_data["num_items"].mean()
    synth_items = synthetic_data["num_items"].mean()
    within_tolerance = abs(synth_items - real_items) / real_items <= tolerance
    details["items_per_transaction"] = {
        "expected": real_items,
        "actual": synth_items,
        "within_tolerance": within_tolerance,
    }

    # Validate channel distribution
    for channel in original_data["channel"].unique():
        real_dist = (original_data["channel"] == channel).mean()
        synth_dist = (synthetic_data["channel"] == channel).mean()
        within_tolerance = abs(synth_dist - real_dist) <= tolerance
        details[f"channel_{channel}"] = {
            "expected": real_dist,
            "actual": synth_dist,
            "within_tolerance": within_tolerance,
        }

    return details


def get_regional_validation_details(
    synthetic_data: pd.DataFrame, original_data: pd.DataFrame, tolerance: float = 0.1
) -> Dict[str, Any]:
    """Get validation details for regional metrics."""
    details = {}

    # Validate regional distribution
    for region in original_data["region"].unique():
        real_dist = (original_data["region"] == region).mean()
        synth_dist = (synthetic_data["region"] == region).mean()
        within_tolerance = abs(synth_dist - real_dist) <= tolerance
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
        within_tolerance = abs(synth_avg - real_avg) / real_avg <= tolerance
        details[f"avg_value_{region}"] = {
            "expected": real_avg,
            "actual": synth_avg,
            "within_tolerance": within_tolerance,
        }

    return details


def calculate_overall_score(validation_results: Dict[str, Any]) -> float:
    """Calculate overall validation score."""
    total_metrics = 0
    passing_metrics = 0

    def count_metrics(details: Dict) -> tuple:
        """Helper to count total and passing metrics in a details dictionary."""
        if not isinstance(details, dict):
            return 0, 0
        total = len(details)
        passing = sum(
            1 for metric in details.values() if metric.get("within_tolerance", False)
        )
        return total, passing

    # Count metrics from all sections
    for section in validation_results.values():
        if isinstance(section, dict):
            for subsection in section.values():
                if isinstance(subsection, dict) and "details" in subsection:
                    t, p = count_metrics(subsection["details"])
                    total_metrics += t
                    passing_metrics += p

    return passing_metrics / total_metrics if total_metrics > 0 else 0.0


def count_total_metrics(validation_results: Dict[str, Any]) -> int:
    """Count total number of metrics being validated."""
    total = 0

    def count_details(details: Dict) -> int:
        return len(details) if isinstance(details, dict) else 0

    for section in validation_results.values():
        if isinstance(section, dict):
            for subsection in section.values():
                if isinstance(subsection, dict) and "details" in subsection:
                    total += count_details(subsection["details"])

    return total


def count_passing_metrics(validation_results: Dict[str, Any]) -> int:
    """Count number of passing metrics."""
    passing = 0

    def count_passing_in_details(details: Dict) -> int:
        return sum(
            1 for metric in details.values() if metric.get("within_tolerance", False)
        )

    for section in validation_results.values():
        if isinstance(section, dict):
            for subsection in section.values():
                if isinstance(subsection, dict) and "details" in subsection:
                    passing += count_passing_in_details(subsection["details"])

    return passing
