"""
Statistical utility functions for validation calculations.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Union, List
import pandas as pd


def calculate_confidence_interval(
    data: Union[np.ndarray, pd.Series], confidence: float = 0.95, method: str = "t"
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a dataset.

    Args:
        data: Array or Series of values
        confidence: Confidence level (default: 0.95)
        method: Method to use ('t' or 'normal')

    Returns:
        Tuple of (lower bound, upper bound)
    """
    if len(data) < 2:
        return (np.nan, np.nan)

    data = np.array(data)
    mean = np.mean(data)
    se = stats.sem(data)

    if method == "t":
        ci = stats.t.interval(confidence, len(data) - 1, mean, se)
    else:
        z_score = stats.norm.ppf((1 + confidence) / 2)
        ci = (mean - z_score * se, mean + z_score * se)

    return float(ci[0]), float(ci[1])


def calculate_correlation_confidence(
    r: float, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for correlation coefficient using Fisher's Z.

    Args:
        r: Correlation coefficient
        n: Sample size
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (lower bound, upper bound)
    """
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_score = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = np.tanh(z - z_score * se)
    ci_upper = np.tanh(z + z_score * se)
    return float(ci_lower), float(ci_upper)


def calculate_effect_size(
    group1: Union[np.ndarray, pd.Series], group2: Union[np.ndarray, pd.Series]
) -> Tuple[float, float]:
    """
    Calculate Cohen's d effect size and its confidence interval.

    Args:
        group1: First group's data
        group2: Second group's data

    Returns:
        Tuple of (effect size, standard error)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_sd

    # Standard error of d
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2)))

    return float(d), float(se)


def calculate_outlier_bounds(
    data: Union[np.ndarray, pd.Series], method: str = "iqr"
) -> Tuple[float, float]:
    """
    Calculate outlier bounds for a dataset.

    Args:
        data: Array or Series of values
        method: Method to use ('iqr' or 'zscore')

    Returns:
        Tuple of (lower bound, upper bound)
    """
    if method == "iqr":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    else:  # zscore
        mean = np.mean(data)
        std = np.std(data)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

    return float(lower_bound), float(upper_bound)


def calculate_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = "correlation",
) -> int:
    """
    Calculate required sample size for desired statistical power.

    Args:
        effect_size: Expected effect size
        power: Desired statistical power (default: 0.8)
        alpha: Significance level (default: 0.05)
        test_type: Type of test ('correlation' or 'ttest')

    Returns:
        Required sample size
    """
    if test_type == "correlation":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = (
            (z_alpha + z_beta) / (0.5 * np.log((1 + effect_size) / (1 - effect_size)))
        ) ** 2 + 3
    else:  # ttest
        n = stats.tt_ind_solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=1,
            alternative="two-sided",
        )

    return int(np.ceil(n))


def analyze_temporal_patterns(
    data: pd.DataFrame, value_column: str, timestamp_column: str = "timestamp"
) -> dict:
    """
    Analyze temporal patterns in data.

    Args:
        data: DataFrame containing temporal data
        value_column: Column containing values to analyze
        timestamp_column: Column containing timestamps

    Returns:
        Dict containing temporal analysis results
    """
    if len(data) == 0:
        return {}

    data = data.copy()
    data["hour"] = pd.to_datetime(data[timestamp_column]).dt.hour
    data["day_of_week"] = pd.to_datetime(data[timestamp_column]).dt.dayofweek
    data["month"] = pd.to_datetime(data[timestamp_column]).dt.month

    results = {
        "hourly": {
            "distribution": data.groupby("hour")[value_column]
            .agg(["mean", "std", "count"])
            .to_dict(orient="index"),
            "peak_hour": int(data.groupby("hour")[value_column].mean().idxmax()),
        },
        "daily": {
            "distribution": data.groupby("day_of_week")[value_column]
            .agg(["mean", "std", "count"])
            .to_dict(orient="index"),
            "peak_day": int(data.groupby("day_of_week")[value_column].mean().idxmax()),
        },
        "monthly": {
            "distribution": data.groupby("month")[value_column]
            .agg(["mean", "std", "count"])
            .to_dict(orient="index"),
            "peak_month": int(data.groupby("month")[value_column].mean().idxmax()),
        },
    }

    # Calculate seasonality
    if len(data) >= 12:  # Need at least a year of data
        monthly_values = data.groupby("month")[value_column].mean()
        season_strength = 1 - np.var(monthly_values - monthly_values.mean()) / np.var(
            monthly_values
        )
        results["seasonality"] = float(season_strength)

    return results
