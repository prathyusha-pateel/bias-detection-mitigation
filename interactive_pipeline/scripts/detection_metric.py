from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    precision_score
)
import numpy as np
import pandas as pd


def process_sensitive_features(sensitive_features):
    """
    Process sensitive features to handle intersectional groups.

    Parameters:
    - sensitive_features: DataFrame, Series, or array-like representing sensitive attributes.

    Returns:
    - Array of combined sensitive feature values as strings for unique group identification.
    """
    if isinstance(sensitive_features, pd.DataFrame):
        return sensitive_features.astype(str).agg('-'.join, axis=1).values
    elif isinstance(sensitive_features, pd.Series):
        return sensitive_features.astype(str).values
    else:
        return np.array(sensitive_features, dtype=str)


def predictive_parity(y_true, y_pred, sensitive_features, sample_weight=None):
    """
    Calculate Predictive Parity as the maximum difference in precision across groups.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sensitive_features = process_sensitive_features(sensitive_features)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    groups = np.unique(sensitive_features)

    def group_precision(y_true_group, y_pred_group, sample_weight_group):
        return precision_score(
            y_true_group,
            y_pred_group,
            sample_weight=sample_weight_group,
            zero_division=0
        )

    group_precisions = {
        group: group_precision(
            y_true[sensitive_features == group],
            y_pred[sensitive_features == group],
            sample_weight[sensitive_features == group] if sample_weight is not None else None
        )
        for group in groups
    }

    return max(abs(group_precisions[g1] - group_precisions[g2]) for g1 in groups for g2 in groups if g1 != g2)


def calibration_difference(y_true, y_prob, sensitive_features, sample_weight=None, n_bins=10):
    """
    Calculate Calibration Difference as the maximum calibration error difference across groups.
    """
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    sensitive_features = process_sensitive_features(sensitive_features)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    else:
        sample_weight = np.array(sample_weight)

    groups = np.unique(sensitive_features)

    def group_calibration_error(y_true_group, y_prob_group, sample_weight_group):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob_group, bins, right=True) - 1
        bin_true, bin_pred = [], []

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            bin_weight = sample_weight_group[mask].sum()

            if bin_weight > 0:
                bin_true.append(np.average(y_true_group[mask], weights=sample_weight_group[mask]))
                bin_pred.append(np.average(y_prob_group[mask], weights=sample_weight_group[mask]))
            else:
                bin_true.append(0)
                bin_pred.append(0)

        return np.abs(np.array(bin_true) - np.array(bin_pred)).mean()

    group_errors = {
        group: group_calibration_error(
            y_true[sensitive_features == group],
            y_prob[sensitive_features == group],
            sample_weight[sensitive_features == group]
        )
        for group in groups
    }

    return max(abs(group_errors[g1] - group_errors[g2]) for g1 in groups for g2 in groups if g1 != g2)


def mean_prediction_parity(y_pred, sensitive_features, sample_weight=None):
    """
    Calculate Mean Prediction Parity as the maximum difference in mean predictions across groups.
    """
    y_pred = np.array(y_pred)
    sensitive_features = process_sensitive_features(sensitive_features)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    groups = np.unique(sensitive_features)

    group_means = {
        group: np.average(
            y_pred[sensitive_features == group],
            weights=sample_weight[sensitive_features == group] if sample_weight is not None else None
        )
        for group in groups
    }

    return max(abs(group_means[g1] - group_means[g2]) for g1 in groups for g2 in groups if g1 != g2)


def error_rate_parity_mse(y_true, y_pred, sensitive_features, sample_weight=None):
    """
    Calculate Error Rate Parity (MSE) as the maximum difference in MSE across groups.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sensitive_features = process_sensitive_features(sensitive_features)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    groups = np.unique(sensitive_features)

    def weighted_mse(y_true_group, y_pred_group, sample_weight_group):
        residuals = y_true_group - y_pred_group
        return np.average(residuals**2, weights=sample_weight_group)

    group_mse = {
        group: weighted_mse(
            y_true[sensitive_features == group],
            y_pred[sensitive_features == group],
            sample_weight[sensitive_features == group] if sample_weight is not None else None
        )
        for group in groups
    }

    return max(abs(group_mse[g1] - group_mse[g2]) for g1 in groups for g2 in groups if g1 != g2)


def error_rate_parity_mae(y_true, y_pred, sensitive_features, sample_weight=None):
    """
    Calculate Error Rate Parity (MAE) as the maximum difference in MAE across groups.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sensitive_features = process_sensitive_features(sensitive_features)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    groups = np.unique(sensitive_features)

    def weighted_mae(y_true_group, y_pred_group, sample_weight_group):
        residuals = np.abs(y_true_group - y_pred_group)
        return np.average(residuals, weights=sample_weight_group)

    group_mae = {
        group: weighted_mae(
            y_true[sensitive_features == group],
            y_pred[sensitive_features == group],
            sample_weight[sensitive_features == group] if sample_weight is not None else None
        )
        for group in groups
    }

    return max(abs(group_mae[g1] - group_mae[g2]) for g1 in groups for g2 in groups if g1 != g2)


def true_positive_rate_difference(y_true, y_pred, sensitive_features, sample_weight=None):
    """
    Calculate True Positive Rate Difference as the maximum difference in TPR across groups.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sensitive_features = process_sensitive_features(sensitive_features)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    groups = np.unique(sensitive_features)

    def group_tpr(y_true_group, y_pred_group):
        tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
        fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    group_tprs = {
        group: group_tpr(
            y_true[sensitive_features == group],
            y_pred[sensitive_features == group]
        )
        for group in groups
    }

    return max(abs(group_tprs[g1] - group_tprs[g2]) for g1 in groups for g2 in groups if g1 != g2)


def false_positive_rate_difference(y_true, y_pred, sensitive_features, sample_weight=None):
    """
    Calculate False Positive Rate Difference as the maximum difference in FPR across groups.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sensitive_features = process_sensitive_features(sensitive_features)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    groups = np.unique(sensitive_features)

    def group_fpr(y_true_group, y_pred_group):
        fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
        tn = np.sum((y_true_group == 0) & (y_pred_group == 0))
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    group_fprs = {
        group: group_fpr(
            y_true[sensitive_features == group],
            y_pred[sensitive_features == group]
        )
        for group in groups
    }

    return max(abs(group_fprs[g1] - group_fprs[g2]) for g1 in groups for g2 in groups if g1 != g2)


def disparate_impact_ratio(y_pred, sensitive_features, sample_weight=None):
    """
    Calculate Disparate Impact Ratio.
    """
    y_pred = np.array(y_pred)
    sensitive_features = process_sensitive_features(sensitive_features)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    groups = np.unique(sensitive_features)

    def selection_rate(y_pred_group, sample_weight_group):
        return np.average(y_pred_group, weights=sample_weight_group)

    group_selection_rates = {
        group: selection_rate(
            y_pred[sensitive_features == group],
            sample_weight[sensitive_features == group] if sample_weight is not None else None
        )
        for group in groups
    }

    min_rate = min(group_selection_rates.values())
    max_rate = max(group_selection_rates.values())
    return min_rate / max_rate if max_rate > 0 else 0.0


def equal_opportunity_difference(y_true, y_pred, sensitive_features, sample_weight=None):
    """
    Calculate Equal Opportunity Difference.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sensitive_features = process_sensitive_features(sensitive_features)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    groups = np.unique(sensitive_features)

    def group_tpr(y_true_group, y_pred_group):
        tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
        fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    group_tprs = {
        group: group_tpr(
            y_true[sensitive_features == group],
            y_pred[sensitive_features == group]
        )
        for group in groups
    }

    return max(abs(group_tprs[g1] - group_tprs[g2]) for g1 in groups for g2 in groups if g1 != g2)
