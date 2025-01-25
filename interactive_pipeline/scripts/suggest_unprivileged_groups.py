import pandas as pd
import numpy as np


class GroupPrivilegeAnalyzer:
    def __init__(self, thresholds=None, metric_weights=None):
        """
        Initialize the analyzer with metric-specific thresholds and weights.
        """
        # Default thresholds for common metrics
        self.thresholds = thresholds or {
            "Accuracy": 0.05,
            "Precision": 0.05,
            "Recall": 0.05,
            "F1 Score": 0.05,
            "Selection Rate": 0.05,
            "Mean Absolute Error": 0.1,
            "Mean Squared Error": 0.2,
            "R-Squared": 0.05,
            "Silhouette Score": 0.05,
            "Davies-Bouldin Index": 0.1,
        }

        # Default weights for metrics
        self.metric_weights = metric_weights or {
            "Accuracy": 1.0,
            "Precision": 0.8,
            "Recall": 0.8,
            "F1 Score": 0.9,
            "Selection Rate": 0.7,
            "Mean Absolute Error": -1.0,  # Negative weight for errors
            "Mean Squared Error": -1.0,
            "R-Squared": 1.0,
            "Silhouette Score": 0.9,
            "Davies-Bouldin Index": -0.5,  # Negative weight for "better is lower" metrics
        }

    def analyze_privilege(self, group_metrics):
        """
        Analyzes privilege and unprivilege based on group-level metrics.

        Parameters:
            group_metrics (pd.DataFrame): DataFrame of group-level metrics with single or multi-index for groups.

        Returns:
            dict: Privileged group, unprivileged group, and reasons for selection.
        """
        # Initialize scores and reasons for each group
        group_scores = {idx: 0 for idx in group_metrics.index}
        detailed_reasons = {idx: [] for idx in group_metrics.index}

        # Iterate through metrics
        for metric in group_metrics.columns:
            max_value = group_metrics[metric].max()
            min_value = group_metrics[metric].min()

            # Determine weight and threshold for this metric
            weight = self.metric_weights.get(metric, 1.0)
            threshold = self.thresholds.get(metric, 0.05)

            # Identify privileged and unprivileged groups for the metric
            privileged_group_idx = group_metrics[group_metrics[metric] == max_value].index[0]
            unprivileged_group_idx = group_metrics[group_metrics[metric] == min_value].index[0]

            # Update scores for privileged and unprivileged groups
            group_scores[privileged_group_idx] += weight
            group_scores[unprivileged_group_idx] -= weight

            # Record reasons if the difference exceeds the threshold
            if abs(max_value - min_value) > threshold:
                for idx, value in group_metrics[metric].items():
                    difference = abs(value - (max_value if idx == unprivileged_group_idx else min_value))
                    detailed_reasons[idx].append(
                        f"{metric}: {value:.2f}"
                        f"with a significant difference of {difference:.2f}."
                    )

        # Determine the final privileged and unprivileged groups based on scores
        most_privileged_group_idx = max(group_scores, key=group_scores.get)
        least_privileged_group_idx = min(group_scores, key=group_scores.get)

        # Handle single-index and multi-index cases
        if isinstance(group_metrics.index, pd.MultiIndex):
            # Multi-index: Convert tuple indices to dictionaries
            privileged_group = dict(zip(group_metrics.index.names, most_privileged_group_idx))
            unprivileged_group = dict(zip(group_metrics.index.names, least_privileged_group_idx))
            #sensitive groups look like this [{'age': 0.0,'race': 0.0}, {'age': 0.0, 'race': 1.0}, {'age':1.0, 'race': 0.0}, {'age':1.0,'race':1.0}]
            sensitive_groups = group_metrics.index.to_frame(index=False).to_dict(orient="records")

        else:
            # Single index: Use the index name and value
            privileged_group = {group_metrics.index.name: most_privileged_group_idx}
            unprivileged_group = {group_metrics.index.name: least_privileged_group_idx}
            sensitive_groups = group_metrics.index.to_frame(index=False).to_dict(orient="records")


        # Combine reasons for the most privileged and unprivileged groups
        privileged_reasons = " | ".join(detailed_reasons[most_privileged_group_idx])
        unprivileged_reasons = " | ".join(detailed_reasons[least_privileged_group_idx])

        return {
            "Privileged Group": privileged_group,
            "Unprivileged Group": unprivileged_group,
            "Reasons": f"Privileged Group Reasons: {privileged_reasons}\nUnprivileged Group Reasons: {unprivileged_reasons}",
            "Sensitive Groups": sensitive_groups
        }


if __name__ == "__main__":
    print("=== Testing Multi-Indexed Group Metrics ===")
    # Multi-indexed groups (age and race)
    group_metrics_multi = pd.DataFrame(
        {
            "Accuracy": [0.9, 0.8, 0.6, 0.5],
            "Precision": [0.7, 0.6, 0.5, 0.4],
            "Recall": [0.8, 0.7, 0.3, 0.2],
            "Mean Absolute Error": [0.1, 0.15, 0.3, 0.4],  # Regression metric
            "Silhouette Score": [0.7, 0.6, 0.3, 0.25],  # Clustering metric
        },
        index=pd.MultiIndex.from_tuples(
            [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)],
            names=["age", "race"],
        ),
    )

    analyzer = GroupPrivilegeAnalyzer()
    result_multi = analyzer.analyze_privilege(group_metrics_multi)

    print("Privileged Group:", result_multi["Privileged Group"])
    print("Unprivileged Group:", result_multi["Unprivileged Group"])
    print("Reasons:", result_multi["Reasons"])
    print("Sensitive Groups:", result_multi["Sensitive Groups"])
    print("\n")

    print("=== Testing Single-Indexed Group Metrics ===")
    # Single-indexed groups (gender)
    group_metrics_single = pd.DataFrame(
        {
            "Accuracy": [0.85, 0.75],
            "Precision": [0.80, 0.70],
            "Recall": [0.78, 0.65],
            "Mean Absolute Error": [0.1, 0.2],  # Regression metric
            "Silhouette Score": [0.7, 0.6],  # Clustering metric
        },
        index=pd.Index([0, 1], name="Gender"),
    )

    result_single = analyzer.analyze_privilege(group_metrics_single)

    print("Privileged Group:", result_single["Privileged Group"])
    print("Unprivileged Group:", result_single["Unprivileged Group"])
    print("Reasons:", result_single["Reasons"])
    print("Sensitive Groups:", result_single["Sensitive Groups"])
