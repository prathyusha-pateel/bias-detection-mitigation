import numpy as np
import pandas as pd


class FairnessMitigationRecommender:
    def __init__(self):
        # Define thresholds for various metrics
        self.thresholds = {
            "Demographic Parity Difference": 0.2,
            "Equalized Odds Difference": 0.1,
            "Predictive Parity (Precision Difference)": 0.1,
            "Disparate Impact": (0.8, 1.25),
            "Statistical Parity Difference": 0.15,
            "Group Error Rate Difference": 0.1,
            "False Negative Rate Difference": 0.1,
            "Group Accuracy Difference": 0.1,
            "Group Precision Difference": 0.1,
            "Group Recall Difference": 0.1
        }
    
    def recommend_techniques(self, overall_metrics, group_metrics, task, fairness_notions):
        """
        Suggests mitigation techniques based on overall and group-level metrics.
        
        Parameters:
            metrics (dict): Dictionary of overall fairness metrics and their values.
            group_metrics (dict): Dictionary of group-level metrics (e.g., {"Accuracy": {"Group 1": 0.9, "Group 2": 0.8}}).
            task (str): Type of ML task ("Classification" or "Regression").
            fairness_notions (list): List of fairness notions to prioritize (e.g., "Demographic Parity").
        
        Returns:
            dict: Recommended mitigation techniques categorized into preprocessing, in-processing, and post-processing.
        """
        preprocessing = set()
        in_processing = set()
        post_processing = set()
        reasons = set()

        # Evaluate overall metrics against thresholds
        if "Demographic Parity" in fairness_notions and overall_metrics.get("Demographic Parity Difference", 0) > self.thresholds["Demographic Parity Difference"]:
            preprocessing.update({"Reweighing", "Disparate Impact Removal"})
            in_processing.update({"Learning Fair Representations"})
            reasons.update({"Large disparity in Demographic Parity indicates an imbalance in outcomes across groups."})
        
        if "Equalized Odds" in fairness_notions and overall_metrics.get("Equalized Odds Difference", 0) > self.thresholds["Equalized Odds Difference"]:
            in_processing.update({"Adversarial Debiasing", "Fairness Regularization"})
            post_processing.update({"Equalized Odds Adjustment"})
            reasons.update({"Equalized Odds Difference indicates disparities in True Positive or False Positive Rates."})

        if "Predictive Parity" in fairness_notions and overall_metrics.get("Predictive Parity (Precision Difference)", 0) > self.thresholds["Predictive Parity (Precision Difference)"]:
            post_processing.update({"Threshold Optimization", "Rejection Option Classification"})
            in_processing.update({"Cost-Sensitive Learning"})
            reasons.update({"Predictive Parity differences indicate unequal Precision across groups."})
            
        if overall_metrics.get("Disparate Impact", 1) < self.thresholds["Disparate Impact"][0] or overall_metrics.get("Disparate Impact", 1) > self.thresholds["Disparate Impact"][1]:
            preprocessing.update({"Disparate Impact Removal", "Synthetic Data Augmentation", "Reweighing"})
            reasons.update({"Disparate Impact ratio indicates disproportionate treatment of sensitive groups."})

        if overall_metrics.get("Statistical Parity Difference", 0) > self.thresholds["Statistical Parity Difference"]:
            preprocessing.update({"Reweighing"})
            in_processing.update({"Adversarial Debiasing"})
            post_processing.update({"Post-Processing Adjustments"})
            reasons.update({"Statistical Parity differences suggest outcome imbalances unrelated to input features."})

        if task == "Regression" and overall_metrics.get("Group Error Rate Difference", 0) > self.thresholds["Group Error Rate Difference"]:
            preprocessing.update({"Preprocessing Adjustments"})
            in_processing.update({"Fair Regression Models", "Regularization-Based Approaches"})
            reasons.update({"Group Error Rate disparities indicate systematic differences in Regression predictions."})

        # Evaluate group-level metrics
        for metric_name, group_values in group_metrics.items():
            max_diff = max(group_values.values()) - min(group_values.values())
            if metric_name == "Accuracy" and max_diff > self.thresholds["Group Accuracy Difference"]:
                in_processing.update({"Fairness Regularization", "Group-Specific Cost-Sensitive Learning"})
                reasons.update({f"Significant group Accuracy differences detected (Max Diff: {max_diff:.2f})."})
                
            if metric_name == "Precision" and max_diff > self.thresholds["Group Precision Difference"]:
                post_processing.update({"Threshold Optimization"})
                in_processing.update({"Adversarial Debiasing"})
                reasons.update({f"Group Precision differences detected (Max Diff: {max_diff:.2f})."})
                
            if metric_name == "Recall" and max_diff > self.thresholds["Group Recall Difference"]:
                preprocessing.update({"Oversampling"})
                in_processing.update({"Cost-Sensitive Learning"})
                post_processing.update({"Threshold Optimization"})
                reasons.update({f"Group Recall differences detected (Max Diff: {max_diff:.2f})."})
                

        # Task-specific adjustments
        if task == "Classification" and overall_metrics.get("False Negative Rate Difference", 0) > self.thresholds["False Negative Rate Difference"]:
            preprocessing.update({"Oversampling"})
            in_processing.update({"Cost-Sensitive Learning"})
            post_processing.update({"Threshold Optimization"})
            reasons.update({"High False Negative Rates in minority groups indicate unfair penalties in critical tasks."})

        reasons = list(reasons) if reasons else ["No significant fairness concerns detected."]
        # Join all the reasons into a single string.
        reasons = " ".join(reasons)

        return {
            "Preprocessing": list(preprocessing),
            "In-Processing": list(in_processing),
            "Post-Processing": list(post_processing),
            "Reasons": reasons
        }

if __name__ == "__main__":

    # Example Usage
    metrics = {
        "Demographic Parity Difference": 0.25,
        "Equalized Odds Difference": 0.08,
        "Predictive Parity (Precision Difference)": 0.12,
        "Disparate Impact": 0.75,
        "Statistical Parity Difference": 0.18,
        "Group Error Rate Difference": 0.05,
        "False Negative Rate Difference": 0.2
    }

    group_metrics = {
        "Accuracy": {"Group 1": 0.9, "Group 2": 0.8, "Group 3": 0.85},
        "Precision": {"Group 1": 0.95, "Group 2": 0.85, "Group 3": 0.9},
        "Recall": {"Group 1": 0.88, "Group 2": 0.75, "Group 3": 0.85}
    }

    task = "Classification"
    fairness_notions = {"Demographic Parity", "Equalized Odds"}

    recommender = FairnessMitigationRecommender()
    recommendations = recommender.recommend_techniques(metrics, group_metrics, task, fairness_notions)

    # Print recommendations
    print("Preprocessing Techniques:", recommendations["Preprocessing"])
    print("In-Processing Techniques:", recommendations["In-Processing"])
    print("Post-Processing Techniques:", recommendations["Post-Processing"])
    print("Reasons:", recommendations["Reasons"])
