from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    equalized_odds_difference,
    demographic_parity_difference,
    
)
    
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    r2_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    mean_squared_error,
    confusion_matrix,
    roc_auc_score
    )
from sklearn.metrics import mean_squared_error
from scripts.detection_metric import equal_opportunity_difference, disparate_impact_ratio,predictive_parity, calibration_difference, error_rate_parity_mse, error_rate_parity_mae, true_positive_rate_difference, false_positive_rate_difference, mean_prediction_parity
import numpy as np
import pandas as pd


class BiasDetector:
    def __init__(self, task, sensitive_features, fairness_notions):
        """
        Initialize the BiasDetector with task, predictions, true labels, sensitive features, 
        fairness notions, and optional predicted probabilities.
        
        Parameters:
        - task (str): Type of task ('Classification', 'Regression', or 'Clustering').
        - y_true (array-like): True labels or outcomes.
        - y_pred (array-like): Predicted labels or outcomes.
        - sensitive_features (array-like): Sensitive feature (protected attribute) for groups.
        - fairness_notions (list of str): List of fairness notions to calculate.
        - y_prob (array-like): Predicted probabilities, used for classification calibration metrics.
        """
        self.task = task
        #self.y_true = np.array(y_true)
        #self.y_pred = np.array(y_pred)
        self.sensitive_features = sensitive_features
        self.fairness_notions = fairness_notions
        #self.y_prob = y_prob
        self.group_metrics_list = []
        self.overall_metrics_list = []
        self.prepare_metric_list()

    def prepare_metric_list(self):
        """
        Prepare a list of metrics to calculate based on the task and fairness notions.
        
        Returns:
        - metrics (list of str): List of metric names to calculate.
        """
        
        
        # Classification Metrics
        if self.task == "Classification":
            
            self.group_metrics_list.append("Accuracy")
            self.group_metrics_list.append("Precision")
            self.group_metrics_list.append("Recall")
            self.group_metrics_list.append("F1 Score")
            self.group_metrics_list.append("Selection Rate")
            
            if "Demographic Parity" in self.fairness_notions:
                self.overall_metrics_list.append("Demographic Parity Difference")
                self.overall_metrics_list.append("Disparate Impact Ratio")
            
            if "Equal Opportunity" in self.fairness_notions:
                self.overall_metrics_list.append("Equal Opportunity Difference (TPR Difference)")
            
            if "Equalized Odds" in self.fairness_notions:
                self.overall_metrics_list.append("Equalized Odds Difference")
                self.overall_metrics_list.append("True Positive Rate Difference")
                self.overall_metrics_list.append("False Positive Rate Difference")
            
            if "Predictive Parity" in self.fairness_notions:
                self.overall_metrics_list.append("Predictive Parity (Precision Difference)")
            
            if "Calibration" in self.fairness_notions:
                self.overall_metrics_list.append("Calibration Difference")
        
        # Regression Metrics (extend if implemented)
        elif self.task == "Regression":
            
            self.group_metrics_list.append("Mean Squared Error")
            self.group_metrics_list.append("Mean Absolute Error")
            self.group_metrics_list.append("R2 Score")
            if "Mean Prediction Parity" in self.fairness_notions:
                self.overall_metrics_list.append("Mean Prediction Parity")
            
            if "Error Rate Parity" in self.fairness_notions:
                self.overall_metrics_list.append("Error Rate Parity (MSE)")
                self.overall_metrics_list.append("Error Rate Parity (MAE)")

        # Clustering Metrics (extend if implemented)
        elif self.task == "Clustering":
            # Add clustering-specific fairness metrics here
            self.group_metrics_list.append("Silhouette Score")
            self.group_metrics_list.append("Davies-Bouldin Index")
            self.group_metrics_list.append("Calinski-Harabasz Index")
            
    
    def prepare_group_metrics(self):
    
        metrics_dict = {
            "Accuracy": accuracy_score,
            "Precision": precision_score,
            "Recall": recall_score,
            "F1 Score": f1_score,
            "Selection Rate": selection_rate,
            "Mean Squared Error": mean_squared_error,
            "Mean Absolute Error": mean_absolute_error,
            "R2 Score": r2_score,
            "Silhouette Score": silhouette_score,
            "Davies-Bouldin Index": davies_bouldin_score,
            "Calinski-Harabasz Index": calinski_harabasz_score
        }
        self.group_metrics_dict = {metric: metrics_dict[metric] for metric in self.group_metrics_list if metric in metrics_dict}

    def calculate_metrics(self,y_true,y_pred, y_prob =None, sample_weight=None):
        """
        Calculate fairness metrics based on selected fairness notions and task.
        
        Returns:
        - metrics (dict): Dictionary of calculated fairness metrics.
        """
        self.prepare_group_metrics()
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        metric_frame = MetricFrame(metrics=self.group_metrics_dict, y_true=y_true, y_pred=y_pred, sensitive_features=self.sensitive_features)
        group_results = metric_frame.by_group
        overall_results = metric_frame.overall.to_dict()
        
        if "Demographic Parity Difference" in self.overall_metrics_list:
            overall_results["Demographic Parity Difference"] = demographic_parity_difference(y_true, y_pred, sensitive_features=self.sensitive_features, sample_weight=sample_weight)
        if "Equalized Odds Difference" in self.overall_metrics_list:
            overall_results["Equalized Odds Difference"] = equalized_odds_difference(y_true, y_pred, sensitive_features=self.sensitive_features, sample_weight=sample_weight)
        if "Disparate Impact Ratio" in self.overall_metrics_list:
            overall_results["Disparate Impact Ratio"] = disparate_impact_ratio(y_pred, sensitive_features=self.sensitive_features, sample_weight=sample_weight)
        if "Equal Opportunity Difference (TPR Difference)" in self.overall_metrics_list:
            overall_results["Equal Opportunity Difference (TPR Difference)"] = equal_opportunity_difference(y_true, y_pred, sensitive_features=self.sensitive_features, sample_weight=sample_weight)
        if "Predictive Parity (Precision Difference)" in self.overall_metrics_list:
            overall_results["Predictive Parity (Precision Difference)"] = predictive_parity(y_true, y_pred, sensitive_features=self.sensitive_features, sample_weight=sample_weight)
        if "Calibration Difference" in self.overall_metrics_list:
            if y_prob is not None:
                overall_results["Calibration Difference"] = calibration_difference(y_true, y_prob, sensitive_features=self.sensitive_features, sample_weight=sample_weight)
        if "Mean Prediction Parity" in self.overall_metrics_list:
            overall_results["Mean Prediction Parity"] = mean_prediction_parity(y_pred, sensitive_features=self.sensitive_features, sample_weight=sample_weight)
        if "Error Rate Parity (MSE)" in self.overall_metrics_list:
            overall_results["Error Rate Parity (MSE)"] = error_rate_parity_mse(y_true, y_pred, sensitive_features=self.sensitive_features,  sample_weight=sample_weight)
        if "Error Rate Parity (MAE)" in self.overall_metrics_list:
            overall_results["Error Rate Parity (MAE)"] = error_rate_parity_mae(y_true, y_pred, sensitive_features=self.sensitive_features,      sample_weight=sample_weight)
        if "True Positive Rate Difference" in self.overall_metrics_list:
            overall_results["True Positive Rate Difference"] = true_positive_rate_difference(y_true, y_pred, sensitive_features=self.sensitive_features, sample_weight=sample_weight)
        if "False Positive Rate Difference" in self.overall_metrics_list:
            overall_results["False Positive Rate Difference"] = false_positive_rate_difference(y_true, y_pred, sensitive_features=self.sensitive_features, sample_weight=sample_weight)
                                                                        
        return group_results, overall_results
    
    
if __name__ == "__main__":
    # Classification Example Data
    y_true_class = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred_class = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    y_prob_class = [0.9, 0.1, 0.8, 0.6, 0.4, 0.9, 0.2, 0.7, 0.5, 0.8]  # Needed for Calibration
    sensitive_features_class = ["black", "white", "black", "white", "white", "black", "white", "black", "black", "white"]
    #sensitive_features_class = np.array([1 if x == "black" else 0 for x in sensitive_features_class])
    sample_weight_class = [1, 2, 1, 2, 1, 1, 1, 2, 1, 1]  # Example sample weights

    fairness_notions_classification = [
        "Demographic Parity",
        "Equal Opportunity",
        "Equalized Odds",
        "Predictive Parity",
        "Calibration"
    ]

    print("=== Classification Metrics ===")
    bias_detector_class = BiasDetector(
        task="Classification",
        sensitive_features=sensitive_features_class,
        fairness_notions=fairness_notions_classification
    )

    group_metrics_class, overall_metrics_class = bias_detector_class.calculate_metrics(
        y_true=y_true_class,
        y_pred=y_pred_class,
        y_prob=y_prob_class,
        sample_weight=None
    )
    
    print("Group Metrics (Classification):")
    print(group_metrics_class)
    print("\nOverall Metrics (Classification):")
    for metric, value in overall_metrics_class.items():
        print(f"{metric}: {value}")

    # Regression Example Data
    y_true_reg = [3.5, 2.1, 4.0, 3.8, 5.5, 2.7, 3.9, 4.2, 5.0, 3.4]
    y_pred_reg = [3.6, 2.0, 4.1, 3.7, 5.6, 2.8, 4.0, 4.3, 5.1, 3.3]
    sensitive_features_reg = ["Group A", "Group B", "Group A", "Group B", "Group B", "Group A", "Group B", "Group A", "Group A", "Group B"]
    #sensitive_features_reg = np.array([1 if x == "Group A" else 0 for x in sensitive_features_reg])
    sample_weight_reg = [1.5, 1, 2, 1, 1, 1.2, 1.3, 1, 1.4, 1]  # Example sample weights

    fairness_notions_regression = [
        "Mean Prediction Parity",
        "Error Rate Parity"
    ]

    print("\n=== Regression Metrics ===")
    bias_detector_reg = BiasDetector(
        task="Regression",
        sensitive_features=sensitive_features_reg,
        fairness_notions=fairness_notions_regression
    )

    group_metrics_reg, overall_metrics_reg = bias_detector_reg.calculate_metrics(
        y_true=y_true_reg,
        y_pred=y_pred_reg,
        y_prob=None,
        sample_weight=None
    )

    print("Group Metrics (Regression):")
    print(group_metrics_reg)
    print("\nOverall Metrics (Regression):")
    for metric, value in overall_metrics_reg.items():
        print(f"{metric}: {value}")
