import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
    selection_rate,
)
from typing import List, Dict, Any
from output import Output
from ai_analyzer import AIAnalyzer
import shap
import os
import matplotlib.pyplot as plt


class KaggleBiasDetector:
    def __init__(
        self,
        data: pd.DataFrame,
        sensitive_features: List[str],
        target_column: str,
        output: Output,
    ):
        self.data = data
        self.sensitive_features = sensitive_features
        self.target_column = target_column
        self.output = output
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self):
        le = LabelEncoder()
        for column in self.data.select_dtypes(include=["object"]):
            self.data[column] = le.fit_transform(self.data[column])

        scaler = StandardScaler()
        self.data["Age"] = scaler.fit_transform(self.data[["Age"]])

        self.data[self.target_column] = (
            self.data[self.target_column] > self.data[self.target_column].median()
        ).astype(int)

    def split_data(self):
        features = [
            col
            for col in self.data.columns
            if col not in [self.target_column, "Customer ID"]
        ]
        X = self.data[features]
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self, model_type="rf"):
        if model_type == "rf":
            self.model = RandomForestClassifier(random_state=42)
        elif model_type == "lr":
            self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def calculate_metrics(self) -> Dict[str, Any]:
        y_pred = self.model.predict(self.X_test)

        metrics = {}
        for sensitive_feature in self.sensitive_features:
            mf = MetricFrame(
                metrics={
                    "selection_rate": selection_rate,
                    "accuracy": lambda y_true, y_pred: (y_true == y_pred).mean(),
                },
                y_true=self.y_test,
                y_pred=y_pred,
                sensitive_features=self.X_test[sensitive_feature],
            )

            metrics[sensitive_feature] = {
                "demographic_parity_difference": demographic_parity_difference(
                    self.y_test,
                    y_pred,
                    sensitive_features=self.X_test[sensitive_feature],
                ),
                "equalized_odds_difference": equalized_odds_difference(
                    self.y_test,
                    y_pred,
                    sensitive_features=self.X_test[sensitive_feature],
                ),
                "selection_rate_disparity": mf.group_max()["selection_rate"]
                - mf.group_min()["selection_rate"],
                "accuracy_disparity": mf.group_max()["accuracy"]
                - mf.group_min()["accuracy"],
            }

        return metrics

    def analyze_feature_importance(self):
        if isinstance(self.model, RandomForestClassifier):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame(
                {"feature": self.X_train.columns, "importance": importance}
            )
            feature_importance = feature_importance.sort_values(
                "importance", ascending=False
            )
            return feature_importance
        elif isinstance(self.model, LogisticRegression):
            importance = np.abs(self.model.coef_[0])
            feature_importance = pd.DataFrame(
                {"feature": self.X_train.columns, "importance": importance}
            )
            feature_importance = feature_importance.sort_values(
                "importance", ascending=False
            )
            return feature_importance
        else:
            return None

    def calculate_shap_values(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        return shap_values

    def detect_bias(self, model_type="rf") -> Dict[str, Any]:
        self.preprocess_data()
        self.split_data()
        self.train_model(model_type)
        metrics = self.calculate_metrics()
        feature_importance = self.analyze_feature_importance()
        shap_values = self.calculate_shap_values()
        return metrics, feature_importance, shap_values

    def generate_report(
        self,
        metrics: Dict[str, Any],
        feature_importance: pd.DataFrame,
        shap_values: np.ndarray,
    ):
        self.output.write("# Kaggle Shopping Trends Bias Detection Report\n")

        for sensitive_feature, feature_metrics in metrics.items():
            self.output.write(f"\n## Bias Metrics for {sensitive_feature}\n")
            for metric_name, value in feature_metrics.items():
                self.output.write(
                    f"- {metric_name.replace('_', ' ').title()}: {value:.4f}\n"
                )

        self.output.write("\n## Feature Importance\n")
        for _, row in feature_importance.head(10).iterrows():
            self.output.write(f"- {row['feature']}: {row['importance']:.4f}\n")

        self.output.write("\n## SHAP Summary\n")
        shap.summary_plot(shap_values, self.X_test, plot_type="bar", show=False)
        plt.savefig(os.path.join(self.output.output_dir, "shap_summary.png"))
        plt.close()
        self.output.write("![SHAP Summary](shap_summary.png)\n")

        self.output.write("\n## Interpretation\n")
        self.output.write(
            "- Demographic Parity Difference: Measures the difference in selection rates between groups. "
            "A value close to 0 indicates better fairness.\n"
        )
        self.output.write(
            "- Equalized Odds Difference: Measures the difference in true positive and false positive rates "
            "between groups. A value close to 0 indicates better fairness.\n"
        )
        self.output.write(
            "- Selection Rate Disparity: The difference between the highest and lowest selection rates "
            "among groups. Lower values indicate more consistent selection across groups.\n"
        )
        self.output.write(
            "- Accuracy Disparity: The difference between the highest and lowest accuracy rates "
            "among groups. Lower values indicate more consistent accuracy across groups.\n"
        )

        self.output.save_to_markdown("kaggle_bias_detection_report.md")


def run_kaggle_bias_detection(file_path: str):
    output_dir = "kaggle_bias_detection_output"
    os.makedirs(output_dir, exist_ok=True)
    output = Output(output_type="both", output_dir=output_dir)

    data = pd.read_csv(file_path)

    detector = KaggleBiasDetector(
        data=data,
        sensitive_features=["Age", "Gender"],
        target_column="Purchase Amount (USD)",
        output=output,
    )

    metrics, feature_importance, shap_values = detector.detect_bias()
    detector.generate_report(metrics, feature_importance, shap_values)

    # Prepare the detection results
    detection_results = {
        "fairness_metrics": metrics,
        "feature_importance": feature_importance.to_dict(),
        "shap_values": (
            shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values
        ),
    }

    # Analyze the results
    ai_analyzer = AIAnalyzer(output)
    analysis = ai_analyzer.analyze_bias_detection(detection_results)

    output.write("\n## AI Analysis\n")
    output.write(analysis)
    output.save_to_markdown("kaggle_ai_analysis_report.md")


if __name__ == "__main__":
    file_path = os.path.join("kaggle_data", "shopping_trends copy.csv")
    run_kaggle_bias_detection(file_path)
