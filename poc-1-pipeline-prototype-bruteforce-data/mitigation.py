import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from typing import Dict, List, Any, Optional, Union
from output import Output
from model import ModelManager
from config import DataProcessingError

import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class MitigationError(DataProcessingError):
    """Base exception class for mitigation errors."""

    pass


class BiasMitigator:
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        features: List[str],
        sensitive_features: List[str],
        output: Output,
    ):
        """Initialize the BiasMitigator with data and configuration."""
        self.data = data
        self.target_column = target_column
        self.features = features
        self.sensitive_features = sensitive_features
        self.output = output
        self.model_manager = ModelManager()
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.y_pred: Optional[np.ndarray] = None
        self.metric_frame: Optional[MetricFrame] = None

    def preprocess_data(self) -> None:
        """Preprocess data with optimized DataFrame operations."""
        try:
            # Pre-allocate columns for categorical variables
            categorical_cols = self.data.select_dtypes(include=["object"]).columns
            encoded_data = {}
            le = LabelEncoder()

            # Process sensitive features first
            for col in self.sensitive_features:
                if col in categorical_cols:
                    encoded_data[col] = le.fit_transform(self.data[col])

            # Process other categorical features
            other_cats = [
                col for col in categorical_cols if col not in self.sensitive_features
            ]
            for col in other_cats:
                dummies = pd.get_dummies(self.data[col], prefix=col)
                for dummy_col in dummies.columns:
                    encoded_data[dummy_col] = dummies[dummy_col]

            # Process numerical features
            numerical_cols = self.data.select_dtypes(
                include=["float64", "int64"]
            ).columns
            for col in numerical_cols:
                if col != self.target_column:
                    series = self.data[col]
                    median = series.median()
                    iqr = series.quantile(0.75) - series.quantile(0.25)
                    if iqr > 0:
                        encoded_data[col] = (series - median) / iqr
                    else:
                        encoded_data[col] = series - median

            # Create new DataFrame at once
            new_data = pd.DataFrame(encoded_data)

            # Add target column
            if self.target_column in self.data.columns:
                median_target = self.data[self.target_column].median()
                new_data[self.target_column] = (
                    self.data[self.target_column] > median_target
                ).astype(int)

            # Replace original data with optimized version
            self.data = new_data.copy()

            # Handle any remaining missing values
            self.data.fillna(0, inplace=True)

        except Exception as e:
            raise DataProcessingError(f"Data preprocessing failed: {str(e)}")

    def split_data(self) -> None:
        """Split the data into training and test sets."""
        try:
            selected_features = self.features + self.sensitive_features
            X = pd.DataFrame()

            for col in selected_features:
                if col in self.sensitive_features:
                    X[col] = self.data[col]
                else:
                    if col in self.data.columns:
                        X[col] = self.data[col]
                    else:
                        prefix = f"{col}_"
                        encoded_cols = [
                            c for c in self.data.columns if c.startswith(prefix)
                        ]
                        if encoded_cols:
                            X[encoded_cols] = self.data[encoded_cols]

            y = self.data[self.target_column]
            self.X_train, self.X_test, self.y_train, self.y_test = (
                self.model_manager.split_data(X, y)
            )
        except Exception as e:
            raise DataProcessingError(f"Data splitting failed: {str(e)}")

    def train_mitigated_model(self, method: str) -> Any:
        """Train model with proper feature name handling."""
        try:
            if method == "reweighing":
                # Store feature names
                feature_names = self.X_train.columns.tolist()

                # Convert to numpy arrays for reweighing
                X_train_array = self.X_train.values
                y_train_array = self.y_train.values

                # Create dataset with protected attributes
                dataset = BinaryLabelDataset(
                    df=self.X_train.join(self.y_train),
                    label_names=[self.y_train.name],
                    protected_attribute_names=self.sensitive_features,
                )

                # Apply reweighing
                privileged_groups = [{sf: 1} for sf in self.sensitive_features]
                unprivileged_groups = [{sf: 0} for sf in self.sensitive_features]

                reweigher = Reweighing(
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups,
                )
                transformed_dataset = reweigher.fit_transform(dataset)

                # Train model with feature names
                model = self.model_manager.get_supervised_model()
                model.fit(
                    pd.DataFrame(transformed_dataset.features, columns=feature_names),
                    transformed_dataset.labels.ravel(),
                )
                return model

            else:
                return self.model_manager.train_mitigated(
                    self.X_train, self.y_train, self.sensitive_features, method
                )

        except Exception as e:
            raise DataProcessingError(f"Model training failed: {str(e)}")

    def make_predictions(self, model: Any) -> np.ndarray:
        """Make predictions using the mitigated model."""
        try:
            return self.model_manager.predict_supervised(model, self.X_test)
        except Exception as e:
            raise DataProcessingError(f"Prediction failed: {str(e)}")

    def calculate_metrics(self) -> None:
        try:
            if any(v is None for v in [self.X_test, self.y_test, self.y_pred]):
                raise MitigationError("Required data not initialized")

            # Convert sensitive features to a dictionary format
            sensitive_features_dict = {}
            for feature in self.sensitive_features:
                sensitive_features_dict[feature] = self.X_test[feature]

            # Enhanced metrics calculation
            def safe_divide(numerator, denominator):
                """Safely divide two numbers, returning 0 if denominator is 0."""
                return numerator / denominator if denominator != 0 else 0

            def true_positive_rate(y_true, y_pred):
                return safe_divide(
                    np.sum((y_true == 1) & (y_pred == 1)), np.sum(y_true == 1)
                )

            def true_negative_rate(y_true, y_pred):
                return safe_divide(
                    np.sum((y_true == 0) & (y_pred == 0)), np.sum(y_true == 0)
                )

            def precision_score(y_true, y_pred):
                return safe_divide(
                    np.sum((y_true == 1) & (y_pred == 1)), np.sum(y_pred == 1)
                )

            metrics = {
                "selection_rate": selection_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "true_positive_rate": lambda y_true, y_pred: (
                    np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
                    if np.sum(y_true == 1) > 0
                    else 0
                ),
                "true_negative_rate": lambda y_true, y_pred: (
                    np.sum((y_true == 0) & (y_pred == 0)) / np.sum(y_true == 0)
                    if np.sum(y_true == 0) > 0
                    else 0
                ),
                "precision": lambda y_true, y_pred: (
                    np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
                    if np.sum(y_pred == 1) > 0
                    else 0
                ),
            }

            self.metric_frame = MetricFrame(
                metrics=metrics,
                y_true=self.y_test,
                y_pred=self.y_pred,
                sensitive_features=sensitive_features_dict,
            )
        except Exception as e:
            raise MitigationError(f"Metrics calculation failed: {str(e)}")

    def get_results(
        self, y_pred: np.ndarray, metric_frame: MetricFrame
    ) -> Dict[str, Any]:
        """Get comprehensive results of bias mitigation."""
        try:
            # Convert metrics_by_group to use string keys instead of tuples
            metrics_by_group = {}
            for index, value in metric_frame.by_group.items():
                # Handle different index structures
                if isinstance(index, tuple):
                    metric_name = index[0]
                    # Join the remaining elements as the group identifier
                    group_identifiers = index[1:]
                    group_str = "_".join(str(g) for g in group_identifiers)
                    key = f"{metric_name}_{group_str}"
                else:
                    key = str(index)

                metrics_by_group[key] = value

            results = {
                "mitigated_metrics_by_group": metrics_by_group,
                "overall_mitigated_metrics": metric_frame.overall.to_dict(),
                "mitigated_demographic_parity_difference": demographic_parity_difference(
                    self.y_test,
                    y_pred,
                    sensitive_features=self.X_test[self.sensitive_features],
                ),
                "mitigated_equalized_odds_difference": equalized_odds_difference(
                    self.y_test,
                    y_pred,
                    sensitive_features=self.X_test[self.sensitive_features],
                ),
            }

            # Add group-specific demographic parity
            for feature in self.sensitive_features:
                feature_values = np.unique(self.X_test[feature])
                group_metrics = {}
                for value in feature_values:
                    mask = self.X_test[feature] == value
                    group_metrics[str(value)] = {  # Convert value to string
                        "selection_rate": np.mean(y_pred[mask]),
                        "base_rate": np.mean(self.y_test[mask]),
                    }
                results[f"{feature}_group_metrics"] = group_metrics

            return results

        except Exception as e:
            raise DataProcessingError(f"Results compilation failed: {str(e)}")

    def run_mitigation(self, method: str) -> Dict[str, Any]:
        """Run the complete bias mitigation pipeline."""
        self.output.write(f"Running bias mitigation using {method} method...")
        try:
            self.preprocess_data()
            self.split_data()
            model = self.train_mitigated_model(method)
            self.y_pred = self.make_predictions(model)
            self.calculate_metrics()
            results = self.get_results(self.y_pred, self.metric_frame)
            self.output.write("Bias mitigation complete.")
            return results
        except Exception as e:
            self.output.write(f"Error in bias mitigation: {str(e)}")
            return {}


class BiasMitigationPipeline:
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        features: List[str],
        sensitive_features: List[str],
        output: Output,
    ):
        """Initialize the BiasMitigationPipeline."""
        self.mitigator = BiasMitigator(
            data, target_column, features, sensitive_features, output
        )
        self.output = output
        self.methods = ["reweighing", "demographic_parity", "equalized_odds"]

    def run_mitigation(self) -> Dict[str, Dict[str, Any]]:
        """Run bias mitigation for all specified methods."""
        results = {}
        for method in self.methods:
            results[method] = self.mitigator.run_mitigation(method)
        return results

    def generate_report(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Generate a comprehensive report of bias mitigation results."""
        try:
            self.output.write("# Bias Mitigation Report\n")

            for method, result in results.items():
                self.output.write(f"\n## {method.replace('_', ' ').title()} Results\n")

                # Metrics by group
                self.output.write("\n### Metrics by Group\n")
                for metric, groups in result["mitigated_metrics_by_group"].items():
                    self.output.write(f"\n#### {metric}\n")
                    for group, value in groups.items():
                        self.output.write(f"{group}: {value:.4f}\n")

                # Overall metrics
                self.output.write("\n### Overall Metrics\n")
                for metric, value in result["overall_mitigated_metrics"].items():
                    self.output.write(f"{metric}: {value:.4f}\n")

                # Fairness metrics
                self.output.write("\n### Fairness Metrics\n")
                self.output.write(
                    f"Demographic Parity Difference: {result['mitigated_demographic_parity_difference']:.4f}\n"
                )
                self.output.write(
                    f"Equalized Odds Difference: {result['mitigated_equalized_odds_difference']:.4f}\n"
                )

                # Group-specific metrics
                for key, metrics in result.items():
                    if key.endswith("_group_metrics"):
                        feature = key.replace("_group_metrics", "")
                        self.output.write(f"\n### {feature} Analysis\n")
                        for group, group_metrics in metrics.items():
                            self.output.write(f"\n#### {group}\n")
                            for metric, value in group_metrics.items():
                                self.output.write(f"{metric}: {value:.4f}\n")

            self.output.save_to_markdown("bias_mitigation_report.md")
        except Exception as e:
            raise DataProcessingError(f"Report generation failed: {str(e)}")

    def compare_methods(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Compare the effectiveness of different bias mitigation methods."""
        try:
            self.output.write("# Bias Mitigation Method Comparison\n")

            # Compare fairness metrics
            fairness_metrics = [
                "demographic_parity_difference",
                "equalized_odds_difference",
            ]
            for metric in fairness_metrics:
                self.output.write(f"\n## {metric.replace('_', ' ').title()}\n")
                self.output.write("| Method | Value |\n|--------|-------|\n")

                for method, result in results.items():
                    metric_key = f"mitigated_{metric}"
                    if metric_key in result:
                        self.output.write(f"| {method} | {result[metric_key]:.4f} |\n")

            # Compare performance metrics
            self.output.write("\n## Performance Metrics\n")
            self.output.write(
                "| Method | Selection Rate | False Positive Rate | False Negative Rate |\n"
            )
            self.output.write(
                "|--------|----------------|-------------------|-------------------|\n"
            )

            for method, result in results.items():
                metrics = result["overall_mitigated_metrics"]
                self.output.write(
                    f"| {method} | {metrics['selection_rate']:.4f} | "
                    f"{metrics['false_positive_rate']:.4f} | "
                    f"{metrics['false_negative_rate']:.4f} |\n"
                )

            self.output.save_to_markdown("bias_mitigation_comparison.md")
        except Exception as e:
            raise DataProcessingError(f"Method comparison failed: {str(e)}")


if __name__ == "__main__":
    # Example usage
    from generate import SyntheticDataGenerator

    try:
        output = Output(output_type="both", output_dir="bias_mitigation_output")
        data_generator = SyntheticDataGenerator(output)
        data = data_generator.generate_synthetic_consumer_data(n_samples=10000)

        # Calculate basket size based on earnings and buying frequency
        frequency_multiplier = {
            "Weekly": 52,
            "Bi-Weekly": 26,
            "Monthly": 12,
            "Quarterly": 4,
        }

        data["Average_Basket_Size"] = data.apply(
            lambda x: x["Weekly_Earnings"]
            * frequency_multiplier.get(x["Buying_Frequency"], 12)
            / 52,
            axis=1,
        )

        pipeline = BiasMitigationPipeline(
            data=data,
            target_column="Average_Basket_Size",
            features=[
                "Age",
                "Weekly_Earnings",
                "Income_Level",
                "Education_Level",
                "Employment_Status",
            ],
            sensitive_features=["Gender", "Race", "Ethnicity"],
            output=output,
        )

        results = pipeline.run_mitigation()
        pipeline.generate_report(results)
        pipeline.compare_methods(results)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
