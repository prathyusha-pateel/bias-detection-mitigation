import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from typing import Dict, Tuple, List, Any, Optional
from output import Output
from model import ModelManager
from config import DataProcessingError


class DetectionError(DataProcessingError):
    """Base exception class for detection errors."""

    pass


class BiasDetector:
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        features: List[str],
        sensitive_features: List[str],
        output: Output,
    ):
        if data.empty:
            raise DetectionError("Input data cannot be empty")
        if not all(
            col in data.columns
            for col in [target_column] + features + sensitive_features
        ):
            raise DetectionError("Missing required columns in input data")

        self.data = data.copy()
        self.target_column = target_column
        self.features = features
        self.sensitive_features = sensitive_features
        self.output = output
        self.model_manager = ModelManager()
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def preprocess_data(self) -> None:
        try:
            # Enhanced preprocessing for new distributions
            le = LabelEncoder()
            categorical_columns = self.data.select_dtypes(include=["object"]).columns

            # Handle categorical features
            for col in categorical_columns:
                if col in self.sensitive_features:
                    # Binary encoding for sensitive features
                    self.data[col] = le.fit_transform(self.data[col])
                else:
                    # One-hot encoding for other categorical features
                    dummies = pd.get_dummies(self.data[col], prefix=col)
                    self.data = pd.concat([self.data, dummies], axis=1)
                    self.data.drop(col, axis=1, inplace=True)

            # Handle numerical features
            numerical_columns = self.data.select_dtypes(
                include=["float64", "int64"]
            ).columns
            for col in numerical_columns:
                if col != self.target_column:
                    # Robust scaling for numerical features
                    median = self.data[col].median()
                    iqr = self.data[col].quantile(0.75) - self.data[col].quantile(0.25)
                    if iqr > 0:
                        self.data[col] = (self.data[col] - median) / iqr

            # Handle missing values with forward fill and backward fill
            self.data = self.data.ffill()
            self.data = self.data.bfill()
            self.data = self.data.fillna(0)  # Fill any remaining NaNs

            # Target variable preprocessing
            if self.target_column in self.data.columns:
                median_target = self.data[self.target_column].median()
                self.data[self.target_column] = (
                    self.data[self.target_column] > median_target
                ).astype(int)

        except Exception as e:
            raise DetectionError(f"Data preprocessing failed: {str(e)}")

    def split_data(self) -> None:
        try:
            # Keep sensitive features in X
            selected_features = self.features + self.sensitive_features

            # Handle one-hot encoding while preserving sensitive features
            X = pd.DataFrame()
            for col in selected_features:
                if col in self.sensitive_features:
                    X[col] = self.data[col]
                else:
                    if col in self.data.columns:
                        X[col] = self.data[col]
                    else:
                        # Handle one-hot encoded columns
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
            raise DetectionError(f"Data splitting failed: {str(e)}")


class SupervisedBiasDetector(BiasDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.y_pred: Optional[np.ndarray] = None
        self.metric_frame: Optional[MetricFrame] = None

    def train_model(self) -> None:
        try:
            if self.X_train is None or self.y_train is None:
                raise DetectionError("Training data not initialized")
            self.model = self.model_manager.train_supervised(self.X_train, self.y_train)
        except Exception as e:
            raise DetectionError(f"Model training failed: {str(e)}")

    def make_predictions(self) -> None:
        try:
            if self.model is None or self.X_test is None:
                raise DetectionError("Model or test data not initialized")
            self.y_pred = self.model_manager.predict_supervised(self.model, self.X_test)
        except Exception as e:
            raise DetectionError(f"Prediction failed: {str(e)}")

    def calculate_metrics(self) -> None:
        try:
            if any(v is None for v in [self.X_test, self.y_test, self.y_pred]):
                raise DetectionError("Required data not initialized")

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
                sensitive_features=sensitive_features_dict,  # Use the dictionary instead of DataFrame
            )
        except Exception as e:
            raise DetectionError(f"Metrics calculation failed: {str(e)}")

    def get_results(self) -> Dict[str, Any]:
        try:
            if any(
                v is None
                for v in [self.y_test, self.y_pred, self.metric_frame, self.X_test]
            ):
                raise DetectionError("Required data not initialized")

            # Convert metrics_by_group to use string keys instead of tuples
            metrics_by_group = {}
            for index, value in self.metric_frame.by_group.items():
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

            # Enhanced results compilation
            results = {
                "overall_metrics": self.metric_frame.overall.to_dict(),
                "metrics_by_group": metrics_by_group,
                "demographic_parity_difference": demographic_parity_difference(
                    self.y_test,
                    self.y_pred,
                    sensitive_features=self.X_test[self.sensitive_features],
                ),
                "equalized_odds_difference": equalized_odds_difference(
                    self.y_test,
                    self.y_pred,
                    sensitive_features=self.X_test[self.sensitive_features],
                ),
            }

            # Add group-specific demographic parity
            for sensitive_feature in self.sensitive_features:
                feature_values = np.unique(self.X_test[sensitive_feature])
                group_metrics = {}
                for value in feature_values:
                    mask = self.X_test[sensitive_feature] == value
                    group_metrics[str(value)] = {  # Convert value to string
                        "selection_rate": np.mean(self.y_pred[mask]),
                        "base_rate": np.mean(self.y_test[mask]),
                    }
                results[f"{sensitive_feature}_group_metrics"] = group_metrics

            return results

        except Exception as e:
            raise DetectionError(f"Results compilation failed: {str(e)}")

    def run_detection(self) -> Dict[str, Any]:
        self.output.write("Running supervised bias detection...")
        try:
            self.preprocess_data()
            self.split_data()
            self.train_model()
            self.make_predictions()
            self.calculate_metrics()
            results = self.get_results()
            self.output.write("Supervised bias detection complete.")
            return results
        except Exception as e:
            error_msg = f"Supervised bias detection failed: {str(e)}"
            self.output.write(error_msg)
            raise DetectionError(error_msg)


class UnsupervisedBiasDetector(BiasDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.y_pred: Optional[np.ndarray] = None
        self.silhouette: Optional[float] = None
        self.calinski_harabasz: Optional[float] = None
        self.cluster_bias: Optional[pd.Series] = None

    def train_model(self) -> None:
        try:
            if self.X_train is None:
                raise DetectionError("Training data not initialized")
            self.model = self.model_manager.train_unsupervised(self.X_train)
        except Exception as e:
            raise DetectionError(f"Unsupervised model training failed: {str(e)}")

    def make_predictions(self) -> None:
        try:
            if self.model is None or self.X_test is None:
                raise DetectionError("Model or test data not initialized")
            self.y_pred = self.model_manager.predict_unsupervised(
                self.model, self.X_test
            )
        except Exception as e:
            raise DetectionError(f"Cluster prediction failed: {str(e)}")

    def calculate_metrics(self) -> None:
        try:
            if any(v is None for v in [self.X_test, self.y_pred]):
                raise DetectionError("Required data not initialized")

            # Calculate clustering quality metrics
            self.silhouette = silhouette_score(self.X_test, self.y_pred)
            self.calinski_harabasz = calinski_harabasz_score(self.X_test, self.y_pred)

            # Enhanced cluster bias calculation
            cluster_bias_metrics = {}
            for sensitive_feature in self.sensitive_features:
                # Calculate distribution of sensitive feature values in each cluster
                cluster_props = (
                    pd.DataFrame(
                        {
                            "cluster": self.y_pred,
                            "sensitive_feature": self.X_test[sensitive_feature],
                        }
                    )
                    .groupby("cluster")["sensitive_feature"]
                    .value_counts(normalize=True)
                    .unstack()
                )

                # Calculate bias metrics for each cluster
                max_diff = cluster_props.max() - cluster_props.min()
                std_dev = cluster_props.std()
                entropy = -(cluster_props * np.log(cluster_props + 1e-10)).sum()

                cluster_bias_metrics[sensitive_feature] = {
                    "max_difference": max_diff.to_dict(),
                    "std_deviation": std_dev.to_dict(),
                    "entropy": entropy,
                }

            self.cluster_bias = pd.DataFrame(cluster_bias_metrics)

        except Exception as e:
            raise DetectionError(f"Metrics calculation failed: {str(e)}")

    def get_results(self) -> Dict[str, Any]:
        try:
            if any(
                v is None
                for v in [self.silhouette, self.calinski_harabasz, self.cluster_bias]
            ):
                raise DetectionError("Required metrics not calculated")

            results = {
                "clustering_metrics": {
                    "silhouette_score": self.silhouette,
                    "calinski_harabasz_score": self.calinski_harabasz,
                },
                "cluster_bias": self.cluster_bias.to_dict(),
                "cluster_sizes": pd.Series(self.y_pred).value_counts().to_dict(),
            }

            # Add cross-cluster analysis
            for sensitive_feature in self.sensitive_features:
                cross_cluster = pd.crosstab(
                    self.y_pred, self.X_test[sensitive_feature], normalize="index"
                )
                results[f"{sensitive_feature}_distribution"] = cross_cluster.to_dict()

            return results

        except Exception as e:
            raise DetectionError(f"Results compilation failed: {str(e)}")

    def run_detection(self) -> Dict[str, Any]:
        self.output.write("Running unsupervised bias detection...")
        try:
            self.preprocess_data()
            self.split_data()
            self.train_model()
            self.make_predictions()
            self.calculate_metrics()
            results = self.get_results()
            self.output.write("Unsupervised bias detection complete.")
            return results
        except Exception as e:
            error_msg = f"Unsupervised bias detection failed: {str(e)}"
            self.output.write(error_msg)
            raise DetectionError(error_msg)


class BiasDetectionPipeline:
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        features: List[str],
        sensitive_features: List[str],
        output: Output,
    ):
        self.supervised_detector = SupervisedBiasDetector(
            data, target_column, features, sensitive_features, output
        )
        self.unsupervised_detector = UnsupervisedBiasDetector(
            data, target_column, features, sensitive_features, output
        )
        self.output = output

    def run_detection(self) -> Dict[str, Any]:
        try:
            supervised_results = self.supervised_detector.run_detection()
            unsupervised_results = self.unsupervised_detector.run_detection()

            return {
                "supervised": supervised_results,
                "unsupervised": unsupervised_results,
            }
        except Exception as e:
            error_msg = f"Bias detection pipeline failed: {str(e)}"
            self.output.write(error_msg)
            raise DetectionError(error_msg)

    def _write_supervised_results(self, results: Dict[str, Any]) -> None:
        self.output.write("### Overall Metrics\n")
        for metric, value in results["overall_metrics"].items():
            self.output.write(f"{metric}: {value:.4f}\n")

        self.output.write("\n### Metrics by Group\n")
        for metric, groups in results["metrics_by_group"].items():
            self.output.write(f"{metric}:\n")
            for group, value in groups.items():
                if not pd.isna(value):  # Only write non-NaN values
                    self.output.write(f"  {group}: {value:.4f}\n")

        self.output.write(
            f"\nDemographic Parity Difference: {results['demographic_parity_difference']:.4f}\n"
        )
        self.output.write(
            f"Equalized Odds Difference: {results['equalized_odds_difference']:.4f}\n"
        )

        for feature in self.supervised_detector.sensitive_features:
            if f"{feature}_group_metrics" in results:
                self.output.write(f"\n### {feature} Group Analysis\n")
                for group, metrics in results[f"{feature}_group_metrics"].items():
                    self.output.write(f"\nGroup {group}:\n")
                    for metric, value in metrics.items():
                        self.output.write(f"  {metric}: {value:.4f}\n")

    def _write_unsupervised_results(self, results: Dict[str, Any]) -> None:
        self.output.write("### Clustering Quality Metrics\n")
        metrics = results["clustering_metrics"]
        self.output.write(f"Silhouette Score: {metrics['silhouette_score']:.4f}\n")
        self.output.write(
            f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}\n"
        )

        self.output.write("\n### Cluster Sizes\n")
        for cluster, size in results["cluster_sizes"].items():
            self.output.write(f"Cluster {cluster}: {size} samples\n")

        self.output.write("\n### Cluster Bias Analysis\n")
        for feature, bias_metrics in results["cluster_bias"].items():
            self.output.write(f"\n{feature} Analysis:\n")
            for metric_type, values in bias_metrics.items():
                self.output.write(f"{metric_type}:\n")
                if isinstance(values, dict):
                    for k, v in values.items():
                        self.output.write(f"  {k}: {v:.4f}\n")
                else:
                    try:
                        entropy_value = (
                            values.mean() if hasattr(values, "mean") else values
                        )
                        self.output.write(f"  {float(entropy_value):.4f}\n")
                    except (TypeError, ValueError) as e:
                        self.output.write(f"  Unable to format value: {str(e)}\n")

        # Write cross-cluster distribution analysis
        for feature in self.unsupervised_detector.sensitive_features:
            if f"{feature}_distribution" in results:
                self.output.write(f"\n### {feature} Distribution Across Clusters\n")
                for cluster, dist in results[f"{feature}_distribution"].items():
                    self.output.write(f"\nCluster {cluster}:\n")
                    for value, prop in dist.items():
                        self.output.write(f"  {value}: {prop:.4f}\n")

    def generate_report(self, results: Dict[str, Any]) -> None:
        try:
            self.output.write("# Bias Detection Report\n")
            self.output.write("\n## Supervised Model Results\n")
            self._write_supervised_results(results["supervised"])
            self.output.write("\n## Unsupervised Model Results\n")
            self._write_unsupervised_results(results["unsupervised"])
            self.output.save_to_markdown("bias_detection_report.md")
        except Exception as e:
            raise DetectionError(f"Report generation failed: {str(e)}")


def test_detection() -> None:
    """Test the BiasDetectionPipeline functionality."""
    from generate import SyntheticDataGenerator
    from output import Output
    import numpy as np

    try:
        output = Output(output_type="both", output_dir="bias_detection_output")
        data_generator = SyntheticDataGenerator(output)
        data = data_generator.generate_synthetic_consumer_data(n_samples=1000)

        # Calculate average basket size
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

        pipeline = BiasDetectionPipeline(
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

        results = pipeline.run_detection()
        pipeline.generate_report(results)

    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_detection()
