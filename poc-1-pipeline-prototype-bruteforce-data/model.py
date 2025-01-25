# File: model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from typing import Tuple, List, Any, Union
import pandas as pd
import numpy as np


class ModelManager:
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelManager.

        Args:
            random_state (int): Random state for reproducibility. Defaults to 42.
        """
        self.random_state = random_state
        self.supervised_model = None
        self.unsupervised_model = None

    def get_supervised_model(self) -> RandomForestClassifier:
        """
        Get or create a supervised model (Random Forest Classifier).

        Returns:
            RandomForestClassifier: The supervised model.
        """
        if self.supervised_model is None:
            self.supervised_model = RandomForestClassifier(
                random_state=self.random_state
            )
        return self.supervised_model

    def get_unsupervised_model(self, n_clusters: int = 5) -> KMeans:
        """
        Get or create an unsupervised model (KMeans).

        Args:
            n_clusters (int): Number of clusters for KMeans. Defaults to 5.

        Returns:
            KMeans: The unsupervised model.
        """
        if self.unsupervised_model is None:
            self.unsupervised_model = KMeans(
                n_clusters=n_clusters, random_state=self.random_state, n_init=10
            )
        return self.unsupervised_model

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data into training and testing sets.

        Args:
            X (pd.DataFrame): Feature dataset.
            y (pd.Series): Target variable.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def train_supervised(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RandomForestClassifier:
        """
        Train a supervised model.

        Args:
            X_train (pd.DataFrame): Training feature dataset.
            y_train (pd.Series): Training target variable.

        Returns:
            RandomForestClassifier: The trained supervised model.
        """
        model = self.get_supervised_model()
        model.fit(X_train, y_train)
        return model

    def train_unsupervised(self, X_train: pd.DataFrame) -> KMeans:
        """
        Train an unsupervised model.

        Args:
            X_train (pd.DataFrame): Training feature dataset.

        Returns:
            KMeans: The trained unsupervised model.
        """
        model = self.get_unsupervised_model()
        model.fit(X_train)
        return model

    def predict_supervised(
        self, model: RandomForestClassifier, X: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions using a supervised model.

        Args:
            model (RandomForestClassifier): The trained supervised model.
            X (pd.DataFrame): Feature dataset for prediction.

        Returns:
            np.ndarray: Predictions.
        """
        return model.predict(X)

    def predict_unsupervised(self, model: KMeans, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions (cluster assignments) using an unsupervised model.

        Args:
            model (KMeans): The trained unsupervised model.
            X (pd.DataFrame): Feature dataset for prediction.

        Returns:
            np.ndarray: Cluster assignments.
        """
        return model.predict(X)

    def train_mitigated(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: List[str],
        method: str,
    ) -> Union[ExponentiatedGradient, RandomForestClassifier]:
        """
        Train a model with bias mitigation applied.

        Args:
            X_train (pd.DataFrame): Training feature dataset.
            y_train (pd.Series): Training target variable.
            sensitive_features (List[str]): List of sensitive feature names.
            method (str): Bias mitigation method to use.

        Returns:
            Union[ExponentiatedGradient, RandomForestClassifier]: The trained mitigated model.

        Raises:
            ValueError: If an unknown mitigation method is specified.
        """
        estimator = self.get_supervised_model()

        if method == "demographic_parity":
            constraint = DemographicParity()
            mitigated_model = ExponentiatedGradient(
                estimator=estimator, constraints=constraint
            )
            mitigated_model.fit(
                X_train, y_train, sensitive_features=X_train[sensitive_features]
            )

        elif method == "equalized_odds":
            constraint = EqualizedOdds()
            mitigated_model = ExponentiatedGradient(
                estimator=estimator, constraints=constraint
            )
            mitigated_model.fit(
                X_train, y_train, sensitive_features=X_train[sensitive_features]
            )

        elif method == "reweighing":
            privileged_groups = [{sf: 1} for sf in sensitive_features]
            unprivileged_groups = [{sf: 0} for sf in sensitive_features]

            dataset = BinaryLabelDataset(
                df=X_train.join(y_train),
                label_names=[y_train.name],
                protected_attribute_names=sensitive_features,
            )

            reweigher = Reweighing(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
            reweighed_dataset = reweigher.fit_transform(dataset)

            mitigated_model = estimator
            mitigated_model.fit(
                reweighed_dataset.features, reweighed_dataset.labels.ravel()
            )

        else:
            raise ValueError(f"Unknown mitigation method: {method}")

        return mitigated_model


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification

    try:
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=2,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
        y = pd.Series(y, name="target")

        model_manager = ModelManager()

        # Test supervised model
        X_train, X_test, y_train, y_test = model_manager.split_data(X, y)
        supervised_model = model_manager.train_supervised(X_train, y_train)
        supervised_predictions = model_manager.predict_supervised(
            supervised_model, X_test
        )
        print("Supervised model predictions:", supervised_predictions[:10])

        # Test unsupervised model
        unsupervised_model = model_manager.train_unsupervised(X_train)
        unsupervised_predictions = model_manager.predict_unsupervised(
            unsupervised_model, X_test
        )
        print("Unsupervised model predictions:", unsupervised_predictions[:10])

        # Test mitigated model
        sensitive_features = ["feature_0", "feature_1"]
        mitigated_model = model_manager.train_mitigated(
            X_train, y_train, sensitive_features, "demographic_parity"
        )
        mitigated_predictions = model_manager.predict_supervised(
            mitigated_model, X_test
        )
        print("Mitigated model predictions:", mitigated_predictions[:10])

    except Exception as e:
        print(f"An error occurred: {str(e)}")
