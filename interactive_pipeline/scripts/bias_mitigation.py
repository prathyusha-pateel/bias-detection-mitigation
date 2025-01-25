import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses INFO and WARNING logs
warnings.filterwarnings("ignore", category=UserWarning, message="`load_boston` has been removed")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover, LFR
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
from scripts.bias_detection import BiasDetector
import math
import streamlit as st
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()



class BiasMitigator:
    def __init__(self, task, sensitive_feature_names, fairness_notions, privileged_groups=None, unprivileged_groups=None):
        self.task = task
        self.sensitive_feature_names = sensitive_feature_names
        self.fairness_notions = fairness_notions
        self.results = []
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        
        
    def reweighing(self, X_train, y_train):
        dataset = BinaryLabelDataset(
            df=pd.concat([X_train, y_train], axis=1),
            label_names=["y"],
            protected_attribute_names=self.sensitive_feature_names,
        )
        rw = Reweighing(privileged_groups=self.privileged_groups, unprivileged_groups=self.unprivileged_groups)
        transformed_dataset = rw.fit_transform(dataset)
        #get back the feature names also and return X_train with the column names
        transformed_dataset.features = pd.DataFrame(transformed_dataset.features, columns=X_train.columns)
        return transformed_dataset.features, transformed_dataset.labels.ravel(), transformed_dataset.instance_weights
    

    def disparate_impact_removal(self, X_train, repair_level=1.0):
        di_remover = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute=self.sensitive_feature_names[0])
        return di_remover.fit_transform(X_train)

    def learning_fair_representations(self, X_train, y_train):
        dataset = BinaryLabelDataset(
            df=pd.concat([X_train, y_train], axis=1),
            label_names=["y"],
            protected_attribute_names=self.sensitive_feature_names,
        )
        lfr = LFR(privileged_groups=self.privileged_groups, unprivileged_groups=self.unprivileged_groups, verbose=1)
        transformed_dataset = lfr.fit_transform(dataset)
        return transformed_dataset.features, transformed_dataset.labels

    def adversarial_debiasing(self, X_train, y_train, sess, instance_weights=None, scope_name="adversarial_debiasing"):
        # Ensure X_train is a pandas DataFrame
        
        if len(self.privileged_groups) != 1 or len(self.unprivileged_groups) != 1:
            combined_privileged_groups = {}
            combined_unprivileged_groups = {}
            #create a unique privileged grouo from all combinations
            
        
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
        
        # Ensure y_train is a 1D pandas Series
        if isinstance(y_train, np.ndarray):
            if y_train.ndim == 2 and y_train.shape[1] == 1:  # Flatten 2D array to 1D
                y_train = y_train.flatten()
            y_train = pd.Series(y_train, name="y")
            
        if instance_weights is None:
            dataset = BinaryLabelDataset(
                df=pd.concat([X_train, y_train], axis=1),
                label_names=["y"],
                protected_attribute_names=self.sensitive_feature_names,
            )
        else:
            dataset = BinaryLabelDataset(
                df=pd.concat([X_train, y_train, pd.Series(instance_weights, name="instance_weights")], axis=1),
                label_names=["y"],
                protected_attribute_names=self.sensitive_feature_names,
                instance_weights_name="instance_weights"
            )
        adv = AdversarialDebiasing(
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            scope_name=scope_name,
            debias=True,
            sess=sess
        )
        adv.fit(dataset)
        return adv


    def fairness_regularization(self, X_train, y_train, sensitive_feature):
        pr = PrejudiceRemover(eta=1.0, sensitive_attr=sensitive_feature)
        pr.fit(X_train, y_train)
        return pr

    def equalized_odds_adjustment(self, X_test, y_test, y_pred):
        eo = EqOddsPostprocessing(privileged_groups=self.privileged_groups, unprivileged_groups=self.unprivileged_groups)
        
        dataset_true = BinaryLabelDataset(
            df=pd.concat([X_test, y_test], axis=1),
            label_names=["y"],
            protected_attribute_names=self.sensitive_feature_names,
        )
        dataset_pred = BinaryLabelDataset(
            df=pd.concat([X_test, pd.Series(y_pred, name="y", index=X_test.index)], axis=1),
            label_names=["y"],
            protected_attribute_names=self.sensitive_feature_names,
        )
        y_pred = eo.fit_predict(dataset_true, dataset_pred).labels.ravel()
        return y_pred

    def evaluate_model(self, y_true, y_pred, sensitive_features, y_prob=None, sample_weight=None):
        if isinstance(sensitive_features, pd.DataFrame):
            sensitive_features = sensitive_features.iloc[:, 0].values
        elif isinstance(sensitive_features, pd.Series):
            sensitive_features = sensitive_features.values

        detector = BiasDetector(self.task, sensitive_features, self.fairness_notions)
        group_results, overall_results = detector.calculate_metrics(y_true, y_pred, y_prob, sample_weight)
        return group_results, overall_results

    def run_pipeline(self, X_train, X_test, y_train, y_test, sensitive_feature_names, base_model, methods):
        sensitive_features = X_test[sensitive_feature_names].iloc[:, 0].values
        
        methods["preprocessing"].insert(0,"None")
        methods["inprocessing"].insert(0,"None")
        methods["postprocessing"].insert(0,"None")
        
        pre_methods = methods["preprocessing"]
        in_methods = methods["inprocessing"]
        post_methods = methods["postprocessing"]
        sess = methods.get("sess")

        for pre_method in pre_methods:
            sample_weight = None
            if pre_method == "None":
                X_train_pre, y_train_pre = X_train, y_train
            if pre_method == "Reweighing":
                X_train_pre, y_train_pre, sample_weight= self.reweighing(X_train, y_train)
            elif pre_method == "Disparate Impact Removal":
                X_train_pre = self.disparate_impact_removal(X_train)
            elif pre_method == "Learning Fair Representations":
                X_train_pre, y_train_pre = self.learning_fair_representations(X_train, y_train)

            for in_method in in_methods:
                y_pred = None
                
                if in_method == "None" and pre_method == "None":
                    model = clone(base_model)
                    model.fit(X_train_pre, y_train_pre)
                    y_pred = model.predict(X_test)
                    
                elif in_method == "None" and pre_method == "Reweighing":
                    model = clone(base_model)
                    model.fit(X_train_pre, y_train_pre, sample_weight=sample_weight)
                    y_pred = model.predict(X_test)
                    print("ypred",y_pred.shape)
                    
                elif in_method == "Adversarial Debiasing" :
                    scope_name = f"adversarial_debiasing_{pre_method.split(' ')[0]}_{post_method.split(' ')[0]}"
                    
                    model = self.adversarial_debiasing(
                            X_train_pre, y_train_pre, sess, instance_weights=sample_weight, scope_name=scope_name
                    )
                    # Ensure BinaryLabelDataset for evaluation if needed
                    dataset_test = BinaryLabelDataset(
                    df=pd.concat([X_test, y_test], axis=1),
                    label_names=["y"],
                    protected_attribute_names=self.sensitive_feature_names,
                )
                    y_pred = model.predict(dataset_test).labels.ravel()
                    
                elif in_method == "Fairness Regularization":
                    model = self.fairness_regularization(X_train_pre, y_train_pre, sensitive_feature_names[0])
                    y_pred = model.predict(X_test)

                sensitive_features = X_test[sensitive_feature_names]
                
                for post_method in post_methods:
                    st.write(f"Preprocessing Technique: {pre_method}")
                    st.write(f"In-processing Technique: {in_method}")
                    st.write(f"Post-processing Technique: {post_method}")
                    print(f"Preprocessing Technique: {pre_method}")
                    print(f"In-processing Technique: {in_method}")
                    print(f"Post-processing Technique: {post_method}")

                    if post_method == "None":
                        y_pred_post = y_pred
                    if post_method == "Equalized Odds Adjustment":
                        y_pred_post = self.equalized_odds_adjustment(X_test,y_test, y_pred)

                    group_metrics, overall_metrics = self.evaluate_model(y_test, y_pred_post, y_prob=None, sensitive_features=sensitive_features, sample_weight=None)
                    
                    st.dataframe(group_metrics)
                    st.dataframe(overall_metrics)

                    self.results.append({
                        "Preprocessing": pre_method,
                        "Inprocessing": in_method,
                        "Postprocessing": post_method,
                        "Group Metrics": group_metrics,
                        "Overall Metrics": overall_metrics,
                    })

    def plot_metrics(self, metrics_to_plot):
    
        # Iterate through each metric
        for metric in metrics_to_plot:
            data = {f"{res['Preprocessing']}+{res['Inprocessing']}+{res['Postprocessing']}": res["Overall Metrics"][metric]
                    for res in self.results}

            # Sort data by values in descending order
            sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
            keys = list(sorted_data.keys())
            values = list(sorted_data.values())

            # Create a separate figure for each metric
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Horizontal bar plot
            ax.barh(keys, values, color='skyblue')

            # Set title and labels
            ax.set_title(f"Comparison of {metric} Across Methods", fontsize=14)
            ax.set_xlabel(metric, fontsize=12)
            ax.set_ylabel("Methods", fontsize=12)

            # Add value labels on the bars
            for i, v in enumerate(values):
                ax.text(v / 2, i, str(round(v, 2)), ha="center", va="center", fontsize=8, color="black")

            st.pyplot(fig)
            plt.savefig(f"interactive_pipeline/plots/{metric}.png")
            





import tensorflow as tf

def main():
    np.random.seed(42)
    from sklearn.model_selection import train_test_split
    X = pd.DataFrame(np.random.rand(1000, 5), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], size=1000), name="y")
    sensitive_feature = pd.Series(np.random.choice([0, 1], size=1000), name="sensitive_feature")
    X["sensitive_feature"] = sensitive_feature

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    sensitive_features = ["sensitive_feature"]
    task = "Classification"
    fairness_notions = ["Demographic Parity", "Equalized Odds"]
    base_model = RandomForestClassifier(random_state=42)
    metrics_to_plot = ["Accuracy", "Equalized Odds Difference", "Demographic Parity Difference"]

    methods = {
        "preprocessing": ["Reweighing"],
        "inprocessing": ["Adversarial Debiasing"],
        "postprocessing": ["Equalized Odds Adjustment"],
        "privileged_groups": [{"sensitive_feature": 1}],
        "unprivileged_groups": [{"sensitive_feature": 0}],
    }

    # Create a TensorFlow session for Adversarial Debiasing
    with tf.compat.v1.Session() as sess:
        methods["sess"] = sess  # Add session to methods dictionary
        sensitive_feature = pd.Series(np.random.choice([0, 1], size=1000), name="sensitive_feature")
        sensitive_features = ["sensitive_feature"]


        mitigator = BiasMitigator(task, sensitive_features, fairness_notions, privileged_groups=methods["privileged_groups"], unprivileged_groups=methods["unprivileged_groups"])
        mitigator.run_pipeline( X_train, X_test, y_train, y_test,sensitive_features, base_model, methods)
        mitigator.plot_metrics(metrics_to_plot)


if __name__ == "__main__":
    main()
