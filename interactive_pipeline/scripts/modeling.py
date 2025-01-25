import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN
import traceback
import streamlit as st
import numpy as np

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, model_list, scoring, cv=5):
        """
        Initialize ModelTrainer with model choices and custom hyperparameter grids.
        
        Parameters:
            model_list (list): List of model names (str) to train and evaluate.
            custom_params (dict): Optional custom parameter grids for models.
            scoring (str or dict): Scoring metric(s) for GridSearchCV.
            cv (int): Number of cross-validation folds.
        """
        self.model_list = model_list
        self.scoring = scoring
        self.cv = cv
        self.models, self.param_grids = self._prepare_models_and_params()
        self.final_hyperparameters = {}
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def _prepare_models_and_params(self):
        model_instances = {
            'Random Forest Classifier': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(),
            'Support Vector Machine': SVC(),
            'K-Nearest Neighbors Classifier': KNeighborsClassifier(),
            'Random Forest Regressor': RandomForestRegressor(),
            'Ridge Regression': Ridge(),
            'Decision Tree Classifier': DecisionTreeClassifier(),
            'Decision Tree Regressor': DecisionTreeRegressor(),
            'KMeans': KMeans(),
            'DBSCAN': DBSCAN()
        }

        default_param_grids = {
            'Random Forest Classifier': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
            'Logistic Regression': {'C': [0.1]},
            'Support Vector Machine': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
            'K-Nearest Neighbors Classifier': {'n_neighbors': [3, 5, 7]},
            'Random Forest Regressor': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
            'Ridge Regression': {'alpha': [0.1, 1.0, 10.0]},
            'Decision Tree Classifier': {'max_depth': [None, 10, 20]},
            'Decision Tree Regressor': {'max_depth': [None, 10, 20]},
            'KMeans': {'n_clusters': [2, 3, 4]},
            'DBSCAN': {'eps': [0.5, 1.0]}
        }
        
        
        models = {}
        param_grids = {}
        for model_name in self.model_list:
            if model_name not in model_instances:
                raise ValueError(f"Invalid model name: {model_name}")
            elif model_name not in default_param_grids:
                raise ValueError(f"Parameter grid not available for {model_name}")
            else:
                models[model_name] = model_instances[model_name]
                param_grids[model_name] = default_param_grids[model_name]

        return models, param_grids
    
    def set_params(self, model_name, params):
        if model_name in self.models:
            return self.models[model_name].set_params(**params)
            
    
        
    def tune_and_train_model(self, model_name, model):
        try:
            param_grid = self.param_grids.get(model_name, {})
            grid_search = GridSearchCV(model, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            #print(f"Best params for {model_name}: {grid_search.best_params_}")
            self.final_hyperparameters[model_name] = grid_search.best_params_
            return grid_search.best_estimator_, grid_search.best_score_
        except Exception as e:
            #print(f"Error tuning {model_name}: {e}")
            traceback.print_exc()
            return None, None

    def evaluate_model(self, model, task):
        try:
            predictions = model.predict(self.X_test)
            if task == 'Classification':
                class_dict = classification_report(self.y_test, predictions, output_dict=True)
                metrics = {
                    "accuracy": accuracy_score(self.y_test, predictions),
                    "precision": class_dict['weighted avg']['precision'],
                    "recall": class_dict['weighted avg']['recall'],
                    "f1": class_dict['weighted avg']['f1-score']
                }
                
            elif task == 'Regression':
                metrics = {
                    "Mean Squared Error": mean_squared_error(self.y_test, predictions),
                    "R² Score": r2_score(self.y_test, predictions)
                }
            else:
                metrics = {}
            
            return metrics
        except Exception as e:
            #print(f"Error evaluating model: {e}")
            traceback.print_exc()
            return None
    

    def plot_model_comparison(self, scores, task):
        """
        Creates a horizontal bar plot comparing model scores.

        Args:
            scores (dict): A dictionary with model names as keys and their performance scores as values.
            task (str): The type of task ('Classification' or 'Regression') to label the x-axis.

        Returns:
            fig (matplotlib.figure.Figure): The generated plot as a matplotlib figure object.
        """
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Unpack model names and scores
        model_names = list(scores.keys())
        model_scores = list(scores.values())
        
        # Plot the horizontal bar chart
        ax.barh(model_names, model_scores, color='skyblue')
        ax.set_xlabel("Accuracy" if task == "Classification" else "R² Score")
        ax.set_title(f"Model Comparison for {task} Task")
        
        # Return the figure for rendering in Streamlit or other applications
        return fig
    
    def plot_confusion_matrix(self, cm, model_name):
        """
        Creates a heatmap for the confusion matrix.

        Args:
            cm (array-like): Confusion matrix values.
            model_name (str): Name of the model for labeling the plot.

        Returns:
            fig (matplotlib.figure.Figure): The generated plot as a matplotlib figure object.
        """
        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix for {model_name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
        # Return the figure for rendering in Streamlit or other applications
        return fig

    def plot_residuals(self, y_test, predictions, model_name):
        """
        Creates a histogram with a kernel density estimate (KDE) for the residuals.

        Args:
            y_test (array-like): Actual values from the test set.
            predictions (array-like): Predicted values from the model.
            model_name (str): Name of the model for labeling the plot.

        Returns:
            fig (matplotlib.figure.Figure): The generated plot as a matplotlib figure object.
        """
        # Calculate residuals
        residuals = y_test - predictions
        
        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram and KDE for residuals
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Residual Plot for {model_name}")
        
        # Return the figure for rendering
        return fig
    
    
    


    def train_and_select_best_model(self, task):
        best_score = float('-inf') if task == 'Regression' else 0
        best_model_name = None
        best_model = None
        best_metrics = None
        scores = {}

        for model_name, model in self.models.items():
            #print(f"\nTraining and tuning {model_name}...")
            tuned_model, score = self.tune_and_train_model(model_name, model)

            if tuned_model:
                metrics = self.evaluate_model(tuned_model, task=task)
                primary_metric = metrics.get("R² Score" if task == "Regression" else "accuracy")
                scores[model_name] = primary_metric

                if task == "Classification":
                    cm = metrics["Confusion Matrix"]
                    self.plot_confusion_matrix(cm, model_name)
                elif task == "Regression":
                    predictions = tuned_model.predict(self.X_test)
                    self.plot_residuals(self.y_test, predictions, model_name)

                if (task == 'Regression' and primary_metric > best_score) or \
                    (task == 'Classification' and primary_metric > best_score):
                    best_score = primary_metric
                    best_model_name = model_name
                    best_model = tuned_model
                    best_metrics = metrics

                #print(f"{model_name} - Test Performance: {primary_metric}")

        self.plot_model_comparison(scores, task)
        #print(f"\nBest Model: {best_model_name} with Performance: {best_score}")
        return best_model_name, best_model, best_metrics
        
            
# Main Script
if __name__ == "__main__":
    task_type = "Classification"  # or "Regression"
    from sklearn.model_selection import train_test_split
    
    if task_type == "Classification":
        data = pd.read_csv('interactive_pipeline/data/bank-additional-full_preprocessed.csv')
        df = data.copy()
        target_column = 'y'
        X = np.ascontiguousarray(df.drop(columns=[target_column]).values)
        y = np.ascontiguousarray(df[target_column].values)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models_to_train = ['Logistic Regression', 'K-Nearest Neighbors Classifier']
        scoring = 'accuracy'
    elif task_type == "Regression":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        target_column = 'target'
        models_to_train = ['Random Forest Regressor', 'Ridge Regression', 'Decision Tree Regressor']
        scoring = 'r2'

    
    trainer = ModelTrainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_list=models_to_train,
        scoring=scoring,
        cv=3
    )
    
    best_model_name, best_model, best_metrics = trainer.train_and_select_best_model(task=task_type)
    print(f"\nBest Model: {best_model_name}")