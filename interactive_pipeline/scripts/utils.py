import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Union, Tuple
from sklearn.model_selection import train_test_split
import streamlit as st

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def recommend_models(task: str):
    """
    Recommend machine learning models based on the task.

    Parameters:
    - task (str): Task for which to recommend models.

    Returns:
    - List[str]: Recommended models.
    """
    if task == "Classification":
        return ["Logistic Regression", "Random Forest Classifier", "Support Vector Machine", "K-Nearest Neighbors Classifier"]
    elif task == "Regression":
        return ["Linear Regression", "Random Forest Regressor", "Gradient Boosting"]
    elif task == "Clustering":
        return ["K-Means", "DBSCAN", "Hierarchical Clustering"]
    else:
        return []
    


def recommend_column_preprocessing(data: pd.DataFrame, target_column: str, sensitive_feature_names: List) -> Dict[str, Dict[str, Union[str, int]]]:
    """
    Provide preprocessing recommendations for columns based on model and data types.

    Parameters:
    - model_choices (list): List of model choices that may influence preprocessing.

    Returns:
    - dict: Recommended preprocessing steps for each column.
    """
    recommendations = {}
    
    #identify numeric and categorical features
    numeric_features = data.select_dtypes(include=[np.number]).columns
    categorical_features = data.select_dtypes(include=[object]).columns 
    
    for col in data.columns:
        col_recommendation = {}

        # Recommended missing values handling
        col_recommendation["missing_values"] = {
            "strategy": "Mean" if col in numeric_features else "Mode",
            "fill_value": None
        }
        # Recommended outlier handling
        if col in numeric_features:
            col_recommendation["outliers"] = "Remove" 
        else:
            col_recommendation["outliers"] = None
        # Recommended binning for numeric columns
        if col in numeric_features:
            if col in sensitive_feature_names or col == target_column:
                col_recommendation["binning"] = {"method": None, "bins": None, "encoding_method": None}

        # Recommended encoding for categorical columns
        if col in categorical_features:
            col_recommendation["encoding"] = "Label" if col == target_column  or col in sensitive_feature_names else "One-Hot"
        elif col in numeric_features:
            col_recommendation["encoding"] = "One-Hot" if data[col].nunique() < 10 else None

        # Recommended scaling
        if col in numeric_features and col_recommendation["encoding"] != "One-Hot" and col not in sensitive_feature_names:
            col_recommendation["scaling"] = "Standard"
        else:
            col_recommendation["scaling"] = None

        recommendations[col] = col_recommendation
        
    keys_to_move = [target_column] + sensitive_feature_names
    
    reordered_dict = {key: recommendations[key] for key in keys_to_move}
    reordered_dict.update({key: value for key, value in recommendations.items() if key not in keys_to_move})

    return reordered_dict
