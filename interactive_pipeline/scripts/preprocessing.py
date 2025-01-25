#preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
import logging
from typing import Optional, List, Dict, Union
from sklearn.model_selection import train_test_split
import streamlit as st

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetProcessor:
    """
    Class to manage and preprocess a dataset with individual transformation steps.
    """

    def __init__(self, data: str):
        try:
            self.data = data
            self.encoders = {}
            self.scalers = {}
            self.imputers = {}
            self.binners = {}
            self.bin_edges = {}
            
        except Exception as e:
            logging.error(f"Error in DatasetProcessor.__init__(): {e}")
            raise

    def handle_missing_values(self, col: str, strategy: str, fill_value: Optional[Union[int, float, str]]):
        """
        Handle missing values for a specified column with the given strategy.

        Parameters:
        - col (str): The column to process for missing values.
        - strategy (str): The strategy to handle missing values (e.g., "mean", "median", "most_frequent", "constant", or "drop").
        - fill_value (any): The value to fill when strategy is "constant".
        """
        try:
            if strategy == "drop":
                # Drop rows with missing values in the specified column
                self.data.dropna(subset=[col], inplace=True)
            elif strategy == "constant":
                # Apply imputation based on specified strategy
                imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                self.data[col] = imputer.fit_transform(self.data[[col]]).ravel()
                self.imputers[col] = imputer
            elif strategy in ["mean", "median", "most_frequent"]:
                # Apply imputation based on specified strategy
                imputer = SimpleImputer(strategy=strategy)
                self.data[col] = imputer.fit_transform(self.data[[col]]).ravel()
                self.imputers[col] = imputer
        except Exception as e:
            logging.error(f"Error in handle_missing_values() for column '{col}': {e}")
            raise

    def handle_outliers(self, col: str, method: str):
        try:
            if method == "Clip":
                lower_bound, upper_bound = self.data[col].quantile([0.05, 0.95])
                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == "Remove":
                if self.data[col].dtype == np.number:
                    q1, q3 = self.data[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
                else:
                    raise ValueError(f"Column '{col}' is not numeric and cannot be processed for outliers.")
        except Exception as e:
            logging.error(f"Error in handle_outliers() for column '{col}': {e}")
            raise

    def apply_scaling(self, col: str, scaling_method: str):
        try:
            if col not in self.scalers:
                if scaling_method == "Standard":
                    self.scalers[col] = StandardScaler()
                elif scaling_method == "Min-Max":
                    self.scalers[col] = MinMaxScaler()
                elif scaling_method == "Robust":
                    self.scalers[col] = RobustScaler()
            self.data[col] = self.scalers[col].fit_transform(self.data[[col]])
        except Exception as e:
            logging.error(f"Error in apply_scaling() for column '{col}': {e}")
            raise

    def apply_encoding(self, col: str, encoding_method: str):
        try:
            if encoding_method == "One-Hot":
                if col not in self.encoders:
                    # Initialize and fit the encoder only if it hasn't been created yet
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    encoder.fit(self.data[[col]])  # Fit encoder on the full data
                    self.encoders[col] = encoder

                encoder = self.encoders[col]
                encoded_data = encoder.transform(self.data[[col]])
                
                # Add encoded columns, ensure alignment
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))
                self.data = self.data.drop(columns=[col])
                self.data = pd.concat([self.data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

                # Add any missing columns (e.g., from OneHotEncoder categories)
                expected_cols = encoder.get_feature_names_out([col])
                for missing_col in expected_cols:
                    if missing_col not in self.data.columns:
                        self.data[missing_col] = 0
                        
            elif encoding_method == "Label":
                if col not in self.encoders:
                    # Initialize and fit the label encoder only if it hasn't been created yet
                    encoder = LabelEncoder()
                    encoder.fit(self.data[col])
                    self.encoders[col] = encoder

                encoder = self.encoders[col]
                self.data[col] = encoder.transform(self.data[col])
            else:
                raise ValueError(f"Encoding method '{encoding_method}' is not supported.")
        except Exception as e:
            logging.error(f"Error in apply_encoding() for column '{col}': {e}")
            raise


    def apply_binning(self, col: str, method: str, bins: int, encoding_method: str):
        """
        Apply binning to a column.

        Parameters:
        - col (str): Column to apply binning.
        - method (str): Binning method ("uniform" or "quantile").
        - bins (int): Number of bins to apply.
        """
        try:
            if col not in self.binners:
                # Initialize and fit the binning only if it hasn't been created yet
                binning = KBinsDiscretizer(n_bins=bins, encode= encoding_method, strategy=method)
                binning.fit(self.data[[col]])
                self.binners[col] = binning
                bin_edges = binning.bin_edges_[0]
                bin_ranges = {i: f"{bin_edges[i]} - {bin_edges[i+1]}" for i in range(len(bin_edges) - 1)}
                self.bin_edges[col] = bin_ranges
                
            binning = self.binners[col]
            binned_data = binning.transform(self.data[[col]])
            binned_cols = binning.get_feature_names_out([col])
            binned_df = pd.DataFrame(binned_data, columns=binned_cols)
            self.data = self.data.drop(columns=[col])
            self.data = pd.concat([self.data.reset_index(drop=True), binned_df.reset_index(drop=True)], axis=1)
            print(f"Bin edges: {binning.bin_edges_}")

        except Exception as e:
            logging.error(f"Error in apply_binning() for column '{col}': {e}")
            raise

    def inverse_transform(self, cols: Optional[List[str]]):
        """
        Perform inverse transformation for encoded or scaled columns.

        Updates self.data by replacing encoded or scaled columns with their original values.
        """
        try:
            for col, encoder in self.encoders.items():
                if col in cols:
                    if isinstance(encoder, OneHotEncoder):
                        # Handle one-hot encoded columns
                        encoded_cols = encoder.get_feature_names_out([col])
                        encoded_data = self.data[encoded_cols]
                        original_data = encoder.inverse_transform(encoded_data)
                        self.data[col] = original_data.flatten()
                        self.data = self.data.drop(columns=encoded_cols)

                    elif isinstance(encoder, LabelEncoder):
                        # Handle label-encoded columns
                        if col in self.data.columns:
                            self.data[col] = encoder.inverse_transform(self.data[col].astype(int))
                
            for col,edges in self.bin_edges.items():
                if col in cols:
                    for i in self.data[col]:
                        for key, value in edges.items():
                            if i == key:
                                self.data[col] = self.data[col].replace(i,value)
                    
        except Exception as e:
            logging.error(f"Error in inverse_transform: {e}")
            raise

    def apply_custom_preprocessing(self, custom_preprocessing: Dict[str, Dict[str, Union[str, int]]]):
        """
        Apply custom preprocessing steps to each column as specified in the custom_preprocessing dictionary.

        Parameters:
        - custom_preprocessing (dict): Dictionary containing preprocessing steps for each column.
        """
        for col, steps in custom_preprocessing.items():
            # Handle missing values
            if steps.get("missing_values"):
                strategy = steps["missing_values"].get("strategy")
                fill_value = steps["missing_values"].get("fill_value")
                self.handle_missing_values(col, strategy=strategy, fill_value=fill_value)
            
            # Handle outliers
            if steps.get("outliers"):
                self.handle_outliers(col, steps["outliers"])

            # Apply binning
            if steps.get("binning"):
                binning_params = steps["binning"]
                method = binning_params.get("method")
                bins = binning_params.get("bins")
                encoding_method = binning_params.get("encoding_method")
                self.apply_binning(col, method=method, bins=bins, encoding_method=encoding_method)
                # st.write(f"Note: Binning edges for {col} are given below.")
                # st.dataframe(self.bin_edges[col])

            # Apply encoding
            if steps.get("encoding"):
                self.apply_encoding(col, steps["encoding"])

            # Apply scaling
            if steps.get("scaling"):
                self.apply_scaling(col, steps["scaling"])

        logging.info("Custom preprocessing applied to dataset.")

# Main execution
if __name__ == "__main__":
    # Load sample dataset
    data = pd.read_csv("interactive_pipeline/data/shopping_trends_copy.csv")
    features = ["Age", "Gender", "Season", "Purchase Amount (USD)"]
    data = data[features]

    # Split data into train and test
    X = data.drop(columns=["Gender"])
    y = data["Gender"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.to_frame()
    print("Unprocessed Training Data:")
    print(X_train)
    y_test = y_test.to_frame()
    print("Unprocessed Test Data:")
    print(X_test)
    # Apply preprocessing
    processor = DatasetProcessor(X_train)  # Initialize with training data
    steps = {"Season": {"encoding": "Label"}}
    processor.apply_custom_preprocessing(steps)

    # Apply the same encoder to test data
    test_processor = DatasetProcessor(X_test)
    test_processor.encoders = processor.encoders
    test_processor.scalers = processor.scalers
    test_processor.imputers = processor.imputers
    test_processor.binners = processor.binners
    test_processor.bin_edges = processor.bin_edges
    test_processor.apply_custom_preprocessing(steps)

    # Print results
    
    print("Processed Training Data:")
    print(processor.data)
   
    print("\nProcessed Test Data:")
    print(test_processor.data)
    
    # Inverse transform encoded data
    test_processor.inverse_transform(X_test.columns)
    print("Inverse transformed data:")
    print(test_processor.data)
    