# app.py
import pandas as pd
import streamlit as st
import itertools
from scripts.utils import recommend_models, recommend_column_preprocessing
from scripts.preprocessing import DatasetProcessor
from scripts.modeling import ModelTrainer
from scripts.bias_detection import BiasDetector
from scripts.suggest_fairness_notions import suggest_fairness_notions
from scripts.suggest_mitigators import FairnessMitigationRecommender
from scripts.bias_mitigation import BiasMitigator
from scripts.suggest_unprivileged_groups import GroupPrivilegeAnalyzer
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import random
# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

disable_eager_execution()

# from scripts.reporting import ReportGenerator

import warnings
# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Title and Introduction
st.title("Interactive Marketing Bias Mitigation Application")
st.write("This application guides you through model selection, bias detection, and bias mitigation processes for marketing datasets.")
st.sidebar.title("User Input Display")
# Step 1: Upload Dataset
st.header("Step 1: Upload Dataset")
file = st.file_uploader("Choose a CSV file", type="csv")

if file:
    # Load dataset and display the first few rows
    data = pd.read_csv(file)
    st.write("## Dataset Preview")
    st.write(data.head())
    
    # Choose features, target Column and sensitive attributes
    st.header("Step 2: Preprocessing and Feature Selection")
    
    st.write("Select the features (columns) you want to use for model training:")

    # Create a list to store selected features
    selected_features = []
    if st.checkbox("Select All Features", key="select_all_features", value=True):
        selected_features = data.columns.tolist()
    else:
        for col in data.columns:
            if st.checkbox(f"{col}", key=f"feature_{col}"):
                selected_features.append(col)
    st.sidebar.write("## Selected Features:")
    # Display selected features
    if selected_features:
        st.sidebar.write(selected_features)
    else:
        st.write("No features selected.")

    # Note: Use selected_features for preprocessing and model training
    data = data[selected_features]

    target_column = st.selectbox("Select Target Column", data.columns, index=0) 
    st.sidebar.write("## Selected Target Column:")
    st.sidebar.write(target_column)
    
    sensitive_feature_names= st.multiselect("Select Sensitive Features", data.columns, default = [data.columns[0]])
    st.sidebar.write("## Selected Sensitive Features:")
    st.sidebar.write(sensitive_feature_names)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Preprocessing Options
    preprocess_required = st.checkbox("## Do you need to preprocess the data?", value=False)
    X_train =train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    if preprocess_required:
        # Detailed Preprocessing Options for Other Columns in Table Format
        st.subheader("Preprocessing Recommendations - Change if needed")
        column_preprocessing_steps = recommend_column_preprocessing(data, target_column, sensitive_feature_names)
        
        # Customize preprocessing for each column in a table-like layout
        custom_preprocessing = {}
        for col, suggested_step in column_preprocessing_steps.items():
            # Initialize column preprocessing dictionary
            custom_preprocessing[col] = {}

            # Set up a row with data type, summary, and preprocessing options in columns
            st.write(f"## Column: {col}")
            
            if col in sensitive_feature_names:
                st.write("This is a sensitive feature. Choose binning if it is a numeric feature.")
            elif col == target_column:
                st.write("This is the target column. Choose Label Encoding if it is a categorical feature. For classification, choose binning if it is a numeric feature.")
                
            col1, col2 = st.columns([2, 2])
            with col1:
                # Display data type and summary
                col_dtype = data[col].dtype
                col_unique_values = data[col].nunique()
                col_summary = data[col].describe()
                st.markdown(f"**Data Type:** {col_dtype}")
                st.write(col_summary)
                numeric_features = data.select_dtypes(include=[np.number]).columns
                if col in numeric_features:
                    st.markdown(f"**Unique Values:** {col_unique_values}")
                
            with col2:
                
                # Missing Values Handling
                missing_value_method = st.selectbox(
                    f"{col} - Missing Values",
                    options=[None, "Drop", "Mean", "Median", "Mode", "Custom"],
                    index=[None, "Drop", "Mean", "Median", "Mode", "Custom"].index(suggested_step.get("missing_values", {}).get("strategy", None))
                )
                if missing_value_method == "Custom":
                    custom_value = st.number_input(f"{col} - Custom Missing Value")
                    custom_preprocessing[col]["missing_values"] = {"strategy": "constant", "fill_value": custom_value}
                elif missing_value_method == "Drop":
                    custom_preprocessing[col]["missing_values"] = {"strategy": "drop", "fill_value": None}
                elif missing_value_method == "Mode":
                    custom_preprocessing[col]["missing_values"] = {"strategy": "most_frequent", "fill_value": None}
                elif missing_value_method != None:
                    custom_preprocessing[col]["missing_values"] = {"strategy": missing_value_method.lower(), "fill_value": None}
                else:
                    custom_preprocessing[col]["missing_values"] = None
            
                # Outlier Handling
                outlier_handling = st.selectbox(
                    f"{col} - Outliers",
                    options=[None, "Clip", "Remove"],
                    index=[None, "Clip", "Remove"].index(suggested_step.get("outliers", None))
                )
                custom_preprocessing[col]["outliers"] = outlier_handling if outlier_handling != None else None

                
                # Binning Options
                if col in numeric_features:
                    if col in sensitive_feature_names or col == target_column:
                        binning_method = st.selectbox(
                            f"{col} - Binning",
                            options=[None, "uniform", "quantile"],
                            index=[None, "uniform", "quantile"].index(
                                suggested_step.get("binning", {}).get("method", None)
                            )
                        )
                        if binning_method != None:
                            num_bins = st.number_input(f"{col} - Bins", min_value=2, step=1)
                            # Mapping user-friendly options to KBinsDiscretizer strategies
                            encoding_method = st.selectbox(
                                f"{col} - Bin Encoding",
                                options=[None, "One-Hot", "Label"],
                                index=[None, "One-Hot", "Label"].index(
                                    suggested_step.get("binning", {}).get("encoding_method", None)
                                )
                            )
                            encode_map = {"One-Hot": "onehot-dense", "Label": "ordinal"}
                            custom_preprocessing[col]["binning"] = {
                                "method": binning_method,
                                "bins": num_bins,
                                "encoding_method": encode_map.get(encoding_method, None)
                            }
                        else:
                            custom_preprocessing[col]["binning"] = None
            
                # Encoding Options (for all columns)
                categorical_features = data.select_dtypes(include=[object]).columns
            
                
                if col in numeric_features and (col in sensitive_feature_names or col == target_column) and binning_method != None:
                    print("Binning method is selected")
                else:
                    if col in numeric_features:    
                        st.write("Note: For numeric features, the default encoding is none. Change it to one-hot if you want to convert it to a categorical feature.")
                    encoding_method = st.selectbox(
                        f"{col} - Encoding",
                        options=[None, "One-Hot", "Label"],
                        index=[None, "One-Hot", "Label"].index(
                            suggested_step.get("encoding", None)
                        )
                    )
                    custom_preprocessing[col]["encoding"] = encoding_method if encoding_method != None else None
                

                # Scaling Options
                if encoding_method == None:
                    if (col in numeric_features and col not in sensitive_feature_names) or ( col in numeric_features and col == target_column and binning_method == None):
                        scaling_method = st.selectbox(
                            f"{col} - Scaling",
                            options=[None, "Standard", "Min-Max", "Robust"],
                            index=[None, "Standard", "Min-Max", "Robust"].index(
                                suggested_step.get("scaling", None)
                            )
                        )
                        custom_preprocessing[col]["scaling"] = scaling_method if scaling_method != None else None
            
        # Apply custom preprocessing steps
        preprocess = st.checkbox("Apply Preprocessing Steps")
        if preprocess:
            train_processor = DatasetProcessor(train_data)
            train_processor.apply_custom_preprocessing(custom_preprocessing)  # Apply all transformations to the training data for features
            
            test_processor = DatasetProcessor(test_data)
            test_processor.encoders = train_processor.encoders
            test_processor.scalers = train_processor.scalers
            test_processor.imputers = train_processor.imputers
            test_processor.binners = train_processor.binners
            test_processor.bin_edges = train_processor.bin_edges
            test_processor.apply_custom_preprocessing(custom_preprocessing)  # Apply all transformations to the test data for features
            
            X_train =train_processor.data.drop(columns=[target_column])
            y_train = train_processor.data[target_column]
            X_test = test_processor.data.drop(columns=[target_column])
            y_test = test_processor.data[target_column]
            
            st.write("## Preprocessed Data Preview: Features")
            st.write(X_train.head())
            if target_column:
                st.write("## Preprocessed Data Preview: Target")
                st.write(y_train.head())
                
            #print the index where NA is present
            print("X_test",X_test[X_test.isna().any(axis=1)])
            print("y_test",y_test[y_test.isna()])
            
            # Validate index alignment
            assert X_train.index.equals(y_train.index), "X_train and y_train indices do not match!"
            assert X_test.index.equals(y_test.index), "X_test and y_test indices do not match!"
            assert X_train.columns.equals(X_test.columns), "X_train and X_test columns do not match!"
            #assert y_train.columns.equals(y_test.columns), "y_train and y_test columns do not match!"
        
    # for group in sensitive_groups:
    #     display_value = ", ".join([f"[{key} - {value}]" for key, value in group.items()])
    #     if st.checkbox(display_value, key=f"unprivileged_{group}"):
    #         unprivileged_groups.append(group)

    # st.write("## Choose Privileged Groups")
    
    # for group in sensitive_groups:
    #     if group in unprivileged_groups:
    #         continue
    #     else:
    #         display_value = ", ".join([f"[{key} - {value}]" for key, value in group.items()])
    #         if st.checkbox(display_value, key=f"privileged_{group}"):
    #             privileged_groups.append(group)
    

    # Streamlit App Configuration

    
    # Step 2: Choose Task and Models
    st.header("Step 3: Base Model Selection")
    task = st.selectbox("Choose a task", ["Classification", "Regression", "Clustering"])
    
    # Model selection with checkboxes
    model_suggestions = recommend_models(task)
    st.write("Select Models to Train")

    
    # Create a list to store selected models
    model_choices = []
    for model in model_suggestions:
        if st.checkbox(f"{model}", key=f"model_{model}"):
            model_choices.append(model)

    # Display selected models
    st.sidebar.write(f"## Selected Task:")
    st.sidebar.write(task)
    if model_choices:
        st.sidebar.write(f"## Selected Models:")
        st.sidebar.write(model_choices)
    else:
        st.sidebar.write("No models selected.")

    default_scoring = {
        'Classification': 'accuracy',
        'Regression': 'r2',
        'Clustering': 'silhouette'
    }
    
    # Step 1: Initialize ModelTrainer
    trainer = ModelTrainer(
        X_train= X_train,
        X_test= X_test,
        y_train= y_train,
        y_test= y_test,
        model_list=model_choices,
        scoring=default_scoring[task],
        cv=5)
    
    # Step 4: Custom Hyperparameters Configuration
    custom_params = {}
    st.write("### Hyperparameter Configuration")
    st.write("Enter custom hyperparameters for each model. All values should be comma-separated.")

    def parse_param_input(param_input, param_type):
        """
        Parses a comma-separated string into a list of specified parameter types, handling 'None' as None.

        Args:
            param_input (str): Comma-separated parameter values as a string.
            param_type (type): Target type for parsing, either int, float, or str.

        Returns:
            list: List of parsed parameter values.
        """
        return [None if value.strip() == "None" else param_type(value.strip()) for value in param_input.split(",")]

    for model_name in model_choices:
        st.write(f"#### {model_name}")
        custom_params[model_name] = {}
        
        for param_name, param_values in trainer.param_grids[model_name].items():
            
            # Convert param_values to strings, handling None, and determine param type based on the first valid type
            param_values_display = [str(v) if v is not None else "None" for v in param_values]
            param_type = type(next((v for v in param_values if v is not None), str))

            # Display input box and parse values according to inferred type
            param_input = st.text_input(
                param_name, 
                ", ".join(param_values_display), 
                key=f"{model_name}_{param_name}_input"
            )
            
            custom_params[model_name][param_name] = parse_param_input(param_input, param_type)
            
            
    trainer.param_grids = custom_params

    # Display the custom parameters dictionary
    st.sidebar.write("## Selected Hyperparameters:")
    for i in custom_params.keys():
        st.sidebar.write(f"## {i}")
        for j in custom_params[i].keys():
            st.sidebar.write(f"**{j}**")
            st.sidebar.write(custom_params[i][j])

    training = st.checkbox("Train and Evaluate Models", value=False)
    if training:
        
        best_score = float('-inf') if task == 'Regression' else 0
        best_model_name = None
        best_hyperparameters = None
        best_metrics = None
        scores = {}

        for model_name, model in trainer.models.items():
            
            tuned_model, score = trainer.tune_and_train_model(model_name, model)

            if tuned_model:
                metrics = trainer.evaluate_model(tuned_model, task=task)
                primary_metric = metrics.get("R² Score" if task == "Regression" else "accuracy")
                scores[model_name] = primary_metric

                # if task == "Classification":
                #     cm = trainer.confusion_matrix(tuned_model)
                #     fig = trainer.plot_confusion_matrix(cm, model_name)
                #     st.pyplot(fig)
                #     plt.savefig("interactive_pipeline/plots/confusion_matrix.png")
                # elif task == "Regression":
                #     predictions = tuned_model.predict(trainer.X_test)
                #     fig = trainer.plot_residuals(trainer.y_test, predictions, model_name)
                #     st.pyplot(fig)
                #     plt.savefig("interactive_pipeline/plots/residuals.png")
                
                if (task == 'Regression' and primary_metric > best_score) or \
                    (task == 'Classification' and primary_metric > best_score):
                    best_score = primary_metric
                    best_model_name = model_name
                    best_hyperparameters = trainer.final_hyperparameters[best_model_name]
                    best_metrics = metrics
                
        #set best model with best hyperparamters
        
        best_model = trainer.set_params(best_model_name, best_hyperparameters)
        best_model.fit(trainer.X_train, trainer.y_train)
        st.write("## Model Comparison")
        fig = trainer.plot_model_comparison(scores, task)
        st.pyplot(fig)
        plt.savefig("interactive_pipeline/plots/model_comparison.png")
        st.write("#### Best Model:")
        st.write(f"{model_name}")
        st.write(f"Final Hyperparameters:")
        st.dataframe(trainer.final_hyperparameters[model_name])
        
                
        y_true = trainer.y_test
        y_pred = best_model.predict(trainer.X_test)
        
        #X_test = pd.DataFrame(X_test, columns= X_test.columns)


        sensitive_features = X_test[sensitive_feature_names]
            #y_prob = best_model.predict_proba(trainer.X_test) if task == "Classification" else None
        
        
        # Step 6: Bias Detection for Best Model (can be selected after review)
        st.header("Step 4: Metrics")
        st.write("Now we calculate the performance and fairness metrics for the best model. Start by selecting the fairness notions for bias detection.")
        
        # fairness_notions = { "Classification":["Demographic Parity", "Equal Opportunity", "Equalized Odds", "Predictive Parity", "Calibration"],
        #                     "Regression": ["Mean Prediction Parity", "Error Rate Parity"],
        #                     }
        
        selected_fairness_notions = suggest_fairness_notions(task)
        #selected_fairness_notions = st.multiselect("Select Fairness Notions", fairness_notions[task])
        if selected_fairness_notions:
            detect = st.checkbox("Calculate Metrics", value=False)
            if detect:
                detector = BiasDetector(task = task, sensitive_features=sensitive_features, fairness_notions=selected_fairness_notions)
                group_results, overall_results = detector.calculate_metrics(y_true, y_pred)
                
                print(group_results)
                st.write("Metrics By group")
                st.dataframe(group_results)
                
                
                analyzer = GroupPrivilegeAnalyzer()
                priv_dict = analyzer.analyze_privilege(group_results)
                st.write("#### Choose Privileged and Unprivileged Groups")
                st.write("Suggested unprivileged and privileged groups are already selected based on the following reasons. But you can change them if needed.")
                st.write(priv_dict["Reasons"])
                sensitive_groups = priv_dict["Sensitive Groups"]
                unprivileged_groups = [st.selectbox("Select Unprivileged Groups", sensitive_groups, index = sensitive_groups.index(priv_dict["Unprivileged Group"]))]
                #remove selected unprivileged groups from the list
                sensitive_groups.remove(unprivileged_groups[0])
                privileged_groups = [st.selectbox("Select Privileged Groups", sensitive_groups, index = sensitive_groups.index(priv_dict["Privileged Group"]))]
                
                st.sidebar.write("## Privileged Groups")
                st.sidebar.write(privileged_groups)
                st.sidebar.write("## Unprivileged Groups")
                st.sidebar.write(unprivileged_groups)
                
                        
                # Step 7: Bias Mitigation
                st.header("Step 5: Bias Mitigation")
                
                st.write("Overall Metrics")
                st.dataframe(overall_results)
                reco = FairnessMitigationRecommender()
                mitigation_technique_dict= reco.recommend_techniques(overall_metrics=overall_results, group_metrics=group_results.to_dict(), task=task, fairness_notions=selected_fairness_notions)
                st.write("#### Select mitigation techniques among the recommended techniques")
                #st.write(mitigation_technique_dict["Reasons"])
                preprocessing_techniques = st.multiselect("Select Preprocessing Techniques", mitigation_technique_dict["Preprocessing"])
                inprocessing_techniquesm = st.multiselect("Select In-Processing Techniques", mitigation_technique_dict["In-Processing"])
                postprocessing_techniques = st.multiselect("Select Post-Processing Techniques", mitigation_technique_dict["Post-Processing"])
                
                methods = {"preprocessing": preprocessing_techniques, "inprocessing": inprocessing_techniquesm, "postprocessing": postprocessing_techniques, "sess": tf.compat.v1.Session()}
                
                mitigate = st.checkbox("Apply Mitigation", value=False)
                if mitigate:
                    with st.spinner("Applying Mitigation Techniques..."):
                        mitigator = BiasMitigator(
                            task = task,
                            sensitive_feature_names = sensitive_feature_names,
                            fairness_notions = selected_fairness_notions,
                            privileged_groups = privileged_groups,
                            unprivileged_groups = unprivileged_groups)

                        mitigator.run_pipeline(X_train,
                                            X_test,
                                            y_train,
                                            y_test,
                                            sensitive_feature_names = sensitive_feature_names, 
                                            base_model = best_model, 
                                            methods = methods)
                        st.write("## Choose which metrics to plot for comparative analysis")
                        metrics_to_plot = st.multiselect("Select Metrics", overall_results.keys())
                        st.write(metrics_to_plot)
                        mitigator.plot_metrics(metrics_to_plot)