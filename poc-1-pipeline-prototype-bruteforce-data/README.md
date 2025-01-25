# Conagra Brands: Bias Detection and Mitigation in AI-Driven Marketing

## Project Overview

This project aims to develop a framework for detecting and mitigating bias in AI-driven marketing models for Conagra Brands. The focus is on ensuring fair treatment of diverse customer segments and preventing disproportionate advantages or disadvantages based on sensitive attributes such as race, gender, or socioeconomic status.

## Table of Contents

- [Conagra Brands: Bias Detection and Mitigation in AI-Driven Marketing](#conagra-brands-bias-detection-and-mitigation-in-ai-driven-marketing)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Modules](#modules)
  - [Kaggle Shopping Trends Data](#kaggle-shopping-trends-data)
  - [Current Implementation](#current-implementation)
  - [Known Issues and Limitations](#known-issues-and-limitations)
  - [Future Improvements](#future-improvements)

## Project Structure

```
conagra/
│
├── ai_analyzer.py
├── config.py
├── detection.py
├── detection_kaggle.py
├── generate.py
├── mitigation.py
├── model.py
├── output.py
├── pipeline.py
├── requirements.txt
├── README.md
└── kaggle_data/
    └── shopping_trends copy.csv
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/jhousteau/conagra-bias-project.git
   cd conagra-bias-detection
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the entire pipeline:

```python
from pipeline import BiasPipeline

pipeline = BiasPipeline(n_samples=10000, output_type="both", output_dir="bias_pipeline_output")
pipeline.run_pipeline()
```

To run the Kaggle Shopping Trends bias detection:

```
python detection_kaggle.py
```

This will generate synthetic data, detect bias, attempt to mitigate bias, and produce reports at each stage.

## Modules

1. **ai_analyzer.py**: Utilizes OpenAI's API to provide AI-driven analysis of the results at various stages.
2. **config.py**: Contains configuration settings and constants used across the project.
3. **detection.py**: Implements bias detection algorithms for both supervised and unsupervised models.
4. **detection_kaggle.py**: Implements bias detection for the Kaggle Shopping Trends dataset.
5. **generate.py**: Generates synthetic consumer data for analysis.
6. **mitigation.py**: Implements various bias mitigation strategies.
7. **model.py**: Defines the machine learning models used in the project.
8. **output.py**: Handles output generation and formatting.
9. **pipeline.py**: Orchestrates the entire bias detection and mitigation process.

## Kaggle Shopping Trends Data

We've incorporated a new dataset from Kaggle: "Shopping Trends" data. This dataset is used to demonstrate bias detection in a real-world scenario.

- **Dataset**: The dataset is located in the `kaggle_data/` folder as `shopping_trends copy.csv`.
- **Analysis**: The `detection_kaggle.py` script is dedicated to analyzing this dataset for potential biases.
- **Output**: Results of the Kaggle data analysis are saved in the `output/` folder, including:
  - `kaggle_bias_detection_report.md`: Detailed report of bias detection results
  - `ai_analysis_report.md`: AI-generated analysis of the bias detection results
  - `shap_summary.png`: SHAP (SHapley Additive exPlanations) summary plot

To run the Kaggle data analysis:
```
python detection_kaggle.py
```

## Current Implementation

The current implementation follows these main steps:

1. **Data Generation**: Synthetic consumer data is generated based on predefined distributions and assumptions.
2. **Bias Detection**: Both supervised and unsupervised models are analyzed for potential biases in the synthetic data.
3. **Bias Mitigation**: Three strategies (reweighing, demographic parity, and equalized odds) are applied to mitigate detected biases in the synthetic data.
4. **Reporting**: Detailed reports are generated at each stage, including AI-driven analysis of the results.
5. **Kaggle Data Analysis**: 
   - The Kaggle Shopping Trends dataset is loaded and preprocessed.
   - A Random Forest Classifier is trained on this real-world data.
   - Bias detection metrics are calculated for sensitive features (Age and Gender).
   - Feature importance and SHAP values are analyzed to understand the model's decision-making process.
   - A comprehensive report is generated, including bias metrics, feature importance, and SHAP summary plots.
   - An AI-driven analysis of the bias detection results is performed and included in the report.

## Known Issues and Limitations

1. Extreme bias (difference of 1.0) is detected in the supervised model for synthetic data, which persists even after mitigation attempts.
2. Some demographic groups have very small sample sizes in the synthetic data, leading to 'nan' values in the results.
3. The bias mitigation strategies show limited effectiveness in reducing overall bias in the synthetic data.
4. The unsupervised model analysis could be expanded to provide more insights into potential biases in clustering results.
5. The Kaggle dataset analysis is currently limited to a binary classification task (high/low purchase amount) and may not capture all nuances of the shopping trends data.
6. The current implementation doesn't apply bias mitigation techniques to the Kaggle dataset, focusing only on detection.
7. The AI analysis of the Kaggle data results may require further refinement to provide more actionable insights.

## Future Improvements

1. Implement robust error handling throughout the codebase.
2. Enhance type safety using type hints and consider using a static type checker like mypy.
3. Improve code documentation with more comprehensive docstrings and inline comments.
4. Explore alternative bias mitigation techniques and fine-tune existing strategies for both synthetic and real-world data.
5. Enhance the data generation process to ensure better representation across all demographic groups.
6. Expand the analysis of the unsupervised model to provide more actionable insights.
7. Extend the Kaggle data analysis to include:
   - Multi-class or regression tasks instead of binary classification.
   - Application of bias mitigation techniques to the real-world data.
   - More sophisticated feature engineering to capture complex relationships in the shopping trends data.
8. Implement intersectional analysis for both synthetic and Kaggle data to detect bias across multiple sensitive attributes simultaneously.
9. Develop a more comprehensive AI analysis system that can provide specific recommendations for addressing detected biases in both synthetic and real-world data.
10. Create a unified pipeline that can seamlessly handle both synthetic and real-world data, allowing for easy comparison of bias detection and mitigation results across different data sources.
11. Implement visualization tools to better illustrate bias patterns and the effects of mitigation strategies in both synthetic and Kaggle data.