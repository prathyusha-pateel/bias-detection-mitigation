---

# Synthetic Marketing Data Generation System

## Overview

A comprehensive system for generating realistic synthetic marketing data with sophisticated validation. The system generates interconnected datasets covering demographics, consumer preferences, marketing campaigns, engagements, and transactions. It includes utilities for exploratory data analysis (EDA) and bias analysis.

## Table of Contents

- [System Requirements](#system-requirements)
  - [Hardware Specifications](#hardware-specifications)
  - [Software Dependencies](#software-dependencies)
  - [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
  - [Generate Complete Synthetic Dataset](#generate-complete-synthetic-dataset)
  - [Run Individual Components](#run-individual-components)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Bias Analysis](#bias-analysis)
- [Architecture](#architecture)
- [Data Generation Pipeline](#data-generation-pipeline)
  - [1. Demographic Data Generation](#1-demographic-data-generation)
  - [2. Consumer Preferences Generation](#2-consumer-preferences-generation)
  - [3. Marketing Campaigns Generation](#3-marketing-campaigns-generation)
  - [4. Transactions Data Generation](#4-transactions-data-generation)
- [Generated Datasets](#generated-datasets)
- [Validation Framework](#validation-framework)
  - [Key Metrics](#key-metrics)
  - [Run Validation](#run-validation)
- [EDA and Bias Analysis](#eda-and-bias-analysis)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda-1)
    - [Features:](#features)
    - [Run EDA:](#run-eda)
  - [Fairness and Bias Analysis](#fairness-and-bias-analysis)
    - [Features:](#features-1)
    - [Run Fairness Analysis:](#run-fairness-analysis)
- [Data Loader with Caching](#data-loader-with-caching)
  - [Features:](#features-2)
  - [Using the Data Loader](#using-the-data-loader)
- [Configuration](#configuration)
  - [Key Configurations in `constants.py`](#key-configurations-in-constantspy)
    - [Model Parameters](#model-parameters)
    - [State Selection](#state-selection)
    - [Validation Thresholds](#validation-thresholds)
- [Best Practices](#best-practices)
- [Limitations and Considerations](#limitations-and-considerations)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## System Requirements

### Hardware Specifications

- **Memory (RAM):** Minimum 16 GB (Recommended: 32 GB+)
- **Processor (CPU):** Multi-core processor with at least 8 cores
- **Storage:** SSD with at least 10 GB free space
- **Graphics Processing Unit (GPU):** Optional, but recommended for deep learning tasks

### Software Dependencies

- **Python 3.8+**
- Python packages: `sdv`, `pandas`, `numpy`, `scikit-learn`, `ydata-profiling`, `aif360`

### Installation

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # Unix/MacOS
env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start Guide

### Generate Complete Synthetic Dataset

```bash
python build_dataset.py
```

### Run Individual Components

```bash
python demographic.py    # Demographic data
python consumer.py       # Consumer preferences
python marketing.py      # Marketing campaigns
python transaction.py    # Transactions data
```

### Exploratory Data Analysis (EDA)

```bash
python eda.py
```

### Bias Analysis

```bash
python fairness.py
```

---

## Architecture

```
synthetic_marketing_data/
├── build_dataset.py          # Main orchestration script
├── data_loader.py            # Efficient data loading with caching
├── demographic.py            # Demographic data generation
├── consumer.py               # Consumer preferences generation
├── marketing.py              # Marketing campaigns generation
├── transaction.py            # Transactions data generation
├── eda.py                    # Exploratory Data Analysis
├── fairness.py               # Bias detection and fairness analysis
├── constants.py              # Configuration constants
├── data/                     # Directory for generated datasets
├── validation/               # Validation scripts and reports
└── utils/                    # Utility functions
```

---

## Data Generation Pipeline

### 1. Demographic Data Generation

```bash
python demographic.py
```

### 2. Consumer Preferences Generation

```bash
python consumer.py
```

### 3. Marketing Campaigns Generation

```bash
python marketing.py
```

### 4. Transactions Data Generation

```bash
python transaction.py
```

---

## Generated Datasets

- **Demographics Data:** `*_demographics.csv`
- **Consumer Preferences Data:** `consumer_preferences.csv`
- **Marketing Campaigns Data:** `marketing_campaigns.csv`
- **Transactions Data:** `transactions.csv`

---

## Validation Framework

Validation ensures statistical validity and alignment with real-world patterns.

### Key Metrics

- Distribution matching
- Correlation preservation
- Demographic parity
- Business rule compliance

### Run Validation

```bash
python validation/validate_data.py
```

---

## EDA and Bias Analysis

### Exploratory Data Analysis (EDA)

The `eda.py` script provides insights into the dataset by generating detailed profiling reports.

#### Features:
- Distribution plots
- Correlation matrices
- Missing value analysis
- Outlier detection

#### Run EDA:
```bash
python eda.py
```

Reports are saved in `eda/reports/` as HTML files for easy viewing.

---

### Fairness and Bias Analysis

The `fairness.py` script uses AI Fairness 360 to analyze the generated data for potential biases.

#### Features:
- Demographic parity checks
- Equalized odds analysis
- Fairness comparison across groups

#### Run Fairness Analysis:
```bash
python fairness.py
```

Results are saved in `fairness/reports/` in JSON format.

---

## Data Loader with Caching

The `data_loader.py` script provides an efficient interface for loading datasets and includes caching capabilities to speed up repeated access.

### Features:
- **Efficient Data Loading:** Handles large datasets efficiently with chunking.
- **Caching:** Automatically caches datasets to minimize reloading times.
- **Error Handling:** Provides informative error messages for missing or corrupted data files.

### Using the Data Loader

Example usage in a script:
```python
from data_loader import DataLoader

# Initialize the data loader
loader = DataLoader(cache_dir="cache/")

# Load a dataset
df = loader.load("data/transactions.csv")

# Perform operations on the dataframe
print(df.head())
```

---

## Configuration

### Key Configurations in `constants.py`

#### Model Parameters
```python
MODEL_PARAMETERS = {
    "epochs": 100,
    "batch_size": 500,
    "generator_dim": (512, 256),
    "discriminator_dim": (512, 256),
}
```

#### State Selection
```python
STATES = ['NY', 'CA', 'TX', 'IL', 'FL', 'WA']
```

#### Validation Thresholds
```python
VALIDATION_THRESHOLDS = {
    "distribution_tolerance": 0.05,
    "correlation_tolerance": 0.10,
}
```

---

## Best Practices

1. **Start Small:** Use smaller state selections and fewer epochs for initial testing.
2. **Monitor Resources:** Track memory and CPU usage during large runs.
3. **Validate Frequently:** Run validation scripts after each major data generation step.
4. **Use Caching:** Leverage the `data_loader` caching feature for faster data access.
5. **Analyze Data Thoroughly:** Use `eda.py` to identify patterns and potential issues early.
6. **Ensure Fairness:** Run `fairness.py` to detect and mitigate biases in the data.

---

## Limitations and Considerations

1. **Synthetic Nature of Data:** While realistic, synthetic data may not fully replicate real-world complexities.
2. **Hardware Requirements:** Larger datasets and higher epochs require more powerful hardware.
3. **Validation Focus:** Primary focus is on statistical and structural validity, which may not capture all business nuances.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- US Census Bureau for demographic data
- SDV for synthetic data generation tools
- AI Fairness 360 for bias analysis

---

## Contact

For questions or support, please contact the project maintainers:

- **James Housteau**
- **Prathyusha Pateel**

--- 

This updated README includes concise instructions for the `data_loader`, `eda.py`, and `fairness.py`, ensuring users can efficiently utilize these features.