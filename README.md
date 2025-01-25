# Conagra Bias Project Environment Setup

This README provides instructions for setting up the **Conagra Bias Project** environment and running various components of the project. Follow these steps to set up and explore the project's different pipelines and applications.

---

## Environment Setup

To set up the project environment (covering all sub-projects), follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/jhousteau/conagra-bias-project.git
   cd conagra-bias-project
   ```

2. Create and activate the Conda environment:
   ```bash
   conda create -n conagra-bias-project python=3.9 -y
   conda activate conagra-bias-project
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Testing the Pipeline for the Original Proof of Concept (POC)

To test the original POC pipeline:

1. Add your OpenAI API key to the `.env` file. Specify the model you want to use.

2. Run the pipeline:
   - To ignore warnings, use:
     ```bash
     PYTHONWARNINGS="ignore::FutureWarning" python poc-1-pipeline-prototype-bruteforce-data/pipeline.py
     ```
   - Otherwise, run:
     ```bash
     python poc-1-pipeline-prototype-bruteforce-data/pipeline.py
     ```

---

## Running POC 2 (Synthetic Data Stats)

Note: Only a few modules from POC 2 are functional. This approach was abandoned in favor of deep learning.

- Run the following commands to execute individual modules:
  ```bash
  python poc-2-synthetic-data-stats/consumer_behavior_data.py
  python poc-2-synthetic-data-stats/customer_channel_and_engagement_preference.py
  python poc-2-synthetic-data-stats/device_shopping_behavior.py
  ```

- The following module is **not functional**:
  ```bash
  python poc-2-synthetic-data-stats/transaction_data_generator.py
  ```

---

## Building and Analyzing the Final Dataset (Deep Learning Approach)

To build and analyze the final dataset using deep learning:

1. By default, the process uses **New York (NY)** data and runs for **1 epoch**.
2. To achieve realistic results:
   - Edit the epoch value in `constants.py` (minimum recommended: **100**).
   - Change the target state in `demographic.py` by modifying the `STATE_*` variable.

3. Note: Running with higher epochs and more states requires significant computational resources (e.g., additional CPU/RAM).

4. Generate the synthetic dataset:
   ```bash
   python final-synthetic_marketing_data/build_dataset.py
   ```
   This trains the GAN model and generates synthetic marketing data.
   - Output location: `output/models/`
      - Saves model checkpoints: `generator_epoch_{n}.pth`, `discriminator_epoch_{n}.pth`
   - Output location: `output/raw/`
      - Generates: `synthetic_data_raw.csv` - Initial synthetic dataset
      - Generates: `training_metrics.json` - Model training history
      - Generates: `generation_config.json` - Parameters used for generation

5. Run exploratory data analysis:
   ```bash
   python final-synthetic_marketing_data/eda.py
   ```
   This will generate visualizations and statistical summaries of the synthetic data.
   - Output location: `output/analysis/eda/`
   - Generates: Distribution plots, correlation matrices, and summary statistics

6. Evaluate fairness metrics:
   ```bash
   python final-synthetic_marketing_data/fairness.py
   ```
   This analyzes the dataset for potential biases across different demographic groups.
   - Output location: `output/analysis/fairness/`
   - Generates: Fairness metric reports and bias visualization plots

7. Export the processed data:
   ```bash
   python final-synthetic_marketing_data/data_export.py [--format FORMAT] [--rows ROWS] [--compress]
   ```
   This creates standardized files ready for downstream analysis.
   - Output location: `output/processed/`
   - Generates: `synthetic_marketing_data.{format}` and `synthetic_marketing_data_metadata.json`

   Arguments:
   - `--format`: Output format (csv, parquet, json). Default: csv
   - `--rows`: Number of rows to generate. Default: 10000
   - `--compress`: Enable compression for output file. Default: False

---

## Running the Interactive Bias Pipeline Application

To run the interactive bias pipeline:

1. Launch the Streamlit app:
   ```bash
   streamlit run interactive_pipeline/app.py
   ```

2. The application provides a user-friendly interface for uploading datasets, detecting bias, and applying mitigation techniques.

3. Run it using the bank-additional-full.csv file from interactive_pipeline/data
4. Choose target column as y and sensitive feature as age. Choose label encoding for both. And preprocess other columns as needed. 
5. Choose Classification as task. Choose High stakes, Demographic parity and equal odds in fairness notions questionnaire. Choose Reweighing, Adversarial Debiasing, Equalized odds adjustment fo rmitigation strategies. 
Even though there are other options, they are not implemented yet. 
---

## Important Notes and Limitations

1. **Current Data Limitations**
   - The exported synthetic data currently lacks referential integrity between related fields
   - While suitable for individual modeling tasks, caution should be used when analyzing relationships between entities

2. **Planned Improvements**
   - Future versions will implement HMASynthesizer (Hierarchical Multi-Agent Synthesizer) to maintain proper relationships and dependencies between data elements
   - This will ensure generated data better reflects real-world entity relationships and constraints

3. **Data Quality Enhancement**
   - For optimal results, the deep learning model should be seeded with real marketing data samples
   - This seeding process allows the model to:
     - Capture authentic patterns and correlations
     - Learn genuine customer behavior nuances
     - Reflect realistic market dynamics
     - Maintain proper business logic constraints

4. **Current Usage Guidelines**
   - The current synthetic dataset is best suited for:
     - Individual feature analysis
     - Single-entity modeling tasks
     - Preliminary data exploration
   - Avoid analyses that rely heavily on inter-entity relationships until the HMASynthesizer implementation is complete

5. **Future Roadmap**
   - Implementation of HMASynthesizer
   - Integration of real data seeding capabilities
   - Enhanced validation of relationship constraints
   - Improved metadata tracking for data lineage

---

## Notes

- **Warning Suppression**: If you encounter deprecation or future warnings, you can suppress them with:
  ```bash
  PYTHONWARNINGS="ignore::FutureWarning"
  ```

- **Module Dependencies**: Each module is dependent on the previous one. Ensure they are executed in the correct order to avoid errors.
