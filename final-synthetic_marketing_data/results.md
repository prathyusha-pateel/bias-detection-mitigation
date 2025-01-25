(conagra) jameshousteau@Calypso synthetic_marketing_data % python build_dataset.py
2024-11-19 21:00:49 - build_dataset - INFO - Starting dataset build process...
2024-11-19 21:00:49 - build_dataset - INFO - Cleaned data directory: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data
2024-11-19 21:00:49 - build_dataset - INFO - Created fresh data directory
2024-11-19 21:00:49 - build_dataset - INFO - Created required subdirectories
2024-11-19 21:00:49 - build_dataset - INFO -
==================== Processing Demographic base data ====================
2024-11-19 21:00:49 - build_dataset - INFO -
==================== Running demographic ====================
2024-11-19 21:00:50 - demographics - INFO - Deleted existing data directory: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data
2024-11-19 21:00:50 - demographics - INFO - Selected states: NY, CA, TX, IL, FL, WA
2024-11-19 21:00:50 - demographics - INFO - Selected states represent:
2024-11-19 21:00:50 - demographics - INFO - - All major US regions
2024-11-19 21:00:50 - demographics - INFO - - Mix of urban/rural populations
2024-11-19 21:00:50 - demographics - INFO - - Diverse economic sectors
2024-11-19 21:00:50 - demographics - INFO - - Different demographic profiles
2024-11-19 21:00:50 - demographics - INFO - - Various education levels
2024-11-19 21:00:50 - demographics - INFO -
Processing state: NY
2024-11-19 21:00:50 - demographics - INFO - ==================================================
2024-11-19 21:00:50 - demographics - INFO - Downloading data for NY...
Downloading data for 2023 1-Year person survey for NY...
2024-11-19 21:00:59 - demographics - INFO -
Raw demographic data saved to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/ny_demographics.csv
2024-11-19 21:00:59 - demographics - INFO -
Processing state: CA
2024-11-19 21:00:59 - demographics - INFO - ==================================================
2024-11-19 21:00:59 - demographics - INFO - Downloading data for CA...
Downloading data for 2023 1-Year person survey for CA...
2024-11-19 21:01:10 - demographics - INFO -
Raw demographic data saved to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/ca_demographics.csv
2024-11-19 21:01:10 - demographics - INFO -
Processing state: TX
2024-11-19 21:01:10 - demographics - INFO - ==================================================
2024-11-19 21:01:10 - demographics - INFO - Downloading data for TX...
Downloading data for 2023 1-Year person survey for TX...
2024-11-19 21:01:18 - demographics - INFO -
Raw demographic data saved to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/tx_demographics.csv
2024-11-19 21:01:18 - demographics - INFO -
Processing state: IL
2024-11-19 21:01:18 - demographics - INFO - ==================================================
2024-11-19 21:01:18 - demographics - INFO - Downloading data for IL...
Downloading data for 2023 1-Year person survey for IL...
2024-11-19 21:01:24 - demographics - INFO -
Raw demographic data saved to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/il_demographics.csv
2024-11-19 21:01:24 - demographics - INFO -
Processing state: FL
2024-11-19 21:01:24 - demographics - INFO - ==================================================
2024-11-19 21:01:24 - demographics - INFO - Downloading data for FL...
Downloading data for 2023 1-Year person survey for FL...
2024-11-19 21:01:31 - demographics - INFO -
Raw demographic data saved to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/fl_demographics.csv
2024-11-19 21:01:31 - demographics - INFO -
Processing state: WA
2024-11-19 21:01:31 - demographics - INFO - ==================================================
2024-11-19 21:01:31 - demographics - INFO - Downloading data for WA...
Downloading data for 2023 1-Year person survey for WA...
2024-11-19 21:01:33 - demographics - INFO -
Raw demographic data saved to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/wa_demographics.csv
2024-11-19 21:01:33 - demographic_validator - INFO - Available columns in combined dataset: SERIALNO, AGEP, SEX, PINCP, ADJINC, PWGTP, STATE, PUMA, SCHL
2024-11-19 21:01:34 - demographics - INFO -
================================================================================
2024-11-19 21:01:34 - demographics - INFO -                          Demographic Validation Results
2024-11-19 21:01:34 - demographics - INFO - ================================================================================

2024-11-19 21:01:34 - demographics - INFO -
----------------------------------------------------------------------------
2024-11-19 21:01:34 - demographics - INFO - DISTRIBUTIONS Validation
2024-11-19 21:01:34 - demographics - INFO - ----------------------------------------------------------------------------
2024-11-19 21:01:34 - demographics - INFO - Metric                    Expected     Actual       Status
2024-11-19 21:01:34 - demographics - INFO - --------------------------------------------------------------------------------
2024-11-19 21:01:34 - demographics - INFO -
numerical_metrics:
2024-11-19 21:01:34 - demographics - INFO - AGEP                      42.88        42.88        ✅
2024-11-19 21:01:34 - demographics - INFO - PINCP                     54954.56     54954.56     ✅
2024-11-19 21:01:34 - demographics - INFO - ADJINC                    1019518.00   1019518.00   ✅
2024-11-19 21:01:34 - demographics - INFO - PWGTP                     99.54        99.54        ✅
2024-11-19 21:01:34 - demographics - INFO -
categorical_metrics:
2024-11-19 21:01:34 - demographics - INFO - SEX                       2: 50.9%, 1: 49.1% 2: 50.9%, 1: 49.1% ✅
2024-11-19 21:01:34 - demographics - INFO - STATE                     distribution with 6 categories distribution with 6 categories ✅
2024-11-19 21:01:34 - demographics - INFO - PUMA                      distribution with 823 categories distribution with 823 categories ✅
2024-11-19 21:01:34 - demographics - INFO - SCHL                      distribution across 24 education levels distribution across 24 education levels ✅
2024-11-19 21:01:34 - demographics - INFO -
----------------------------------------------------------------------------
2024-11-19 21:01:34 - demographics - INFO - SUMMARY
2024-11-19 21:01:34 - demographics - INFO - ----------------------------------------------------------------------------
2024-11-19 21:01:34 - demographics - INFO - Overall Score:          100.00% ✅
2024-11-19 21:01:34 - demographics - INFO - Total Metrics Checked:  8
2024-11-19 21:01:34 - demographics - INFO - Passing Metrics:        8
2024-11-19 21:01:34 - demographics - INFO -
Validation results saved to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/validation_results.json
2024-11-19 21:01:34 - demographics - INFO -
Validation results saved to validation_results.json
2024-11-19 21:01:34 - build_dataset - INFO - ✓ demographic completed successfully in 45.0s
2024-11-19 21:01:34 - build_dataset - INFO - ✓ Output verification passed for demographic
2024-11-19 21:01:34 - build_dataset - INFO - ✓ Completed Demographic base data generation
2024-11-19 21:01:34 - build_dataset - INFO -
==================== Processing Consumer preferences ====================
2024-11-19 21:01:34 - build_dataset - INFO - Checking dependencies for consumer...
2024-11-19 21:01:34 - build_dataset - INFO - ✓ All dependencies satisfied for consumer
2024-11-19 21:01:34 - build_dataset - INFO -
==================== Running consumer ====================
2024-11-19 21:01:37 - consumer - INFO - Starting consumer preferences generation...
2024-11-19 21:01:37 - consumer - INFO - Loading demographic data...
2024-11-19 21:01:37 - consumer - INFO - Reading il_demographics.csv
2024-11-19 21:01:37 - consumer - INFO - Reading ny_demographics.csv
2024-11-19 21:01:37 - consumer - INFO - Reading ca_demographics.csv
2024-11-19 21:01:37 - consumer - INFO - Reading tx_demographics.csv
2024-11-19 21:01:37 - consumer - INFO - Reading fl_demographics.csv
2024-11-19 21:01:37 - consumer - INFO - Reading wa_demographics.csv
2024-11-19 21:01:37 - consumer - INFO - Loaded 1,326,172 demographic records
2024-11-19 21:01:37 - consumer - INFO - Creating initial preference dataset...
2024-11-19 21:01:37 - consumer - INFO - Processing 1,326,172 demographic records
2024-11-19 21:01:37 - consumer - INFO - Generated consumer IDs
2024-11-19 21:01:38 - consumer - INFO - Age groups distribution:
2024-11-19 21:01:38 - consumer - INFO -   55+: 724,686 consumers (54.6%)
2024-11-19 21:01:38 - consumer - INFO -   35-54: 325,969 consumers (24.6%)
2024-11-19 21:01:38 - consumer - INFO -   18-34: 275,517 consumers (20.8%)
2024-11-19 21:01:38 - consumer - INFO - Created initial preferences for 1,326,172 consumers
2024-11-19 21:01:38 - consumer - INFO - Creating metadata...
2024-11-19 21:01:40 - consumer - INFO - Deleted existing metadata file: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/consumer.json
2024-11-19 21:01:40 - consumer - INFO - Saved metadata to: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/consumer.json
2024-11-19 21:01:40 - consumer - INFO -
Training synthetic data model...
2024-11-19 21:01:40 - consumer - INFO -
Training Configuration:
2024-11-19 21:01:40 - consumer - INFO - --------------------------------------------------
2024-11-19 21:01:40 - consumer - INFO - Batch size: 1326168
2024-11-19 21:01:40 - consumer - INFO - PAC: 8
2024-11-19 21:01:40 - consumer - INFO - Epochs: 100
2024-11-19 21:01:40 - consumer - INFO - Generator LR: 0.0002
2024-11-19 21:01:40 - consumer - INFO - Discriminator LR: 0.0001
2024-11-19 21:01:40 - consumer - INFO - Training data shape: (1326172, 21)
2024-11-19 21:01:40 - consumer - INFO - --------------------------------------------------
Gen. (-1.30) | Discrim. (0.05): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [6:08:02<00:00, 220.83s/it]
Training Progress:   0%|                                                                                                                               | 0/100 [6:21:28<?, ?it/s]
2024-11-20 03:23:09 - consumer - INFO -
Training Summary:
2024-11-20 03:23:09 - consumer - INFO - --------------------------------------------------
2024-11-20 03:23:09 - consumer - INFO - Total time: 22888.9s
2024-11-20 03:23:09 - consumer - INFO - Avg time per epoch: 228.9s
2024-11-20 03:23:09 - consumer - INFO - --------------------------------------------------
2024-11-20 03:23:09 - consumer - INFO - Generating 1,326,172 synthetic records...
Generating report ...

(1/2) Evaluating Data Validity: |███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:02<00:00, 10.47it/s]|
Data Validity Score: 100.0%

(2/2) Evaluating Data Structure: |████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 41.47it/s]|
Data Structure Score: 100.0%

Overall Score (Average): 100.0%

2024-11-20 03:24:08 - consumer - INFO -
================================================================================
2024-11-20 03:24:08 - consumer - INFO -                      Consumer Preference Validation Results
2024-11-20 03:24:08 - consumer - INFO - ================================================================================

2024-11-20 03:24:08 - consumer - INFO -
----------------------------------------------------------------------------
2024-11-20 03:24:08 - consumer - INFO - DISTRIBUTIONS Validation
2024-11-20 03:24:08 - consumer - INFO - ----------------------------------------------------------------------------
2024-11-20 03:24:08 - consumer - INFO - Metric                    Expected     Actual       Status
2024-11-20 03:24:08 - consumer - INFO - --------------------------------------------------------------------------------
2024-11-20 03:24:08 - consumer - INFO -
numerical_metrics:
2024-11-20 03:24:08 - consumer - INFO - social_media_engagement_rate 72.02%       71.16%       ✅
2024-11-20 03:24:08 - consumer - INFO - online_shopping_rate      54.06%       66.21%       ❌
2024-11-20 03:24:08 - consumer - INFO - loyalty_memberships       16.70        18.16        ❌
2024-11-20 03:24:08 - consumer - INFO -
categorical_metrics:
2024-11-20 03:24:08 - consumer - INFO - age_group                 55+: 54.6%, 35-54: 24.6%, 18-34: 20.8% 55+: 34.9%, 35-54: 32.8%, 18-34: 32.3% ❌
2024-11-20 03:24:08 - consumer - INFO -
----------------------------------------------------------------------------
2024-11-20 03:24:08 - consumer - INFO - PREFERENCES Validation
2024-11-20 03:24:08 - consumer - INFO - ----------------------------------------------------------------------------
2024-11-20 03:24:08 - consumer - INFO - Metric                    Expected     Actual       Status
2024-11-20 03:24:08 - consumer - INFO - --------------------------------------------------------------------------------
2024-11-20 03:24:08 - consumer - INFO -
product_metrics:
2024-11-20 03:24:08 - consumer - INFO - social_media_engagement   72.02%       71.16%       ✅
2024-11-20 03:24:08 - consumer - INFO -
----------------------------------------------------------------------------
2024-11-20 03:24:08 - consumer - INFO - SUMMARY
2024-11-20 03:24:08 - consumer - INFO - ----------------------------------------------------------------------------
2024-11-20 03:24:08 - consumer - INFO - Overall Score:          40.00% ❌
2024-11-20 03:24:08 - consumer - INFO - Total Metrics Checked:  5
2024-11-20 03:24:08 - consumer - INFO - Passing Metrics:        2
2024-11-20 03:24:08 - consumer - INFO - Synthetic data generation completed
2024-11-20 03:24:23 - consumer - INFO -
Saved results:
2024-11-20 03:24:23 - consumer - INFO -   Data: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/consumer/20241120_032408/consumer.csv (457.98 MB)
2024-11-20 03:24:23 - consumer - INFO -   Metadata: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/consumer/20241120_032408/metadata.json (2.81 KB)
2024-11-20 03:24:23 - consumer - INFO -   Model: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/consumer/20241120_032408/model.pkl (32.36 MB)
2024-11-20 03:24:23 - consumer - INFO - Consumer preferences generation completed successfully
2024-11-20 03:24:23 - build_dataset - INFO - ✓ consumer completed successfully in 22968.7s
2024-11-20 03:24:23 - build_dataset - INFO - ✓ Output verification passed for consumer
2024-11-20 03:24:23 - build_dataset - INFO - ✓ Completed Consumer preferences generation
2024-11-20 03:24:23 - build_dataset - INFO -
==================== Processing Marketing engagement ====================
2024-11-20 03:24:23 - build_dataset - INFO - Checking dependencies for marketing...
2024-11-20 03:24:23 - build_dataset - INFO - ✓ All dependencies satisfied for marketing
2024-11-20 03:24:23 - build_dataset - INFO -
==================== Running marketing ====================
2024-11-20 03:24:23 - marketing - INFO - Starting marketing engagement data generation...
2024-11-20 03:24:23 - marketing - INFO - Loading input data...
2024-11-20 03:24:24 - marketing - INFO - Loaded 1,326,172 demographic records
2024-11-20 03:24:26 - marketing - INFO - Loaded 1,326,172 consumer preference records from /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/consumer/20241120_032408/consumer.csv
2024-11-20 03:24:26 - marketing - INFO - Creating 50 initial campaign patterns...
2024-11-20 03:24:26 - marketing - INFO - Creating metadata for campaigns...
2024-11-20 03:24:29 - marketing - INFO - Deleted existing metadata file: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/campaigns.json
2024-11-20 03:24:29 - marketing - INFO - Saved metadata to: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/campaigns.json
2024-11-20 03:24:29 - marketing - INFO - Training campaigns generation model...
2024-11-20 03:24:29 - marketing - INFO - Training data shape: (50, 14)
2024-11-20 03:24:29 - marketing - INFO - Starting model training with 100 epochs...
Gen. (0.85) | Discrim. (0.01): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 68.11it/s]
2024-11-20 03:24:32 - marketing - INFO - Model training completed in 0:00:02.511583
2024-11-20 03:24:32 - marketing - INFO - Generating 132,617 synthetic campaigns records...
2024-11-20 03:24:46 - marketing - INFO - Generated 132,617 campaigns records
2024-11-20 03:24:46 - marketing - INFO - Creating initial engagement patterns...
2024-11-20 03:24:46 - marketing - INFO - Creating metadata for engagements...
2024-11-20 03:24:46 - marketing - INFO - Deleted existing metadata file: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/engagements.json
2024-11-20 03:24:46 - marketing - INFO - Saved metadata to: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/engagements.json
2024-11-20 03:24:46 - marketing - INFO - Training engagements generation model...
2024-11-20 03:24:46 - marketing - INFO - Training data shape: (49, 11)
2024-11-20 03:24:46 - marketing - INFO - Starting model training with 100 epochs...
Gen. (0.63) | Discrim. (0.13): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 83.57it/s]
2024-11-20 03:24:48 - marketing - INFO - Model training completed in 0:00:01.336525
2024-11-20 03:24:48 - marketing - INFO - Generating 3,978,510 synthetic engagements records...
2024-11-20 03:26:10 - marketing - INFO - Generated 3,978,510 engagements records
2024-11-20 03:26:10 - marketing - INFO - Creating initial loyalty patterns...
2024-11-20 03:26:11 - marketing - INFO - Creating metadata for loyalties...
2024-11-20 03:26:11 - marketing - INFO - Deleted existing metadata file: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/loyalties.json
2024-11-20 03:26:11 - marketing - INFO - Saved metadata to: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/loyalties.json
2024-11-20 03:26:11 - marketing - INFO - Training loyalties generation model...
2024-11-20 03:26:11 - marketing - INFO - Training data shape: (82, 8)
2024-11-20 03:26:11 - marketing - INFO - Starting model training with 100 epochs...
Gen. (0.80) | Discrim. (-0.12): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 82.94it/s]
2024-11-20 03:26:12 - marketing - INFO - Model training completed in 0:00:01.295733
2024-11-20 03:26:12 - marketing - INFO - Generating 530,468 synthetic loyalties records...
2024-11-20 03:26:20 - marketing - INFO - Generated 530,468 loyalties records
2024-11-20 03:26:21 - marketing - INFO - Saved campaigns.csv to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/marketing_engagement/20241120_032620/campaigns.csv
2024-11-20 03:26:37 - marketing - INFO - Saved engagements.csv to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/marketing_engagement/20241120_032620/engagements.csv
2024-11-20 03:26:38 - marketing - INFO - Saved loyalties.csv to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/marketing_engagement/20241120_032620/loyalties.csv
2024-11-20 03:26:38 - marketing - INFO - Saved metadata to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/marketing_engagement/20241120_032620/metadata.json
Generating report ...

(1/2) Evaluating Data Validity: |██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 423.60it/s]|
Data Validity Score: 100.0%

(2/2) Evaluating Data Structure: |██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2050.00it/s]|
Data Structure Score: 100.0%

Overall Score (Average): 100.0%

2024-11-20 03:26:38 - marketing - INFO -
================================================================================
2024-11-20 03:26:38 - marketing - INFO -                        Campaigns Data Validation Results
2024-11-20 03:26:38 - marketing - INFO - ================================================================================

2024-11-20 03:26:38 - marketing - INFO -
----------------------------------------------------------------------------
2024-11-20 03:26:38 - marketing - INFO - DISTRIBUTIONS Validation
2024-11-20 03:26:38 - marketing - INFO - ----------------------------------------------------------------------------
2024-11-20 03:26:38 - marketing - INFO - Metric                    Expected     Actual       Status
2024-11-20 03:26:38 - marketing - INFO - --------------------------------------------------------------------------------
2024-11-20 03:26:38 - marketing - INFO -
numerical_metrics:
2024-11-20 03:26:38 - marketing - INFO - target_age_min            34.32        32.83        ❌
2024-11-20 03:26:38 - marketing - INFO - target_age_max            51.52        52.97        ❌
2024-11-20 03:26:38 - marketing - INFO - target_income_min         48500.00     50715.68     ❌
2024-11-20 03:26:38 - marketing - INFO - target_income_max         125500.00    97694.91     ❌
2024-11-20 03:26:38 - marketing - INFO - base_engagement_rate      4.97%        3.05%        ✅
2024-11-20 03:26:38 - marketing - INFO - budget                    26956.52     21294.91     ❌
2024-11-20 03:26:38 - marketing - INFO -
categorical_metrics:
2024-11-20 03:26:38 - marketing - INFO - campaign_type             distribution with 6 categories distribution with 6 categories ✅
2024-11-20 03:26:38 - marketing - INFO - primary_channel           distribution with 6 categories distribution with 6 categories ✅
2024-11-20 03:26:38 - marketing - INFO - creative_type             distribution with 5 categories distribution with 5 categories ✅
2024-11-20 03:26:38 - marketing - INFO -
----------------------------------------------------------------------------
2024-11-20 03:26:38 - marketing - INFO - SUMMARY
2024-11-20 03:26:38 - marketing - INFO - ----------------------------------------------------------------------------
2024-11-20 03:26:38 - marketing - INFO - Overall Score:          100.00% ✅
2024-11-20 03:26:38 - marketing - INFO - Total Metrics Checked:  9
2024-11-20 03:26:38 - marketing - INFO - Passing Metrics:        4
Generating report ...

(1/2) Evaluating Data Validity: |███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 12.13it/s]|
Data Validity Score: 100.0%

(2/2) Evaluating Data Structure: |██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2198.27it/s]|
Data Structure Score: 100.0%

Overall Score (Average): 100.0%

2024-11-20 03:26:39 - marketing - INFO -
================================================================================
2024-11-20 03:26:39 - marketing - INFO -                       Engagements Data Validation Results
2024-11-20 03:26:39 - marketing - INFO - ================================================================================

2024-11-20 03:26:39 - marketing - INFO -
----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - DISTRIBUTIONS Validation
2024-11-20 03:26:39 - marketing - INFO - ----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - Metric                    Expected     Actual       Status
2024-11-20 03:26:39 - marketing - INFO - --------------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO -
numerical_metrics:
2024-11-20 03:26:39 - marketing - INFO - impressions               10254.92     12928.15     ❌
2024-11-20 03:26:39 - marketing - INFO - clicks                    70.80        28.99        ❌
2024-11-20 03:26:39 - marketing - INFO - engagement_rate           0.67%        0.51%        ✅
2024-11-20 03:26:39 - marketing - INFO - time_spent_minutes        2.15         1.73         ❌
2024-11-20 03:26:39 - marketing - INFO - conversion_rate           0.07%        0.17%        ✅
2024-11-20 03:26:39 - marketing - INFO - campaign_day              100.10       110.34       ❌
2024-11-20 03:26:39 - marketing - INFO - campaign_progress         48.22%       67.59%       ❌
2024-11-20 03:26:39 - marketing - INFO -
categorical_metrics:
2024-11-20 03:26:39 - marketing - INFO - day_of_week               distribution with 7 categories distribution with 7 categories ❌
2024-11-20 03:26:39 - marketing - INFO -
----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - ENGAGEMENT Validation
2024-11-20 03:26:39 - marketing - INFO - ----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - Metric                    Expected     Actual       Status
2024-11-20 03:26:39 - marketing - INFO - --------------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO -
metrics:
2024-11-20 03:26:39 - marketing - INFO -
temporal:
2024-11-20 03:26:39 - marketing - INFO - day_0                     0.90%        0.52%        ✅
2024-11-20 03:26:39 - marketing - INFO - day_1                     1.03%        0.50%        ✅
2024-11-20 03:26:39 - marketing - INFO - day_2                     1.04%        0.50%        ✅
2024-11-20 03:26:39 - marketing - INFO - day_3                     0.39%        0.48%        ✅
2024-11-20 03:26:39 - marketing - INFO - day_4                     0.86%        0.52%        ✅
2024-11-20 03:26:39 - marketing - INFO - day_5                     0.65%        0.52%        ✅
2024-11-20 03:26:39 - marketing - INFO - day_6                     0.63%        0.52%        ✅
2024-11-20 03:26:39 - marketing - INFO -
----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - SUMMARY
2024-11-20 03:26:39 - marketing - INFO - ----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - Overall Score:          100.00% ✅
2024-11-20 03:26:39 - marketing - INFO - Total Metrics Checked:  15
2024-11-20 03:26:39 - marketing - INFO - Passing Metrics:        9
Generating report ...

(1/2) Evaluating Data Validity: |█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 88.94it/s]|
Data Validity Score: 100.0%

(2/2) Evaluating Data Structure: |██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2303.30it/s]|
Data Structure Score: 100.0%

Overall Score (Average): 100.0%

2024-11-20 03:26:39 - marketing - INFO -
================================================================================
2024-11-20 03:26:39 - marketing - INFO -                        Loyalties Data Validation Results
2024-11-20 03:26:39 - marketing - INFO - ================================================================================

2024-11-20 03:26:39 - marketing - INFO -
----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - DISTRIBUTIONS Validation
2024-11-20 03:26:39 - marketing - INFO - ----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - Metric                    Expected     Actual       Status
2024-11-20 03:26:39 - marketing - INFO - --------------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO -
numerical_metrics:
2024-11-20 03:26:39 - marketing - INFO - points_balance            1143.15      457.87       ❌
2024-11-20 03:26:39 - marketing - INFO - lifetime_points           2548.33      3351.55      ❌
2024-11-20 03:26:39 - marketing - INFO - redemption_rate           35.00%       35.00%       ✅
2024-11-20 03:26:39 - marketing - INFO -
categorical_metrics:
2024-11-20 03:26:39 - marketing - INFO - status                    semi_active: 52.4%, active: 47.6% semi_active: 53.8%, active: 46.2% ✅
2024-11-20 03:26:39 - marketing - INFO - tier                      Bronze: 100.0% Bronze: 100.0% ✅
2024-11-20 03:26:39 - marketing - INFO - age_group                 55+: 35.4%, 18-34: 34.1%, 35-54: 30.5% 55+: 35.9%, 35-54: 32.1%, 18-34: 32.0% ✅
2024-11-20 03:26:39 - marketing - INFO -
----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - LOYALTY Validation
2024-11-20 03:26:39 - marketing - INFO - ----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - Metric                    Expected     Actual       Status
2024-11-20 03:26:39 - marketing - INFO - --------------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO -
program_metrics:
2024-11-20 03:26:39 - marketing - INFO -
----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - SUMMARY
2024-11-20 03:26:39 - marketing - INFO - ----------------------------------------------------------------------------
2024-11-20 03:26:39 - marketing - INFO - Overall Score:          100.00% ✅
2024-11-20 03:26:39 - marketing - INFO - Total Metrics Checked:  6
2024-11-20 03:26:39 - marketing - INFO - Passing Metrics:        4
2024-11-20 03:26:39 - marketing - INFO - Marketing engagement data generation completed successfully
2024-11-20 03:26:39 - build_dataset - INFO - ✓ marketing completed successfully in 136.4s
2024-11-20 03:26:39 - build_dataset - INFO - ✓ Output verification passed for marketing
2024-11-20 03:26:39 - build_dataset - INFO - ✓ Completed Marketing engagement generation
2024-11-20 03:26:39 - build_dataset - INFO -
==================== Processing Transaction data ====================
2024-11-20 03:26:39 - build_dataset - INFO - Checking dependencies for transaction...
2024-11-20 03:26:39 - build_dataset - INFO - ✓ All dependencies satisfied for transaction
2024-11-20 03:26:39 - build_dataset - INFO -
==================== Running transaction ====================
2024-11-20 03:26:39 - transactions - INFO - Starting transaction data generation...
2024-11-20 03:26:39 - transactions - INFO - Loading input data...
2024-11-20 03:26:40 - transactions - INFO - Loaded 1,326,172 demographic records
2024-11-20 03:26:42 - transactions - INFO - Loaded 1,326,172 consumer preference records
2024-11-20 03:26:42 - transactions - INFO - Loaded 132,617 campaign records
2024-11-20 03:26:42 - transactions - INFO - Generating product catalog...
2024-11-20 03:26:42 - transactions - INFO - Generated catalog with 71 products
2024-11-20 03:26:42 - transactions - INFO -
Product Catalog Summary:
2024-11-20 03:26:42 - transactions - INFO - Categories: ready_to_eat, snacks, sustainable, family_size, healthy_alternatives, traditional
2024-11-20 03:26:42 - transactions - INFO - Price range: $3.11 - $15.74
2024-11-20 03:26:42 - transactions - INFO -
Products per category:
2024-11-20 03:26:42 - transactions - INFO -   ready_to_eat: 10 products
2024-11-20 03:26:42 - transactions - INFO -   snacks: 12 products
2024-11-20 03:26:42 - transactions - INFO -   sustainable: 13 products
2024-11-20 03:26:42 - transactions - INFO -   family_size: 13 products
2024-11-20 03:26:42 - transactions - INFO -   healthy_alternatives: 12 products
2024-11-20 03:26:42 - transactions - INFO -   traditional: 11 products
2024-11-20 03:26:42 - transactions - INFO - Creating initial transaction patterns...
2024-11-20 03:55:09 - transactions - INFO - Created 132617 seed transactions
2024-11-20 03:55:09 - transactions - INFO - Region distribution in transactions:
2024-11-20 03:55:09 - transactions - INFO -   South: 51689 (39.0%)
2024-11-20 03:55:09 - transactions - INFO -   West: 47313 (35.7%)
2024-11-20 03:55:09 - transactions - INFO -   Northeast: 20882 (15.7%)
2024-11-20 03:55:09 - transactions - INFO -   Midwest: 12733 (9.6%)
2024-11-20 03:55:09 - transactions - INFO - Creating metadata for transactions...
2024-11-20 03:55:09 - transactions - INFO - Deleted existing metadata file: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/transactions.json
2024-11-20 03:55:09 - transactions - INFO - Saved metadata to: /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/metadata/transactions.json
2024-11-20 03:55:09 - transactions - INFO -
Training synthetic data model for transactions...
2024-11-20 03:55:09 - transactions - INFO - Starting model training with configuration:
2024-11-20 03:55:09 - transactions - INFO -   Batch size: 132616
2024-11-20 03:55:09 - transactions - INFO -   PAC: 8
2024-11-20 03:55:09 - transactions - INFO -   Epochs: 100
2024-11-20 03:55:09 - transactions - INFO -   Generator LR: 0.0002
2024-11-20 03:55:09 - transactions - INFO -   Discriminator LR: 0.0001
2024-11-20 03:55:09 - transactions - INFO -   Training data shape: (132617, 10)
python(49331) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49332) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49333) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49334) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49335) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49336) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49337) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49338) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49339) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49340) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49341) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(49342) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Gen. (-0.07) | Discrim. (0.07): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [05:05<00:00,  3.05s/it]
2024-11-20 04:00:28 - transactions - INFO -
Training completed:
2024-11-20 04:00:28 - transactions - INFO -   Total time: 318.9s
2024-11-20 04:00:28 - transactions - INFO -   Avg time per epoch: 3.2s
2024-11-20 04:00:28 - transactions - INFO - Generating 3978516 synthetic transactions...
2024-11-20 04:01:42 - transactions - INFO - transactions generation completed
2024-11-20 04:01:42 - transactions - INFO - Generating transaction details...
2024-11-20 04:01:42 - transactions - INFO - Product catalog contains 71 products
2024-11-20 04:01:42 - transactions - INFO - Product categories available:
2024-11-20 04:01:42 - transactions - INFO -   - ready_to_eat
2024-11-20 04:01:42 - transactions - INFO -   - snacks
2024-11-20 04:01:42 - transactions - INFO -   - sustainable
2024-11-20 04:01:42 - transactions - INFO -   - family_size
2024-11-20 04:01:42 - transactions - INFO -   - healthy_alternatives
2024-11-20 04:01:42 - transactions - INFO -   - traditional
2024-11-20 04:01:46 - transactions - INFO - Processing transaction 0/3978516
2024-11-20 04:01:48 - transactions - INFO - Processing transaction 10000/3978516
2024-11-20 04:01:51 - transactions - INFO - Processing transaction 20000/3978516
2024-11-20 04:01:54 - transactions - INFO - Processing transaction 30000/3978516
2024-11-20 04:01:57 - transactions - INFO - Processing transaction 40000/3978516
2024-11-20 04:02:00 - transactions - INFO - Processing transaction 50000/3978516
2024-11-20 04:02:03 - transactions - INFO - Processing transaction 60000/3978516
2024-11-20 04:02:06 - transactions - INFO - Processing transaction 70000/3978516
2024-11-20 04:02:09 - transactions - INFO - Processing transaction 80000/3978516
2024-11-20 04:02:12 - transactions - INFO - Processing transaction 90000/3978516
2024-11-20 04:02:14 - transactions - INFO - Processing transaction 100000/3978516
2024-11-20 04:02:17 - transactions - INFO - Processing transaction 110000/3978516
2024-11-20 04:02:20 - transactions - INFO - Processing transaction 120000/3978516
2024-11-20 04:02:23 - transactions - INFO - Processing transaction 130000/3978516
2024-11-20 04:02:26 - transactions - INFO - Processing transaction 140000/3978516
2024-11-20 04:02:29 - transactions - INFO - Processing transaction 150000/3978516
2024-11-20 04:02:32 - transactions - INFO - Processing transaction 160000/3978516
2024-11-20 04:02:34 - transactions - INFO - Processing transaction 170000/3978516
2024-11-20 04:02:37 - transactions - INFO - Processing transaction 180000/3978516
2024-11-20 04:02:40 - transactions - INFO - Processing transaction 190000/3978516
2024-11-20 04:02:43 - transactions - INFO - Processing transaction 200000/3978516
2024-11-20 04:02:46 - transactions - INFO - Processing transaction 210000/3978516
2024-11-20 04:02:49 - transactions - INFO - Processing transaction 220000/3978516
2024-11-20 04:02:51 - transactions - INFO - Processing transaction 230000/3978516
2024-11-20 04:02:54 - transactions - INFO - Processing transaction 240000/3978516
2024-11-20 04:02:57 - transactions - INFO - Processing transaction 250000/3978516
2024-11-20 04:03:00 - transactions - INFO - Processing transaction 260000/3978516
2024-11-20 04:03:03 - transactions - INFO - Processing transaction 270000/3978516
2024-11-20 04:03:06 - transactions - INFO - Processing transaction 280000/3978516
2024-11-20 04:03:08 - transactions - INFO - Processing transaction 290000/3978516
2024-11-20 04:03:11 - transactions - INFO - Processing transaction 300000/3978516
2024-11-20 04:03:14 - transactions - INFO - Processing transaction 310000/3978516
2024-11-20 04:03:17 - transactions - INFO - Processing transaction 320000/3978516
2024-11-20 04:03:20 - transactions - INFO - Processing transaction 330000/3978516
2024-11-20 04:03:23 - transactions - INFO - Processing transaction 340000/3978516
2024-11-20 04:03:26 - transactions - INFO - Processing transaction 350000/3978516
2024-11-20 04:03:28 - transactions - INFO - Processing transaction 360000/3978516
2024-11-20 04:03:31 - transactions - INFO - Processing transaction 370000/3978516
2024-11-20 04:03:34 - transactions - INFO - Processing transaction 380000/3978516
2024-11-20 04:03:37 - transactions - INFO - Processing transaction 390000/3978516
2024-11-20 04:03:40 - transactions - INFO - Processing transaction 400000/3978516
2024-11-20 04:03:43 - transactions - INFO - Processing transaction 410000/3978516
2024-11-20 04:03:45 - transactions - INFO - Processing transaction 420000/3978516
2024-11-20 04:03:48 - transactions - INFO - Processing transaction 430000/3978516
2024-11-20 04:03:51 - transactions - INFO - Processing transaction 440000/3978516
2024-11-20 04:03:54 - transactions - INFO - Processing transaction 450000/3978516
2024-11-20 04:03:57 - transactions - INFO - Processing transaction 460000/3978516
2024-11-20 04:04:00 - transactions - INFO - Processing transaction 470000/3978516
2024-11-20 04:04:03 - transactions - INFO - Processing transaction 480000/3978516
2024-11-20 04:04:05 - transactions - INFO - Processing transaction 490000/3978516
2024-11-20 04:04:08 - transactions - INFO - Processing transaction 500000/3978516
2024-11-20 04:04:11 - transactions - INFO - Processing transaction 510000/3978516
2024-11-20 04:04:14 - transactions - INFO - Processing transaction 520000/3978516
2024-11-20 04:04:17 - transactions - INFO - Processing transaction 530000/3978516
2024-11-20 04:04:20 - transactions - INFO - Processing transaction 540000/3978516
2024-11-20 04:04:23 - transactions - INFO - Processing transaction 550000/3978516
2024-11-20 04:04:26 - transactions - INFO - Processing transaction 560000/3978516
2024-11-20 04:04:28 - transactions - INFO - Processing transaction 570000/3978516
2024-11-20 04:04:31 - transactions - INFO - Processing transaction 580000/3978516
2024-11-20 04:04:34 - transactions - INFO - Processing transaction 590000/3978516
2024-11-20 04:04:37 - transactions - INFO - Processing transaction 600000/3978516
2024-11-20 04:04:40 - transactions - INFO - Processing transaction 610000/3978516
2024-11-20 04:04:43 - transactions - INFO - Processing transaction 620000/3978516
2024-11-20 04:04:45 - transactions - INFO - Processing transaction 630000/3978516
2024-11-20 04:04:48 - transactions - INFO - Processing transaction 640000/3978516
2024-11-20 04:04:51 - transactions - INFO - Processing transaction 650000/3978516
2024-11-20 04:04:54 - transactions - INFO - Processing transaction 660000/3978516
2024-11-20 04:04:57 - transactions - INFO - Processing transaction 670000/3978516
2024-11-20 04:05:00 - transactions - INFO - Processing transaction 680000/3978516
2024-11-20 04:05:02 - transactions - INFO - Processing transaction 690000/3978516
2024-11-20 04:05:05 - transactions - INFO - Processing transaction 700000/3978516
2024-11-20 04:05:08 - transactions - INFO - Processing transaction 710000/3978516
2024-11-20 04:05:11 - transactions - INFO - Processing transaction 720000/3978516
2024-11-20 04:05:14 - transactions - INFO - Processing transaction 730000/3978516
2024-11-20 04:05:17 - transactions - INFO - Processing transaction 740000/3978516
2024-11-20 04:05:19 - transactions - INFO - Processing transaction 750000/3978516
2024-11-20 04:05:23 - transactions - INFO - Processing transaction 760000/3978516
2024-11-20 04:05:25 - transactions - INFO - Processing transaction 770000/3978516
2024-11-20 04:05:28 - transactions - INFO - Processing transaction 780000/3978516
2024-11-20 04:05:31 - transactions - INFO - Processing transaction 790000/3978516
2024-11-20 04:05:34 - transactions - INFO - Processing transaction 800000/3978516
2024-11-20 04:05:37 - transactions - INFO - Processing transaction 810000/3978516
2024-11-20 04:05:40 - transactions - INFO - Processing transaction 820000/3978516
2024-11-20 04:05:42 - transactions - INFO - Processing transaction 830000/3978516
2024-11-20 04:05:45 - transactions - INFO - Processing transaction 840000/3978516
2024-11-20 04:05:48 - transactions - INFO - Processing transaction 850000/3978516
2024-11-20 04:05:51 - transactions - INFO - Processing transaction 860000/3978516
2024-11-20 04:05:54 - transactions - INFO - Processing transaction 870000/3978516
2024-11-20 04:05:57 - transactions - INFO - Processing transaction 880000/3978516
2024-11-20 04:06:00 - transactions - INFO - Processing transaction 890000/3978516
2024-11-20 04:06:03 - transactions - INFO - Processing transaction 900000/3978516
2024-11-20 04:06:05 - transactions - INFO - Processing transaction 910000/3978516
2024-11-20 04:06:08 - transactions - INFO - Processing transaction 920000/3978516
2024-11-20 04:06:11 - transactions - INFO - Processing transaction 930000/3978516
2024-11-20 04:06:14 - transactions - INFO - Processing transaction 940000/3978516
2024-11-20 04:06:17 - transactions - INFO - Processing transaction 950000/3978516
2024-11-20 04:06:20 - transactions - INFO - Processing transaction 960000/3978516
2024-11-20 04:06:22 - transactions - INFO - Processing transaction 970000/3978516
2024-11-20 04:06:25 - transactions - INFO - Processing transaction 980000/3978516
2024-11-20 04:06:28 - transactions - INFO - Processing transaction 990000/3978516
2024-11-20 04:06:31 - transactions - INFO - Processing transaction 1000000/3978516
2024-11-20 04:06:34 - transactions - INFO - Processing transaction 1010000/3978516
2024-11-20 04:06:37 - transactions - INFO - Processing transaction 1020000/3978516
2024-11-20 04:06:40 - transactions - INFO - Processing transaction 1030000/3978516
2024-11-20 04:06:42 - transactions - INFO - Processing transaction 1040000/3978516
2024-11-20 04:06:45 - transactions - INFO - Processing transaction 1050000/3978516
2024-11-20 04:06:48 - transactions - INFO - Processing transaction 1060000/3978516
2024-11-20 04:06:51 - transactions - INFO - Processing transaction 1070000/3978516
2024-11-20 04:06:54 - transactions - INFO - Processing transaction 1080000/3978516
2024-11-20 04:06:57 - transactions - INFO - Processing transaction 1090000/3978516
2024-11-20 04:06:59 - transactions - INFO - Processing transaction 1100000/3978516
2024-11-20 04:07:02 - transactions - INFO - Processing transaction 1110000/3978516
2024-11-20 04:07:05 - transactions - INFO - Processing transaction 1120000/3978516
2024-11-20 04:07:08 - transactions - INFO - Processing transaction 1130000/3978516
2024-11-20 04:07:11 - transactions - INFO - Processing transaction 1140000/3978516
2024-11-20 04:07:14 - transactions - INFO - Processing transaction 1150000/3978516
2024-11-20 04:07:16 - transactions - INFO - Processing transaction 1160000/3978516
2024-11-20 04:07:19 - transactions - INFO - Processing transaction 1170000/3978516
2024-11-20 04:07:22 - transactions - INFO - Processing transaction 1180000/3978516
2024-11-20 04:07:25 - transactions - INFO - Processing transaction 1190000/3978516
2024-11-20 04:07:28 - transactions - INFO - Processing transaction 1200000/3978516
2024-11-20 04:07:31 - transactions - INFO - Processing transaction 1210000/3978516
2024-11-20 04:07:33 - transactions - INFO - Processing transaction 1220000/3978516
2024-11-20 04:07:36 - transactions - INFO - Processing transaction 1230000/3978516
2024-11-20 04:07:39 - transactions - INFO - Processing transaction 1240000/3978516
2024-11-20 04:07:42 - transactions - INFO - Processing transaction 1250000/3978516
2024-11-20 04:07:45 - transactions - INFO - Processing transaction 1260000/3978516
2024-11-20 04:07:48 - transactions - INFO - Processing transaction 1270000/3978516
2024-11-20 04:07:50 - transactions - INFO - Processing transaction 1280000/3978516
2024-11-20 04:07:53 - transactions - INFO - Processing transaction 1290000/3978516
2024-11-20 04:07:56 - transactions - INFO - Processing transaction 1300000/3978516
2024-11-20 04:07:59 - transactions - INFO - Processing transaction 1310000/3978516
2024-11-20 04:08:02 - transactions - INFO - Processing transaction 1320000/3978516
2024-11-20 04:08:05 - transactions - INFO - Processing transaction 1330000/3978516
2024-11-20 04:08:08 - transactions - INFO - Processing transaction 1340000/3978516
2024-11-20 04:08:11 - transactions - INFO - Processing transaction 1350000/3978516
2024-11-20 04:08:13 - transactions - INFO - Processing transaction 1360000/3978516
2024-11-20 04:08:16 - transactions - INFO - Processing transaction 1370000/3978516
2024-11-20 04:08:19 - transactions - INFO - Processing transaction 1380000/3978516
2024-11-20 04:08:22 - transactions - INFO - Processing transaction 1390000/3978516
2024-11-20 04:08:25 - transactions - INFO - Processing transaction 1400000/3978516
2024-11-20 04:08:28 - transactions - INFO - Processing transaction 1410000/3978516
2024-11-20 04:08:30 - transactions - INFO - Processing transaction 1420000/3978516
2024-11-20 04:08:33 - transactions - INFO - Processing transaction 1430000/3978516
2024-11-20 04:08:36 - transactions - INFO - Processing transaction 1440000/3978516
2024-11-20 04:08:39 - transactions - INFO - Processing transaction 1450000/3978516
2024-11-20 04:08:42 - transactions - INFO - Processing transaction 1460000/3978516
2024-11-20 04:08:44 - transactions - INFO - Processing transaction 1470000/3978516
2024-11-20 04:08:47 - transactions - INFO - Processing transaction 1480000/3978516
2024-11-20 04:08:50 - transactions - INFO - Processing transaction 1490000/3978516
2024-11-20 04:08:53 - transactions - INFO - Processing transaction 1500000/3978516
2024-11-20 04:08:56 - transactions - INFO - Processing transaction 1510000/3978516
2024-11-20 04:08:59 - transactions - INFO - Processing transaction 1520000/3978516
2024-11-20 04:09:02 - transactions - INFO - Processing transaction 1530000/3978516
2024-11-20 04:09:04 - transactions - INFO - Processing transaction 1540000/3978516
2024-11-20 04:09:07 - transactions - INFO - Processing transaction 1550000/3978516
2024-11-20 04:09:10 - transactions - INFO - Processing transaction 1560000/3978516
2024-11-20 04:09:13 - transactions - INFO - Processing transaction 1570000/3978516
2024-11-20 04:09:16 - transactions - INFO - Processing transaction 1580000/3978516
2024-11-20 04:09:19 - transactions - INFO - Processing transaction 1590000/3978516
2024-11-20 04:09:21 - transactions - INFO - Processing transaction 1600000/3978516
2024-11-20 04:09:24 - transactions - INFO - Processing transaction 1610000/3978516
2024-11-20 04:09:27 - transactions - INFO - Processing transaction 1620000/3978516
2024-11-20 04:09:30 - transactions - INFO - Processing transaction 1630000/3978516
2024-11-20 04:09:33 - transactions - INFO - Processing transaction 1640000/3978516
2024-11-20 04:09:36 - transactions - INFO - Processing transaction 1650000/3978516
2024-11-20 04:09:38 - transactions - INFO - Processing transaction 1660000/3978516
2024-11-20 04:09:41 - transactions - INFO - Processing transaction 1670000/3978516
2024-11-20 04:09:44 - transactions - INFO - Processing transaction 1680000/3978516
2024-11-20 04:09:47 - transactions - INFO - Processing transaction 1690000/3978516
2024-11-20 04:09:50 - transactions - INFO - Processing transaction 1700000/3978516
2024-11-20 04:09:53 - transactions - INFO - Processing transaction 1710000/3978516
2024-11-20 04:09:56 - transactions - INFO - Processing transaction 1720000/3978516
2024-11-20 04:09:59 - transactions - INFO - Processing transaction 1730000/3978516
2024-11-20 04:10:01 - transactions - INFO - Processing transaction 1740000/3978516
2024-11-20 04:10:04 - transactions - INFO - Processing transaction 1750000/3978516
2024-11-20 04:10:07 - transactions - INFO - Processing transaction 1760000/3978516
2024-11-20 04:10:10 - transactions - INFO - Processing transaction 1770000/3978516
2024-11-20 04:10:13 - transactions - INFO - Processing transaction 1780000/3978516
2024-11-20 04:10:16 - transactions - INFO - Processing transaction 1790000/3978516
2024-11-20 04:10:18 - transactions - INFO - Processing transaction 1800000/3978516
2024-11-20 04:10:21 - transactions - INFO - Processing transaction 1810000/3978516
2024-11-20 04:10:24 - transactions - INFO - Processing transaction 1820000/3978516
2024-11-20 04:10:27 - transactions - INFO - Processing transaction 1830000/3978516
2024-11-20 04:10:30 - transactions - INFO - Processing transaction 1840000/3978516
2024-11-20 04:10:33 - transactions - INFO - Processing transaction 1850000/3978516
2024-11-20 04:10:35 - transactions - INFO - Processing transaction 1860000/3978516
2024-11-20 04:10:38 - transactions - INFO - Processing transaction 1870000/3978516
2024-11-20 04:10:41 - transactions - INFO - Processing transaction 1880000/3978516
2024-11-20 04:10:44 - transactions - INFO - Processing transaction 1890000/3978516
2024-11-20 04:10:47 - transactions - INFO - Processing transaction 1900000/3978516
2024-11-20 04:10:50 - transactions - INFO - Processing transaction 1910000/3978516
2024-11-20 04:10:52 - transactions - INFO - Processing transaction 1920000/3978516
2024-11-20 04:10:55 - transactions - INFO - Processing transaction 1930000/3978516
2024-11-20 04:10:58 - transactions - INFO - Processing transaction 1940000/3978516
2024-11-20 04:11:01 - transactions - INFO - Processing transaction 1950000/3978516
2024-11-20 04:11:04 - transactions - INFO - Processing transaction 1960000/3978516
2024-11-20 04:11:07 - transactions - INFO - Processing transaction 1970000/3978516
2024-11-20 04:11:10 - transactions - INFO - Processing transaction 1980000/3978516
2024-11-20 04:11:12 - transactions - INFO - Processing transaction 1990000/3978516
2024-11-20 04:11:15 - transactions - INFO - Processing transaction 2000000/3978516
2024-11-20 04:11:18 - transactions - INFO - Processing transaction 2010000/3978516
2024-11-20 04:11:21 - transactions - INFO - Processing transaction 2020000/3978516
2024-11-20 04:11:24 - transactions - INFO - Processing transaction 2030000/3978516
2024-11-20 04:11:27 - transactions - INFO - Processing transaction 2040000/3978516
2024-11-20 04:11:29 - transactions - INFO - Processing transaction 2050000/3978516
2024-11-20 04:11:32 - transactions - INFO - Processing transaction 2060000/3978516
2024-11-20 04:11:35 - transactions - INFO - Processing transaction 2070000/3978516
2024-11-20 04:11:38 - transactions - INFO - Processing transaction 2080000/3978516
2024-11-20 04:11:41 - transactions - INFO - Processing transaction 2090000/3978516
2024-11-20 04:11:44 - transactions - INFO - Processing transaction 2100000/3978516
2024-11-20 04:11:46 - transactions - INFO - Processing transaction 2110000/3978516
2024-11-20 04:11:49 - transactions - INFO - Processing transaction 2120000/3978516
2024-11-20 04:11:52 - transactions - INFO - Processing transaction 2130000/3978516
2024-11-20 04:11:55 - transactions - INFO - Processing transaction 2140000/3978516
2024-11-20 04:11:58 - transactions - INFO - Processing transaction 2150000/3978516
2024-11-20 04:12:01 - transactions - INFO - Processing transaction 2160000/3978516
2024-11-20 04:12:04 - transactions - INFO - Processing transaction 2170000/3978516
2024-11-20 04:12:07 - transactions - INFO - Processing transaction 2180000/3978516
2024-11-20 04:12:09 - transactions - INFO - Processing transaction 2190000/3978516
2024-11-20 04:12:12 - transactions - INFO - Processing transaction 2200000/3978516
2024-11-20 04:12:15 - transactions - INFO - Processing transaction 2210000/3978516
2024-11-20 04:12:18 - transactions - INFO - Processing transaction 2220000/3978516
2024-11-20 04:12:21 - transactions - INFO - Processing transaction 2230000/3978516
2024-11-20 04:12:24 - transactions - INFO - Processing transaction 2240000/3978516
2024-11-20 04:12:27 - transactions - INFO - Processing transaction 2250000/3978516
2024-11-20 04:12:30 - transactions - INFO - Processing transaction 2260000/3978516
2024-11-20 04:12:32 - transactions - INFO - Processing transaction 2270000/3978516
2024-11-20 04:12:35 - transactions - INFO - Processing transaction 2280000/3978516
2024-11-20 04:12:38 - transactions - INFO - Processing transaction 2290000/3978516
2024-11-20 04:12:41 - transactions - INFO - Processing transaction 2300000/3978516
2024-11-20 04:12:44 - transactions - INFO - Processing transaction 2310000/3978516
2024-11-20 04:12:46 - transactions - INFO - Processing transaction 2320000/3978516
2024-11-20 04:12:50 - transactions - INFO - Processing transaction 2330000/3978516
2024-11-20 04:12:52 - transactions - INFO - Processing transaction 2340000/3978516
2024-11-20 04:12:55 - transactions - INFO - Processing transaction 2350000/3978516
2024-11-20 04:12:58 - transactions - INFO - Processing transaction 2360000/3978516
2024-11-20 04:13:01 - transactions - INFO - Processing transaction 2370000/3978516
2024-11-20 04:13:04 - transactions - INFO - Processing transaction 2380000/3978516
2024-11-20 04:13:07 - transactions - INFO - Processing transaction 2390000/3978516
2024-11-20 04:13:10 - transactions - INFO - Processing transaction 2400000/3978516
2024-11-20 04:13:13 - transactions - INFO - Processing transaction 2410000/3978516
2024-11-20 04:13:15 - transactions - INFO - Processing transaction 2420000/3978516
2024-11-20 04:13:18 - transactions - INFO - Processing transaction 2430000/3978516
2024-11-20 04:13:21 - transactions - INFO - Processing transaction 2440000/3978516
2024-11-20 04:13:24 - transactions - INFO - Processing transaction 2450000/3978516
2024-11-20 04:13:27 - transactions - INFO - Processing transaction 2460000/3978516
2024-11-20 04:13:30 - transactions - INFO - Processing transaction 2470000/3978516
2024-11-20 04:13:32 - transactions - INFO - Processing transaction 2480000/3978516
2024-11-20 04:13:36 - transactions - INFO - Processing transaction 2490000/3978516
2024-11-20 04:13:38 - transactions - INFO - Processing transaction 2500000/3978516
2024-11-20 04:13:41 - transactions - INFO - Processing transaction 2510000/3978516
2024-11-20 04:13:44 - transactions - INFO - Processing transaction 2520000/3978516
2024-11-20 04:13:47 - transactions - INFO - Processing transaction 2530000/3978516
2024-11-20 04:13:50 - transactions - INFO - Processing transaction 2540000/3978516
2024-11-20 04:13:53 - transactions - INFO - Processing transaction 2550000/3978516
2024-11-20 04:13:56 - transactions - INFO - Processing transaction 2560000/3978516
2024-11-20 04:13:59 - transactions - INFO - Processing transaction 2570000/3978516
2024-11-20 04:14:01 - transactions - INFO - Processing transaction 2580000/3978516
2024-11-20 04:14:04 - transactions - INFO - Processing transaction 2590000/3978516
2024-11-20 04:14:07 - transactions - INFO - Processing transaction 2600000/3978516
2024-11-20 04:14:10 - transactions - INFO - Processing transaction 2610000/3978516
2024-11-20 04:14:13 - transactions - INFO - Processing transaction 2620000/3978516
2024-11-20 04:14:16 - transactions - INFO - Processing transaction 2630000/3978516
2024-11-20 04:14:18 - transactions - INFO - Processing transaction 2640000/3978516
2024-11-20 04:14:21 - transactions - INFO - Processing transaction 2650000/3978516
2024-11-20 04:14:24 - transactions - INFO - Processing transaction 2660000/3978516
2024-11-20 04:14:27 - transactions - INFO - Processing transaction 2670000/3978516
2024-11-20 04:14:30 - transactions - INFO - Processing transaction 2680000/3978516
2024-11-20 04:14:33 - transactions - INFO - Processing transaction 2690000/3978516
2024-11-20 04:14:36 - transactions - INFO - Processing transaction 2700000/3978516
2024-11-20 04:14:38 - transactions - INFO - Processing transaction 2710000/3978516
2024-11-20 04:14:41 - transactions - INFO - Processing transaction 2720000/3978516
2024-11-20 04:14:44 - transactions - INFO - Processing transaction 2730000/3978516
2024-11-20 04:14:47 - transactions - INFO - Processing transaction 2740000/3978516
2024-11-20 04:14:50 - transactions - INFO - Processing transaction 2750000/3978516
2024-11-20 04:14:52 - transactions - INFO - Processing transaction 2760000/3978516
2024-11-20 04:14:55 - transactions - INFO - Processing transaction 2770000/3978516
2024-11-20 04:14:58 - transactions - INFO - Processing transaction 2780000/3978516
2024-11-20 04:15:01 - transactions - INFO - Processing transaction 2790000/3978516
2024-11-20 04:15:04 - transactions - INFO - Processing transaction 2800000/3978516
2024-11-20 04:15:07 - transactions - INFO - Processing transaction 2810000/3978516
2024-11-20 04:15:10 - transactions - INFO - Processing transaction 2820000/3978516
2024-11-20 04:15:12 - transactions - INFO - Processing transaction 2830000/3978516
2024-11-20 04:15:15 - transactions - INFO - Processing transaction 2840000/3978516
2024-11-20 04:15:18 - transactions - INFO - Processing transaction 2850000/3978516
2024-11-20 04:15:21 - transactions - INFO - Processing transaction 2860000/3978516
2024-11-20 04:15:24 - transactions - INFO - Processing transaction 2870000/3978516
2024-11-20 04:15:27 - transactions - INFO - Processing transaction 2880000/3978516
2024-11-20 04:15:30 - transactions - INFO - Processing transaction 2890000/3978516
2024-11-20 04:15:32 - transactions - INFO - Processing transaction 2900000/3978516
2024-11-20 04:15:35 - transactions - INFO - Processing transaction 2910000/3978516
2024-11-20 04:15:38 - transactions - INFO - Processing transaction 2920000/3978516
2024-11-20 04:15:41 - transactions - INFO - Processing transaction 2930000/3978516
2024-11-20 04:15:44 - transactions - INFO - Processing transaction 2940000/3978516
2024-11-20 04:15:47 - transactions - INFO - Processing transaction 2950000/3978516
2024-11-20 04:15:50 - transactions - INFO - Processing transaction 2960000/3978516
2024-11-20 04:15:53 - transactions - INFO - Processing transaction 2970000/3978516
2024-11-20 04:15:56 - transactions - INFO - Processing transaction 2980000/3978516
2024-11-20 04:15:58 - transactions - INFO - Processing transaction 2990000/3978516
2024-11-20 04:16:01 - transactions - INFO - Processing transaction 3000000/3978516
2024-11-20 04:16:04 - transactions - INFO - Processing transaction 3010000/3978516
2024-11-20 04:16:07 - transactions - INFO - Processing transaction 3020000/3978516
2024-11-20 04:16:10 - transactions - INFO - Processing transaction 3030000/3978516
2024-11-20 04:16:13 - transactions - INFO - Processing transaction 3040000/3978516
2024-11-20 04:16:16 - transactions - INFO - Processing transaction 3050000/3978516
2024-11-20 04:16:18 - transactions - INFO - Processing transaction 3060000/3978516
2024-11-20 04:16:21 - transactions - INFO - Processing transaction 3070000/3978516
2024-11-20 04:16:24 - transactions - INFO - Processing transaction 3080000/3978516
2024-11-20 04:16:27 - transactions - INFO - Processing transaction 3090000/3978516
2024-11-20 04:16:30 - transactions - INFO - Processing transaction 3100000/3978516
2024-11-20 04:16:33 - transactions - INFO - Processing transaction 3110000/3978516
2024-11-20 04:16:36 - transactions - INFO - Processing transaction 3120000/3978516
2024-11-20 04:16:39 - transactions - INFO - Processing transaction 3130000/3978516
2024-11-20 04:16:41 - transactions - INFO - Processing transaction 3140000/3978516
2024-11-20 04:16:44 - transactions - INFO - Processing transaction 3150000/3978516
2024-11-20 04:16:47 - transactions - INFO - Processing transaction 3160000/3978516
2024-11-20 04:16:50 - transactions - INFO - Processing transaction 3170000/3978516
2024-11-20 04:16:53 - transactions - INFO - Processing transaction 3180000/3978516
2024-11-20 04:16:56 - transactions - INFO - Processing transaction 3190000/3978516
2024-11-20 04:16:59 - transactions - INFO - Processing transaction 3200000/3978516
2024-11-20 04:17:01 - transactions - INFO - Processing transaction 3210000/3978516
2024-11-20 04:17:04 - transactions - INFO - Processing transaction 3220000/3978516
2024-11-20 04:17:07 - transactions - INFO - Processing transaction 3230000/3978516
2024-11-20 04:17:10 - transactions - INFO - Processing transaction 3240000/3978516
2024-11-20 04:17:13 - transactions - INFO - Processing transaction 3250000/3978516
2024-11-20 04:17:16 - transactions - INFO - Processing transaction 3260000/3978516
2024-11-20 04:17:19 - transactions - INFO - Processing transaction 3270000/3978516
2024-11-20 04:17:21 - transactions - INFO - Processing transaction 3280000/3978516
2024-11-20 04:17:24 - transactions - INFO - Processing transaction 3290000/3978516
2024-11-20 04:17:27 - transactions - INFO - Processing transaction 3300000/3978516
2024-11-20 04:17:30 - transactions - INFO - Processing transaction 3310000/3978516
2024-11-20 04:17:33 - transactions - INFO - Processing transaction 3320000/3978516
2024-11-20 04:17:36 - transactions - INFO - Processing transaction 3330000/3978516
2024-11-20 04:17:38 - transactions - INFO - Processing transaction 3340000/3978516
2024-11-20 04:17:41 - transactions - INFO - Processing transaction 3350000/3978516
2024-11-20 04:17:44 - transactions - INFO - Processing transaction 3360000/3978516
2024-11-20 04:17:47 - transactions - INFO - Processing transaction 3370000/3978516
2024-11-20 04:17:50 - transactions - INFO - Processing transaction 3380000/3978516
2024-11-20 04:17:53 - transactions - INFO - Processing transaction 3390000/3978516
2024-11-20 04:17:55 - transactions - INFO - Processing transaction 3400000/3978516
2024-11-20 04:17:58 - transactions - INFO - Processing transaction 3410000/3978516
2024-11-20 04:18:01 - transactions - INFO - Processing transaction 3420000/3978516
2024-11-20 04:18:04 - transactions - INFO - Processing transaction 3430000/3978516
2024-11-20 04:18:07 - transactions - INFO - Processing transaction 3440000/3978516
2024-11-20 04:18:10 - transactions - INFO - Processing transaction 3450000/3978516
2024-11-20 04:18:12 - transactions - INFO - Processing transaction 3460000/3978516
2024-11-20 04:18:15 - transactions - INFO - Processing transaction 3470000/3978516
2024-11-20 04:18:18 - transactions - INFO - Processing transaction 3480000/3978516
2024-11-20 04:18:21 - transactions - INFO - Processing transaction 3490000/3978516
2024-11-20 04:18:24 - transactions - INFO - Processing transaction 3500000/3978516
2024-11-20 04:18:27 - transactions - INFO - Processing transaction 3510000/3978516
2024-11-20 04:18:30 - transactions - INFO - Processing transaction 3520000/3978516
2024-11-20 04:18:33 - transactions - INFO - Processing transaction 3530000/3978516
2024-11-20 04:18:35 - transactions - INFO - Processing transaction 3540000/3978516
2024-11-20 04:18:38 - transactions - INFO - Processing transaction 3550000/3978516
2024-11-20 04:18:41 - transactions - INFO - Processing transaction 3560000/3978516
2024-11-20 04:18:44 - transactions - INFO - Processing transaction 3570000/3978516
2024-11-20 04:18:47 - transactions - INFO - Processing transaction 3580000/3978516
2024-11-20 04:18:50 - transactions - INFO - Processing transaction 3590000/3978516
2024-11-20 04:18:52 - transactions - INFO - Processing transaction 3600000/3978516
2024-11-20 04:18:55 - transactions - INFO - Processing transaction 3610000/3978516
2024-11-20 04:18:58 - transactions - INFO - Processing transaction 3620000/3978516
2024-11-20 04:19:01 - transactions - INFO - Processing transaction 3630000/3978516
2024-11-20 04:19:04 - transactions - INFO - Processing transaction 3640000/3978516
2024-11-20 04:19:07 - transactions - INFO - Processing transaction 3650000/3978516
2024-11-20 04:19:09 - transactions - INFO - Processing transaction 3660000/3978516
2024-11-20 04:19:12 - transactions - INFO - Processing transaction 3670000/3978516
2024-11-20 04:19:15 - transactions - INFO - Processing transaction 3680000/3978516
2024-11-20 04:19:18 - transactions - INFO - Processing transaction 3690000/3978516
2024-11-20 04:19:21 - transactions - INFO - Processing transaction 3700000/3978516
2024-11-20 04:19:24 - transactions - INFO - Processing transaction 3710000/3978516
2024-11-20 04:19:26 - transactions - INFO - Processing transaction 3720000/3978516
2024-11-20 04:19:29 - transactions - INFO - Processing transaction 3730000/3978516
2024-11-20 04:19:32 - transactions - INFO - Processing transaction 3740000/3978516
2024-11-20 04:19:35 - transactions - INFO - Processing transaction 3750000/3978516
2024-11-20 04:19:38 - transactions - INFO - Processing transaction 3760000/3978516
2024-11-20 04:19:41 - transactions - INFO - Processing transaction 3770000/3978516
2024-11-20 04:19:43 - transactions - INFO - Processing transaction 3780000/3978516
2024-11-20 04:19:46 - transactions - INFO - Processing transaction 3790000/3978516
2024-11-20 04:19:49 - transactions - INFO - Processing transaction 3800000/3978516
2024-11-20 04:19:52 - transactions - INFO - Processing transaction 3810000/3978516
2024-11-20 04:19:55 - transactions - INFO - Processing transaction 3820000/3978516
2024-11-20 04:19:58 - transactions - INFO - Processing transaction 3830000/3978516
2024-11-20 04:20:00 - transactions - INFO - Processing transaction 3840000/3978516
2024-11-20 04:20:03 - transactions - INFO - Processing transaction 3850000/3978516
2024-11-20 04:20:06 - transactions - INFO - Processing transaction 3860000/3978516
2024-11-20 04:20:09 - transactions - INFO - Processing transaction 3870000/3978516
2024-11-20 04:20:12 - transactions - INFO - Processing transaction 3880000/3978516
2024-11-20 04:20:15 - transactions - INFO - Processing transaction 3890000/3978516
2024-11-20 04:20:18 - transactions - INFO - Processing transaction 3900000/3978516
2024-11-20 04:20:20 - transactions - INFO - Processing transaction 3910000/3978516
2024-11-20 04:20:23 - transactions - INFO - Processing transaction 3920000/3978516
2024-11-20 04:20:26 - transactions - INFO - Processing transaction 3930000/3978516
2024-11-20 04:20:29 - transactions - INFO - Processing transaction 3940000/3978516
2024-11-20 04:20:32 - transactions - INFO - Processing transaction 3950000/3978516
2024-11-20 04:20:34 - transactions - INFO - Processing transaction 3960000/3978516
2024-11-20 04:20:37 - transactions - INFO - Processing transaction 3970000/3978516
2024-11-20 04:20:49 - transactions - INFO - Generated 13,421,606 transaction detail records
2024-11-20 04:20:49 - transactions - INFO -
Transaction Details Summary:
2024-11-20 04:20:50 - transactions - INFO - Total unique products used: 71
2024-11-20 04:20:50 - transactions - INFO - Average line total: $15.34
2024-11-20 04:20:50 - transactions - INFO -
Category distribution:
2024-11-20 04:20:50 - transactions - INFO -   family_size: 18.3%
2024-11-20 04:20:50 - transactions - INFO -   sustainable: 18.3%
2024-11-20 04:20:50 - transactions - INFO -   snacks: 16.9%
2024-11-20 04:20:50 - transactions - INFO -   healthy_alternatives: 16.9%
2024-11-20 04:20:50 - transactions - INFO -   traditional: 15.5%
2024-11-20 04:20:50 - transactions - INFO -   ready_to_eat: 14.1%
2024-11-20 04:20:51 - transactions - INFO - Generating transaction details...
2024-11-20 04:20:51 - transactions - INFO - Product catalog contains 71 products
2024-11-20 04:20:51 - transactions - INFO - Product categories available:
2024-11-20 04:20:51 - transactions - INFO -   - ready_to_eat
2024-11-20 04:20:51 - transactions - INFO -   - snacks
2024-11-20 04:20:51 - transactions - INFO -   - sustainable
2024-11-20 04:20:51 - transactions - INFO -   - family_size
2024-11-20 04:20:51 - transactions - INFO -   - healthy_alternatives
2024-11-20 04:20:51 - transactions - INFO -   - traditional
2024-11-20 04:20:51 - transactions - INFO - Processing transaction 0/132617
2024-11-20 04:20:54 - transactions - INFO - Processing transaction 10000/132617
2024-11-20 04:20:56 - transactions - INFO - Processing transaction 20000/132617
2024-11-20 04:20:59 - transactions - INFO - Processing transaction 30000/132617
2024-11-20 04:21:02 - transactions - INFO - Processing transaction 40000/132617
2024-11-20 04:21:04 - transactions - INFO - Processing transaction 50000/132617
2024-11-20 04:21:07 - transactions - INFO - Processing transaction 60000/132617
2024-11-20 04:21:09 - transactions - INFO - Processing transaction 70000/132617
2024-11-20 04:21:12 - transactions - INFO - Processing transaction 80000/132617
2024-11-20 04:21:14 - transactions - INFO - Processing transaction 90000/132617
2024-11-20 04:21:17 - transactions - INFO - Processing transaction 100000/132617
2024-11-20 04:21:19 - transactions - INFO - Processing transaction 110000/132617
2024-11-20 04:21:22 - transactions - INFO - Processing transaction 120000/132617
2024-11-20 04:21:24 - transactions - INFO - Processing transaction 130000/132617
2024-11-20 04:21:25 - transactions - INFO - Generated 397,074 transaction detail records
2024-11-20 04:21:25 - transactions - INFO -
Transaction Details Summary:
2024-11-20 04:21:25 - transactions - INFO - Total unique products used: 71
2024-11-20 04:21:25 - transactions - INFO - Average line total: $15.30
2024-11-20 04:21:25 - transactions - INFO -
Category distribution:
2024-11-20 04:21:25 - transactions - INFO -   sustainable: 18.4%
2024-11-20 04:21:25 - transactions - INFO -   family_size: 18.3%
2024-11-20 04:21:25 - transactions - INFO -   snacks: 17.0%
2024-11-20 04:21:25 - transactions - INFO -   healthy_alternatives: 16.9%
2024-11-20 04:21:25 - transactions - INFO -   traditional: 15.5%
2024-11-20 04:21:25 - transactions - INFO -   ready_to_eat: 14.0%
2024-11-20 04:21:28 - transactions - INFO -
================================================================================
2024-11-20 04:21:28 - transactions - INFO -                       Transaction Data Validation Results
2024-11-20 04:21:28 - transactions - INFO - ================================================================================

2024-11-20 04:21:28 - transactions - INFO -
----------------------------------------------------------------------------
2024-11-20 04:21:28 - transactions - INFO - TRANSACTIONS Validation
2024-11-20 04:21:28 - transactions - INFO - ----------------------------------------------------------------------------
2024-11-20 04:21:28 - transactions - INFO - Metric                    Expected     Actual       Status
2024-11-20 04:21:28 - transactions - INFO - --------------------------------------------------------------------------------
2024-11-20 04:21:28 - transactions - INFO -
numerical_metrics:
2024-11-20 04:21:28 - transactions - INFO - transaction_value         46.67        59.47        ❌
2024-11-20 04:21:28 - transactions - INFO - num_items                 2.99         3.37         ❌
2024-11-20 04:21:28 - transactions - INFO -
categorical_metrics:
2024-11-20 04:21:28 - transactions - INFO - channel                   desktop: 39.8%, mobile: 30.3%, in_store: 29.9% desktop: 38.2%, mobile: 31.2%, in_store: 30.6% ✅
2024-11-20 04:21:28 - transactions - INFO - state                     distribution with 6 categories distribution with 6 categories ✅
2024-11-20 04:21:28 - transactions - INFO - region                    distribution with 4 categories distribution with 4 categories ✅
2024-11-20 04:21:28 - transactions - INFO - age_group                 55+: 35.1%, 35-54: 32.6%, 18-34: 32.3% 55+: 34.7%, 18-34: 33.2%, 35-54: 32.1% ✅
2024-11-20 04:21:28 - transactions - INFO - buying_frequency          medium: 49.9%, low: 30.0%, high: 20.1% medium: 47.4%, low: 30.4%, high: 22.2% ✅
2024-11-20 04:21:28 - transactions - INFO -
transaction_metrics:
2024-11-20 04:21:28 - transactions - INFO - average_transaction_value 46.67        59.47        ❌
2024-11-20 04:21:28 - transactions - INFO - items_per_transaction     2.99         3.37         ✅
2024-11-20 04:21:28 - transactions - INFO - channel_in_store          29.91%       30.60%       ✅
2024-11-20 04:21:28 - transactions - INFO - channel_mobile            30.31%       31.20%       ✅
2024-11-20 04:21:28 - transactions - INFO - channel_desktop           39.78%       38.21%       ✅
2024-11-20 04:21:28 - transactions - INFO -
regional_metrics:
2024-11-20 04:21:28 - transactions - INFO - region_South              38.98%       36.60%       ✅
2024-11-20 04:21:28 - transactions - INFO - avg_value_South           41.41        59.09        ❌
2024-11-20 04:21:28 - transactions - INFO - region_West               35.68%       31.94%       ✅
2024-11-20 04:21:28 - transactions - INFO - avg_value_West            50.61        59.63        ❌
2024-11-20 04:21:28 - transactions - INFO - region_Midwest            9.60%        12.71%       ✅
2024-11-20 04:21:28 - transactions - INFO - avg_value_Midwest         43.58        60.72        ❌
2024-11-20 04:21:28 - transactions - INFO - region_Northeast          15.75%       18.75%       ✅
2024-11-20 04:21:28 - transactions - INFO - avg_value_Northeast       52.66        59.11        ❌
2024-11-20 04:21:28 - transactions - INFO -
----------------------------------------------------------------------------
2024-11-20 04:21:28 - transactions - INFO - TRANSACTION_DETAILS Validation
2024-11-20 04:21:28 - transactions - INFO - ----------------------------------------------------------------------------
2024-11-20 04:21:28 - transactions - INFO - Metric                    Expected     Actual       Status
2024-11-20 04:21:28 - transactions - INFO - --------------------------------------------------------------------------------
2024-11-20 04:21:28 - transactions - INFO -
numerical_metrics:
2024-11-20 04:21:28 - transactions - INFO - quantity                  2.00         2.00         ✅
2024-11-20 04:21:28 - transactions - INFO - unit_price                7.65         7.67         ✅
2024-11-20 04:21:28 - transactions - INFO - line_total                15.30        15.34        ✅
2024-11-20 04:21:28 - transactions - INFO -
categorical_metrics:
2024-11-20 04:21:28 - transactions - INFO - category                  distribution with 6 categories distribution with 6 categories ✅
2024-11-20 04:21:28 - transactions - INFO - subcategory               distribution with 19 categories distribution with 19 categories ✅
2024-11-20 04:21:28 - transactions - INFO -
----------------------------------------------------------------------------
2024-11-20 04:21:28 - transactions - INFO - SUMMARY
2024-11-20 04:21:28 - transactions - INFO - ----------------------------------------------------------------------------
2024-11-20 04:21:28 - transactions - INFO - Overall Score:          72.00% ❌
2024-11-20 04:21:28 - transactions - INFO - Total Metrics Checked:  25
2024-11-20 04:21:28 - transactions - INFO - Passing Metrics:        18
2024-11-20 04:21:39 - transactions - INFO - Saved transactions.csv to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/transactions/20241120_042128/transactions.csv
2024-11-20 04:21:59 - transactions - INFO - Saved transaction_details.csv to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/transactions/20241120_042128/transaction_details.csv
2024-11-20 04:21:59 - transactions - INFO - Saved product_catalog.csv to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/transactions/20241120_042128/product_catalog.csv
2024-11-20 04:21:59 - transactions - INFO - Saved metadata to /Users/jameshousteau/source_code/conagra/synthetic_marketing_data/data/transactions/20241120_042128/metadata.json
2024-11-20 04:21:59 - transactions - INFO - Transaction data generation completed successfully
2024-11-20 04:21:59 - build_dataset - INFO - ✓ transaction completed successfully in 3319.8s
2024-11-20 04:21:59 - build_dataset - INFO - ✓ Output verification passed for transaction
2024-11-20 04:21:59 - build_dataset - INFO - ✓ Completed Transaction data generation
2024-11-20 04:21:59 - build_dataset - INFO -
✓ Dataset build completed successfully
(conagra) jameshousteau@Calypso synthetic_marketing_data %