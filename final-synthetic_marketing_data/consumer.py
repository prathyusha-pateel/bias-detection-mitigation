"""Consumer preference behavior data generator.

Generates synthetic consumer preference data using CTGAN/SDV to maintain proper
statistical relationships with demographic data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

from constants import (
    METADATA_CONFIGURATIONS,
    MODEL_PARAMETERS,
    VALIDATION_THRESHOLDS,
    NOISE_FACTORS,
)

from validation.consumer.consumer_benchmarks import (
    SOCIAL_MEDIA_ADOPTION,
    PLATFORM_ENGAGEMENT,
    DEVICE_PATTERNS,
    PRODUCT_PREFERENCES,
    INCOME_CORRELATIONS,
    ONLINE_ADOPTION,
    LOYALTY_METRICS,
)

from utils.logging import setup_logging
from utils.validation_display import display_validation_metrics
from utils.validation_metrics import (
    get_numerical_validation_details,
    get_categorical_validation_details,
    count_total_metrics,
    count_passing_metrics,
)
from utils.encoders import NumpyEncoder

import time
from tqdm import tqdm
import json

# Initialize global logger
logger = setup_logging(module_name="consumer")


def load_demographic_data(data_dir: Path, logger) -> pd.DataFrame:
    """Load and combine demographic data files."""
    logger.info("Loading demographic data...")

    demo_files = list(data_dir.glob("*_demographics.csv"))
    if not demo_files:
        raise FileNotFoundError(
            "No demographic data found. Please run demographic.py first."
        )

    dfs = []
    for file in demo_files:
        logger.info(f"Reading {file.name}")
        df = pd.read_csv(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df):,} demographic records")
    return combined_df


def create_initial_preferences(demographic_data: pd.DataFrame, logger) -> pd.DataFrame:
    """Create initial preference dataset based on demographic patterns."""
    logger.info("Creating initial preference dataset...")
    logger.info(f"Processing {len(demographic_data):,} demographic records")

    preferences = pd.DataFrame()
    preferences["consumer_id"] = demographic_data.index.map(lambda x: f"C{int(x):06d}")
    logger.info("Generated consumer IDs")

    # Age groups for preference mapping - use consistent age groups
    age_groups = pd.cut(
        demographic_data["AGEP"],
        bins=[17, 34, 54, float("inf")],
        labels=["18-34", "35-54", "55+"],
    )

    # Handle NaN values and convert to string
    most_common_age = age_groups.mode()[0]
    preferences["age_group"] = age_groups.fillna(most_common_age).astype(str)

    # Log age group distribution
    logger.info("Age groups distribution:")
    for group in preferences["age_group"].value_counts().items():
        logger.info(
            f"  {group[0]}: {group[1]:,} consumers ({group[1]/len(preferences)*100:.1f}%)"
        )

    # Normalize income for correlations
    income_normalized = (
        demographic_data["PINCP"] - demographic_data["PINCP"].mean()
    ) / demographic_data["PINCP"].std()

    # Product Category Preferences
    for age_group, product_prefs in PRODUCT_PREFERENCES.items():
        mask = preferences["age_group"] == age_group
        for product, rate in product_prefs.items():
            noise_factor = NOISE_FACTORS[age_group]
            preferences.loc[mask, f"{product}_preference"] = (
                rate * (1 + np.random.normal(0, noise_factor, mask.sum()))
            ).clip(0, 1)

    # Fill missing product preferences
    product_columns = [col for col in preferences.columns if "_preference" in col]
    for col in product_columns:
        if col not in preferences.columns:
            preferences[col] = 0
        preferences[col] = preferences[col].fillna(0)

    # Online Shopping Adoption - Use direct mapping since benchmarks now use same age groups
    for age_group, rate in ONLINE_ADOPTION.items():
        mask = preferences["age_group"] == age_group
        noise_factor = NOISE_FACTORS[age_group]
        preferences.loc[mask, "online_shopping_rate"] = np.random.normal(
            rate, noise_factor, mask.sum()
        ).clip(0, 1)

    # Loyalty Program Behavior
    preferences["loyalty_memberships"] = np.random.normal(
        LOYALTY_METRICS["average_memberships"],
        LOYALTY_METRICS["average_memberships"] * 0.1,
        len(preferences),
    ).clip(0)

    for program, rate in LOYALTY_METRICS["program_engagement"].items():
        preferences[f"{program}_loyalty"] = np.random.normal(
            rate, 0.05, len(preferences)
        ).clip(0, 1)

    # Social Media Engagement with age-specific noise factors
    for age_group, rate in SOCIAL_MEDIA_ADOPTION.items():
        mask = preferences["age_group"] == age_group
        noise_factor = NOISE_FACTORS[age_group]
        preferences.loc[mask, "social_media_engagement_rate"] = (
            rate * (1 + np.random.normal(0, noise_factor, mask.sum()))
        ).clip(0, 1)

    # Platform-specific engagement
    for platform, rate in PLATFORM_ENGAGEMENT.items():
        preferences[f"{platform}_engagement"] = np.random.normal(
            rate, rate * 0.1, len(preferences)
        ).clip(0, 1)

    # Device usage patterns
    for device, rate in DEVICE_PATTERNS.items():
        preferences[f"{device}_usage"] = np.random.normal(
            rate, 0.05, len(preferences)
        ).clip(0, 1)

    # Apply income correlations
    for col, correlation in INCOME_CORRELATIONS.items():
        if col in preferences.columns and pd.api.types.is_numeric_dtype(
            preferences[col]
        ):
            preferences[col] = preferences[col] * (
                1 + (correlation * income_normalized)
            )
            preferences[col] = preferences[col].clip(0, preferences[col].max())

    logger.info(f"Created initial preferences for {len(preferences):,} consumers")
    return preferences


def create_metadata(initial_preferences: pd.DataFrame) -> Metadata:
    """Create metadata for synthesizer using SDV's recommended approach."""
    logger.info("Creating metadata...")

    metadata = Metadata()
    metadata = Metadata.detect_from_dataframe(
        data=initial_preferences, table_name="consumer"
    )

    # Get consumer-specific configuration
    consumer_config = METADATA_CONFIGURATIONS["consumer"]

    # Update specific column types based on configurations
    metadata.update_column(column_name=consumer_config["id_column"], sdtype="id")

    # Update categorical columns from configuration
    for col in consumer_config["categorical_columns"]:
        if col in initial_preferences.columns:
            metadata.update_column(column_name=col, sdtype="categorical")

    # Update numerical columns with bounds
    for col in consumer_config["numerical_columns"]:
        if col in initial_preferences.columns:
            metadata.update_column(
                column_name=col, sdtype="numerical", computer_representation="Float"
            )

    # Update preference columns
    for col in consumer_config["preference_columns"]:
        col_name = f"{col}_preference"
        if col_name in initial_preferences.columns:
            metadata.update_column(
                column_name=col_name,
                sdtype="numerical",
                computer_representation="Float",
            )

    metadata.validate()

    # Save metadata
    metadata_dir = Path(__file__).parent / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    metadata_file = metadata_dir / "consumer.json"

    if metadata_file.exists():
        metadata_file.unlink()
        logger.info(f"Deleted existing metadata file: {metadata_file}")

    metadata.save_to_json(metadata_file)
    logger.info(f"Saved metadata to: {metadata_file}")

    return metadata


def save_results(
    synthetic_data: pd.DataFrame,
    metadata: Metadata,
    synthesizer: CTGANSynthesizer,
    output_dir: Path,
) -> None:
    """Save synthetic data, metadata, and model with versioning."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create version directory
    version_dir = output_dir / timestamp
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save synthetic data
    data_file = version_dir / "consumer.csv"
    synthetic_data.to_csv(data_file, index=False)

    # Save metadata
    metadata_file = version_dir / "metadata.json"
    metadata.save_to_json(metadata_file)

    # Save model
    model_file = version_dir / "model.pkl"
    synthesizer.save(model_file)

    # Log saved files
    logger.info("\nSaved results:")
    logger.info(f"  Data: {data_file} ({data_file.stat().st_size / 1024**2:.2f} MB)")
    logger.info(
        f"  Metadata: {metadata_file} ({metadata_file.stat().st_size / 1024:.2f} KB)"
    )
    logger.info(f"  Model: {model_file} ({model_file.stat().st_size / 1024**2:.2f} MB)")


def _get_preference_validation_details(
    synthetic_data: pd.DataFrame,
    original_data: pd.DataFrame,
) -> Dict[str, Any]:
    """Get validation details for preference metrics."""
    details = {}

    # Validate product preferences
    for product in PRODUCT_PREFERENCES.keys():
        col = f"{product}_preference"
        if col in synthetic_data.columns:
            real_val = original_data[col].mean()
            synth_val = synthetic_data[col].mean()
            within_tolerance = (
                abs(synth_val - real_val)
                <= VALIDATION_THRESHOLDS["preference"]["tolerance"]
            )
            details[product] = {
                "expected": real_val,
                "actual": synth_val,
                "within_tolerance": within_tolerance,
            }

    # Validate social media engagement
    real_engagement = original_data["social_media_engagement_rate"].mean()
    synth_engagement = synthetic_data["social_media_engagement_rate"].mean()
    within_tolerance = (
        abs(synth_engagement - real_engagement)
        <= VALIDATION_THRESHOLDS["engagement"]["tolerance"]
    )
    details["social_media_engagement"] = {
        "expected": real_engagement,
        "actual": synth_engagement,
        "within_tolerance": within_tolerance,
    }

    return details


def validate_synthetic_data(
    synthetic_data: pd.DataFrame,
    original_data: pd.DataFrame,
    logger,
) -> None:
    """Validate synthetic data using benchmarks."""
    # Format results for standardized display
    validation_results = {
        "distributions": {
            "numerical_metrics": {
                "details": get_numerical_validation_details(
                    synthetic_data, original_data, METADATA_CONFIGURATIONS["consumer"]
                )
            },
            "categorical_metrics": {
                "details": get_categorical_validation_details(
                    synthetic_data, original_data, METADATA_CONFIGURATIONS["consumer"]
                )
            },
        },
        "preferences": {
            "product_metrics": {
                "details": _get_preference_validation_details(
                    synthetic_data, original_data
                )
            }
        },
    }

    # Calculate metrics
    total_metrics = count_total_metrics(validation_results)
    passing_metrics = count_passing_metrics(validation_results)

    # Add metadata
    validation_results["metadata"] = {
        "overall_score": passing_metrics / total_metrics if total_metrics > 0 else 0.0,
        "total_metrics_checked": total_metrics,
        "passing_metrics": passing_metrics,
    }

    # Display results
    display_validation_metrics(
        validation_results, logger, "Consumer Preference Validation Results"
    )

    # Create validation_results directory in validation folder
    validation_dir = (
        Path(__file__).resolve().parent / "validation" / "validation_results"
    )
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Save validation results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = validation_dir / f"consumer_validation_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(validation_results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nValidation results saved to {results_file}")


def train_model(
    initial_preferences: pd.DataFrame, metadata: Metadata
) -> CTGANSynthesizer:
    """Train CTGAN model on initial preferences data."""
    logger.info("\nTraining synthetic data model...")

    # Calculate batch_size to be divisible by pac
    pac = MODEL_PARAMETERS["pac"]
    batch_size = (len(initial_preferences) // pac) * pac
    if batch_size == 0:
        batch_size = pac

    # Initialize CTGAN with parameters from constants
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=MODEL_PARAMETERS["epochs"],
        batch_size=batch_size,
        log_frequency=True,  # Enable detailed logging
        verbose=True,  # Enable verbose output
        generator_dim=MODEL_PARAMETERS["generator_dim"],
        discriminator_dim=MODEL_PARAMETERS["discriminator_dim"],
        generator_lr=MODEL_PARAMETERS["generator_lr"],
        discriminator_lr=MODEL_PARAMETERS["discriminator_lr"],
        generator_decay=MODEL_PARAMETERS["generator_decay"],
        discriminator_decay=MODEL_PARAMETERS["discriminator_decay"],
        embedding_dim=MODEL_PARAMETERS["embedding_dim"],
        pac=pac,
    )

    # Log training configuration
    logger.info("\nTraining Configuration:")
    logger.info("-" * 50)
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"PAC: {pac}")
    logger.info(f"Epochs: {MODEL_PARAMETERS['epochs']}")
    logger.info(f"Generator LR: {MODEL_PARAMETERS['generator_lr']}")
    logger.info(f"Discriminator LR: {MODEL_PARAMETERS['discriminator_lr']}")
    logger.info(f"Training data shape: {initial_preferences.shape}")
    logger.info("-" * 50)

    # Train with timing and progress tracking
    start_time = time.time()
    try:
        with tqdm(total=MODEL_PARAMETERS["epochs"], desc="Training Progress") as pbar:

            def progress_callback(epoch, *args):
                pbar.update(1)
                if epoch % 10 == 0:  # Log every 10 epochs
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Epoch {epoch}/{MODEL_PARAMETERS['epochs']} "
                        f"({elapsed:.1f}s elapsed)"
                    )

            # Fit the model with progress callback
            synthesizer._fit_callback = progress_callback
            synthesizer.fit(initial_preferences)

        # Log training summary
        total_time = time.time() - start_time
        logger.info("\nTraining Summary:")
        logger.info("-" * 50)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Avg time per epoch: {total_time/MODEL_PARAMETERS['epochs']:.1f}s")
        logger.info("-" * 50)

        return synthesizer

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


def generate_synthetic_data(
    synthesizer: CTGANSynthesizer,
    num_samples: int,
    original_data: pd.DataFrame,
    logger,
    metadata: Metadata,
) -> pd.DataFrame:
    """Generate synthetic preferences using trained CTGAN model."""
    logger.info(f"Generating {num_samples:,} synthetic records...")

    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_samples)

    # Post-process to ensure valid ranges
    for col in synthetic_data.columns:
        if pd.api.types.is_numeric_dtype(synthetic_data[col]):
            if "preference" in col or "rate" in col:
                synthetic_data[col] = synthetic_data[col].clip(0, 1)
            elif "monthly_purchases" in col:
                synthetic_data[col] = synthetic_data[col].clip(0, 12)
            elif "basket_size" in col:
                synthetic_data[col] = synthetic_data[col].clip(10, 500)
            else:
                synthetic_data[col] = synthetic_data[col].clip(0)

    # Validate the synthetic data
    validate_synthetic_data(synthetic_data, original_data, logger)

    logger.info("Synthetic data generation completed")
    return synthetic_data


def main():
    """Main execution function."""
    # Set up directories
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    output_dir = data_dir / "consumer"

    # Create directories
    for directory in [data_dir, output_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Setup logging - no file logging
    logger = setup_logging(log_dir=None, log_to_file=False, module_name="consumer")

    try:
        logger.info("Starting consumer preferences generation...")

        # Load demographic data
        demographic_data = load_demographic_data(data_dir, logger)

        # Create initial preferences
        initial_preferences = create_initial_preferences(demographic_data, logger)

        # Create metadata and train model
        metadata = create_metadata(initial_preferences)
        model = train_model(initial_preferences, metadata)

        # Generate synthetic data and validate against initial preferences
        synthetic_preferences = generate_synthetic_data(
            model, len(demographic_data), initial_preferences, logger, metadata
        )

        # Save results
        save_results(synthetic_preferences, metadata, model, output_dir)

        logger.info("Consumer preferences generation completed successfully")

    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
