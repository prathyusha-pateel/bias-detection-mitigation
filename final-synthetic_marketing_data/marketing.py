"""
Marketing Engagement Data Generator

Generates synthetic marketing campaign, engagement, and loyalty data using CTGAN/SDV 
to maintain proper statistical relationships.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import run_diagnostic
from utils.logging import setup_logging
from constants import (
    METADATA_CONFIGURATIONS,
    MODEL_PARAMETERS,
    CREATIVE_ELEMENTS,
    VALIDATION_THRESHOLDS,
    REGIONAL_VARIATIONS,
)
from validation.marketing.marketing_benchmarks import (
    ENGAGEMENT_METRICS,
    LOYALTY_METRICS,
)
from utils.validation_metrics import (
    get_numerical_validation_details,
    get_categorical_validation_details,
    get_temporal_validation_details,
    count_total_metrics,
    count_passing_metrics,
)
from utils.validation_display import display_validation_metrics
from utils.encoders import NumpyEncoder
import json

# Initialize global logger
logger = setup_logging(module_name="marketing")


def load_input_data(data_dir: Path, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load demographic and consumer preference data."""
    logger.info("Loading input data...")

    # Load demographic data
    demo_files = list(data_dir.glob("*_demographics.csv"))
    if not demo_files:
        raise FileNotFoundError(
            "No demographic data found. Please run demographic.py first."
        )
    demographic_data = pd.concat(
        [pd.read_csv(f) for f in demo_files], ignore_index=True
    )
    logger.info(f"Loaded {len(demographic_data):,} demographic records")

    # Try multiple paths for consumer preferences
    preference_data = None
    possible_paths = [
        data_dir / "consumer" / "consumer.csv",  # Direct path
        data_dir / "consumers" / "consumers.csv",  # Alternative directory
        data_dir / "consumer_preferences.csv",  # Root directory
    ]

    # Check version directories if main paths don't exist
    consumer_dirs = [data_dir / "consumer", data_dir / "consumers"]

    for consumer_dir in consumer_dirs:
        if consumer_dir.exists():
            version_dirs = list(consumer_dir.glob("*"))
            if version_dirs:
                latest_dir = max(version_dirs, key=lambda x: x.name)
                possible_paths.append(latest_dir / "consumer.csv")
                possible_paths.append(latest_dir / "consumers.csv")

    # Try each possible path
    for path in possible_paths:
        if path.exists():
            try:
                preference_data = pd.read_csv(path)
                logger.info(
                    f"Loaded {len(preference_data):,} consumer preference records from {path}"
                )
                break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {str(e)}")

    # If no preference data found, generate basic preferences
    if preference_data is None:
        logger.warning(
            "No consumer preference data found. Generating basic preferences..."
        )
        preference_data = generate_basic_preferences(demographic_data, logger)

        # Save the generated preferences
        consumer_dir = data_dir / "consumer"
        consumer_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = consumer_dir / timestamp / "consumer.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        preference_data.to_csv(save_path, index=False)
        logger.info(f"Saved generated preferences to {save_path}")

    return demographic_data, preference_data


def generate_basic_preferences(demographic_data: pd.DataFrame, logger) -> pd.DataFrame:
    """Generate basic consumer preferences when no existing data is found."""
    logger.info("Generating basic consumer preferences...")

    preferences = pd.DataFrame()
    preferences["consumer_id"] = demographic_data.index.map(lambda x: f"C{int(x):06d}")

    # Create age groups
    preferences["age_group"] = pd.cut(
        demographic_data["AGEP"],
        bins=[17, 34, 54, float("inf")],
        labels=["18-34", "35-54", "55+"],
    ).astype(str)

    # Generate basic preference metrics
    for metric in ["online_shopping_rate", "social_media_engagement_rate"]:
        preferences[metric] = np.random.uniform(0.1, 0.9, len(preferences))

    # Generate basic product preferences
    product_categories = [
        "ready_to_eat",
        "snacks",
        "sustainable",
        "family_size",
        "healthy_alternatives",
        "traditional",
    ]
    for category in product_categories:
        preferences[f"{category}_preference"] = np.random.uniform(
            0.1, 0.9, len(preferences)
        )

    # Generate loyalty metrics
    preferences["loyalty_memberships"] = np.random.randint(0, 5, len(preferences))

    logger.info(f"Generated basic preferences for {len(preferences):,} consumers")
    return preferences


def create_initial_campaign_patterns(
    demographic_data: pd.DataFrame, preference_data: pd.DataFrame, num_seeds: int = 50
) -> pd.DataFrame:
    """Create initial campaign patterns for CTGAN training."""
    logger.info(f"Creating {num_seeds} initial campaign patterns...")

    campaigns = []
    campaign_types = [
        "Brand_Awareness",
        "Product_Launch",
        "Seasonal_Promotion",
        "Loyalty_Rewards",
        "Re-engagement",
        "Cross_sell",
    ]

    channels = {
        "Email": ENGAGEMENT_METRICS["email"]["open_rate"],
        "Facebook": ENGAGEMENT_METRICS["social_media"]["facebook"],
        "Instagram": ENGAGEMENT_METRICS["social_media"]["instagram"],
        "Twitter": ENGAGEMENT_METRICS["social_media"]["twitter"],
        "Display_Ads": 0.0015,
        "Search_Ads": 0.0320,
    }

    creative_types = {
        "Static_Image": 1.0,
        "Carousel": 1.2,
        "Video": 1.5,
        "Interactive": 1.8,
        "Story": 1.3,
    }

    start_date = datetime(2024, 1, 1)

    for i in range(num_seeds):
        campaign_type = np.random.choice(campaign_types)
        primary_channel = np.random.choice(list(channels.keys()))
        creative_type = np.random.choice(list(creative_types.keys()))

        duration_days = np.random.randint(7, 31)
        start = start_date + timedelta(days=np.random.randint(0, 365 - duration_days))
        end = start + timedelta(days=duration_days)

        target_age_min = np.random.choice([18, 25, 35, 45, 55])
        target_age_max = target_age_min + np.random.choice([10, 15, 20, 25])

        target_income_max = np.random.choice([75000, 100000, 150000, 200000])

        campaign = {
            "campaign_id": f"C{i+1:04d}",
            "campaign_type": campaign_type,
            "primary_channel": primary_channel,
            "creative_type": creative_type,
            "creative_elements": generate_creative_elements(creative_type),
            "start_date": start,
            "end_date": end,
            "target_age_min": target_age_min,
            "target_age_max": target_age_max,
            "target_income_min": np.random.choice([25000, 50000, 75000]),
            "target_income_max": target_income_max,
            "base_engagement_rate": channels[primary_channel]
            * creative_types[creative_type],
            "budget": np.random.randint(5000, 50000),
            "target_regions": np.random.choice(
                list(REGIONAL_VARIATIONS.keys()),
                size=np.random.randint(1, 5),
                replace=False,
            ).tolist(),
        }
        campaigns.append(campaign)

    return pd.DataFrame(campaigns)


def create_initial_engagement_patterns(
    campaign_data: pd.DataFrame, num_days_per_campaign: int = 10
) -> pd.DataFrame:
    """Create initial engagement patterns for CTGAN training."""
    logger.info("Creating initial engagement patterns...")

    engagement_records = []
    engagement_id = 1  # Initialize counter for unique IDs

    for _, campaign in campaign_data.sample(n=min(len(campaign_data), 10)).iterrows():
        duration = (campaign.end_date - campaign.start_date).days

        # Sample days throughout campaign
        for day in range(0, duration, max(1, duration // num_days_per_campaign)):
            current_date = campaign.start_date + timedelta(days=day)

            # Base metrics based on channel benchmarks
            base_rate = campaign.base_engagement_rate

            # Apply temporal factors
            dow_multiplier = {0: 0.9, 1: 1.0, 2: 1.1, 3: 1.2, 4: 1.1, 5: 0.8, 6: 0.7}[
                current_date.weekday()
            ]

            lifecycle_position = day / duration
            lifecycle_multiplier = 1.0 - abs(0.5 - lifecycle_position)

            # Calculate metrics
            impressions = np.random.normal(10000, 2000)
            engagement_rate = base_rate * dow_multiplier * lifecycle_multiplier
            clicks = int(impressions * engagement_rate)

            time_spent = np.random.normal(
                ENGAGEMENT_METRICS["time_spent_minutes"].get(
                    f"{campaign.primary_channel.lower()}_content",
                    ENGAGEMENT_METRICS["time_spent_minutes"]["social_content"],
                ),
                0.5,
            )

            record = {
                "engagement_id": f"E{engagement_id:06d}",  # Add unique engagement ID
                "campaign_id": campaign.campaign_id,  # Keep as foreign key
                "date": current_date,
                "impressions": max(0, int(impressions)),
                "clicks": max(0, clicks),
                "engagement_rate": engagement_rate,
                "time_spent_minutes": max(0, time_spent),
                "conversion_rate": engagement_rate * np.random.uniform(0.05, 0.15),
                "day_of_week": current_date.weekday(),
                "campaign_day": day,
                "campaign_progress": day / duration,
            }
            engagement_records.append(record)
            engagement_id += 1  # Increment the ID counter

    return pd.DataFrame(engagement_records)


def create_initial_loyalty_patterns(
    preference_data: pd.DataFrame, num_seeds: int = 100
) -> pd.DataFrame:
    """Create initial loyalty patterns for CTGAN training."""
    logger.info("Creating initial loyalty patterns...")

    loyalty_records = []
    enrollment_dist = LOYALTY_METRICS["program_enrollment"]
    redemption_rates = LOYALTY_METRICS["redemption_rates"]

    for _ in range(num_seeds):
        consumer = preference_data.sample(n=1).iloc[0]

        status = np.random.choice(
            ["active", "semi_active", "inactive"],
            p=[
                enrollment_dist["active"],
                enrollment_dist["semi_active"],
                enrollment_dist["inactive"],
            ],
        )

        if status != "inactive":
            enrollment_date = datetime.now() - timedelta(days=np.random.randint(1, 730))

            age_group = consumer["age_group"]
            group = (
                "Gen_Z"
                if age_group in ["18-29"]
                else (
                    "Millennial"
                    if age_group in ["30-49"]
                    else "Gen_X" if age_group in ["50-64"] else "Boomer"
                )
            )

            base_points = np.random.normal(5000, 1500)
            redemption_factor = redemption_rates.get(group, 0.5)
            points_factor = {"active": 1.0, "semi_active": 0.4}[status]

            points_balance = int(base_points * points_factor * redemption_factor)

            record = {
                "consumer_id": consumer["consumer_id"],
                "enrollment_date": enrollment_date,
                "status": status,
                "points_balance": max(0, points_balance),
                "lifetime_points": int(points_balance * np.random.uniform(1.5, 3.0)),
                "tier": (
                    "Gold"
                    if points_balance > 7500
                    else "Silver" if points_balance > 3000 else "Bronze"
                ),
                "age_group": age_group,
                "redemption_rate": redemption_factor,
            }
            loyalty_records.append(record)

    return pd.DataFrame(loyalty_records)


def create_metadata(initial_data: pd.DataFrame, data_type: str) -> Metadata:
    """Create metadata for synthesizer using SDV.

    Args:
        initial_data: Initial training data
        data_type: Type of data ('campaigns', 'engagements', or 'loyalties')
    """
    logger.info(f"Creating metadata for {data_type}...")

    # Convert list columns to strings to avoid unhashable type error
    list_cols = initial_data.select_dtypes(include=["object"]).columns
    for col in list_cols:
        if isinstance(initial_data[col].iloc[0], list):
            initial_data[col] = initial_data[col].apply(lambda x: ",".join(map(str, x)))

    # Get configuration for this data type
    config = METADATA_CONFIGURATIONS[data_type]

    # Create metadata instance and detect from dataframe
    metadata = Metadata()
    metadata = Metadata.detect_from_dataframe(data=initial_data, table_name=data_type)

    # Update ID column type without setting primary key again
    metadata.update_column(column_name=config["id_column"], sdtype="id")

    # Update foreign keys if they exist
    if "foreign_keys" in config:
        for fk in config["foreign_keys"]:
            metadata.update_column(column_name=fk, sdtype="id")

    # Update categorical columns
    for col in config.get("categorical_columns", []):
        if col in initial_data.columns:
            metadata.update_column(column_name=col, sdtype="categorical")

    # Update datetime columns
    for col in config.get("datetime_columns", []):
        if col in initial_data.columns:
            metadata.update_column(column_name=col, sdtype="datetime")

    # Update numerical columns
    for col in config.get("numerical_columns", []):
        if col in initial_data.columns:
            metadata.update_column(
                column_name=col, sdtype="numerical", computer_representation="Float"
            )

    # Update text columns
    for col in config.get("text_columns", []):
        if col in initial_data.columns:
            metadata.update_column(column_name=col, sdtype="text")

    metadata.validate()

    # Save with versioning
    metadata_dir = Path(__file__).parent / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    metadata_file = metadata_dir / f"{data_type}.json"

    if metadata_file.exists():
        metadata_file.unlink()
        logger.info(f"Deleted existing metadata file: {metadata_file}")

    metadata.save_to_json(metadata_file)
    logger.info(f"Saved metadata to: {metadata_file}")

    return metadata


def generate_creative_elements(creative_type: str) -> List[str]:
    """Generate list of creative elements based on type.

    Args:
        creative_type: Type of creative content (e.g., 'Static_Image', 'Video')

    Returns:
        List of creative elements appropriate for the given type
    """
    # Get available elements for the creative type
    available_elements = CREATIVE_ELEMENTS.get(creative_type, [])

    if not available_elements:
        logger.warning(
            f"Unknown creative type: {creative_type}. Using default elements."
        )
        return ["default_creative"]

    # Select random number of elements (between 1 and 3)
    num_elements = np.random.randint(1, min(4, len(available_elements) + 1))

    # Randomly select elements
    selected_elements = np.random.choice(
        available_elements, size=num_elements, replace=False
    ).tolist()

    logger.debug(
        f"Generated {num_elements} elements for {creative_type}: {selected_elements}"
    )

    return selected_elements


def train_model(
    training_data: pd.DataFrame, metadata: Metadata, data_type: str = None
) -> CTGANSynthesizer:
    """Train CTGAN synthesizer model."""
    logger.info(f"Training {data_type} generation model...")
    logger.info(f"Training data shape: {training_data.shape}")

    # Initialize CTGAN synthesizer with parameters
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=MODEL_PARAMETERS["epochs"],
        batch_size=MODEL_PARAMETERS["batch_size"],
        log_frequency=MODEL_PARAMETERS["log_frequency"],
        verbose=MODEL_PARAMETERS["verbose"],
    )

    # Train the model with progress logging
    logger.info(f"Starting model training with {MODEL_PARAMETERS['epochs']} epochs...")
    start_time = datetime.now()
    synthesizer.fit(training_data)
    training_time = datetime.now() - start_time
    logger.info(f"Model training completed in {training_time}")

    return synthesizer


def generate_synthetic_data(
    synthesizer: CTGANSynthesizer, num_samples: int, data_type: str
) -> pd.DataFrame:
    """Generate synthetic data using trained model."""
    logger.info(f"Generating {num_samples:,} synthetic {data_type} records...")

    synthetic_data = synthesizer.sample(num_samples)

    # Post-process based on data type
    if data_type == "campaigns":
        synthetic_data["budget"] = synthetic_data["budget"].clip(0)
        synthetic_data["base_engagement_rate"] = synthetic_data[
            "base_engagement_rate"
        ].clip(0, 1)
        synthetic_data = synthetic_data.sort_values("start_date")

    elif data_type == "engagement":
        synthetic_data["impressions"] = synthetic_data["impressions"].clip(0)
        synthetic_data["clicks"] = synthetic_data["clicks"].clip(0)
        synthetic_data["engagement_rate"] = synthetic_data["engagement_rate"].clip(0, 1)
        synthetic_data["conversion_rate"] = synthetic_data["conversion_rate"].clip(0, 1)
        synthetic_data = synthetic_data.sort_values("date")

    elif data_type == "loyalty":
        synthetic_data["points_balance"] = synthetic_data["points_balance"].clip(0)
        synthetic_data["lifetime_points"] = synthetic_data["lifetime_points"].clip(0)
        synthetic_data["redemption_rate"] = synthetic_data["redemption_rate"].clip(0, 1)
        synthetic_data = synthetic_data.sort_values("enrollment_date")

    logger.info(f"Generated {len(synthetic_data):,} {data_type} records")
    return synthetic_data


class MarketingEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Pandas Timestamps and other special types."""

    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(
    campaigns: pd.DataFrame,
    engagements: pd.DataFrame,
    loyalties: pd.DataFrame,
    output_dir: Path,
    logger,
) -> None:
    """Save generated data with version directory structure."""
    # Create version directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = output_dir / timestamp
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets in version directory with standardized names
    files = {
        "campaigns.csv": campaigns,
        "engagements.csv": engagements,
        "loyalties.csv": loyalties,
    }

    for filename, df in files.items():
        filepath = version_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {filename} to {filepath}")

    # Generate and save metadata
    metadata = {
        "generation_timestamp": timestamp,
        "record_counts": {
            "campaigns": len(campaigns),
            "engagements": len(engagements),
            "loyalties": len(loyalties),
        },
        "date_range": {
            "start": campaigns["start_date"].min().strftime("%Y-%m-%d"),
            "end": campaigns["end_date"].max().strftime("%Y-%m-%d"),
        },
    }

    metadata_file = version_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, cls=MarketingEncoder)

    logger.info(f"Saved metadata to {metadata_file}")


def _get_engagement_validation_details(
    synthetic_data: pd.DataFrame,
    original_data: pd.DataFrame,
) -> Dict[str, Any]:
    """Get validation details for engagement metrics."""
    details = {}

    # Validate engagement rates
    for channel, metrics in ENGAGEMENT_METRICS.items():
        if isinstance(metrics, dict):
            for metric_name, expected_value in metrics.items():
                col = f"{channel}_{metric_name}"
                if col in synthetic_data.columns:
                    real_val = original_data[col].mean()
                    synth_val = synthetic_data[col].mean()
                    within_tolerance = (
                        abs(synth_val - real_val)
                        <= VALIDATION_THRESHOLDS["engagement"]["tolerance"]
                    )
                    details[f"{channel}_{metric_name}"] = {
                        "expected": expected_value,
                        "actual": synth_val,
                        "within_tolerance": within_tolerance,
                    }

    return details


def _get_loyalty_validation_details(
    synthetic_data: pd.DataFrame,
    original_data: pd.DataFrame,
) -> Dict[str, Any]:
    """Get validation details for loyalty metrics."""
    details = {}

    # Validate loyalty program metrics
    for program, rate in LOYALTY_METRICS["program_engagement"].items():
        col = f"{program}_loyalty"
        if col in synthetic_data.columns:
            real_val = original_data[col].mean()
            synth_val = synthetic_data[col].mean()
            within_tolerance = (
                abs(synth_val - real_val)
                <= VALIDATION_THRESHOLDS["loyalty"]["tolerance"]
            )
            details[f"loyalty_{program}"] = {
                "expected": rate,
                "actual": synth_val,
                "within_tolerance": within_tolerance,
            }

    return details


def validate_synthetic_data(
    synthetic_data: pd.DataFrame,
    original_data: pd.DataFrame,
    logger,
    data_type: str = None,
) -> None:
    """Enhanced validation using SDV evaluation tools."""
    # Get configuration for this data type
    config = METADATA_CONFIGURATIONS[data_type]

    # Create validation metadata
    metadata = create_validation_metadata(synthetic_data, data_type, config)

    # Run diagnostic checks
    diagnostic_report = run_diagnostic(
        real_data=original_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
    )

    # Format results for standardized display
    validation_results = {
        "distributions": {
            "numerical_metrics": {
                "details": get_numerical_validation_details(
                    synthetic_data, original_data, config
                )
            },
            "categorical_metrics": {
                "details": get_categorical_validation_details(
                    synthetic_data, original_data, config
                )
            },
        }
    }

    # Add type-specific validation results
    if data_type == "engagements":
        validation_results["engagement"] = {
            "metrics": {
                "details": _get_engagement_validation_details(
                    synthetic_data, original_data
                )
            },
            "temporal": {
                "details": get_temporal_validation_details(
                    synthetic_data, original_data
                )
            },
        }
    elif data_type == "loyalties":
        validation_results["loyalty"] = {
            "program_metrics": {
                "details": _get_loyalty_validation_details(
                    synthetic_data, original_data
                )
            }
        }

    # Add metadata
    validation_results["metadata"] = {
        "overall_score": diagnostic_report.get_score(),
        "total_metrics_checked": count_total_metrics(validation_results),
        "passing_metrics": count_passing_metrics(validation_results),
    }

    # Display validation results
    display_validation_metrics(
        validation_results, logger, f"{data_type.title()} Data Validation Results"
    )

    # Create validation_results directory in validation folder
    validation_dir = (
        Path(__file__).resolve().parent / "validation" / "validation_results"
    )
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Save validation results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = validation_dir / f"{data_type}_validation_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(validation_results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nValidation results saved to {results_file}")


def create_validation_metadata(
    synthetic_data: pd.DataFrame, data_type: str, config: dict
) -> Metadata:
    """Create validation metadata for SDV diagnostics."""
    metadata = Metadata()
    metadata.add_table(table_name=data_type)

    # Add columns based on configuration
    metadata.add_column(
        table_name=data_type, column_name=config["id_column"], sdtype="id"
    )

    for col in config.get("categorical_columns", []):
        if col in synthetic_data.columns:
            metadata.add_column(
                table_name=data_type, column_name=col, sdtype="categorical"
            )

    for col in config.get("numerical_columns", []):
        if col in synthetic_data.columns:
            metadata.add_column(
                table_name=data_type, column_name=col, sdtype="numerical"
            )

    for col in config.get("datetime_columns", []):
        if col in synthetic_data.columns:
            metadata.add_column(
                table_name=data_type, column_name=col, sdtype="datetime"
            )

    metadata.set_primary_key(table_name=data_type, column_name=config["id_column"])

    return metadata


def main():
    """Main execution function."""
    # Set up directories
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    output_dir = data_dir / "marketing_engagement"

    # Create directories
    for directory in [data_dir, output_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(module_name="marketing")

    try:
        logger.info("Starting marketing engagement data generation...")

        # Load input data
        demographic_data, preference_data = load_input_data(data_dir, logger)

        # 1. Generate campaigns first
        initial_campaigns = create_initial_campaign_patterns(
            demographic_data, preference_data
        )
        campaign_metadata = create_metadata(initial_campaigns, "campaigns")
        campaign_model = train_model(initial_campaigns, campaign_metadata, "campaigns")
        campaign_data = generate_synthetic_data(
            campaign_model, int(len(demographic_data) * 0.1), "campaigns"
        )

        # 2. Generate engagement data
        initial_engagement = create_initial_engagement_patterns(campaign_data)
        engagement_metadata = create_metadata(initial_engagement, "engagements")
        engagement_model = train_model(
            initial_engagement, engagement_metadata, "engagements"
        )
        engagement_data = generate_synthetic_data(
            engagement_model, len(campaign_data) * 30, "engagements"
        )

        # 3. Generate loyalty data
        initial_loyalty = create_initial_loyalty_patterns(preference_data)
        loyalty_metadata = create_metadata(initial_loyalty, "loyalties")
        loyalty_model = train_model(initial_loyalty, loyalty_metadata, "loyalties")
        loyalty_data = generate_synthetic_data(
            loyalty_model, int(len(preference_data) * 0.4), "loyalties"
        )

        # 4. Save all results after everything is generated
        save_results(
            campaign_data,
            engagement_data,
            loyalty_data,
            output_dir,
            logger,
        )

        # 5. Validate each type of data
        validate_synthetic_data(campaign_data, initial_campaigns, logger, "campaigns")
        validate_synthetic_data(
            engagement_data, initial_engagement, logger, "engagements"
        )
        validate_synthetic_data(loyalty_data, initial_loyalty, logger, "loyalties")

        logger.info("Marketing engagement data generation completed successfully")

    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
