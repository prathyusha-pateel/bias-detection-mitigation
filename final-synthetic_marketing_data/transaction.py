"""
Transaction Data Generator

Generates synthetic transaction data, product information, and purchase channel data
using CTGAN/SDV to maintain proper statistical relationships.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import json
import time
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import run_diagnostic

from constants import (
    TRANSACTION_METRICS,
    BUYING_PATTERNS,
    REGIONAL_VARIATIONS,
    STATE_TO_REGION,
    MODEL_PARAMETERS,
    METADATA_CONFIGURATIONS,
    VALIDATION_THRESHOLDS,
    FIPS_TO_STATE,
)
from utils.logging import setup_logging
from utils.validation_display import display_validation_metrics
from utils.validation_metrics import (
    get_numerical_validation_details,
    get_categorical_validation_details,
    get_transaction_validation_details,
    get_regional_validation_details,
    count_total_metrics,
    count_passing_metrics,
)
from utils.encoders import NumpyEncoder

# Initialize global logger
logger = setup_logging(module_name="transactions")


def generate_product_catalog(logger) -> pd.DataFrame:
    """Generate synthetic product catalog."""
    logger.info("Generating product catalog...")

    products = []
    categories = {
        "ready_to_eat": ["Frozen Meals", "Snack Packs", "Instant Meals"],
        "snacks": ["Chips", "Crackers", "Nuts", "Popcorn"],
        "sustainable": ["Organic Meals", "Plant Based", "Eco Friendly"],
        "family_size": ["Bulk Meals", "Party Packs", "Value Boxes"],
        "healthy_alternatives": ["Low Cal", "High Protein", "Gluten Free"],
        "traditional": ["Classic Meals", "Home Style", "Comfort Food"],
    }

    product_id = 1
    for category, subcategories in categories.items():
        for subcategory in subcategories:
            # Generate 3-5 products per subcategory
            num_products = np.random.randint(3, 6)

            base_price_range = {
                "ready_to_eat": (3.99, 7.99),
                "snacks": (2.99, 5.99),
                "sustainable": (4.99, 9.99),
                "family_size": (8.99, 15.99),
                "healthy_alternatives": (4.99, 8.99),
                "traditional": (5.99, 10.99),
            }.get(category, (4.99, 9.99))

            for _ in range(num_products):
                base_price = np.random.uniform(base_price_range[0], base_price_range[1])

                product = {
                    "product_id": f"P{product_id:04d}",
                    "category": category,
                    "subcategory": subcategory,
                    "base_price": round(base_price, 2),
                    "min_order_quantity": 1,
                    "max_order_quantity": np.random.choice([5, 10, 20]),
                }
                products.append(product)
                product_id += 1

    catalog_df = pd.DataFrame(products)

    # Log catalog statistics
    logger.info(f"Generated catalog with {len(catalog_df)} products")
    logger.info("\nProduct Catalog Summary:")
    logger.info(f"Categories: {', '.join(catalog_df['category'].unique())}")
    logger.info(
        f"Price range: ${catalog_df['base_price'].min():.2f} - ${catalog_df['base_price'].max():.2f}"
    )
    logger.info("\nProducts per category:")
    for cat in catalog_df["category"].unique():
        count = len(catalog_df[catalog_df["category"] == cat])
        logger.info(f"  {cat}: {count} products")

    return catalog_df


def load_input_data(
    data_dir: Path, logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and combine demographic data files."""
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

    # Load consumer preferences with versioning support
    preference_data = None
    consumer_dir = data_dir / "consumer"
    if consumer_dir.exists():
        # Get latest version directory
        version_dirs = sorted(
            consumer_dir.glob("*"), key=lambda x: x.name, reverse=True
        )
        if version_dirs:
            pref_file = version_dirs[0] / "consumer.csv"
            if pref_file.exists():
                preference_data = pd.read_csv(pref_file)
                logger.info(
                    f"Loaded {len(preference_data):,} consumer preference records"
                )

    if preference_data is None:
        raise FileNotFoundError(
            "No consumer preference data found. Please run consumer.py first."
        )

    # Load campaign data with versioning support
    campaign_data = None
    marketing_dir = data_dir / "marketing_engagement"
    if marketing_dir.exists():
        # Get latest version directory
        version_dirs = sorted(
            marketing_dir.glob("*"), key=lambda x: x.name, reverse=True
        )
        if version_dirs:
            campaign_file = version_dirs[0] / "campaigns.csv"
            if campaign_file.exists():
                campaign_data = pd.read_csv(campaign_file)
                logger.info(f"Loaded {len(campaign_data):,} campaign records")
            else:
                logger.warning(f"No campaign file found in {version_dirs[0]}")
        else:
            logger.warning(f"No version directories found in {marketing_dir}")
    else:
        logger.warning(f"Marketing directory not found: {marketing_dir}")

    if campaign_data is None:
        logger.warning("No campaign data found. Using simplified campaign data...")
        campaign_data = generate_basic_campaigns(demographic_data, logger)

    return demographic_data, preference_data, campaign_data


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


def generate_basic_campaigns(demographic_data: pd.DataFrame, logger) -> pd.DataFrame:
    """Generate basic campaign data when no existing data is found."""
    logger.info("Generating basic campaign data...")

    num_campaigns = max(
        10, len(demographic_data) // 10000
    )  # 1 campaign per 10k consumers
    campaigns = []

    campaign_types = ["Brand_Awareness", "Product_Launch", "Seasonal_Promotion"]
    channels = ["Email", "Social_Media", "Display_Ads"]

    for i in range(num_campaigns):
        campaign = {
            "campaign_id": f"C{i+1:04d}",
            "campaign_type": np.random.choice(campaign_types),
            "channel": np.random.choice(channels),
            "start_date": datetime.now() - timedelta(days=np.random.randint(1, 365)),
            "duration_days": np.random.randint(7, 31),
            "target_audience": np.random.choice(["All", "Young", "Adult", "Senior"]),
            "budget": np.random.randint(5000, 50000),
        }
        campaigns.append(campaign)

    campaign_df = pd.DataFrame(campaigns)
    logger.info(f"Generated {len(campaign_df)} basic campaign records")
    return campaign_df


def create_initial_transactions(
    demographic_data: pd.DataFrame,
    preference_data: pd.DataFrame,
    product_catalog: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """Create initial transaction data based on demographic patterns."""
    logger.info("Creating initial transaction patterns...")

    transactions = []
    num_seeds = len(demographic_data) // 10  # Create 10% of demographic size as seeds

    for transaction_id in range(1, num_seeds + 1):
        # Select random consumer
        consumer = preference_data.sample(n=1).iloc[0]
        consumer_demo = demographic_data[
            demographic_data.index == int(consumer.name)
        ].iloc[0]

        # Get consumer's state and region
        state_fips = int(consumer_demo["STATE"])
        state = FIPS_TO_STATE.get(state_fips)

        if state is None:
            logger.warning(
                f"FIPS code {state_fips} not found in mapping, using random region"
            )
            region = np.random.choice(list(set(STATE_TO_REGION.values())))
        else:
            region = STATE_TO_REGION.get(state)
            if region is None:
                logger.warning(
                    f"State {state} not found in region mapping, using random region"
                )
                region = np.random.choice(list(set(STATE_TO_REGION.values())))

        regional_mult = REGIONAL_VARIATIONS[region]["baseline_multiplier"]

        # Determine buying frequency based on BUYING_PATTERNS
        freq_dist = BUYING_PATTERNS["frequency"]
        frequency = np.random.choice(list(freq_dist.keys()), p=list(freq_dist.values()))

        # Generate base transaction values
        base_value = np.random.normal(
            TRANSACTION_METRICS["average_value"]["mean"],
            TRANSACTION_METRICS["average_value"]["std_dev"],
        )
        transaction_value = base_value * regional_mult

        # Channel selection based on age group
        if consumer["age_group"] in ["18-29", "30-49"]:
            channel_probs = {"mobile": 0.6, "desktop": 0.3, "in_store": 0.1}
        else:
            channel_probs = {"mobile": 0.3, "desktop": 0.4, "in_store": 0.3}

        channel = np.random.choice(
            list(channel_probs.keys()), p=list(channel_probs.values())
        )

        # Create transaction with validated region
        transaction = {
            "transaction_id": transaction_id,  # Add transaction_id
            "consumer_id": consumer["consumer_id"],
            "transaction_date": datetime.now()
            - timedelta(days=np.random.randint(1, 365)),
            "channel": channel,
            "transaction_value": round(transaction_value, 2),
            "num_items": np.random.randint(1, 6),
            "state": state,
            "region": region,
            "age_group": consumer["age_group"],
            "buying_frequency": frequency,
        }
        transactions.append(transaction)

    transactions_df = pd.DataFrame(transactions)
    logger.info(f"Created {len(transactions_df)} seed transactions")

    # Log region distribution
    region_dist = transactions_df["region"].value_counts()
    logger.info("Region distribution in transactions:")
    for region, count in region_dist.items():
        logger.info(f"  {region}: {count} ({count/len(transactions_df):.1%})")

    return transactions_df


def create_metadata(initial_data: pd.DataFrame, data_type: str) -> Metadata:
    """Create metadata for synthesizer using SDV's recommended approach."""
    logger.info(f"Creating metadata for {data_type}...")

    # Create metadata instance
    metadata = Metadata()
    metadata.add_table(table_name=data_type)

    # Detect column types from data
    for column in initial_data.columns:
        # Handle ID columns
        if column in ["transaction_id", "consumer_id", "product_id", "detail_id"]:
            metadata.add_column(table_name=data_type, column_name=column, sdtype="id")
            if column == f"{data_type}_id":  # Set primary key based on table type
                metadata.set_primary_key(table_name=data_type, column_name=column)
            continue

        # Handle datetime columns
        if pd.api.types.is_datetime64_any_dtype(initial_data[column]):
            metadata.add_column(
                table_name=data_type, column_name=column, sdtype="datetime"
            )
            continue

        # Handle numerical columns
        if pd.api.types.is_numeric_dtype(initial_data[column]):
            metadata.add_column(
                table_name=data_type, column_name=column, sdtype="numerical"
            )
            continue

        # Handle categorical columns
        metadata.add_column(
            table_name=data_type, column_name=column, sdtype="categorical"
        )

    metadata.validate()

    # Save metadata with versioning
    metadata_dir = Path(__file__).parent / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    metadata_file = metadata_dir / f"{data_type}.json"

    if metadata_file.exists():
        metadata_file.unlink()
        logger.info(f"Deleted existing metadata file: {metadata_file}")

    metadata.save_to_json(metadata_file)
    logger.info(f"Saved metadata to: {metadata_file}")

    return metadata


def train_model(
    initial_data: pd.DataFrame, metadata: Metadata, data_type: str
) -> CTGANSynthesizer:
    """Train CTGAN model on initial data."""
    logger.info(f"\nTraining synthetic data model for {data_type}...")

    # Calculate batch_size to be divisible by pac
    pac = MODEL_PARAMETERS["pac"]
    batch_size = (len(initial_data) // pac) * pac
    if batch_size == 0:
        batch_size = pac

    # Initialize CTGAN with parameters from constants
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=MODEL_PARAMETERS["epochs"],
        batch_size=batch_size,
        log_frequency=MODEL_PARAMETERS["log_frequency"],
        verbose=MODEL_PARAMETERS["verbose"],
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
    logger.info(f"Starting model training with configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  PAC: {pac}")
    logger.info(f"  Epochs: {MODEL_PARAMETERS['epochs']}")
    logger.info(f"  Generator LR: {MODEL_PARAMETERS['generator_lr']}")
    logger.info(f"  Discriminator LR: {MODEL_PARAMETERS['discriminator_lr']}")
    logger.info(f"  Training data shape: {initial_data.shape}")

    # Train with timing
    start_time = time.time()
    try:
        synthesizer.fit(initial_data)

        total_time = time.time() - start_time
        logger.info("\nTraining completed:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(
            f"  Avg time per epoch: {total_time/MODEL_PARAMETERS['epochs']:.1f}s"
        )

        return synthesizer

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


def generate_transaction_details(
    transactions: pd.DataFrame, product_catalog: pd.DataFrame, logger
) -> pd.DataFrame:
    """Generate transaction detail records for each transaction."""
    logger.info("Generating transaction details...")

    details = []
    total_transactions = len(transactions)

    # Log product catalog info for debugging
    logger.info(f"Product catalog contains {len(product_catalog)} products")
    logger.info("Product categories available:")
    for cat in product_catalog["category"].unique():
        logger.info(f"  - {cat}")

    for idx, transaction in transactions.iterrows():
        if idx % 10000 == 0:  # Progress logging
            logger.info(f"Processing transaction {idx}/{total_transactions}")

        num_items = int(transaction["num_items"])

        # Select products from the entire catalog instead of filtering
        try:
            # Generate details for each item in the transaction
            for _ in range(num_items):
                # Sample a random product
                product = product_catalog.sample(n=1)

                # Get base price and apply regional adjustments
                base_price = product["base_price"].iloc[0]

                # Apply regional pricing
                region = transaction["region"]
                price_mult = REGIONAL_VARIATIONS[region]["premium_brand_preference"]
                final_price = round(base_price * price_mult, 2)

                # Determine quantity (1-3 items)
                quantity = np.random.randint(1, 4)

                detail = {
                    "transaction_id": transaction.name,
                    "product_id": product["product_id"].iloc[0],
                    "quantity": quantity,
                    "unit_price": final_price,
                    "line_total": round(final_price * quantity, 2),
                    "category": product["category"].iloc[0],
                    "subcategory": product["subcategory"].iloc[0],
                }
                details.append(detail)

        except Exception as e:
            logger.error(f"Error processing transaction {idx}: {str(e)}")
            logger.error(f"Transaction data: {transaction.to_dict()}")
            raise

    details_df = pd.DataFrame(details)
    logger.info(f"Generated {len(details_df):,} transaction detail records")

    # Log summary statistics
    logger.info("\nTransaction Details Summary:")
    logger.info(f"Total unique products used: {details_df['product_id'].nunique()}")
    logger.info(f"Average line total: ${details_df['line_total'].mean():.2f}")
    logger.info("\nCategory distribution:")
    category_dist = details_df["category"].value_counts(normalize=True)
    for cat, pct in category_dist.items():
        logger.info(f"  {cat}: {pct:.1%}")

    return details_df


import json
import pandas as pd
import numpy as np
from datetime import datetime


class TransactionEncoder(json.JSONEncoder):
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
    transactions: pd.DataFrame,
    details: pd.DataFrame,
    product_catalog: pd.DataFrame,
    output_dir: Path,
    logger,
) -> None:
    """Save generated data with version directory structure."""
    # Create version directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = output_dir / timestamp
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets with standardized names
    files = {
        "transactions.csv": transactions,
        "transaction_details.csv": details,
        "product_catalog.csv": product_catalog,
    }

    for filename, df in files.items():
        filepath = version_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {filename} to {filepath}")

    # Generate and save metadata
    metadata = {
        "generation_timestamp": timestamp,
        "record_counts": {
            "transactions": len(transactions),
            "transaction_details": len(details),
            "products": len(product_catalog),
        },
    }

    metadata_file = version_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")


def generate_synthetic_data(
    model: CTGANSynthesizer, num_samples: int, data_type: str, logger
) -> pd.DataFrame:
    """Generate synthetic transactions using the trained model."""
    logger.info(f"Generating {num_samples} synthetic {data_type}...")
    synthetic_data = model.sample(num_samples)
    logger.info(f"{data_type} generation completed")
    return synthetic_data


def validate_synthetic_data(
    synthetic_transactions: pd.DataFrame,
    synthetic_details: pd.DataFrame,
    original_transactions: pd.DataFrame,
    original_details: pd.DataFrame,
    product_catalog: pd.DataFrame,
    logger,
) -> None:
    """Validate synthetic transaction data using validation modules."""
    # Get configurations for both transaction types
    transaction_config = METADATA_CONFIGURATIONS.get("transactions", {})
    detail_config = METADATA_CONFIGURATIONS.get("transaction_details", {})

    try:
        # Format results for standardized display
        validation_results = {
            "transactions": {
                "numerical_metrics": {
                    "details": get_numerical_validation_details(
                        synthetic_transactions,
                        original_transactions,
                        transaction_config,
                    )
                },
                "categorical_metrics": {
                    "details": get_categorical_validation_details(
                        synthetic_transactions,
                        original_transactions,
                        transaction_config,
                    )
                },
                "transaction_metrics": {
                    "details": get_transaction_validation_details(
                        synthetic_transactions,
                        original_transactions,
                        VALIDATION_THRESHOLDS["transaction"]["value_difference"],
                    )
                },
                "regional_metrics": {
                    "details": get_regional_validation_details(
                        synthetic_transactions,
                        original_transactions,
                        VALIDATION_THRESHOLDS["regional"]["tolerance"],
                    )
                },
            },
            "transaction_details": {
                "numerical_metrics": {
                    "details": get_numerical_validation_details(
                        synthetic_details, original_details, detail_config
                    )
                },
                "categorical_metrics": {
                    "details": get_categorical_validation_details(
                        synthetic_details, original_details, detail_config
                    )
                },
            },
        }

        # Calculate metrics
        total_metrics = count_total_metrics(validation_results)
        passing_metrics = count_passing_metrics(validation_results)

        # Add metadata
        validation_results["metadata"] = {
            "overall_score": (
                passing_metrics / total_metrics if total_metrics > 0 else 0.0
            ),
            "total_metrics_checked": total_metrics,
            "passing_metrics": passing_metrics,
            "thresholds_used": {
                "transaction": VALIDATION_THRESHOLDS["transaction"],
                "regional": VALIDATION_THRESHOLDS["regional"],
            },
        }

        # Display validation results
        display_validation_metrics(
            validation_results, logger, "Transaction Data Validation Results"
        )

        # Create validation_results directory in validation folder
        validation_dir = (
            Path(__file__).resolve().parent / "validation" / "validation_results"
        )
        validation_dir.mkdir(parents=True, exist_ok=True)

        # Save validation results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = validation_dir / f"transaction_validation_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(validation_results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\nValidation results saved to {results_file}")

    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        logger.error(f"Validation results structure: {validation_results}")
        raise


def main():
    """Main execution function."""
    # Set up directories
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    output_dir = data_dir / "transactions"

    # Create directories
    for directory in [data_dir, output_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Setup logging - no file logging
    logger = setup_logging(module_name="transactions")

    try:
        logger.info("Starting transaction data generation...")

        # Load input data
        demographic_data, preference_data, campaign_data = load_input_data(
            data_dir, logger
        )

        # Generate product catalog
        product_catalog = generate_product_catalog(logger)

        # Create initial transaction patterns
        initial_transactions = create_initial_transactions(
            demographic_data, preference_data, product_catalog, logger
        )

        # Create metadata and train model
        metadata = create_metadata(initial_transactions, "transactions")
        model = train_model(initial_transactions, metadata, "transactions")

        # Generate synthetic transactions
        synthetic_transactions = generate_synthetic_data(
            model,
            num_samples=len(demographic_data) * 3,
            data_type="transactions",
            logger=logger,
        )

        # Generate transaction details for both synthetic and original data
        synthetic_details = generate_transaction_details(
            synthetic_transactions, product_catalog, logger
        )
        original_details = generate_transaction_details(
            initial_transactions, product_catalog, logger
        )

        # Validate synthetic data - pass product_catalog
        validate_synthetic_data(
            synthetic_transactions,
            synthetic_details,
            initial_transactions,
            original_details,
            product_catalog,
            logger,
        )

        # Save results
        save_results(
            synthetic_transactions,
            synthetic_details,
            product_catalog,
            output_dir,
            logger,
        )

        logger.info("Transaction data generation completed successfully")

    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
