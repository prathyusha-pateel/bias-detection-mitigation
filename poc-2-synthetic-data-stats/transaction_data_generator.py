# transaction_data_generator.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import uuid
import random
from typing import Dict, List

from utils.transaction_constants import (
    REGIONAL_VARIATIONS,
    CATEGORY_PREFERENCES,
    CATEGORY_PRICE_EFFECTS,
    SEASONAL_PATTERNS,
    STATE_CODE_TO_REGION,
    get_event_multipliers,
    get_season,
    is_special_event,
    PRODUCT_CATEGORIES,
)


def setup_logging(log_dir: Path, log_to_file: bool = False) -> None:
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]

    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"transaction_generation_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# In transaction_data_generator.py


def prepare_consumer_data(df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """Prepare consumer data for transaction generation."""
    try:
        # Make a copy to avoid modifying original
        df = df.copy()

        # Print columns for debugging
        logging.info(f"Available columns: {df.columns.tolist()}")

        # Handle region mapping
        if "Region" in df.columns:
            df["REGION"] = df["Region"]
        elif "region" in df.columns:
            df["REGION"] = df["region"]
        elif "STATE" in df.columns:
            df["REGION"] = df["STATE"].map(STATE_CODE_TO_REGION)
        elif "State" in df.columns:
            df["REGION"] = df["State"].str.upper().map(STATE_CODE_TO_REGION)
        else:
            # If no state/region info, default to Midwest since data is from Illinois
            df["REGION"] = "Midwest"
            logging.warning("No state/region information found. Defaulting to Midwest.")

        # Ensure all required columns exist
        required_columns = ["REGION", "AGE", "INCOME", "BUYING_FREQUENCY"]
        for col in required_columns:
            if col not in df.columns:
                logging.warning(
                    f"Missing required column {col}. Adding default values."
                )
                if col == "REGION":
                    df[col] = "Midwest"
                elif col == "AGE":
                    df[col] = 35  # Default age
                elif col == "INCOME":
                    df[col] = 50000  # Default income
                elif col == "BUYING_FREQUENCY":
                    df[col] = "Monthly"  # Default frequency

        # Sample if needed
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        # Verify REGION column exists and has valid values
        if "REGION" not in df.columns:
            raise ValueError("REGION column not properly set during preparation")

        logging.info(f"Prepared data shape: {df.shape}")
        logging.info(f"Final columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        logging.error(f"Error preparing consumer data: {str(e)}")
        raise


def select_category(region: str) -> str:
    """Select category based on available product categories."""
    # Get list of all product categories from transaction_constants
    categories = list(PRODUCT_CATEGORIES.keys())

    # For now, use equal weights for all categories
    weights = [1.0] * len(categories)

    # Apply regional adjustments if in a specific season
    if region in REGIONAL_VARIATIONS:
        # Adjust weights based on regional characteristics
        region_data = REGIONAL_VARIATIONS[region]

        # If region has specific store type weights, adjust category preferences
        if "store_type_weights" in region_data:
            # Slight boost to certain categories based on store type prevalence
            if region_data["store_type_weights"].get("Convenience Store", 0) > 0.2:
                # Boost snacks and ready meals in regions with more convenience stores
                weights[categories.index("Snacks")] *= 1.2
                weights[categories.index("Ready Meals")] *= 1.2

    return random.choices(categories, weights=weights, k=1)[0]


def calculate_transaction_value(
    base_value: float, region: str, category: str, date: datetime
) -> float:
    """Calculate transaction value with all multipliers."""
    # Get regional characteristics with fallback
    region_data = get_region_characteristics(region)

    # Apply regional price index as the base multiplier
    value = base_value * region_data["price_index"]

    # Get season for seasonal adjustments
    season = get_season(date)

    # Apply category price effects based on season
    if season in CATEGORY_PRICE_EFFECTS["Season"]:
        seasonal_multiplier = CATEGORY_PRICE_EFFECTS["Season"][season].get(
            category, 1.0
        )
        value *= seasonal_multiplier

    # Apply store type effects (using average of store type multipliers for the region)
    if "store_type_weights" in region_data:
        store_multipliers = []
        for store_type, weight in region_data["store_type_weights"].items():
            if store_type in CATEGORY_PRICE_EFFECTS["Store_Type"]:
                store_mult = CATEGORY_PRICE_EFFECTS["Store_Type"][store_type].get(
                    category, 1.0
                )
                store_multipliers.append(store_mult * weight)
        if store_multipliers:
            value *= sum(store_multipliers)

    # Apply event multipliers
    event_mults = get_event_multipliers(date)
    value *= event_mults["value"]

    # Add small random variation (±2%)
    value *= random.uniform(0.98, 1.02)

    return round(value, 2)


def should_generate_transaction(date: datetime, region: str) -> bool:
    """Determine if a transaction should be generated."""
    # Base daily probability to achieve ~2.8 transactions per week
    base_prob = 0.4

    # Apply seasonal adjustment
    season = get_season(date)
    seasonal_pattern = SEASONAL_PATTERNS[season]
    seasonal_factor = seasonal_pattern["transaction_frequency_multiplier"]
    prob = base_prob * seasonal_factor

    # Apply event boost if applicable
    event_mults = get_event_multipliers(date)
    prob *= event_mults["transaction"]

    # Weekend boost (20%)
    if date.weekday() >= 5:
        prob *= 1.2

    return random.random() < prob


def is_online_transaction(region: str, date: datetime) -> bool:
    """Determine if transaction is online based on region and date."""
    # Base probabilities from research
    BASE_ONLINE_RATES = {
        "Northeast": 0.22,
        "West": 0.25,
        "Midwest": 0.18,
        "South": 0.16,
    }

    base_prob = BASE_ONLINE_RATES[region]

    # Apply event boost if applicable
    event_mults = get_event_multipliers(date)
    prob = base_prob * event_mults["online"]

    # Weekend boost
    if date.weekday() >= 5:  # Saturday or Sunday
        prob *= 1.1

    return random.random() < prob


def generate_transactions(
    consumers_df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Generate transaction data for given consumers and date range."""
    logging.info("Generating enhanced transaction data...")

    transactions = []
    dates = pd.date_range(start_date, end_date)
    BASE_VALUE = 50.00

    for idx, consumer in consumers_df.iterrows():
        if idx % 100 == 0:
            logging.info(f"Processing consumer {idx}/{len(consumers_df)}...")

        region = consumer["REGION"]

        for date in dates:
            # Determine if transaction occurs
            if not should_generate_transaction(date, region):
                continue

            # Get season for this date
            season = get_season(date)

            # Select category
            category = select_category(region)

            # Calculate value with seasonal adjustments
            value = calculate_transaction_value(BASE_VALUE, region, category, date)
            value *= SEASONAL_PATTERNS[season]["basket_size_multiplier"]

            # Determine channel
            is_online = is_online_transaction(region, date)

            transactions.append(
                {
                    "transaction_id": str(uuid.uuid4()),
                    "consumer_id": consumer.name,
                    "date": date,
                    "region": region,
                    "category": category,
                    "value": value,
                    "channel": "online" if is_online else "in_store",
                    "season": season,
                }
            )

    df = pd.DataFrame(transactions)
    return df


def main():
    """Main function to generate and validate transaction data."""
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    log_dir = script_dir / "logs"

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_dir, log_to_file=True)

    try:
        # Load consumer data - use il_demographics.csv instead of consumer_information.csv
        input_file = data_dir / "il_demographics.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        df = pd.read_csv(input_file)
        logging.info(f"Loaded consumer data. Shape: {df.shape}")

        # Prepare consumer data
        df = prepare_consumer_data(df, sample_size=1000)

        # Generate transactions for 2023
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        transactions_df = generate_transactions(df, start_date, end_date)

        # Save enhanced transaction dataset
        output_file = data_dir / "enhanced_transaction_data.csv"
        transactions_df.to_csv(output_file, index=False)
        logging.info(f"\nEnhanced transaction data saved to {output_file}")

        # Log final statistics
        logging.info(f"\nFinal dataset shape: {transactions_df.shape}")

        # Log column info
        logging.info("Generated columns:")
        for col in sorted(transactions_df.columns):
            non_null = transactions_df[col].count()
            logging.info(f"- {col}: {non_null:,} non-null values")

        # Log regional distribution
        logging.info("\nFinal Regional Distribution:")
        region_dist = transactions_df["region"].value_counts(normalize=True)
        for region, pct in region_dist.items():
            logging.info(f"{region}: {pct*100:.1f}%")

        # Log channel distribution
        logging.info("\nChannel Distribution:")
        channel_dist = transactions_df["channel"].value_counts(normalize=True)
        for channel, pct in channel_dist.items():
            logging.info(f"{channel}: {pct*100:.1f}%")

        # Log category distribution by region
        logging.info("\nCategory Distribution by Region:")
        for region in transactions_df["region"].unique():
            region_data = transactions_df[transactions_df["region"] == region]
            logging.info(f"\n{region}:")
            cat_dist = region_data["category"].value_counts(normalize=True)
            for cat, pct in cat_dist.items():
                logging.info(f"- {cat}: {pct*100:.1f}%")

    except Exception as e:
        logging.error(f"Script failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
