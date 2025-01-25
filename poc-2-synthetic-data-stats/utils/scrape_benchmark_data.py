import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Define the raw marketing benchmarks
PRODUCT_CATEGORIES = {
    "category_name": [
        "Snacks",
        "Frozen Foods",
        "Condiments & Sauces",
        "Refrigerated Foods",
        "Grocery Staples",
    ],
    "industry_share": [  # Example shares from public reports
        0.25,
        0.30,
        0.15,
        0.20,
        0.10,
    ],
}

COMMUNICATION_CHANNELS = {
    "channel_name": ["Email", "SMS", "Social Media", "Direct Mail", "Mobile App"],
    "adoption_rate": [0.85, 0.54, 0.72, 0.35, 0.48],  # From HubSpot/Salesforce reports
    "engagement_rate": [0.21, 0.19, 0.06, 0.28, 0.12],
}

ENGAGEMENT_PREFERENCES = {
    "type": [
        "Digital Coupons",
        "Email Newsletters",
        "Social Media Interactions",
        "Loyalty Programs",
        "Mobile App Usage",
    ],
    "usage_rate": [0.65, 0.45, 0.38, 0.52, 0.31],  # Industry benchmarks
    "response_rate": [0.23, 0.18, 0.08, 0.44, 0.15],
}


def setup_logging(log_dir: Path) -> None:
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"marketing_benchmarks_{timestamp}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def save_raw_benchmarks(output_dir: Path) -> None:
    """Save raw marketing benchmarks to CSV files."""
    try:
        # Save product categories
        pd.DataFrame(PRODUCT_CATEGORIES).to_csv(
            output_dir / "raw_product_categories.csv", index=False
        )
        logging.info("Saved raw product categories data")

        # Save communication channels
        pd.DataFrame(COMMUNICATION_CHANNELS).to_csv(
            output_dir / "raw_communication_channels.csv", index=False
        )
        logging.info("Saved raw communication channels data")

        # Save engagement preferences
        pd.DataFrame(ENGAGEMENT_PREFERENCES).to_csv(
            output_dir / "raw_engagement_preferences.csv", index=False
        )
        logging.info("Saved raw engagement preferences data")

    except Exception as e:
        logging.error(f"Failed to save raw benchmarks: {e}")
        raise


def main():
    # Set up directories
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data" / "marketing_benchmarks"
    log_dir = script_dir / "logs"

    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_dir)

    try:
        logging.info("Starting to save raw marketing benchmarks")
        save_raw_benchmarks(data_dir)
        logging.info("Successfully saved all raw marketing benchmarks")

    except Exception as e:
        logging.error(f"Script failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
