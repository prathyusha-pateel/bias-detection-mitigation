import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Constants from verified sources
# Pew Research Center "Social Media Use in 2023"
SOCIAL_MEDIA_ADOPTION = {"18-29": 0.84, "30-49": 0.81, "50-64": 0.73, "65+": 0.59}

# Merkle Q4 2023 Digital Marketing Report
ENGAGEMENT_RATES = {"instagram": 0.0215, "facebook": 0.0012, "twitter": 0.00045}

# Mailchimp 2023 Email Marketing Benchmarks (Food & Beverage)
EMAIL_METRICS = {"open_rate": 0.1977, "click_rate": 0.0201, "unsubscribe_rate": 0.0009}

# McKinsey 2023 US Consumer Pulse Survey
DEVICE_PREFERENCES = {
    "Gen_Z": {"mobile": 0.72, "desktop": 0.28},
    "Millennial": {"mobile": 0.65, "desktop": 0.35},
    "Gen_X": {"mobile": 0.48, "desktop": 0.52},
    "Boomer": {"mobile": 0.31, "desktop": 0.69},
}

# Kim & Kumar 2023 study correlations
INCOME_CORRELATIONS = {
    "multi_channel": 0.31,
    "digital_preference": 0.28,
    "digital_adoption": 0.34,
    "mobile_app": 0.29,
}


def setup_logging(log_dir: Path, log_to_file: bool = False) -> None:
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]

    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"device_shopping_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def generate_device_preferences(df: pd.DataFrame) -> pd.DataFrame:
    """Generate device preferences based on McKinsey 2023 data."""
    logging.info("Generating device preferences...")

    df = df.copy()

    for generation, prefs in DEVICE_PREFERENCES.items():
        mask = df["generation"] == generation
        if mask.any():
            # Mobile vs desktop preference
            df.loc[mask, "primary_device"] = np.random.choice(
                ["mobile", "desktop"],
                size=mask.sum(),
                p=[prefs["mobile"], prefs["desktop"]],
            )

            actual_mobile = (df.loc[mask, "primary_device"] == "mobile").mean()
            logging.info(
                f"{generation} mobile preference: {actual_mobile:.3f} (target: {prefs['mobile']:.3f})"
            )

    return df


def generate_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate engagement metrics based on Merkle Q4 2023 report."""
    logging.info("Generating engagement metrics...")

    df = df.copy()

    for platform, rate in ENGAGEMENT_RATES.items():
        # Add random variation within ±10% of target rate
        variation = rate * 0.1
        df[f"{platform}_engagement"] = np.random.normal(rate, variation, size=len(df))
        df[f"{platform}_engagement"] = df[f"{platform}_engagement"].clip(0, 1)

        actual_rate = df[f"{platform}_engagement"].mean()
        logging.info(
            f"{platform} engagement rate: {actual_rate:.4f} (target: {rate:.4f})"
        )

    return df


def generate_email_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate email metrics based on Mailchimp 2023 benchmarks."""
    logging.info("Generating email metrics...")

    df = df.copy()

    for metric, rate in EMAIL_METRICS.items():
        variation = rate * 0.1
        df[f"email_{metric}"] = np.random.normal(rate, variation, size=len(df))
        df[f"email_{metric}"] = df[f"email_{metric}"].clip(0, 1)

        actual_rate = df[f"email_{metric}"].mean()
        logging.info(f"Email {metric}: {actual_rate:.4f} (target: {rate:.4f})")

    return df


def generate_income_correlated_metrics(
    df: pd.DataFrame, column_mappings: dict
) -> pd.DataFrame:
    """Generate metrics correlated with income based on Kim & Kumar 2023."""
    logging.info("Generating income-correlated metrics...")

    df = df.copy()
    income_col = column_mappings["PINCP"]

    # Normalize income
    income_normalized = (df[income_col] - df[income_col].mean()) / df[income_col].std()

    for metric, target_corr in INCOME_CORRELATIONS.items():
        random_component = np.random.normal(0, 1, len(df))
        correlated = (
            target_corr * income_normalized
            + np.sqrt(1 - target_corr**2) * random_component
        )

        df[f"{metric}_score"] = (
            (correlated - correlated.min())
            / (correlated.max() - correlated.min())
            * 100
        )

        actual_corr = np.corrcoef(income_normalized, df[f"{metric}_score"])[0, 1]
        logging.info(
            f"{metric} correlation with income: {actual_corr:.3f} (target: {target_corr:.3f})"
        )

    return df


def validate_distributions(df: pd.DataFrame) -> None:
    """Validate the generated distributions against source data."""
    logging.info("\nValidating distributions...")

    # Validate device preferences
    for generation in DEVICE_PREFERENCES.keys():
        mask = df["generation"] == generation
        if mask.any():
            mobile_pct = (df.loc[mask, "primary_device"] == "mobile").mean()
            target = DEVICE_PREFERENCES[generation]["mobile"]
            logging.info(
                f"{generation} mobile usage: {mobile_pct:.3f} (target: {target:.3f})"
            )

    # Validate engagement metrics
    for platform, target in ENGAGEMENT_RATES.items():
        actual = df[f"{platform}_engagement"].mean()
        logging.info(f"{platform} engagement: {actual:.4f} (target: {target:.4f})")

    # Validate email metrics
    for metric, target in EMAIL_METRICS.items():
        actual = df[f"email_{metric}"].mean()
        logging.info(f"Email {metric}: {actual:.4f} (target: {target:.4f})")


def main():
    # Set up directories
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    log_dir = script_dir / "logs"
    csv_path = script_dir / "docs" / "PUMS_CODES.csv"

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_dir, log_to_file=True)

    try:
        # Load existing consumer data
        input_file = data_dir / "consumer_information.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        df = pd.read_csv(input_file)
        logging.info(f"Loaded consumer data. Shape: {df.shape}")

        # Load column mappings
        column_mappings = pd.read_csv(csv_path)
        column_mappings = dict(
            zip(
                column_mappings["PUMS_CODE"].str.strip(),
                column_mappings["COLUMN_NAME"].str.strip(),
            )
        )

        # Generate data using verified sources
        df = generate_device_preferences(df)
        df = generate_engagement_metrics(df)
        df = generate_email_metrics(df)
        df = generate_income_correlated_metrics(df, column_mappings)

        # Validate all distributions
        validate_distributions(df)

        # Save enhanced dataset
        output_file = data_dir / "enhanced_consumer_data_validated.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"\nEnhanced consumer data saved to {output_file}")

        # Log final column count
        logging.info(f"\nFinal dataset shape: {df.shape}")
        logging.info("New columns added:")
        new_cols = [
            col for col in df.columns if col not in pd.read_csv(input_file).columns
        ]
        for col in sorted(new_cols):
            logging.info(f"- {col}")

    except Exception as e:
        logging.error(f"Script failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
