import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Define the PUMS codes we need (matching data.py structure)
FEATURES = [
    "SERIALNO",  # Unique identifier
    "AGEP",  # Age
    "PINCP",  # Personal income
    "STATE",  # State
    "PUMA",  # Public Use Microdata Area
    "SCHL",  # Educational attainment
    "SEX",  # Gender
    "RAC1P",  # Race
]


def load_column_mappings(csv_path: Path) -> dict:
    """Load column mappings from a CSV file."""
    df = pd.read_csv(csv_path)
    # Create a clean mapping dictionary from PUMS_CODE to COLUMN_NAME
    return dict(zip(df["PUMS_CODE"].str.strip(), df["COLUMN_NAME"].str.strip()))


def setup_logging(log_dir: Path, log_to_file: bool = False) -> None:
    """Set up logging consistent with data.py"""
    handlers = [logging.StreamHandler()]

    if log_to_file:
        print("Creating log directory and file")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"consumer_generation_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# Statistical patterns remain the same as before...
AGE_PRODUCT_PREFERENCES = {
    "18-34": {"ready_to_eat": 0.38, "snacks": 0.42, "sustainable": 0.45},
    "35-54": {"family_size": 0.45, "healthy_alternatives": 0.35, "sustainable": 0.38},
    "55+": {"traditional": 0.52, "health_conscious": 0.48, "sustainable": 0.29},
}

DEVICE_PREFERENCES = {
    "Gen_Z": {"mobile": 0.72, "desktop": 0.28},
    "Millennial": {"mobile": 0.65, "desktop": 0.35},
    "Gen_X": {"mobile": 0.48, "desktop": 0.52},
    "Boomer": {"mobile": 0.31, "desktop": 0.69},
}

INCOME_CORRELATIONS = {
    "multi_channel": 0.31,
    "digital_preference": 0.28,
    "digital_adoption": 0.34,
}

PRICE_SENSITIVITY = {
    "premium_brands": -0.38,
    "private_label": -0.22,
    "sustainable": -0.15,
}


def validate_pums_data(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """Validate PUMS data has required columns and proper values. Returns filtered dataframe."""
    age_col = column_mappings["AGEP"]
    income_col = column_mappings["PINCP"]

    required_columns = [age_col, income_col]

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Log age statistics before filtering
    logging.info(f"Age statistics before filtering:")
    logging.info(f"Min age: {df[age_col].min()}")
    logging.info(f"Max age: {df[age_col].max()}")
    logging.info(f"Total records: {len(df)}")

    # Filter for adults (18+)
    df_filtered = df[df[age_col] >= 18].copy()

    # Log filtering results
    logging.info(f"Records after age filtering: {len(df_filtered)}")
    logging.info(f"Removed {len(df) - len(df_filtered)} records with age < 18")

    # Validate remaining age values
    if df_filtered[age_col].max() > 100:
        logging.warning(
            f"Found {len(df_filtered[df_filtered[age_col] > 100])} records with age > 100"
        )

    # Log income statistics before cleaning
    logging.info("\nIncome statistics before cleaning:")
    logging.info(f"Min income: ${df_filtered[income_col].min():,.2f}")
    logging.info(f"Max income: ${df_filtered[income_col].max():,.2f}")
    logging.info(f"Median income: ${df_filtered[income_col].median():,.2f}")
    logging.info(f"Mean income: ${df_filtered[income_col].mean():,.2f}")

    # Handle negative income values
    negative_mask = df_filtered[income_col] < 0
    n_negative = negative_mask.sum()

    if n_negative > 0:
        logging.warning(f"\nFound {n_negative} negative income values")
        logging.info("Income statistics for negative income records:")
        neg_df = df_filtered[negative_mask]
        logging.info(f"Age distribution of negative income records:")
        logging.info(neg_df[age_col].value_counts().sort_index().to_string())

        # Replace negative incomes with median income
        median_income = df_filtered[df_filtered[income_col] >= 0][income_col].median()
        df_filtered.loc[negative_mask, income_col] = median_income
        logging.info(
            f"\nReplaced negative incomes with median income: ${median_income:,.2f}"
        )

    # Log final income statistics
    logging.info("\nIncome statistics after cleaning:")
    logging.info(f"Min income: ${df_filtered[income_col].min():,.2f}")
    logging.info(f"Max income: ${df_filtered[income_col].max():,.2f}")
    logging.info(f"Median income: ${df_filtered[income_col].median():,.2f}")
    logging.info(f"Mean income: ${df_filtered[income_col].mean():,.2f}")

    # Check for null values
    null_counts = df_filtered[required_columns].isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}"
        )

    return df_filtered


def assign_generation(age):
    """Assign generation based on age"""
    if 18 <= age <= 24:
        return "Gen_Z"
    elif 25 <= age <= 40:
        return "Millennial"
    elif 41 <= age <= 56:
        return "Gen_X"
    else:
        return "Boomer"


def generate_consumers(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """Generate consumer preferences based on age"""
    logging.info("Generating consumer preferences...")

    df = df.copy()
    age_col = column_mappings["AGEP"]

    # Create age groups using the mapped age column
    df["age_group"] = pd.cut(
        df[age_col], bins=[17, 34, 54, float("inf")], labels=["18-34", "35-54", "55+"]
    )

    # Generate product preferences
    for age_group, prefs in AGE_PRODUCT_PREFERENCES.items():
        mask = df["age_group"] == age_group
        for product, prob in prefs.items():
            df.loc[mask, f"pref_{product}"] = np.random.binomial(
                1, prob, size=mask.sum()
            )

    # Generate communication preferences using age
    df["generation"] = df[age_col].apply(assign_generation)

    # Log generation distribution
    gen_dist = df["generation"].value_counts()
    logging.info("Generation distribution:")
    for gen, count in gen_dist.items():
        logging.info(f"{gen}: {count} ({count/len(df)*100:.1f}%)")

    # Generate device preferences
    for gen, prefs in DEVICE_PREFERENCES.items():
        mask = df["generation"] == gen
        if mask.any():
            df.loc[mask, "preferred_device"] = np.random.choice(
                ["mobile", "desktop"],
                size=mask.sum(),
                p=[prefs["mobile"], prefs["desktop"]],
            )

    return df


def _generate_correlated_variable(base_variable, correlation):
    """Generate a new variable with specified correlation"""
    # Ensure input is properly normalized
    base_normalized = (base_variable - base_variable.mean()) / base_variable.std()

    # Generate random component
    random_component = np.random.normal(0, 1, len(base_variable))

    # Generate correlated variable
    correlated = (
        correlation * base_normalized + np.sqrt(1 - correlation**2) * random_component
    )

    # Scale back to reasonable range (0-100 scale)
    return (correlated - correlated.min()) / (correlated.max() - correlated.min()) * 100


def generate_behavioral_data(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """Generate behavioral data based on income correlations"""
    logging.info("Generating behavioral data...")

    df = df.copy()
    income_col = column_mappings["PINCP"]

    # Clean income data
    df[income_col] = pd.to_numeric(df[income_col], errors="coerce")
    df[income_col] = df[income_col].fillna(df[income_col].median())

    # Log income statistics
    logging.info(f"Income statistics:")
    logging.info(f"Median income: ${df[income_col].median():,.2f}")
    logging.info(f"Mean income: ${df[income_col].mean():,.2f}")

    # Generate correlated variables
    df["buying_frequency"] = _generate_correlated_variable(
        df[income_col], INCOME_CORRELATIONS["multi_channel"]
    )
    df["avg_basket_size"] = _generate_correlated_variable(
        df[income_col], -PRICE_SENSITIVITY["premium_brands"]
    )
    df["brand_affinity"] = _generate_correlated_variable(
        df[income_col], -PRICE_SENSITIVITY["private_label"]
    )

    return df


def validate_correlations(
    df: pd.DataFrame, column_mappings: dict, threshold: float = 0.02
) -> None:
    """Validate generated data maintains required correlations"""
    logging.info("Validating correlations...")
    income_col = column_mappings["PINCP"]

    # Clean and normalize income
    income_normalized = (df[income_col] - df[income_col].mean()) / df[income_col].std()

    # Calculate and validate correlations
    correlations = {
        "multi_channel": np.corrcoef(income_normalized, df["buying_frequency"])[0, 1],
        "avg_basket_size": np.corrcoef(income_normalized, df["avg_basket_size"])[0, 1],
        "private_label": np.corrcoef(income_normalized, df["brand_affinity"])[0, 1],
    }

    # Expected correlations from constants
    expected_correlations = {
        "multi_channel": INCOME_CORRELATIONS["multi_channel"],
        "avg_basket_size": -PRICE_SENSITIVITY["premium_brands"],
        "private_label": -PRICE_SENSITIVITY["private_label"],
    }

    # Log and validate each correlation
    for var, actual_corr in correlations.items():
        expected_corr = expected_correlations[var]
        logging.info(
            f"Correlation for {var}: {actual_corr:.3f} (expected: {expected_corr:.3f})"
        )

        if abs(actual_corr - expected_corr) > threshold:
            logging.warning(
                f"Correlation for {var} ({actual_corr:.3f}) differs from "
                f"expected ({expected_corr:.3f}) by more than {threshold}"
            )


def main():
    # Set up directories following data.py pattern
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    log_dir = script_dir / "logs"
    csv_path = script_dir / "docs" / "PUMS_CODES.csv"

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_dir)

    try:
        # Load column mappings
        column_mappings = load_column_mappings(csv_path)
        logging.info("Successfully loaded column mappings")

        # Load PUMS data
        pums_data = pd.read_csv(data_dir / "il_demographics.csv")
        logging.info(f"Loaded PUMS data shape: {pums_data.shape}")

        # Validate and filter PUMS data
        pums_data = validate_pums_data(pums_data, column_mappings)
        logging.info("PUMS data validation successful")

        # Generate additional consumer information
        df = generate_consumers(pums_data, column_mappings)
        df = generate_behavioral_data(df, column_mappings)

        # Validate correlations
        validate_correlations(df, column_mappings)

        # Save enhanced dataset
        output_path = data_dir / "consumer_information.csv"
        df.to_csv(output_path, index=False)
        logging.info(f"Enhanced consumer data saved to {output_path}")

    except Exception as e:
        logging.error(f"Script failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
