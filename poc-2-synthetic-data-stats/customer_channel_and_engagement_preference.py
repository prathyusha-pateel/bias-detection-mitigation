import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import shutil

# Constants from Statistical Requirements
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

CHANNEL_PREFERENCES = {
    "18-29": {
        "social_media": 0.84,
        "email": 0.92,
        "instagram": 0.78,
        "facebook": 0.65,
        "twitter": 0.45,  # Added Twitter
    },
    "30-49": {
        "social_media": 0.81,
        "email": 0.89,
        "instagram": 0.55,
        "facebook": 0.68,
        "twitter": 0.35,
    },
    "50-64": {
        "social_media": 0.73,
        "email": 0.85,
        "instagram": 0.32,
        "facebook": 0.70,
        "twitter": 0.25,
    },
    "65+": {
        "social_media": 0.59,
        "email": 0.80,
        "instagram": 0.15,
        "facebook": 0.65,
        "twitter": 0.15,
    },
}

ENGAGEMENT_METRICS = {
    "instagram": 0.0215,  # 2.15% engagement rate
    "facebook": 0.0012,  # 0.12% engagement rate
    "twitter": 0.00045,  # 0.045% engagement rate
    "email": {
        "open_rate": 0.1977,  # 19.77% open rate
        "click_rate": 0.0201,  # 2.01% click rate
        "unsubscribe": 0.0009,  # 0.09% unsubscribe rate
    },
}

PEAK_ENGAGEMENT_TIMES = {
    "Gen_Z": {"start": 20, "end": 23},  # 8PM-11PM
    "Millennial": {"start": 20, "end": 23},  # 8PM-11PM
    "Gen_X": {"start": 18, "end": 21},  # 6PM-9PM
    "Boomer": {"start": 14, "end": 17},  # 2PM-5PM
}

LINKEDIN_BASE_RATES = {
    "Gen_Z": 0.35,  # 35% for Gen Z
    "Millennial": 0.45,  # 45% for Millennials
    "Gen_X": 0.40,  # 40% for Gen X
    "Boomer": 0.30,  # 30% for Boomers
}


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


def load_column_mappings(csv_path: Path) -> dict:
    """Load column mappings from PUMS_CODES.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Column mappings file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return dict(zip(df["PUMS_CODE"].str.strip(), df["COLUMN_NAME"].str.strip()))


def validate_pums_data(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """Validate and clean PUMS data."""
    age_col = column_mappings["AGEP"]
    income_col = column_mappings["PINCP"]

    required_columns = [age_col, income_col]

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Log age statistics before filtering
    logging.info("\nAge statistics before filtering:")
    logging.info(f"Min age: {df[age_col].min()}")
    logging.info(f"Max age: {df[age_col].max()}")
    logging.info(f"Total records: {len(df)}")

    # Filter for adults (18+)
    df_filtered = df[df[age_col] >= 18].copy()

    # Log filtering results
    logging.info(f"Records after age filtering: {len(df_filtered)}")
    logging.info(f"Removed {len(df) - len(df_filtered)} records with age < 18")

    # Log income statistics before cleaning
    logging.info("\nIncome statistics before cleaning:")
    logging.info(f"Min income: ${df_filtered[income_col].min():,.2f}")
    logging.info(f"Max income: ${df_filtered[income_col].max():,.2f}")
    logging.info(f"Median income: ${df_filtered[income_col].median():,.2f}")
    logging.info(f"Mean income: ${df_filtered[income_col].mean():,.2f}")

    # Handle negative incomes
    negative_mask = df_filtered[income_col] < 0
    n_negative = negative_mask.sum()

    if n_negative > 0:
        logging.warning(f"\nFound {n_negative} negative income values")
        logging.info("Age distribution of negative income records:")
        logging.info(
            df_filtered[negative_mask][age_col].value_counts().sort_index().to_string()
        )

        # Replace negative incomes with median
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
    """Assign generation based on age."""
    if 18 <= age <= 24:
        return "Gen_Z"
    elif 25 <= age <= 40:
        return "Millennial"
    elif 41 <= age <= 56:
        return "Gen_X"
    else:
        return "Boomer"


def _generate_correlated_variable(
    base_variable: pd.Series, correlation: float
) -> pd.Series:
    """Generate a new variable with specified correlation to base_variable."""
    # Ensure input is properly normalized
    if not isinstance(base_variable, pd.Series):
        base_variable = pd.Series(base_variable)

    # Generate random component
    random_component = pd.Series(
        np.random.normal(0, 1, len(base_variable)), index=base_variable.index
    )

    # Generate correlated variable
    correlated = (
        correlation * base_variable + np.sqrt(1 - correlation**2) * random_component
    )

    # Scale to 0-100 range
    return (correlated - correlated.min()) / (correlated.max() - correlated.min()) * 100


def generate_consumers(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """Generate consumer preferences based on age"""
    logging.info("\nGenerating consumer preferences...")

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
    logging.info("\nGeneration distribution:")
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


def generate_correlated_series(
    base_series: pd.Series,
    target_correlation: float,
    positive_only: bool = False,
    random_state: int = None,
) -> pd.Series:
    """
    Generate a new series with a specified correlation to the base series.

    Args:
        base_series: The series to correlate with
        target_correlation: Desired correlation coefficient
        positive_only: If True, ensures all values are non-negative
        random_state: Random seed for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random normal variable
    random_series = pd.Series(
        np.random.normal(0, 1, len(base_series)), index=base_series.index
    )

    # Normalize base series
    base_normalized = (base_series - base_series.mean()) / base_series.std()

    # Generate correlated series using the formula:
    # new = r * base + sqrt(1-r^2) * random
    correlated = (
        target_correlation * base_normalized
        + np.sqrt(1 - target_correlation**2) * random_series
    )

    # Scale to reasonable range (0-100)
    scaled = (
        (correlated - correlated.min()) / (correlated.max() - correlated.min()) * 100
    )

    if positive_only:
        scaled = scaled.clip(lower=0)

    return scaled


def generate_linkedin_correlation(
    df: pd.DataFrame, education_col: str, social_mask: pd.Series
) -> pd.Series:
    """Generate LinkedIn usage with realistic education distribution and target correlation"""
    # Convert education to numeric and normalize
    edu_series = pd.to_numeric(df.loc[social_mask, education_col])
    edu_normalized = (edu_series - edu_series.mean()) / edu_series.std()
    generation_series = df.loc[social_mask, "generation"]

    # Initialize output series
    linkedin_binary = pd.Series(False, index=social_mask[social_mask].index)

    # Education quartile probabilities (increasing but not extreme)
    EDU_QUARTILE_MULTIPLIERS = {
        "Q1": 0.4,  # 40% of base rate
        "Q2": 0.8,  # 80% of base rate
        "Q3": 1.2,  # 120% of base rate
        "Q4": 1.6,  # 160% of base rate
    }

    # Process each generation separately
    for gen, base_rate in LINKEDIN_BASE_RATES.items():
        gen_mask = generation_series == gen
        if not gen_mask.any():
            continue

        # Get education scores and quartiles for this generation
        gen_edu = edu_normalized[gen_mask]
        edu_quartiles = pd.qcut(gen_edu, q=4, labels=["Q1", "Q2", "Q3", "Q4"])

        # Assign LinkedIn usage by education quartile
        for quartile, multiplier in EDU_QUARTILE_MULTIPLIERS.items():
            quartile_mask = edu_quartiles == quartile
            if not quartile_mask.any():
                continue

            # Calculate probability for this quartile
            prob = base_rate * multiplier
            prob = min(max(prob, 0.1), 0.9)  # Ensure probability is between 0.1 and 0.9

            # Assign based on probability and education ranking within quartile
            quartile_edu = gen_edu[quartile_mask]
            n_users = len(quartile_edu)
            n_active = int(n_users * prob)

            if n_active > 0:
                # Sort by education within quartile
                sorted_indices = quartile_edu.sort_values(ascending=False).index[
                    :n_active
                ]
                linkedin_binary[sorted_indices] = True

                # Add small random noise to maintain correlation but avoid perfect separation
                if len(sorted_indices) > 1:
                    noise_indices = np.random.choice(
                        sorted_indices,
                        size=min(len(sorted_indices) // 5, 10),
                        replace=False,
                    )
                    for idx in noise_indices:
                        linkedin_binary[idx] = False

    # Calculate and log statistics
    final_corr = edu_normalized.corr(linkedin_binary.astype(float))
    logging.info(f"\nGenerated LinkedIn-Education correlation: {final_corr:.3f}")

    # Log usage by education quartile
    edu_quartiles_all = pd.qcut(edu_normalized, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    for quartile in ["Q1", "Q2", "Q3", "Q4"]:
        mask = edu_quartiles_all == quartile
        usage_rate = linkedin_binary[mask].mean() * 100
        logging.info(f"LinkedIn usage in education {quartile}: {usage_rate:.1f}%")

    return linkedin_binary


def validate_linkedin_rates(df: pd.DataFrame, social_mask: pd.Series) -> None:
    """Validate LinkedIn usage rates by generation"""
    logging.info("\nLinkedIn usage rates by generation:")
    for gen, expected_rate in LINKEDIN_BASE_RATES.items():
        gen_mask = (df["generation"] == gen) & social_mask
        if gen_mask.any():
            actual_rate = df.loc[gen_mask, "uses_linkedin"].mean()
            logging.info(
                f"{gen}: {actual_rate*100:.1f}% " f"(target: {expected_rate*100:.1f}%)"
            )
            if abs(actual_rate - expected_rate) > 0.05:
                logging.warning(
                    f"LinkedIn usage rate for {gen} ({actual_rate:.2f}) "
                    f"differs from expected ({expected_rate:.2f})"
                )


def generate_behavioral_data(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """Generate behavioral data based on income correlations"""
    logging.info("\nGenerating behavioral data...")

    df = df.copy()
    income_col = column_mappings["PINCP"]

    # Clean income data
    df[income_col] = pd.to_numeric(df[income_col], errors="coerce")
    df[income_col] = df[income_col].fillna(df[income_col].median())

    # Normalize income for correlation generation
    income_normalized = (df[income_col] - df[income_col].mean()) / df[income_col].std()

    # Log income statistics
    logging.info(f"\nIncome statistics:")
    logging.info(f"Median income: ${df[income_col].median():,.2f}")
    logging.info(f"Mean income: ${df[income_col].mean():,.2f}")

    # Generate correlated variables
    df["buying_frequency"] = generate_correlated_series(
        income_normalized, INCOME_CORRELATIONS["multi_channel"], positive_only=True
    )

    df["avg_basket_size"] = generate_correlated_series(
        income_normalized, -PRICE_SENSITIVITY["premium_brands"], positive_only=True
    )

    df["brand_affinity"] = generate_correlated_series(
        income_normalized, -PRICE_SENSITIVITY["private_label"], positive_only=True
    )

    # Validate correlations
    for var, expected in {
        "buying_frequency": INCOME_CORRELATIONS["multi_channel"],
        "avg_basket_size": -PRICE_SENSITIVITY["premium_brands"],
        "brand_affinity": -PRICE_SENSITIVITY["private_label"],
    }.items():
        actual = df[var].corr(income_normalized)
        logging.info(
            f"Generated {var} correlation: {actual:.3f} (target: {expected:.3f})"
        )

    return df


def validate_input_data(df: pd.DataFrame, column_mappings: dict) -> None:
    """Validate input data for channel generation."""
    required_cols = ["generation", "age_group"]
    education_col = column_mappings.get("SCHL")

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for channel generation: {missing_cols}"
        )

    # Validate generation assignments
    valid_generations = {"Gen_Z", "Millennial", "Gen_X", "Boomer"}
    invalid_gens = set(df["generation"].unique()) - valid_generations
    if invalid_gens:
        raise ValueError(f"Invalid generation values found: {invalid_gens}")

    # Log distribution checks
    logging.info("\nValidating input data distributions:")
    gen_dist = df["generation"].value_counts(normalize=True)
    for gen, pct in gen_dist.items():
        logging.info(f"{gen}: {pct*100:.1f}%")

    if education_col and education_col in df.columns:
        logging.info("\nEducation level distribution:")
        edu_dist = df[education_col].value_counts(normalize=True)
        for edu, pct in edu_dist.items():
            logging.info(f"Level {edu}: {pct*100:.1f}%")


def generate_communication_preferences(
    df: pd.DataFrame, column_mappings: dict
) -> pd.DataFrame:
    """Generate communication preferences and engagement patterns"""
    logging.info("\nGenerating communication preferences...")

    # Validate input data first
    validate_input_data(df, column_mappings)

    df = df.copy()
    age_col = column_mappings["AGEP"]
    education_col = column_mappings.get("SCHL")

    # Create more granular age groups for channel preferences
    df["channel_age_group"] = pd.cut(
        df[age_col],
        bins=[17, 29, 49, 64, float("inf")],
        labels=["18-29", "30-49", "50-64", "65+"],
    )

    logging.info("\nChannel age group distribution:")
    age_dist = df["channel_age_group"].value_counts(normalize=True)
    for age_group, pct in age_dist.items():
        logging.info(f"{age_group}: {pct*100:.1f}%")

    # First generate social media usage as base for other platforms
    for age_group, prefs in CHANNEL_PREFERENCES.items():
        mask = df["channel_age_group"] == age_group
        if not mask.any():
            logging.warning(f"No records found for age group {age_group}")
            continue

        # Generate social media usage first
        df.loc[mask, "uses_social_media"] = np.random.binomial(
            1, prefs["social_media"], size=mask.sum()
        )

        # Then generate platform-specific usage conditional on social media use
        social_mask = mask & (df["uses_social_media"] == 1)
        if social_mask.any():
            for platform in ["instagram", "facebook", "twitter"]:
                df.loc[social_mask, f"uses_{platform}"] = np.random.binomial(
                    1, prefs[platform], size=social_mask.sum()
                )

        # Generate email usage separately (not conditional on social media)
        df.loc[mask, "uses_email"] = np.random.binomial(
            1, prefs["email"], size=mask.sum()
        )

    # Generate engagement rates
    for channel, rate in ENGAGEMENT_METRICS.items():
        if isinstance(rate, dict):  # Email has multiple metrics
            for metric, value in rate.items():
                mask = df["uses_email"] == 1
                if not mask.any():
                    logging.warning(f"No email users found for {metric}")
                    continue

                df.loc[mask, f"email_{metric}"] = np.random.normal(
                    value, value * 0.1, size=mask.sum()
                ).clip(0, 1)

                actual_mean = df.loc[mask, f"email_{metric}"].mean()
                logging.info(
                    f"Generated {metric}: {actual_mean:.3f} (target: {value:.3f})"
                )
        else:
            mask = df[f"uses_{channel}"] == 1
            if not mask.any():
                logging.warning(f"No {channel} users found")
                continue

            df.loc[mask, f"{channel}_engagement"] = np.random.normal(
                rate, rate * 0.1, size=mask.sum()
            ).clip(0, 1)

            actual_rate = df.loc[mask, f"{channel}_engagement"].mean()
            logging.info(
                f"Generated {channel} engagement: {actual_rate:.3f} (target: {rate:.3f})"
            )

    # Generate peak engagement times
    def generate_peak_time(generation):
        if generation not in PEAK_ENGAGEMENT_TIMES:
            logging.warning(f"No peak time defined for generation {generation}")
            return None

        time_range = PEAK_ENGAGEMENT_TIMES[generation]
        mean_time = (time_range["start"] + time_range["end"]) / 2
        peak_time = float(np.random.normal(mean_time, 1.0))  # 1 hour standard deviation
        return np.clip(peak_time, 0, 24)

    df["peak_engagement_hour"] = df["generation"].apply(generate_peak_time)

    if education_col and education_col in df.columns:
        social_mask = df["uses_social_media"] == 1
        if social_mask.any():
            df.loc[social_mask, "uses_linkedin"] = generate_linkedin_correlation(
                df, education_col, social_mask
            )

            # Validate correlation
            edu_series = pd.to_numeric(df.loc[social_mask, education_col])
            edu_normalized = (edu_series - edu_series.mean()) / edu_series.std()
            actual_corr = edu_normalized.corr(
                df.loc[social_mask, "uses_linkedin"].astype(float)
            )

            logging.info(
                f"\nLinkedIn-Education correlation: {actual_corr:.3f} (target: 0.42)"
            )

            if abs(actual_corr - 0.42) > 0.05:  # 5% tolerance
                logging.warning(
                    f"LinkedIn-Education correlation ({actual_corr:.2f}) "
                    f"differs from expected (0.42)"
                )

    # Log comprehensive statistics
    log_channel_statistics(df)

    return df


def _format_time(hour: float) -> str:
    """Format hour as HH:MM."""
    hours = int(hour)
    minutes = int((hour % 1) * 60)
    return f"{hours:02d}:{minutes:02d}"


def log_channel_statistics(df: pd.DataFrame) -> None:
    """Log comprehensive statistics about generated channel preferences"""
    # Channel adoption by generation
    logging.info("\nChannel adoption by generation:")
    for gen in sorted(df["generation"].unique()):
        gen_mask = df["generation"] == gen
        logging.info(f"\n{gen}:")
        for channel in CHANNEL_PREFERENCES["18-29"].keys():
            if f"uses_{channel}" in df.columns:
                adoption = df.loc[gen_mask, f"uses_{channel}"].mean() * 100
                logging.info(f"- {channel}: {adoption:.1f}%")
        if "uses_linkedin" in df.columns:
            linkedin_adoption = df.loc[gen_mask, "uses_linkedin"].mean() * 100
            logging.info(f"- linkedin: {linkedin_adoption:.1f}%")

    # Engagement metrics
    logging.info("\nEngagement metrics by generation:")
    for gen in sorted(df["generation"].unique()):
        gen_mask = df["generation"] == gen
        logging.info(f"\n{gen}:")
        for channel in ENGAGEMENT_METRICS.keys():
            if channel == "email":
                for metric in ["open_rate", "click_rate", "unsubscribe"]:
                    if f"email_{metric}" in df.columns:
                        mean_rate = df.loc[gen_mask, f"email_{metric}"].mean() * 100
                        logging.info(f"- Email {metric}: {mean_rate:.2f}%")
            else:
                if f"{channel}_engagement" in df.columns:
                    mean_rate = df.loc[gen_mask, f"{channel}_engagement"].mean() * 100
                    logging.info(f"- {channel}: {mean_rate:.3f}%")

    # Peak engagement times
    logging.info("\nPeak engagement time distribution (24-hour format):")
    for gen in sorted(df["generation"].unique()):
        gen_mask = df["generation"] == gen
        if "peak_engagement_hour" in df.columns:
            peak_times = df.loc[gen_mask, "peak_engagement_hour"]
            mean_time = peak_times.mean()
            std_time = peak_times.std()

            logging.info(
                f"{gen}: {_format_time(mean_time)} " f"(± {std_time:.1f} hours)"
            )


def validate_correlations(
    df: pd.DataFrame, column_mappings: dict, threshold: float = 0.02
) -> None:
    """Validate generated data maintains required correlations"""
    logging.info("\nValidating correlations...")
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
    # Set up directories
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

        # Validate and clean PUMS data
        pums_data = validate_pums_data(pums_data, column_mappings)
        logging.info("PUMS data validation successful")

        # Generate consumer preferences and behavioral data
        df = generate_consumers(pums_data, column_mappings)
        df = generate_behavioral_data(df, column_mappings)

        # Generate communication preferences
        df = generate_communication_preferences(df, column_mappings)

        # Final validation
        validate_correlations(df, column_mappings)

        # Log final dataset statistics
        logging.info("\nFinal dataset summary:")
        logging.info(f"Shape: {df.shape}")
        logging.info("\nColumns generated:")
        for col in sorted(df.columns):
            non_null = df[col].count()
            logging.info(f"- {col}: {non_null:,} non-null values")

        # Save enhanced dataset
        output_path = data_dir / "consumer_information.csv"
        df.to_csv(output_path, index=False)
        logging.info(f"\nEnhanced consumer data saved to {output_path}")

    except Exception as e:
        logging.error(f"Script failed with error: {e}")
        logging.error(f"Error details: {type(e).__name__}")
        raise


if __name__ == "__main__":
    main()
