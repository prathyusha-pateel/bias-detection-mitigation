"""
Core Demographics Data Processing

This module handles the acquisition and processing of demographic data from PUMS sources.
Uses folktables ACSDataSource for data acquisition and integrates with validation pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from folktables import ACSDataSource
import shutil
import json
from typing import Dict, Optional, Any
from datetime import datetime
from sdv.metadata import Metadata

from constants import (
    FEATURES,
    STATES_FULL,
    STATES_MEDIUM,
    STATES_TEST,
    ACS_CONFIG,
    METADATA_CONFIGURATIONS,
)
from utils.logging import setup_logging
from validation.demographic.demographic_validator import DemographicValidator
from utils.encoders import NumpyEncoder
from utils.validation_display import display_validation_metrics
from utils.validation_metrics import (
    get_numerical_validation_details,
    get_categorical_validation_details,
    calculate_overall_score,
    count_total_metrics,
    count_passing_metrics,
)

NUMPY_FLOAT = np.float64


def check_data_requirements(df: pd.DataFrame, required_features: list, logger) -> bool:
    """
    Verify that DataFrame contains all required features.

    Args:
        df: DataFrame to check
        required_features: List of required column names
        logger: Logger instance

    Returns:
        Boolean indicating if all requirements are met
    """
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
        logger.error(f"Missing required features: {missing_features}")
        return False
    return True


def process_state_data(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Process raw state data with basic cleaning and transformations.

    Args:
        df: Raw state data
        logger: Logger instance

    Returns:
        Processed DataFrame
    """
    df = df.copy()

    # Basic cleaning
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # First convert to numeric, then cast to float64
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    # Handle income adjustment
    if "ADJINC" in df.columns and "PINCP" in df.columns:
        # Check if ADJINC is constant (same year data)
        if df["ADJINC"].nunique() == 1:
            # No need to apply row-by-row adjustment if constant
            logger.info("ADJINC is constant (single year data) - no adjustment needed")
        else:
            # Apply adjustment only if we have multiple years
            df["PINCP"] = df["PINCP"].astype(np.float64) * (
                df["ADJINC"].astype(np.float64) / 1000000
            )
            logger.info("Applied income adjustment factor (ADJINC) for multiple years")

    return df


def display_validation_results(
    results: Dict,
    synthetic_data: pd.DataFrame,
    original_data: pd.DataFrame,
    config: Dict[str, Any],
    logger,
    output_dir: Path,
) -> None:
    """Display validation results in a readable format."""
    # Ensure all numeric data is float64
    synthetic_data = synthetic_data.copy()
    original_data = original_data.copy()

    for df in [synthetic_data, original_data]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(np.float64)

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

    # Add metadata
    total_metrics = count_total_metrics(validation_results)
    passing_metrics = count_passing_metrics(validation_results)

    validation_results["metadata"] = {
        "overall_score": calculate_overall_score(validation_results),
        "total_metrics_checked": total_metrics,
        "passing_metrics": passing_metrics,
    }

    # Display validation results
    display_validation_metrics(
        validation_results, logger, "Demographic Validation Results"
    )

    # Create validation_results directory in validation folder
    validation_dir = (
        Path(__file__).resolve().parent / "validation" / "validation_results"
    )
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Save validation results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = validation_dir / f"demographic_validation_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(validation_results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nValidation results saved to {results_file}")


def download_filtered_pums_data(
    state: str, output_dir: Path, logger
) -> Optional[pd.DataFrame]:
    """Download and validate PUMS data for specified state."""
    try:
        logger.info(f"\nProcessing state: {state}")
        logger.info("=" * 50)

        # Create a custom progress bar for the download
        from tqdm import tqdm
        import requests
        from urllib.parse import urlparse
        import os

        # Create a custom ACSDataSource class with progress bar
        class ACSDataSourceWithProgress(ACSDataSource):
            def _download_data(self, download_url, download_path, show_progress=True):
                """Download with progress bar."""
                if show_progress:
                    response = requests.get(download_url, stream=True)
                    total_size = int(response.headers.get("content-length", 0))

                    # Ensure directory exists
                    os.makedirs(os.path.dirname(download_path), exist_ok=True)

                    with open(download_path, "wb") as file, tqdm(
                        desc=f"Downloading {state}",
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        for data in response.iter_content(chunk_size=1024):
                            size = file.write(data)
                            pbar.update(size)
                else:
                    super()._download_data(download_url, download_path)

        # Download and process data using configuration from constants
        data_source = ACSDataSourceWithProgress(
            survey_year=ACS_CONFIG["survey_year"],
            horizon=ACS_CONFIG["horizon"],
            survey=ACS_CONFIG["survey"],
            root_dir=str(output_dir),
        )

        logger.info(f"Downloading data for {state}...")

        # Get all available columns first
        df = data_source.get_data(states=[state], download=True)

        if df is None or df.empty:
            raise ValueError(f"No data retrieved for {state}")

        # Check which required features are missing
        missing_features = [feat for feat in FEATURES if feat not in df.columns]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")

        # Keep all required columns plus DENSITY if calculated
        columns_to_keep = FEATURES + (["DENSITY"] if "DENSITY" in df.columns else [])
        df = process_state_data(df[columns_to_keep], logger)

        # Save raw data
        output_file = output_dir / f"{state.lower()}_demographics.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nRaw demographic data saved to {output_file}")

        return df

    except Exception as e:
        logger.error(f"Failed to process {state}: {str(e)}")
        return None


def delete_data_folder(data_dir: Path, logger) -> None:
    """
    Delete the data directory if it exists.

    Args:
        data_dir: Directory to delete
        logger: Logger instance
    """
    if data_dir.exists() and data_dir.is_dir():
        shutil.rmtree(data_dir)
        logger.info(f"Deleted existing data directory: {data_dir}")


def validate_synthetic_data(
    synthetic_data: pd.DataFrame, original_data: pd.DataFrame, logger
) -> None:
    """Validate synthetic demographic data."""
    # Convert numeric columns to float64 before validation
    synthetic_data = synthetic_data.copy()
    original_data = original_data.copy()

    for df in [synthetic_data, original_data]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(np.float64)

    # Get configuration
    config = METADATA_CONFIGURATIONS.get("demographics", {})

    try:
        # Run validations
        validation_results = {}  # Initialize results dictionary

        # Display validation results with all required parameters
        display_validation_results(
            validation_results, synthetic_data, original_data, config, logger, Path(".")
        )

    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        raise


def create_metadata(initial_data: pd.DataFrame, data_type: str) -> Metadata:
    """Create metadata for synthesizer using SDV's recommended approach."""
    logger.info(f"Creating metadata for {data_type}...")

    # Create metadata instance
    metadata = Metadata()
    # ... existing metadata creation code ...

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


def main():
    """Main execution function."""
    # Set up directories
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    output_dir = data_dir  # Use data_dir as output_dir

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(module_name="demographics")

    try:
        # Parse command line arguments
        validate_only = False

        if not validate_only:
            # Clean data directory
            delete_data_folder(data_dir, logger)
            data_dir.mkdir(parents=True, exist_ok=True)

        # Choose state set based on environment or parameters
        # STATES_FULL Or STATES_MEDIUM or STATES_TEST depending on needs
        states = STATES_TEST

        logger.info(f"Selected states: {', '.join(states)}")
        logger.info("Selected states represent:")
        logger.info("- All major US regions")
        logger.info("- Mix of urban/rural populations")
        logger.info("- Diverse economic sectors")
        logger.info("- Different demographic profiles")
        logger.info("- Various education levels")

        state_data = {}

        if validate_only:
            # Load existing data files
            logger.info("Loading existing state data files...")
            for state in states:
                file_path = data_dir / f"{state.lower()}_demographics.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.upper()
                    state_data[state] = df
                else:
                    logger.warning(f"No existing data file found for {state}")
        else:
            # Download and process new data
            for state in states:
                df = download_filtered_pums_data(state, data_dir, logger)
                if df is not None:
                    df.columns = df.columns.str.upper()
                    state_data[state] = df

        # Create and run the demographic validator
        if state_data:
            # Convert numeric columns to float64 before validation
            for state, df in state_data.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    df[col] = df[col].astype(np.float64)
                state_data[state] = df

            demographic_validator = DemographicValidator(state_data)
            validation_results = demographic_validator.validate_all()

            # Create combined DataFrame from state_data
            combined_data = pd.concat(state_data.values(), ignore_index=True)

            # Display validation results with the actual data and output directory
            display_validation_results(
                validation_results,
                combined_data,
                combined_data,
                METADATA_CONFIGURATIONS["demographics"],
                logger,
                output_dir,
            )

            # Save validation results
            with open(data_dir / "validation_results.json", "w") as f:
                json.dump(validation_results, f, indent=2, cls=NumpyEncoder)

            logger.info("\nValidation results saved to validation_results.json")
        else:
            logger.warning("No state data available for validation")

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
