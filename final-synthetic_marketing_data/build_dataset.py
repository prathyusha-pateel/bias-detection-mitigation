"""
Master orchestration script for generating synthetic datasets.

Coordinates the generation of:
1. Demographic data
2. Consumer preference data 
3. Marketing/engagement data
4. Transaction data

Ensures proper sequencing and validates dependencies between datasets.
"""

import sys
from pathlib import Path
import time
import importlib
import shutil
import logging
import os

from utils.logging import setup_logging
from constants import DEFAULT_DATA_DIR, DATA_SUBDIRS, FILE_PATTERNS

# Initialize master logger
logger = setup_logging(log_dir=None, log_to_file=False, module_name="build_dataset")


def run_module(module_name: str, logger: logging.Logger) -> bool:
    """
    Run a specific data generation module and handle any errors.

    Args:
        module_name: Name of the module to run
        logger: Logger instance

    Returns:
        bool: True if successful, False if failed
    """
    try:
        logger.info(f"\n{'='*20} Running {module_name} {'='*20}")
        start_time = time.time()

        # Import and run the module
        module = importlib.import_module(module_name)
        if hasattr(module, "main"):
            module.main()  # The module's main function should handle its own logging
        else:
            logger.error(f"No main() function found in {module_name}")
            return False

        elapsed_time = time.time() - start_time
        logger.info(f"✓ {module_name} completed successfully in {elapsed_time:.1f}s")
        return True

    except Exception as e:
        logger.error(f"✗ {module_name} failed with error: {str(e)}")
        return False


def verify_output(data_dir: Path, module_name: str, logger) -> bool:
    """
    Verify that a module produced its expected output files.
    """
    # Create the verification map using regular dictionary construction
    verification_map = {
        "demographic": lambda d: bool(list(d.glob("*demographics.csv"))),
    }

    # Add other patterns from FILE_PATTERNS
    for name, pattern in FILE_PATTERNS.items():
        if name != "demographic":  # Skip demographic since we already defined it
            verification_map[name] = lambda d, p=pattern: bool(list(d.glob(p)))

    verify_func = verification_map.get(module_name)
    if not verify_func:
        logger.error(f"No verification function defined for {module_name}")
        return False

    if verify_func(data_dir):
        logger.info(f"✓ Output verification passed for {module_name}")
        return True
    else:
        logger.error(f"✗ Required output files missing for {module_name}")
        return False


def clean_data_directory(data_dir: Path, logger) -> None:
    """
    Clean the data directory before running the build process.
    Uses shutil.rmtree to handle nested directories.

    Args:
        data_dir: Path to data directory
        logger: Logger instance
    """
    try:
        if data_dir.exists():
            # Use shutil.rmtree to remove directory and all contents
            shutil.rmtree(data_dir)
            logger.info(f"Cleaned data directory: {data_dir}")

        # Create fresh data directory
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created fresh data directory")

        # Create required subdirectories
        for subdir in DATA_SUBDIRS:
            (data_dir / subdir).mkdir(parents=True, exist_ok=True)
        logger.info("Created required subdirectories")

    except Exception as e:
        logger.error(f"Failed to clean data directory: {str(e)}")
        raise


def check_dependencies(data_dir: Path, module_name: str, logger) -> bool:
    """
    Check if required dependencies exist before running a module.

    Args:
        data_dir: Path to data directory
        module_name: Name of the module to check
        logger: Logger instance

    Returns:
        bool: True if dependencies exist, False otherwise
    """
    dependency_map = {
        "demographic": [],  # No dependencies
        "consumer": [
            # Requires demographic data
            lambda d: bool(list(d.glob("*_demographics.csv"))),
        ],
        "marketing": [
            # Requires demographic and consumer data
            lambda d: bool(list(d.glob("*_demographics.csv"))),
            lambda d: bool(list((d / "consumer").glob("*/consumer.csv"))),
        ],
        "transaction": [
            # Requires all previous data
            lambda d: bool(list(d.glob("*_demographics.csv"))),
            lambda d: bool(list((d / "consumer").glob("*/consumer.csv"))),
            lambda d: bool(list((d / "marketing_engagement").glob("*/campaigns.csv"))),
        ],
    }

    checks = dependency_map.get(module_name, [])
    if not checks:
        return True

    logger.info(f"Checking dependencies for {module_name}...")
    for check in checks:
        if not check(data_dir):
            logger.error(f"Missing required dependency files for {module_name}")
            return False

    logger.info(f"✓ All dependencies satisfied for {module_name}")
    return True


def main():
    """Main execution function."""
    # Get data directory from environment variable or use default
    data_dir = Path(os.getenv("SYNTHETIC_DATA_DIR", DEFAULT_DATA_DIR))

    logger.info("Starting dataset build process...")

    try:
        # Clean data directory
        clean_data_directory(data_dir, logger)

        # 1. Generate demographic data
        logger.info("\n" + "=" * 20 + " Processing Demographic base data " + "=" * 20)
        if not check_dependencies(data_dir, "demographic", logger):
            raise Exception("Dependency check failed for demographic")
        if not run_module("demographic", logger):
            raise Exception("Failed to generate demographic data")
        if not verify_output(data_dir, "demographic", logger):
            raise Exception("Output verification failed for demographic")
        logger.info("✓ Completed Demographic base data generation")

        # 2. Generate consumer preferences
        logger.info("\n" + "=" * 20 + " Processing Consumer preferences " + "=" * 20)
        if not check_dependencies(data_dir, "consumer", logger):
            raise Exception("Dependency check failed for consumer")
        if not run_module("consumer", logger):
            raise Exception("Failed to generate consumer data")
        if not verify_output(data_dir, "consumer", logger):
            raise Exception("Output verification failed for consumer")
        logger.info("✓ Completed Consumer preferences generation")

        # 3. Generate marketing/campaign data
        logger.info("\n" + "=" * 20 + " Processing Marketing engagement " + "=" * 20)
        if not check_dependencies(data_dir, "marketing", logger):
            raise Exception("Dependency check failed for marketing")
        if not run_module("marketing", logger):
            raise Exception("Failed to generate marketing data")
        if not verify_output(data_dir, "marketing", logger):
            raise Exception("Output verification failed for marketing")
        logger.info("✓ Completed Marketing engagement generation")

        # 4. Generate transaction data
        logger.info("\n" + "=" * 20 + " Processing Transaction data " + "=" * 20)
        if not check_dependencies(data_dir, "transaction", logger):
            raise Exception("Dependency check failed for transaction")
        if not run_module("transaction", logger):
            raise Exception("Failed to generate transaction data")
        if not verify_output(data_dir, "transaction", logger):
            raise Exception("Output verification failed for transaction")
        logger.info("✓ Completed Transaction data generation")

        logger.info("\n✓ Dataset build completed successfully")

    except Exception as e:
        logger.error("\n✗ Dataset build failed")
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
