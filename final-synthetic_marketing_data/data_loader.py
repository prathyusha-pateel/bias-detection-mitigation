"""
Data Loader

Loads and combines all generated synthetic data into a unified pandas data model.
Handles versioning and provides access to the complete synthetic dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pickle
import snappy
from tqdm import tqdm

from utils.logging import setup_logging

logger = setup_logging(module_name="data_loader")


class SyntheticDataLoader:
    """Loads and manages synthetic marketing data."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        test_mode: bool = False,
        cache_duration: int = 24,
    ):
        """Initialize the data loader.

        Args:
            data_dir: Path to data directory. If None, uses default path.
            test_mode: If True, uses data_test directory instead of data directory
            cache_duration: Number of hours before cache expires (default 24)
        """
        base_dir = Path(__file__).parent
        if test_mode:
            self.data_dir = data_dir or base_dir / "data_test"
            logger.info("Running in test mode using data_test directory")
        else:
            self.data_dir = data_dir or base_dir / "data"
            logger.info("Using production data directory")

        # Create directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache settings
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_duration = timedelta(hours=cache_duration)

        self.datasets: Dict[str, pd.DataFrame] = {}
        self._load_status = {}
        self.test_mode = test_mode

    def _get_cache_path(self) -> Path:
        """Get path to cache file based on test mode."""
        return (
            self.cache_dir / f"data_cache_{'test' if self.test_mode else 'prod'}.snappy"
        )

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache exists and is within validity period."""
        if not cache_path.exists():
            return False

        # Check cache age
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < self.cache_duration

    def _save_to_cache(self):
        """Save loaded datasets to cache file with snappy compression."""
        cache_path = self._get_cache_path()
        try:
            cache_data = {
                "datasets": self.datasets,
                "load_status": self._load_status,
                "timestamp": datetime.now(),
            }

            # Pickle and compress the data with progress bar
            pickled_data = pickle.dumps(cache_data)
            with tqdm(
                total=len(pickled_data), desc="Compressing", unit="B", unit_scale=True
            ) as pbar:
                from snappy import compress

                compressed_data = compress(pickled_data)
                pbar.update(len(pickled_data))

            # Write compressed data
            with open(cache_path, "wb") as f:
                f.write(compressed_data)

            # Log cache info
            original_size = len(pickled_data) / (1024 * 1024)  # MB
            compressed_size = len(compressed_data) / (1024 * 1024)  # MB
            compression_ratio = (1 - compressed_size / original_size) * 100

            logger.info(f"Data cached successfully to {cache_path}")
            logger.info(f"Original size: {original_size:.1f}MB")
            logger.info(f"Compressed size: {compressed_size:.1f}MB")
            logger.info(f"Compression ratio: {compression_ratio:.1f}%")

        except Exception as e:
            logger.warning(f"Failed to cache data: {str(e)}")

    def _load_from_cache(self) -> bool:
        """Load datasets from compressed cache if available and valid."""
        cache_path = self._get_cache_path()

        if not self._is_cache_valid(cache_path):
            return False

        try:
            # Read and decompress the data with progress bar
            with open(cache_path, "rb") as f:
                compressed_data = f.read()

            with tqdm(
                total=len(compressed_data),
                desc="Decompressing",
                unit="B",
                unit_scale=True,
            ) as pbar:
                from snappy import decompress

                decompressed_data = decompress(compressed_data)
                pbar.update(len(compressed_data))

            cache_data = pickle.loads(decompressed_data)

            self.datasets = cache_data["datasets"]
            self._load_status = cache_data["load_status"]
            cache_age = datetime.now() - cache_data["timestamp"]

            # Log cache info
            compressed_size = len(compressed_data) / (1024 * 1024)  # MB
            decompressed_size = len(decompressed_data) / (1024 * 1024)  # MB

            logger.info(
                f"Loaded data from cache ({cache_age.total_seconds()/3600:.1f} hours old)"
            )
            logger.info(f"Compressed size: {compressed_size:.1f}MB")
            logger.info(f"Decompressed size: {decompressed_size:.1f}MB")

            return True

        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            return False

    def get_latest_version(self, subdirectory: Path) -> Optional[Path]:
        """Get the latest version directory based on timestamp."""
        if not subdirectory.exists():
            return None

        # Look for directories that match the timestamp pattern YYYYMMDD_HHMMSS
        version_dirs = [
            d
            for d in subdirectory.iterdir()
            if d.is_dir() and len(d.name) == 15 and "_" in d.name
        ]

        if not version_dirs:
            logger.debug(f"No version directories found in {subdirectory}")
            logger.debug(
                f"Available directories: {[d.name for d in subdirectory.iterdir() if d.is_dir()]}"
            )
            return None

        # Sort by name (timestamp) and return the latest
        return max(version_dirs, key=lambda x: x.name)

    def load_demographics(self) -> Optional[pd.DataFrame]:
        """Load demographic data from CSV files."""
        try:
            demo_files = list(self.data_dir.glob("*_demographics.csv"))
            if not demo_files:
                logger.warning("No demographic data files found")
                return None

            dfs = []
            for file in demo_files:
                logger.info(f"Loading demographic data from {file.name}")
                df = pd.read_csv(file)
                dfs.append(df)

            combined_df = pd.concat(dfs, ignore_index=True)
            self.datasets["demographics"] = combined_df
            self._load_status["demographics"] = True
            return combined_df

        except Exception as e:
            logger.error(f"Error loading demographic data: {str(e)}")
            self._load_status["demographics"] = False
            return None

    def load_consumer_data(self) -> Optional[pd.DataFrame]:
        """Load consumer preference data from latest version."""
        try:
            consumer_dir = self.data_dir / "consumer"
            latest_version = self.get_latest_version(consumer_dir)

            if not latest_version:
                logger.warning("No consumer data versions found")
                return None

            consumer_file = latest_version / "consumer.csv"
            if not consumer_file.exists():
                logger.warning(f"No consumer data file found in {latest_version}")
                return None

            logger.info(f"Loading consumer data from {consumer_file}")
            df = pd.read_csv(consumer_file)
            self.datasets["consumer"] = df
            self._load_status["consumer"] = True
            return df

        except Exception as e:
            logger.error(f"Error loading consumer data: {str(e)}")
            self._load_status["consumer"] = False
            return None

    def load_transaction_data(
        self,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load transaction data, details, and product catalog from latest version."""
        try:
            transaction_dir = self.data_dir / "transactions"
            latest_version = self.get_latest_version(transaction_dir)

            if not latest_version:
                logger.warning("No transaction data versions found")
                return None, None, None

            # Load transactions
            transactions_file = latest_version / "transactions.csv"
            details_file = latest_version / "transaction_details.csv"
            catalog_file = latest_version / "product_catalog.csv"

            transactions = (
                pd.read_csv(transactions_file) if transactions_file.exists() else None
            )
            details = pd.read_csv(details_file) if details_file.exists() else None
            catalog = pd.read_csv(catalog_file) if catalog_file.exists() else None

            if transactions is not None:
                self.datasets["transactions"] = transactions
                self._load_status["transactions"] = True
            if details is not None:
                self.datasets["transaction_details"] = details
                self._load_status["transaction_details"] = True
            if catalog is not None:
                self.datasets["product_catalog"] = catalog
                self._load_status["product_catalog"] = True

            return transactions, details, catalog

        except Exception as e:
            logger.error(f"Error loading transaction data: {str(e)}")
            self._load_status["transactions"] = False
            return None, None, None

    def load_marketing_data(
        self,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load marketing campaign, engagement, and loyalty data from latest version."""
        try:
            marketing_dir = self.data_dir / "marketing_engagement"
            latest_version = self.get_latest_version(marketing_dir)

            if not latest_version:
                logger.warning("No marketing data versions found")
                return None, None, None

            # Load marketing data
            campaigns_file = latest_version / "campaigns.csv"
            engagements_file = latest_version / "engagements.csv"
            loyalties_file = latest_version / "loyalties.csv"

            campaigns = pd.read_csv(campaigns_file) if campaigns_file.exists() else None
            engagements = (
                pd.read_csv(engagements_file) if engagements_file.exists() else None
            )
            loyalties = pd.read_csv(loyalties_file) if loyalties_file.exists() else None

            if campaigns is not None:
                self.datasets["campaigns"] = campaigns
                self._load_status["campaigns"] = True
            if engagements is not None:
                self.datasets["engagements"] = engagements
                self._load_status["engagements"] = True
            if loyalties is not None:
                self.datasets["loyalties"] = loyalties
                self._load_status["loyalties"] = True

            return campaigns, engagements, loyalties

        except Exception as e:
            logger.error(f"Error loading marketing data: {str(e)}")
            self._load_status["marketing"] = False
            return None, None, None

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets, using cache if available."""
        logger.info("Loading synthetic marketing datasets...")

        # Try to load from cache first
        if self._load_from_cache():
            return self.datasets

        logger.info("Cache not available or expired, loading from source files...")

        # Load each dataset type
        self.load_demographics()
        self.load_consumer_data()
        self.load_transaction_data()
        self.load_marketing_data()

        # Report loading status
        logger.info("\nData loading status:")
        for dataset, status in self._load_status.items():
            status_msg = "✓ Loaded" if status else "✗ Failed"
            logger.info(f"{dataset}: {status_msg}")

        # Cache the loaded data
        self._save_to_cache()

        return self.datasets

    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get a specific dataset by name."""
        return self.datasets.get(name)

    def get_load_status(self) -> Dict[str, bool]:
        """Get the loading status of all datasets."""
        return self._load_status

    def get_data_path(self) -> Path:
        """Get the current data directory path."""
        return self.data_dir

    def is_test_mode(self) -> bool:
        """Check if loader is running in test mode."""
        return self.test_mode

    def generate_summary_stats(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Generate summary statistics for a dataset.

        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset for context

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            "record_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_mb": df.memory_usage().sum() / 1024**2,
            "null_counts": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Add dataset-specific metrics
        if dataset_name == "demographics":
            summary.update(
                {
                    "age_range": f"{df['AGEP'].min()}-{df['AGEP'].max()}",
                    "states_covered": len(df["STATE"].unique()),
                    "income_stats": {
                        "mean": df["PINCP"].mean(),
                        "median": df["PINCP"].median(),
                        "std": df["PINCP"].std(),
                    },
                }
            )

        elif dataset_name == "consumer":
            if "age_group" in df.columns:
                summary["age_distribution"] = df["age_group"].value_counts().to_dict()
            if "loyalty_memberships" in df.columns:
                summary["avg_loyalty_memberships"] = df["loyalty_memberships"].mean()

        elif dataset_name == "transactions":
            if "transaction_value" in df.columns:
                summary["transaction_stats"] = {
                    "avg_value": df["transaction_value"].mean(),
                    "total_value": df["transaction_value"].sum(),
                    "transaction_count": len(df),
                }

        elif dataset_name == "campaigns":
            if "campaign_type" in df.columns:
                summary["campaign_types"] = df["campaign_type"].value_counts().to_dict()
            if "budget" in df.columns:
                summary["total_budget"] = df["budget"].sum()

        return summary

    def print_dataset_summary(self, name: str, df: pd.DataFrame, indent: str = ""):
        """Print detailed summary for a dataset."""
        if df is None:
            logger.info(f"{indent}{name}: Not loaded")
            return

        stats = self.generate_summary_stats(df, name)

        logger.info(f"{indent}{name}:")
        logger.info(f"{indent}  Records: {stats['record_count']:,}")
        logger.info(f"{indent}  Columns: {len(df.columns)}")
        logger.info(f"{indent}  Memory: {stats['memory_usage_mb']:.2f} MB")

        # Print null counts if any exist
        null_cols = {k: v for k, v in stats["null_counts"].items() if v > 0}
        if null_cols:
            logger.info(f"{indent}  Columns with null values:")
            for col, count in null_cols.items():
                logger.info(f"{indent}    {col}: {count:,} nulls")

        # Print dataset-specific metrics
        if name == "demographics":
            logger.info(f"{indent}  Age Range: {stats['age_range']}")
            logger.info(f"{indent}  States Covered: {stats['states_covered']}")
            logger.info(f"{indent}  Income Statistics:")
            logger.info(f"{indent}    Mean: ${stats['income_stats']['mean']:,.2f}")
            logger.info(f"{indent}    Median: ${stats['income_stats']['median']:,.2f}")

        elif name == "consumer":
            if "age_distribution" in stats:
                logger.info(f"{indent}  Age Distribution:")
                for age_group, count in stats["age_distribution"].items():
                    logger.info(f"{indent}    {age_group}: {count:,}")
            if "avg_loyalty_memberships" in stats:
                logger.info(
                    f"{indent}  Avg Loyalty Memberships: {stats['avg_loyalty_memberships']:.2f}"
                )

        elif name == "transactions":
            if "transaction_stats" in stats:
                logger.info(f"{indent}  Transaction Statistics:")
                logger.info(
                    f"{indent}    Average Value: ${stats['transaction_stats']['avg_value']:,.2f}"
                )
                logger.info(
                    f"{indent}    Total Value: ${stats['transaction_stats']['total_value']:,.2f}"
                )

        elif name == "campaigns":
            if "campaign_types" in stats:
                logger.info(f"{indent}  Campaign Types:")
                for campaign_type, count in stats["campaign_types"].items():
                    logger.info(f"{indent}    {campaign_type}: {count:,}")
            if "total_budget" in stats:
                logger.info(f"{indent}  Total Budget: ${stats['total_budget']:,.2f}")

        logger.info("")

    def print_full_summary(self):
        """Print comprehensive summary of all loaded datasets."""
        logger.info("\n" + "=" * 50)
        logger.info("SYNTHETIC MARKETING DATA SUMMARY")
        logger.info("=" * 50 + "\n")

        # Group related datasets
        dataset_groups = {
            "Core Data": ["demographics", "consumer"],
            "Transaction Data": [
                "transactions",
                "transaction_details",
                "product_catalog",
            ],
            "Marketing Data": ["campaigns", "engagements", "loyalties"],
        }

        for group_name, dataset_names in dataset_groups.items():
            logger.info(f"{group_name}:")
            logger.info("-" * 40)
            for name in dataset_names:
                df = self.get_dataset(name)
                self.print_dataset_summary(name, df, indent="  ")
            logger.info("")

        # Print overall statistics
        total_records = sum(len(df) for df in self.datasets.values() if df is not None)
        total_memory = (
            sum(
                df.memory_usage().sum()
                for df in self.datasets.values()
                if df is not None
            )
            / 1024**2
        )

        logger.info("Overall Statistics:")
        logger.info("-" * 40)
        logger.info(f"Total Datasets: {len(self.datasets)}")
        logger.info(f"Total Records: {total_records:,}")
        logger.info(f"Total Memory Usage: {total_memory:.2f} MB")
        logger.info("\n" + "=" * 50)

    def clear_cache(self):
        """Clear the cached data."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            cache_path.unlink()
            logger.info("Cache cleared successfully")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Load synthetic marketing data")
    parser.add_argument("--test", action="store_true", help="Use test data directory")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear cached data before loading"
    )
    parser.add_argument(
        "--cache-duration",
        type=int,
        default=24,
        help="Cache duration in hours (default: 24)",
    )
    args = parser.parse_args()

    try:
        # Initialize loader with test mode if specified
        loader = SyntheticDataLoader(
            test_mode=args.test, cache_duration=args.cache_duration
        )

        if args.clear_cache:
            loader.clear_cache()

        # Get absolute path and log directory information
        data_path = loader.get_data_path().resolve()
        logger.info("=" * 50)
        logger.info("DATA DIRECTORY INFORMATION")
        logger.info("=" * 50)
        logger.info(f"Mode: {'Test' if args.test else 'Production'}")
        logger.info(f"Base Directory: {data_path.parent}")
        logger.info(f"Data Directory: {data_path}")
        logger.info(f"Directory exists: {'Yes' if data_path.exists() else 'No'}")
        if data_path.exists():
            contents = list(data_path.glob("*"))
            logger.info(f"Number of files/directories: {len(contents)}")
            logger.info("Contents:")
            for item in contents:
                logger.info(f"  - {item.name}")
        logger.info("=" * 50 + "\n")

        # Load all datasets
        datasets = loader.load_all()

        # Print comprehensive summary
        loader.print_full_summary()

    except Exception as e:
        logger.error(f"Error in data loading: {str(e)}")
        raise


if __name__ == "__main__":
    main()
