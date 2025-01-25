"""
Data Export Utility

Exports synthetic marketing data in various formats (CSV, Parquet, SQLite) with validation,
proper data typing, and comprehensive documentation.
"""

import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List
import json
from datetime import datetime
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataExporter:
    """Enhanced utility for exporting synthetic data in various formats."""

    def __init__(self, loader):
        """Initialize with a DataLoader instance."""
        self.loader = loader
        self.datasets = loader.load_all()
        self._validate_datasets()

    def _validate_datasets(self) -> None:
        """Validate loaded datasets for expected structure."""
        required_datasets = {
            "demographics": ["SERIALNO", "AGEP", "SEX", "PINCP"],
            "consumer": ["consumer_id", "age_group"],
            "transactions": ["transaction_id", "consumer_id"],
            "product_catalog": ["product_id", "category"],
        }

        for name, required_cols in required_datasets.items():
            if name not in self.datasets:
                logger.warning(f"Missing required dataset: {name}")
                continue

            df = self.datasets[name]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing required columns in {name}: {missing_cols}")

    def export_csv(self, output_dir: Path) -> None:
        """Export as standardized CSV files with enhanced validation."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define standardized column mappings
        column_maps = {
            "demographics": ["SERIALNO", "AGEP", "SEX", "PINCP", "STATE", "PUMA"],
            "consumer": [
                "consumer_id",
                "age_group",
                "online_shopping_rate",
                "social_media_engagement_rate",
                "loyalty_memberships",
            ],
            "transactions": [
                "transaction_id",
                "consumer_id",
                "transaction_date",
                "channel",
                "transaction_value",
                "state",
                "region",
            ],
            "transaction_details": [
                "transaction_id",
                "product_id",
                "quantity",
                "unit_price",
                "line_total",
            ],
            "product_catalog": ["product_id", "category", "subcategory", "base_price"],
            "campaigns": [
                "campaign_id",
                "campaign_type",
                "start_date",
                "end_date",
                "target_age_min",
                "target_age_max",
                "budget",
            ],
            "engagements": [
                "engagement_id",
                "campaign_id",
                "date",
                "impressions",
                "clicks",
                "conversion_rate",
            ],
        }

        for name, columns in tqdm(column_maps.items(), desc="Exporting CSV files"):
            if name in self.datasets:
                df = self.datasets[name]
                if df.empty:
                    logger.warning(f"Dataset {name} is empty")
                    continue

                # Validate columns
                existing_cols = df.columns
                cols_to_export = [col for col in columns if col in existing_cols]
                if len(cols_to_export) != len(columns):
                    missing = set(columns) - set(existing_cols)
                    logger.warning(f"Missing columns in {name}: {missing}")

                # Export with type preservation
                output_file = output_dir / f"{name}.csv"
                df[cols_to_export].to_csv(
                    output_file, index=False, date_format="%Y-%m-%d %H:%M:%S"
                )

                logger.info(f"Exported {name} to {output_file}")

        # Export schema documentation
        self._export_schema_docs(output_dir)

    def export_parquet(self, output_dir: Path) -> None:
        """Export as Parquet files with compression and validation."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for name in tqdm(self.datasets, desc="Exporting Parquet files"):
            df = self.datasets[name]
            if df.empty:
                logger.warning(f"Dataset {name} is empty")
                continue

            output_file = output_dir / f"{name}.parquet"
            try:
                df.to_parquet(output_file, compression="snappy", index=False)
                logger.info(f"Exported {name} to {output_file}")
            except Exception as e:
                logger.error(f"Failed to export {name} to parquet: {e}")

    def export_sqlite(self, output_file: Path) -> None:
        """Export as SQLite database."""
        conn = sqlite3.connect(str(output_file))

        try:
            # Start transaction
            conn.execute("BEGIN TRANSACTION")

            # Create tables and insert data
            for table in tqdm(
                self.datasets.keys(), desc="Creating and populating tables"
            ):
                df = self.datasets[table]
                if df.empty:
                    logger.warning(f"Table {table} is empty")
                    continue

                # Drop existing table if it exists
                conn.execute(f"DROP TABLE IF EXISTS {table}")

                # Create table from DataFrame
                df.to_sql(table, conn, if_exists="replace", index=False)
                logger.info(f"Created and populated table: {table}")

            # Commit transaction
            conn.commit()
            logger.info(f"\nSuccessfully exported to SQLite database: {output_file}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to export to SQLite: {e}")
            raise
        finally:
            conn.close()

    def _export_schema_docs(self, output_dir: Path) -> None:
        """Export enhanced schema documentation."""
        docs = []
        docs.append("# Synthetic Marketing Data Schema\n")

        type_descriptions = {
            "datetime64[ns]": "Timestamp",
            "float64": "Decimal number",
            "int64": "Integer",
            "object": "Text/String",
            "bool": "True/False",
            "category": "Categorical value",
        }

        for name, df in self.datasets.items():
            docs.append(f"## {name}\n")
            docs.append("| Column | Type | Description |")
            docs.append("|--------|------|-------------|")

            for col in df.columns:
                dtype = str(df[col].dtype)
                type_desc = type_descriptions.get(dtype, dtype)
                desc = self._get_column_description(name, col)
                docs.append(f"| {col} | {type_desc} | {desc} |")
            docs.append("\n")

            # Add sample values for categorical columns
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(cat_cols) > 0:
                docs.append("### Categorical Values\n")
                for col in cat_cols:
                    unique_vals = df[col].unique()
                    if len(unique_vals) < 10:  # Only show if reasonable number
                        docs.append(f"**{col}**: {', '.join(map(str, unique_vals))}\n")

            docs.append("\n")

        with open(output_dir / "schema.md", "w") as f:
            f.write("\n".join(docs))

        logger.info(f"Exported schema documentation to {output_dir}/schema.md")

    def _get_column_description(self, table: str, column: str) -> str:
        """Get description for a column based on common naming patterns."""
        descriptions = {
            "consumer_id": "Unique identifier for consumer",
            "transaction_id": "Unique identifier for transaction",
            "product_id": "Unique identifier for product",
            "campaign_id": "Unique identifier for marketing campaign",
            "engagement_id": "Unique identifier for engagement record",
            "age_group": "Age group category",
            "transaction_value": "Total value of transaction",
            "unit_price": "Price per unit",
            "quantity": "Number of units purchased",
            "category": "Product category",
            "subcategory": "Product subcategory",
            "SERIALNO": "Demographic record identifier",
            "AGEP": "Age of person",
            "SEX": "Gender",
            "PINCP": "Total person's income",
            "STATE": "State code",
            "online_shopping_rate": "Rate of online shopping activity",
            "social_media_engagement_rate": "Rate of social media engagement",
            "loyalty_memberships": "Number of loyalty program memberships",
            "channel": "Transaction channel (e.g., online, in-store)",
            "region": "Geographic region",
            "base_price": "Base price of product",
            "campaign_type": "Type of marketing campaign",
            "start_date": "Campaign start date",
            "end_date": "Campaign end date",
            "target_age_min": "Minimum target age for campaign",
            "target_age_max": "Maximum target age for campaign",
            "budget": "Campaign budget",
            "impressions": "Number of campaign impressions",
            "clicks": "Number of clicks on campaign",
            "conversion_rate": "Campaign conversion rate",
            "line_total": "Total line item value",
        }
        return descriptions.get(column, "No description available")


if __name__ == "__main__":
    import argparse
    from data_loader import SyntheticDataLoader

    parser = argparse.ArgumentParser(description="Export synthetic marketing data")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear cached data before loading"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test data instead of production data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../exported_data",
        help="Base directory for exported data",
    )
    parser.add_argument(
        "--format",
        choices=["all", "csv", "parquet", "sqlite"],
        default="all",
        help="Export format",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize loader with options
        loader = SyntheticDataLoader(
            test_mode=args.test_mode, cache_duration=24  # 24 hour cache duration
        )

        if args.clear_cache:
            loader.clear_cache()
            logger.info("Cache cleared")

        # Initialize exporter
        exporter = DataExporter(loader)

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(args.output_dir) / timestamp

        # Export in requested format(s)
        if args.format in ["all", "csv"]:
            logger.info("Exporting CSV files...")
            exporter.export_csv(base_dir / "csv")

        if args.format in ["all", "parquet"]:
            logger.info("Exporting Parquet files...")
            exporter.export_parquet(base_dir / "parquet")

        if args.format in ["all", "sqlite"]:
            logger.info("Exporting SQLite database...")
            exporter.export_sqlite(base_dir / "synthetic_data.db")

        logger.info(f"\nData exported to {base_dir}")
        logger.info(f"Export completed at: {datetime.now()}")

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise
