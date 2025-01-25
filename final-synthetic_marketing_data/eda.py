"""
Enhanced Exploratory Data Analysis for Marketing Data
Uses ydata-profiling (formerly pandas-profiling) for automated analysis
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from data_loader import SyntheticDataLoader
from ydata_profiling import ProfileReport


class MarketingEDA:
    def __init__(self, test_mode: bool = False):
        self.loader = SyntheticDataLoader(test_mode=test_mode)
        self.output_dir = Path("eda_outputs")
        self.output_dir.mkdir(exist_ok=True)

    def generate_profile_reports(self):
        """Generate profile reports for each dataset"""
        datasets = self.loader.load_all()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"profile_reports_{timestamp}"
        report_dir.mkdir(exist_ok=True)

        logging.info(f"Generating profile reports in {report_dir}")

        # Group related datasets
        dataset_groups = {
            "demographics": ["demographics"],
            "consumer": ["consumer"],
            "marketing": ["campaigns", "engagements", "loyalties"],
            "transactions": ["transactions", "transaction_details", "product_catalog"],
        }

        for group_name, dataset_names in dataset_groups.items():
            logging.info(f"\nProcessing {group_name} datasets...")

            # Create a combined dataframe for related datasets if needed
            group_dfs = {}
            for name in dataset_names:
                if name in datasets:
                    df = datasets[name]
                    logging.info(f"Processing {name} with shape {df.shape}")

                    # Generate individual report with corrected parameters
                    report = ProfileReport(
                        df, title=f"{name.title()} Analysis", explorative=True
                    )
                    report.to_file(report_dir / f"{name}_report.html")
                    group_dfs[name] = df
                else:
                    logging.warning(f"Dataset {name} not found")

            # If we have multiple datasets in the group, try to create a combined report
            if len(group_dfs) > 1:
                try:
                    # Attempt to merge datasets based on common keys
                    merged_df = self._merge_related_datasets(group_dfs)
                    if merged_df is not None:
                        logging.info(f"Generating combined report for {group_name}")
                        combined_report = ProfileReport(
                            merged_df,
                            title=f"Combined {group_name.title()} Analysis",
                            explorative=True,
                        )
                        combined_report.to_file(
                            report_dir / f"{group_name}_combined_report.html"
                        )
                except Exception as e:
                    logging.error(
                        f"Error creating combined report for {group_name}: {str(e)}"
                    )

        logging.info(f"\nAll reports generated in {report_dir}")
        return report_dir

    def _merge_related_datasets(self, datasets: dict) -> pd.DataFrame:
        """Attempt to merge related datasets based on common keys"""
        if not datasets:
            return None

        # Common merge keys for different dataset types
        merge_keys = {
            "consumer_id": ["demographics", "consumer", "transactions"],
            "transaction_id": ["transactions", "transaction_details"],
            "campaign_id": ["campaigns", "engagements"],
        }

        # Start with the first dataset
        base_name = list(datasets.keys())[0]
        merged_df = datasets[base_name].copy()

        # Try to merge other datasets
        for name, df in list(datasets.items())[1:]:
            # Find common columns that could be merge keys
            common_cols = set(merged_df.columns) & set(df.columns)
            merge_key = None

            # Check if we have a predefined merge key
            for key, related_datasets in merge_keys.items():
                if (
                    key in common_cols
                    and base_name in related_datasets
                    and name in related_datasets
                ):
                    merge_key = key
                    break

            if merge_key:
                try:
                    merged_df = merged_df.merge(
                        df,
                        on=merge_key,
                        how="left",
                        suffixes=(f"_{base_name}", f"_{name}"),
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not merge {name} using {merge_key}: {str(e)}"
                    )

        return merged_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate marketing data profile reports"
    )
    parser.add_argument("--test", action="store_true", help="Use test data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    eda = MarketingEDA(test_mode=args.test)
    report_dir = eda.generate_profile_reports()
    print(f"\nReports have been generated in: {report_dir}")
