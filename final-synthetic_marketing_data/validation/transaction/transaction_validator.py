"""Main Transaction Data Validator"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json

from validation.transaction.value_validator import TransactionValueValidator
from validation.transaction.channel_validator import ChannelValidator
from validation.transaction.product_validator import ProductValidator
from validation.transaction.transaction_benchmarks import VALIDATION_THRESHOLDS
from utils.encoders import NumpyEncoder
from utils.logging import log_validation_step


class TransactionValidator:
    """Master validator for transaction data."""

    def __init__(
        self,
        transaction_data: pd.DataFrame,
        detail_data: Optional[pd.DataFrame] = None,
        product_data: Optional[pd.DataFrame] = None,
    ):
        """Initialize with transaction and optional related data."""
        self.validators = {
            "value": TransactionValueValidator(
                transaction_data, detail_data, product_data
            ),
            "channel": ChannelValidator(transaction_data, detail_data, product_data),
            "product": ProductValidator(transaction_data, detail_data, product_data),
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_validation(self) -> Dict:
        """Run all validations and return combined results."""
        self.logger.info("\nStarting Comprehensive Transaction Data Validation")
        self.logger.info("=" * 80)

        try:
            results = {
                "value": self.validators["value"].validate(),
                "channel": self.validators["channel"].validate(),
                "product": self.validators["product"].validate(),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "validator_version": "1.0",
                },
            }

            # Calculate and add summary metrics
            summary = self._calculate_summary_metrics(results)
            results["metadata"].update(summary)

            # Log overall summary
            self._log_validation_summary(summary)

            return results

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise

    def _calculate_summary_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive summary metrics."""
        total_checks = 0
        passing_checks = 0
        category_scores = {}

        for category, category_results in results.items():
            if category == "metadata":
                continue

            category_total = 0
            category_passing = 0

            for metric_group in category_results.values():
                if isinstance(metric_group, dict):
                    for metric in metric_group.values():
                        if isinstance(metric, dict) and "within_tolerance" in metric:
                            category_total += 1
                            if metric["within_tolerance"]:
                                category_passing += 1

            if category_total > 0:
                category_scores[category] = category_passing / category_total
                total_checks += category_total
                passing_checks += category_passing

        return {
            "total_metrics_checked": total_checks,
            "passing_metrics": passing_checks,
            "overall_score": passing_checks / total_checks if total_checks > 0 else 0.0,
            "category_scores": category_scores,
        }

    def _log_validation_summary(self, summary: Dict) -> None:
        """Log validation summary with clear formatting."""
        self.logger.info("\nVALIDATION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Total Metrics Checked: {summary['total_metrics_checked']}")
        self.logger.info(f"Passing Metrics: {summary['passing_metrics']}")
        self.logger.info(f"Overall Score: {summary['overall_score']:.1%}")

        if summary["category_scores"]:
            self.logger.info("\nCategory Scores:")
            for category, score in summary["category_scores"].items():
                self.logger.info(f"  {category}: {score:.1%}")

    def save_results(self, output_path: Path) -> None:
        """Save validation results to JSON file."""
        results = self.run_validation()

        with open(output_path, "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)

        self.logger.info(f"\nDetailed validation results saved to: {output_path}")


def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Set up directories
        script_dir = Path(__file__).resolve().parent
        data_dir = script_dir.parent.parent / "data"
        output_dir = data_dir / "validation_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find transaction data files
        transaction_files = list(data_dir.glob("transactions_*.csv"))
        detail_files = list(data_dir.glob("transaction_details_*.csv"))
        product_files = list(data_dir.glob("product_catalog_*.csv"))

        if not transaction_files:
            raise FileNotFoundError("No transaction data files found")

        # Process each transaction file
        for file_path in transaction_files:
            logger.info(f"Validating {file_path.name}")

            # Load data
            transactions = pd.read_csv(file_path)

            # Find corresponding detail and product files
            timestamp = file_path.stem.split("_")[-1]
            detail_file = next((f for f in detail_files if timestamp in f.stem), None)
            product_file = next((f for f in product_files if timestamp in f.stem), None)

            details = pd.read_csv(detail_file) if detail_file else None
            products = pd.read_csv(product_file) if product_file else None

            # Create validator and run validation
            validator = TransactionValidator(transactions, details, products)
            results = validator.run_validation()

            # Save results
            output_path = output_dir / f"{file_path.stem}_validation.json"
            with open(output_path, "w") as f:
                json.dump(results, f, cls=NumpyEncoder, indent=2)

            logger.info(f"Validation score: {results['metadata']['overall_score']:.2%}")

        logger.info("Transaction data validation completed successfully")

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
