"""Base validator for transaction data validation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging
from pathlib import Path


class BaseTransactionValidator(ABC):
    """Base validator for transaction data validation."""

    def __init__(
        self,
        transaction_data: pd.DataFrame,
        detail_data: Optional[pd.DataFrame] = None,
        product_data: Optional[pd.DataFrame] = None,
    ):
        """Initialize with transaction data and optional related data."""
        self.transaction_data = transaction_data
        self.detail_data = detail_data
        self.product_data = product_data
        self.logger = logging.getLogger(__name__)
        self.results_dir = Path(__file__).parent / "validation_results"
        self.results_dir.mkdir(exist_ok=True)

    @abstractmethod
    def _perform_validation(self) -> Dict[str, Any]:
        """Perform specific validation checks."""
        pass

    def validate(self) -> Dict[str, Any]:
        """Run validation and return results."""
        try:
            results = self._perform_validation()
            self._log_validation_results(results)
            return results
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise

    def _log_validation_results(self, results: Dict[str, Any]) -> None:
        """Log validation results."""
        self.logger.info("\nValidation Results:")
        for category, metrics in results.items():
            self.logger.info(f"\n{category}:")
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric}: {value}")
            else:
                self.logger.info(f"  {metrics}")
