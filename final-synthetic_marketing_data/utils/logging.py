"""Logging configuration and utilities for validation."""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import sys


def setup_logging(
    module_name: str,
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    log_to_file: bool = False,
) -> logging.Logger:
    """Configure and return logger instance.

    Args:
        module_name: Name to identify the logger
        log_dir: Optional directory for log files
        log_level: Logging level
        log_to_file: Whether to save logs to file

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(module_name)

    # Only configure if no handlers exist
    if not logger.handlers:
        logger.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if requested
        if log_to_file and log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{module_name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def log_validation_step(logger: logging.Logger, step_name: str) -> None:
    """Log the start of a validation step.

    Args:
        logger: Logger instance
        step_name: Name of the validation step
    """
    separator = "=" * 50
    logger.info(f"\n{separator}")
    logger.info(f"Starting: {step_name}")
    logger.info(separator)


def log_validation_error(
    logger: logging.Logger,
    context: str,
    error: Exception,
    additional_info: Optional[Dict] = None,
) -> None:
    """Log validation errors with context.

    Args:
        logger: Logger instance
        context: Context where the error occurred
        error: The exception that was raised
        additional_info: Optional additional information
    """
    logger.error(f"Error in {context}: {str(error)}")
    if additional_info:
        logger.error("Additional information:")
        for key, value in additional_info.items():
            logger.error(f"  {key}: {value}")


def log_metric_validation(
    logger: logging.Logger,
    metric_name: str,
    expected: float,
    actual: float,
    difference: float,
    passed: bool,
    additional_info: Optional[Dict] = None,
) -> None:
    """Log individual metric validation results.

    Args:
        logger: Logger instance
        metric_name: Name of the metric being validated
        expected: Expected value
        actual: Actual value
        difference: Difference between expected and actual
        passed: Whether the validation passed
        additional_info: Optional additional information
    """
    result = "✅" if passed else "❌"

    if 0 <= expected <= 1 and 0 <= actual <= 1:
        # Format as percentages for values between 0 and 1
        logger.info(
            f"{metric_name:<30} Expected: {expected:>7.2%} "
            f"Actual: {actual:>7.2%} Diff: {difference:>7.2%} {result}"
        )
    else:
        # Format as decimals for other values
        logger.info(
            f"{metric_name:<30} Expected: {expected:>7.2f} "
            f"Actual: {actual:>7.2f} Diff: {difference:>7.2f} {result}"
        )

    if additional_info:
        for key, value in additional_info.items():
            logger.info(f"  {key}: {value}")


def log_validation_summary(
    logger: logging.Logger, results: Dict[str, Any], runtime: float
) -> None:
    """Log validation summary results.

    Args:
        logger: Logger instance
        results: Validation results dictionary
        runtime: Validation runtime in seconds
    """
    separator = "-" * 50
    logger.info(f"\n{separator}")
    logger.info("VALIDATION SUMMARY")
    logger.info(separator)

    metadata = results.get("metadata", {})

    # Overall metrics
    total_metrics = metadata.get("total_metrics_checked", 0)
    passing_metrics = metadata.get("passing_metrics", 0)
    success_rate = (passing_metrics / total_metrics * 100) if total_metrics > 0 else 0

    logger.info(f"Total Metrics Checked: {total_metrics}")
    logger.info(f"Passing Metrics: {passing_metrics}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    logger.info(f"Runtime: {runtime:.1f} seconds")

    # Category-specific results
    if "category_scores" in metadata:
        logger.info("\nCategory Results:")
        for category, score in metadata["category_scores"].items():
            result = "✅" if score >= 0.8 else "❌"
            logger.info(f"  {category:<20} {score:>7.1%} {result}")

    logger.info(separator)


def log_data_quality(logger: logging.Logger, quality_metrics: Dict[str, Any]) -> None:
    """Log data quality metrics.

    Args:
        logger: Logger instance
        quality_metrics: Dictionary of data quality metrics
    """
    logger.info("\nDATA QUALITY METRICS")
    logger.info("-" * 50)

    # Missing values
    if "missing_values" in quality_metrics:
        logger.info("\nMissing Values:")
        for col, rate in quality_metrics["missing_values"].items():
            logger.info(f"  {col:<30} {rate:>7.2%}")

    # Duplicates
    if "duplicates" in quality_metrics:
        logger.info(f"\nDuplicate Records: {quality_metrics['duplicates']}")

    # Outliers
    if "outliers" in quality_metrics:
        logger.info("\nOutliers:")
        for col, counts in quality_metrics["outliers"].items():
            logger.info(
                f"  {col:<30} Below: {counts['below']:>5} Above: {counts['above']:>5}"
            )

    # Value ranges
    if "value_ranges" in quality_metrics:
        logger.info("\nValue Ranges:")
        for col, stats in quality_metrics["value_ranges"].items():
            logger.info(
                f"  {col:<30} "
                f"Min: {stats['min']:>8.2f} "
                f"Max: {stats['max']:>8.2f} "
                f"Mean: {stats['mean']:>8.2f} "
                f"Std: {stats['std']:>8.2f}"
            )


def log_progress(
    logger: logging.Logger, current: int, total: int, message: str = "Progress"
) -> None:
    """Log progress of long-running operations.

    Args:
        logger: Logger instance
        current: Current progress count
        total: Total items to process
        message: Progress message prefix
    """
    progress = (current / total) * 100
    if current % max(1, total // 20) == 0:  # Log every 5%
        logger.info(f"{message}: {current}/{total} ({progress:.1f}%)")
