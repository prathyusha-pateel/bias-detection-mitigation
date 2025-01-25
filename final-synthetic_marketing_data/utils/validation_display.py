"""Utility functions for standardized validation output display."""

from typing import Dict, Any
import logging
from itertools import zip_longest


def display_validation_metrics(
    results: Dict[str, Any], logger: logging.Logger, title: str = "Validation Results"
) -> None:
    """Display validation metrics in a standardized format."""

    # Define fixed column widths
    metric_width = 35
    value_width = 15
    status_width = 10
    total_width = 80
    distribution_width = 25  # Width for each column in distribution display

    def print_header(text: str) -> None:
        """Print a section header."""
        logger.info("\n" + "=" * total_width)
        logger.info(text)
        logger.info("=" * total_width + "\n")

    def print_columns() -> None:
        """Print column headers."""
        logger.info(
            f"{'Metric':<{metric_width}} "
            f"{'Expected':<{value_width}} "
            f"{'Actual':<{value_width}} "
            f"{'Status':<{status_width}}"
        )
        logger.info("-" * total_width)

    def format_distribution(dist_dict: Dict, other_dict: Dict = None) -> None:
        """Format and display distribution in columns."""
        # Sort items by value
        items = sorted(dist_dict.items(), key=lambda x: float(x[1]), reverse=True)

        # If distribution has more than 10 items, show only top 10 with summary
        if len(items) > 10:
            logger.info("Top 10 categories (sorted by frequency):")
            items = items[:10]
            logger.info(f"{'Category':<15} {'Percentage':>10}")
            logger.info("-" * 25)
            for k, v in items:
                logger.info(f"{str(k):<15} {float(v)*100:>9.2f}%")
            logger.info(f"\n... and {len(dist_dict) - 10} more categories")
        else:
            # For smaller distributions, show all items
            logger.info(f"{'Category':<15} {'Percentage':>10}")
            logger.info("-" * 25)
            for k, v in items:
                logger.info(f"{str(k):<15} {float(v)*100:>9.2f}%")

    def display_metrics(metrics: Dict, prefix: str = "") -> None:
        """Display metrics recursively."""
        for key, value in metrics.items():
            if key == "details":
                for metric_name, metric_data in value.items():
                    expected = metric_data.get("expected", "None")
                    actual = metric_data.get("actual", "None")
                    status = (
                        "✅" if metric_data.get("within_tolerance", False) else "❌"
                    )

                    # Handle distribution dictionaries
                    if isinstance(expected, dict):
                        logger.info(f"\n{metric_name} Distribution:")
                        logger.info("-" * total_width)
                        logger.info("Expected:")
                        format_distribution(expected)
                        logger.info("\nActual:")
                        format_distribution(actual)
                        logger.info(f"\nStatus: {status}")
                        logger.info("-" * total_width)
                    else:
                        # Regular metrics display
                        logger.info(
                            f"{metric_name:<{metric_width}} "
                            f"{format_value(expected):<{value_width}} "
                            f"{format_value(actual):<{value_width}} "
                            f"{status:<{status_width}}"
                        )
            elif isinstance(value, dict):
                if key not in ["metadata", "details"]:
                    logger.info(f"\n{key.upper()} Metrics:")
                    logger.info("-" * total_width)
                    display_metrics(value, prefix + "  ")

    def format_value(value: Any) -> str:
        """Format values consistently."""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    # Display main header
    print_header(title)

    # Process each section
    for section_name, section_data in results.items():
        if section_name != "metadata":
            logger.info(f"\n{section_name.upper()} Validation")
            logger.info("-" * total_width)
            print_columns()
            display_metrics(section_data)

    # Display summary if available
    if "metadata" in results:
        metadata = results["metadata"]
        logger.info("\nSUMMARY")
        logger.info("-" * total_width)

        score = metadata.get("overall_score", 0)
        status = "✅" if score >= 0.8 else "❌"

        logger.info(f"Overall Score:          {score*100:>6.2f}% {status}")
        logger.info(
            f"Total Metrics Checked:  {metadata.get('total_metrics_checked', 0)}"
        )
        logger.info(f"Passing Metrics:        {metadata.get('passing_metrics', 0)}")
