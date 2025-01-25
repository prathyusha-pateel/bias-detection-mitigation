"""Campaign data validator"""

from typing import Dict
import pandas as pd
import logging
from validation.marketing.base_validator import BaseValidator
from validation.marketing.marketing_benchmarks import (
    CAMPAIGN_DISTRIBUTIONS,
    VALIDATION_THRESHOLDS,
)
from utils.validation_display import display_validation_metrics


class CampaignValidator(BaseValidator):
    def __init__(self, campaigns: pd.DataFrame):
        """Initialize with only campaign data"""
        self.campaigns = campaigns
        self.logger = logging.getLogger(self.__class__.__name__)

        # Add validation of required columns
        required_columns = ["campaign_type", "campaign_id"]
        missing_columns = [
            col for col in required_columns if col not in campaigns.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. Available columns: {campaigns.columns.tolist()}"
            )

    def validate(self) -> Dict:
        """Validate campaign data structure and patterns"""
        self.logger.info("\nValidating Campaign Data")

        results = {
            "campaign_metrics": {
                "type_distribution": {
                    "details": self._validate_distribution(
                        self.campaigns["campaign_type"].value_counts(normalize=True),
                        CAMPAIGN_DISTRIBUTIONS.get("campaign_types"),
                    )
                }
            }
        }

        # Only validate channels if the column exists
        if "primary_channel" in self.campaigns.columns:
            results["channel_metrics"] = {
                "channel_distribution": {
                    "details": self._validate_distribution(
                        self.campaigns["primary_channel"].value_counts(normalize=True),
                        CAMPAIGN_DISTRIBUTIONS.get("channels"),
                    )
                }
            }

        # Only validate creative types if the column exists
        if "creative_type" in self.campaigns.columns:
            results["creative_metrics"] = {
                "creative_distribution": {
                    "details": self._validate_distribution(
                        self.campaigns["creative_type"].value_counts(normalize=True),
                        CAMPAIGN_DISTRIBUTIONS.get("creative_types"),
                    )
                }
            }

        # Add metadata
        total_metrics = sum(len(section["details"]) for section in results.values())
        passing_metrics = sum(
            sum(
                1
                for detail in section["details"].values()
                if detail["within_tolerance"]
            )
            for section in results.values()
        )

        results["metadata"] = {
            "total_metrics_checked": total_metrics,
            "passing_metrics": passing_metrics,
            "overall_score": (
                (passing_metrics / total_metrics * 100) if total_metrics > 0 else 0
            ),
        }

        # Use standardized display
        display_validation_metrics(results, self.logger, "Campaign Validation Results")

        return results
