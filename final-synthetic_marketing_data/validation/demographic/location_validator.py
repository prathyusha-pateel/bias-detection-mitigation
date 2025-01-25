"""Location distribution validator"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from validation.demographic.base_validator import BaseValidator
from validation.demographic.demographics_benchmarks import (
    LOCATION_DISTRIBUTION,
    VALIDATION_THRESHOLDS,
)


class LocationValidator(BaseValidator):
    """Validator for location-related demographic metrics."""

    def _perform_validation(self) -> Dict[str, Any]:
        results = {
            "regional_distribution": self._validate_regional_distribution(),
        }
        return results

    def _validate_regional_distribution(self) -> Dict[str, Any]:
        """Validate regional distribution."""
        if "STATE" not in self.df.columns:
            return {"error": "Missing STATE (state) column"}

        # Ensure state codes are properly formatted
        self.df["STATE"] = self.df["STATE"].astype(str).str.zfill(2)

        # Create region mapping
        region_mapping = {}
        for region, states in {
            "Northeast": ["09", "23", "25", "33", "34", "36", "42", "44", "50"],
            "Midwest": [
                "17",
                "18",
                "19",
                "20",
                "26",
                "27",
                "29",
                "31",
                "38",
                "39",
                "46",
                "55",
            ],
            "South": [
                "01",
                "05",
                "10",
                "11",
                "12",
                "13",
                "21",
                "22",
                "24",
                "28",
                "37",
                "40",
                "45",
                "47",
                "48",
                "51",
                "54",
            ],
            "West": [
                "02",
                "04",
                "06",
                "08",
                "15",
                "16",
                "30",
                "32",
                "35",
                "41",
                "49",
                "53",
                "56",
            ],
        }.items():
            for state in states:
                region_mapping[state] = region

        self.df["region"] = self.df["STATE"].map(region_mapping)

        # Use weighted distribution
        regional_dist = (
            self.df.groupby("region")["PWGTP"]
            .sum()
            .div(self.df["PWGTP"].sum())
            .to_dict()
        )

        # If testing single state, only validate that region's proportion
        unique_states = self.df["STATE"].unique()
        if len(unique_states) == 1:
            state = unique_states[0]
            region = region_mapping.get(state)
            if region:
                expected_dist = {region: LOCATION_DISTRIBUTION["regions"][region]}
                return self._validate_distribution(
                    pd.Series(regional_dist),
                    expected_dist,
                    tolerance=VALIDATION_THRESHOLDS["regional"]["tolerance"],
                )
            else:
                return {"error": f"Unknown region for state code {state}"}

        # Otherwise validate full distribution
        return self._validate_distribution(
            pd.Series(regional_dist),
            LOCATION_DISTRIBUTION["regions"],
            tolerance=VALIDATION_THRESHOLDS["regional"]["tolerance"],
        )
