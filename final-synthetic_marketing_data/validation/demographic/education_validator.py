"""Education distribution validator"""

import pandas as pd
from typing import Dict, Any
from validation.demographic.base_validator import BaseValidator
from validation.demographic.demographics_benchmarks import (
    EDUCATION_DISTRIBUTION,
    VALIDATION_THRESHOLDS,
)


class EducationValidator(BaseValidator):
    """Validator for education-related demographic metrics."""

    def _perform_validation(self) -> Dict[str, Any]:
        results = {
            "education_distribution": self._validate_education_distribution(),
        }
        return results

    def _validate_education_distribution(self) -> Dict[str, Any]:
        """Validate education level distribution."""
        if "SCHL" not in self.df.columns:
            return {"error": "Missing SCHL (education) column"}

        # Map PUMS education codes to our benchmark categories
        education_mapping = {
            # No High School
            1: "Less than high school",  # No schooling
            2: "Less than high school",  # Nursery/Preschool
            3: "Less than high school",  # Kindergarten
            4: "Less than high school",  # Grade 1
            5: "Less than high school",  # Grade 2
            6: "Less than high school",  # Grade 3
            7: "Less than high school",  # Grade 4
            8: "Less than high school",  # Grade 5
            9: "Less than high school",  # Grade 6
            10: "Less than high school",  # Grade 7
            11: "Less than high school",  # Grade 8
            12: "Less than high school",  # Grade 9
            13: "Less than high school",  # Grade 10
            14: "Less than high school",  # Grade 11
            15: "Less than high school",  # 12th grade - no diploma
            # High School
            16: "High school",  # Regular high school diploma
            17: "High school",  # GED or alternative credential
            # Some College
            18: "Some college",  # Some college, < 1 year
            19: "Some college",  # 1+ years college, no degree
            # Associates
            20: "Associates",  # Associate's degree
            # Bachelor's
            21: "Bachelors",  # Bachelor's degree
            # Graduate
            22: "Graduate",  # Master's degree
            23: "Graduate",  # Professional degree
            24: "Graduate",  # Doctorate degree
        }

        # Add age filter to exclude young people from education stats
        working_age_df = self.df[self.df["AGEP"] >= 25]  # Only include 25+ years old

        # Calculate distribution using person weights
        education_dist = (
            working_age_df.assign(
                edu_level=working_age_df["SCHL"].map(education_mapping)
            )
            .groupby("edu_level")["PWGTP"]
            .sum()
            .div(working_age_df["PWGTP"].sum())
            .reindex(EDUCATION_DISTRIBUTION["attainment_levels"].keys())
            .fillna(0)
            .to_dict()
        )

        # Add sample size info
        validation_results = self._validate_distribution(
            pd.Series(education_dist),
            EDUCATION_DISTRIBUTION["attainment_levels"],
            tolerance=VALIDATION_THRESHOLDS["education"]["tolerance"],
        )

        # Add additional context
        validation_results["metadata"] = {
            "sample_size": len(working_age_df),
            "working_age_only": True,
            "min_age": 25,
            "weighted_calculation": "PWGTP" in self.df.columns,
        }

        # Add summary stats
        validation_results["summary"] = {
            "higher_education_rate": sum(
                education_dist[level] for level in ["Bachelors", "Graduate"]
            ),
            "high_school_or_higher": sum(
                education_dist[level]
                for level in [
                    "High school",
                    "Some college",
                    "Associates",
                    "Bachelors",
                    "Graduate",
                ]
            ),
        }

        return validation_results
