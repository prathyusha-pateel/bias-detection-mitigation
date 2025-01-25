#!/usr/bin/env python3
"""
PUMA to ZIP Code Mapper
Maps Public Use Microdata Areas (PUMAs) to ZIP codes using 2020 Census Tract relationship files.
"""

import os
import pandas as pd
import requests
import logging
from typing import List, Optional
from pathlib import Path


class PUMAZipMapper:
    """Maps PUMA codes to ZIP codes using Census Bureau relationship files"""

    def __init__(self, year: str = "2020") -> None:
        """
        Initialize the PUMA to ZIP code mapper.

        Args:
            year (str): Census year for geographic relationship files
        """
        self.year = year
        self.base_dir = Path(
            os.getcwd()
        )  # Save directly in the current working directory
        self.data_dir = self.base_dir

        # Alternative access URLs for relationship files
        self.tract_url = "https://usa.ipums.org/usa/volii/pumas20.shtml"  # IPUMS USA link for manual download
        self.zip_tract_url = "https://mcdc.missouri.edu/geography/PUMAs.html"  # Missouri Census Data Center for manual options

        # Initialize data frames
        self.tract_puma_df: Optional[pd.DataFrame] = None
        self.zip_tract_df: Optional[pd.DataFrame] = None
        self.puma_zip_mapping: Optional[pd.DataFrame] = None

    def load_relationship_files(self) -> None:
        """
        Load Census geographic relationship files. If files are not available, instruct on manual download options.
        """
        try:
            # Define file paths directly in the working directory
            tract_file = (
                self.data_dir / "2020_Census_Tract_to_2020_PUMA.csv"
            )  # Adjust to expected name post-manual download
            zip_tract_file = (
                self.data_dir / "2020_ZCTA_to_2020_Census_Tract.csv"
            )  # Adjust to expected name post-manual download

            # Check if manual download is required for the tract-to-PUMA file
            if not tract_file.exists():
                logging.warning(
                    f"{tract_file} not found. Please download the 2020 Tract to PUMA file manually from IPUMS USA ({self.tract_url}) or Missouri Census Data Center ({self.zip_tract_url}) and place it in the script directory."
                )
            else:
                self.tract_puma_df = pd.read_csv(
                    tract_file,
                    dtype={
                        "STATEFP10": str,
                        "COUNTYFP10": str,
                        "TRACTCE10": str,
                        "PUMA5CE20": str,
                    },
                )

            # Check if manual download is required for the ZIP-to-tract file
            if not zip_tract_file.exists():
                logging.warning(
                    f"{zip_tract_file} not found. Please download the 2020 ZIP to Tract file manually from Missouri Census Data Center ({self.zip_tract_url}) and place it in the script directory."
                )
            else:
                self.zip_tract_df = pd.read_csv(
                    zip_tract_file,
                    dtype={
                        "ZCTA5CE20": str,
                        "TRACTCE20": str,
                        "STATEFP20": str,
                        "COUNTYFP20": str,
                    },
                )

            if self.tract_puma_df is not None and self.zip_tract_df is not None:
                self._create_puma_zip_mapping()
                logging.info("Loaded all relationship files successfully")
            else:
                logging.warning(
                    "Could not load relationship files. Using simplified mapping."
                )
                self._create_simplified_mapping()

        except Exception as e:
            logging.error(f"Error loading relationship files: {str(e)}")
            self._create_simplified_mapping()

    def _create_simplified_mapping(self) -> None:
        """
        Create a simplified PUMA-ZIP mapping when relationship files are unavailable.
        """
        logging.info("Creating simplified PUMA-ZIP mapping")
        self.puma_zip_mapping = pd.DataFrame(
            {"state_code": [], "puma_code": [], "zip_codes": []}
        )

    def _create_puma_zip_mapping(self) -> None:
        """
        Create mapping between PUMAs and ZIP codes.
        """
        try:
            if self.tract_puma_df is None or self.zip_tract_df is None:
                raise ValueError("Relationship files not loaded")

            # Rename columns to match between datasets
            self.tract_puma_df = self.tract_puma_df.rename(
                columns={
                    "STATEFP10": "STATEFP",
                    "COUNTYFP10": "COUNTYFP",
                    "TRACTCE10": "TRACTCE",
                    "PUMA5CE20": "PUMA5CE",
                }
            )
            self.zip_tract_df = self.zip_tract_df.rename(
                columns={
                    "STATEFP20": "STATEFP",
                    "COUNTYFP20": "COUNTYFP",
                    "TRACTCE20": "TRACTCE",
                    "ZCTA5CE20": "ZCTA5",
                }
            )

            # Merge tract-PUMA and tract-ZIP relationships
            self.puma_zip_mapping = pd.merge(
                self.tract_puma_df,
                self.zip_tract_df,
                on=["STATEFP", "COUNTYFP", "TRACTCE"],
            )

            self.puma_zip_mapping = (
                self.puma_zip_mapping.groupby(["STATEFP", "PUMA5CE"])["ZCTA5"]
                .agg(list)
                .reset_index()
                .rename(
                    columns={
                        "STATEFP": "state_code",
                        "PUMA5CE": "puma_code",
                        "ZCTA5": "zip_codes",
                    }
                )
            )

            logging.info("Created PUMA to ZIP mapping successfully")

        except Exception as e:
            logging.error(f"Error creating PUMA-ZIP mapping: {str(e)}")
            self._create_simplified_mapping()

    def get_zip_codes(self, state_code: str, puma_code: str) -> List[str]:
        """
        Get list of ZIP codes for a given PUMA.
        """
        try:
            if self.puma_zip_mapping is None:
                self.load_relationship_files()

            mask = (self.puma_zip_mapping["state_code"] == state_code) & (
                self.puma_zip_mapping["puma_code"] == str(puma_code).zfill(5)
            )
            result = self.puma_zip_mapping[mask]

            if result.empty:
                state_prefix = f"{int(state_code):02d}"
                puma_prefix = f"{int(puma_code):05d}"
                return [f"{state_prefix}{puma_prefix[:3]}xx"]

            return result.iloc[0]["zip_codes"]

        except Exception as e:
            logging.error(f"Error getting ZIP codes: {str(e)}")
            state_prefix = f"{int(state_code):02d}"
            puma_prefix = f"{int(puma_code):05d}"
            return [f"{state_prefix}{puma_prefix[:3]}xx"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mapper = PUMAZipMapper()
    mapper.load_relationship_files()
    print("ZIP Codes for PUMA 01234 in state 06:", mapper.get_zip_codes("06", "01234"))
