"""
Demographics validation benchmarks and thresholds.

Contains benchmark values and validation thresholds for demographic data validation,
sourced from census data and demographic research.

Sources:
1. US Census Bureau Current Population Survey (CPS) 2023
   https://www.census.gov/data/tables/2023/demo/age-and-sex/2023-age-sex-composition.html

2. US Census Bureau American Community Survey (ACS) 2023
   https://www.census.gov/programs-surveys/acs/data.html

3. US Census Bureau Income and Poverty Report 2023
   https://www.census.gov/library/publications/2023/demo/p60-276.html

4. US Census Bureau Educational Attainment 2023
   https://www.census.gov/data/tables/2023/demo/educational-attainment/cps-detailed-tables.html

5. US Census Bureau Geographic Mobility Report 2023
   https://www.census.gov/data/tables/2023/demo/geographic-mobility/cps-2023.html
"""

# Age distribution benchmarks from Census Bureau 2023 (Source #1)
AGE_DISTRIBUTION = {
    "age_groups": {
        "0-18": 0.22,
        "19-25": 0.09,
        "26-35": 0.14,
        "36-50": 0.19,
        "51-65": 0.20,
        "65+": 0.16,
    },
    "generations": {
        "Gen Z": 0.20,  # 0-25 years (born 1997-2012)
        "Millennials": 0.25,  # 26-41 years (born 1981-1996)
        "Gen X": 0.20,  # 42-57 years (born 1965-1980)
        "Boomers": 0.23,  # 58-76 years (born 1946-1964)
        "Silent": 0.12,  # 77+ years (born before 1946)
    },
}

# Income distribution benchmarks from Census Bureau 2023 (Source #3)
INCOME_DISTRIBUTION = {
    "income_brackets": {
        "<25k": 0.20,  # Bottom quintile
        "25k-50k": 0.20,  # Second quintile
        "50k-75k": 0.20,  # Middle quintile
        "75k-100k": 0.15,  # Fourth quintile part 1
        "100k-150k": 0.15,  # Fourth quintile part 2
        ">150k": 0.10,  # Top quintile
    },
    "gini_coefficient": 0.485,  # US Census Bureau 2023
    "correlations": {
        "income_education": 0.42,  # Correlation between income and education level
        "income_age": 0.35,  # Correlation between income and age
    },
}

# Education attainment from Census Bureau 2023 (Source #4)
EDUCATION_DISTRIBUTION = {
    "attainment_levels": {
        "Less than high school": 0.10,  # No high school diploma
        "High school": 0.28,  # High school or equivalent
        "Some college": 0.15,  # Some college, no degree
        "Associates": 0.10,  # Associate's degree
        "Bachelors": 0.25,  # Bachelor's degree
        "Graduate": 0.12,  # Graduate or professional degree
    },
    "higher_education_by_age": {  # Bachelor's degree or higher by age group
        "25-34": 0.41,
        "35-44": 0.39,
        "45-64": 0.35,
        "65+": 0.30,
    },
}

# Location/Geographic distribution from Census Bureau 2023 (Source #5)
LOCATION_DISTRIBUTION = {
    "urban_rural": {
        1: 0.83,  # Metropolitan area
        2: 0.10,  # Micropolitan area
        3: 0.07,  # Neither
    },
    "regions": {"Northeast": 0.17, "Midwest": 0.21, "South": 0.38, "West": 0.24},
    "density_distribution": {
        "Rural": 0.15,  # < 100 people per square mile
        "Suburban-Low": 0.25,  # 100-1,000 people per square mile
        "Suburban-High": 0.30,  # 1,000-5,000 people per square mile
        "Urban": 0.20,  # 5,000-10,000 people per square mile
        "Metro-Core": 0.10,  # > 10,000 people per square mile
    },
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    "distribution": {
        "tolerance": 0.05,  # 5% tolerance for distribution comparisons
        "min_sample_size": 1000,
    },
    "correlation": {
        "tolerance": 0.10,  # Correlation coefficient tolerance
        "min_sample_size": 100,
    },
    "gini": {
        "tolerance": 0.05,  # Gini coefficient tolerance
        "min_sample_size": 1000,
    },
    "education": {
        "tolerance": 0.05,  # Education rate tolerance
        "min_sample_size": 500,
    },
    "regional": {
        "tolerance": 0.05,  # Regional distribution tolerance
        "min_sample_size": 1000,
    },
    "urban_rural": {
        "tolerance": 0.05,  # Urban/rural distribution tolerance
        "min_sample_size": 1000,
    },
    "population_density": {
        "tolerance": 0.05,  # Population density distribution tolerance
        "min_sample_size": 1000,
    },
}

# Metadata about the benchmarks
BENCHMARK_METADATA = {
    "version": "1.0",
    "last_updated": "2024-01",
    "sources": [
        "US Census Bureau ACS 2023",
        "Bureau of Labor Statistics 2023",
        "Pew Research Center 2023",
        "Federal Reserve Economic Data 2023",
    ],
    "notes": [
        "All distributions are based on most recent available data",
        "Correlations are derived from multiple studies",
        "Regional distributions account for recent population shifts",
        "Education benchmarks reflect post-pandemic trends",
    ],
}

# Education code mappings for PUMS data
EDUCATION_CODES = {
    "mapping": {
        "Less than high school": list(range(1, 16)),  # Codes 1-15
        "High school": [16, 17],  # Regular HS diploma and GED
        "Some college": [18, 19],  # Some college, no degree
        "Associates": [20],  # Associate's degree
        "Bachelors": [21],  # Bachelor's degree
        "Graduate": [22, 23, 24],  # Master's, Professional, Doctorate
    },
    "higher_education_threshold": 21,  # Bachelor's degree or higher
}

# Major US Cities Data (Source: Census Bureau 2023)
MAJOR_CITIES = {
    "New York City": {
        "state": "36",  # NY
        "puma_codes": [
            "03701",
            "03702",
            "03703",
            "03704",
            "03705",  # Manhattan
            "03801",
            "03802",
            "03803",
            "03804",  # Bronx
            "03901",
            "03902",
            "03903",
            "03904",  # Brooklyn
            "04001",
            "04002",
            "04003",
            "04004",  # Queens
            "04101",
            "04102",  # Staten Island
        ],
        "population_share": 0.0255,  # 2.55% of US population
        "demographics": {
            "median_income": 70663,
            "median_age": 36.9,
            "education_bachelors_or_higher": 0.37,
        },
    },
    "Los Angeles": {
        "state": "06",  # CA
        "puma_codes": [
            "04701",
            "04702",
            "04703",
            "04704",
            "04705",
            "04706",
            "04707",
            "04708",
            "04709",
            "04710",
        ],
        "population_share": 0.0121,  # 1.21% of US population
        "demographics": {
            "median_income": 65290,
            "median_age": 35.9,
            "education_bachelors_or_higher": 0.33,
        },
    },
    "Chicago": {
        "state": "17",  # IL
        "puma_codes": [
            "03201",
            "03202",
            "03203",
            "03204",
            "03205",
            "03206",
            "03207",
            "03208",
            "03209",
            "03210",
        ],
        "population_share": 0.0086,  # 0.86% of US population
        "demographics": {
            "median_income": 62097,
            "median_age": 35.2,
            "education_bachelors_or_higher": 0.39,
        },
    },
    "Houston": {
        "state": "48",  # TX
        "puma_codes": [
            "02301",
            "02302",
            "02303",
            "02304",
            "02305",
            "02306",
            "02307",
            "02308",
        ],
        "population_share": 0.0073,  # 0.73% of US population
        "demographics": {
            "median_income": 53600,
            "median_age": 33.4,
            "education_bachelors_or_higher": 0.32,
        },
    },
    "Phoenix": {
        "state": "04",  # AZ
        "puma_codes": ["01001", "01002", "01003", "01004", "01005", "01006", "01007"],
        "population_share": 0.0051,  # 0.51% of US population
        "demographics": {
            "median_income": 60931,
            "median_age": 34.2,
            "education_bachelors_or_higher": 0.29,
        },
    },
}

# Major Cities Validation Thresholds
MAJOR_CITIES_THRESHOLDS = {
    "population_share": {
        "tolerance": 0.002,  # 0.2 percentage point tolerance for population share
        "min_sample_size": 1000,
    },
    "demographics": {
        "median_income": {
            "tolerance": 0.10,  # 10% tolerance for median income
            "min_sample_size": 500,
        },
        "median_age": {
            "tolerance": 0.05,  # 5% tolerance for median age
            "min_sample_size": 500,
        },
        "education": {
            "tolerance": 0.05,  # 5% tolerance for education rates
            "min_sample_size": 500,
        },
    },
}

# Update BENCHMARK_METADATA with new source
BENCHMARK_METADATA["sources"].append(
    "US Census Bureau Population Estimates for Incorporated Places: 2023"
)
BENCHMARK_METADATA["notes"].append(
    "Major cities data includes PUMA-level geographic identifiers"
)
