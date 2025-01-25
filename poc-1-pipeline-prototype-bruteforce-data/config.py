from enum import Enum
from typing import Dict, List, Tuple, Union
import math
from dataclasses import dataclass


# [Exception classes remain unchanged]
class DataProcessingError(Exception):
    pass


class ValidationError(DataProcessingError):
    pass


class ConfigurationError(DataProcessingError):
    pass


# [Enum classes remain unchanged]
class EmploymentStatus(Enum):
    EMPLOYED = "Employed"
    UNEMPLOYED = "Unemployed"
    NOT_IN_LABOR_FORCE = "Not in labor force"


class IncomeLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class EducationLevel(Enum):
    LESS_THAN_HIGH_SCHOOL = "Less than high school"
    HIGH_SCHOOL = "High School"
    SOME_COLLEGE = "Some College/Associate"
    BACHELOR = "Bachelor"
    ADVANCED = "Advanced"


class Race(Enum):
    WHITE = "White"
    BLACK = "Black or African American"
    ASIAN = "Asian"
    OTHER = "Other"


class Ethnicity(Enum):
    HISPANIC = "Hispanic or Latino"
    NOT_HISPANIC = "Not Hispanic or Latino"


class Gender(Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"


class BuyingFrequency(Enum):
    WEEKLY = "Weekly"
    BI_WEEKLY = "Bi-Weekly"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"


AGE_DISTRIBUTION: Dict[str, float] = {
    "16-19": 0.055,  # ~5.5% matches labor force participation
    "20-24": 0.085,  # ~8.5% matches workforce demographics
    "25-34": 0.210,  # Increased to 21% per labor force data
    "35-44": 0.205,  # Increased to 20.5% based on employment stats
    "45-54": 0.185,  # Increased to 18.5% to match workforce share
    "55-64": 0.155,  # Slightly increased to 15.5% per data
    "65+": 0.105,  # Reduced to 10.5% to match labor force participation
}


GENDER_DISTRIBUTION = {
    "Male": 0.484,  # Adjust down from 0.464
    "Female": 0.507,  # Adjust down from 0.534
    "Other": 0.009,  # Consistent with population estimates
}

RACE_DISTRIBUTION: Dict[str, float] = {
    "White": 0.765,
    "Black or African American": 0.130,
    "Asian": 0.066,
    "Other": 0.039,
}

HISPANIC_DISTRIBUTION: Dict[str, float] = {
    "Hispanic or Latino": 0.179,
    "Not Hispanic or Latino": 0.821,
}

# Regional Distribution
REGIONS: Dict[str, List[str]] = {
    "Northeast": ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "PA", "NJ"],
    "Midwest": ["OH", "MI", "IN", "WI", "IL", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    "South": [
        "DE",
        "MD",
        "DC",
        "VA",
        "WV",
        "NC",
        "SC",
        "GA",
        "FL",
        "KY",
        "TN",
        "MS",
        "AL",
        "LA",
        "AR",
        "TX",
        "OK",
    ],
    "West": [
        "MT",
        "ID",
        "WY",
        "CO",
        "NM",
        "AZ",
        "UT",
        "NV",
        "CA",
        "OR",
        "WA",
        "AK",
        "HI",
    ],
}

REGION_WEIGHTS: Dict[str, float] = {
    "Northeast": 0.17,
    "Midwest": 0.21,
    "South": 0.38,
    "West": 0.24,
}

EDUCATION_DISTRIBUTION = {
    "Less than high school": 0.100,
    "High School": 0.270,
    "Some College/Associate": 0.290,
    "Bachelor": 0.220,  # Increase
    "Advanced": 0.120,  # Increase
}

EDUCATION_BY_RACE: Dict[str, Dict[str, float]] = {
    "White": {
        "Less than high school": 0.093,
        "High School": 0.258,
        "Some College/Associate": 0.306,
        "Bachelor": 0.233,
        "Advanced": 0.110,
    },
    "Black or African American": {
        "Less than high school": 0.138,
        "High School": 0.228,
        "Some College/Associate": 0.333,
        "Bachelor": 0.226,  # Increased from 0.187
        "Advanced": 0.075,
    },
    "Asian": {
        "Less than high school": 0.065,
        "High School": 0.210,
        "Some College/Associate": 0.274,
        "Bachelor": 0.320,
        "Advanced": 0.131,  # Increased from 0.120 to make sum 1.0
    },
    "Hispanic or Latino": {
        "Less than high school": 0.185,
        "High School": 0.295,
        "Some College/Associate": 0.265,
        "Bachelor": 0.190,
        "Advanced": 0.065,
    },
}

# Employment Distributions
EMPLOYMENT_STATUS_DISTRIBUTION: Dict[str, float] = {
    "Employed": 0.603,
    "Unemployed": 0.036,
    "Not in labor force": 0.361,
}

EMPLOYMENT_RATE_BY_RACE: Dict[str, float] = {
    "White": 0.603,
    "Black or African American": 0.596,  # Exact match
    "Asian": 0.631,  # Exact match
    "Hispanic or Latino": 0.634,
}

UNEMPLOYMENT_RATE_BY_RACE: Dict[str, float] = {
    "White": 0.033,
    "Black or African American": 0.055,
    "Asian": 0.030,
    "Hispanic or Latino": 0.046,
}

INCOME_DISTRIBUTION: Dict[str, float] = {
    "Low": 0.30,  # Decrease
    "Medium": 0.50,  # Increase
    "High": 0.20,
}

INCOME_RANGES: Dict[str, Tuple[int, int]] = {
    "Low": (400, 1100),  # Wider range
    "Medium": (1101, 1900),  # Wider range
    "High": (1901, 3000),
}

MEDIAN_WEEKLY_EARNINGS: Dict[str, Dict[str, float]] = {
    "White": {"Male": 1225, "Female": 1021},
    "Black or African American": {"Male": 970, "Female": 889},
    "Asian": {"Male": 1635, "Female": 1299},
    "Hispanic or Latino": {"Male": 915, "Female": 800},
}

# Labor Force Participation
LABOR_FORCE_PARTICIPATION: Dict[str, Dict[str, float]] = {
    "White": {"Male": 0.681, "Female": 0.573},
    "Black or African American": {"Male": 0.656, "Female": 0.610},
    "Asian": {"Male": 0.728, "Female": 0.581},
    "Hispanic or Latino": {"Male": 0.795, "Female": 0.617},
}

# Adjustment Factors
EDUCATION_EMPLOYMENT_ADJUSTMENT: Dict[str, float] = {
    "Less than high school": 0.8,
    "High School": 1.0,
    "Some College/Associate": 1.1,
    "Bachelor": 1.2,
    "Advanced": 1.3,
}

AGE_EMPLOYMENT_ADJUSTMENT: Dict[str, float] = {
    "16-19": 0.7,
    "20-24": 0.9,
    "25-34": 1.1,
    "35-44": 1.2,
    "45-54": 1.2,
    "55-64": 1.0,
    "65+": 0.5,
}

DEMOGRAPHIC_EARNINGS_ADJUSTMENTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "Education": {
        "Less than high school": {
            "White": 1.0,
            "Black or African American": 0.89,
            "Asian": 1.02,  # Reduced
            "Hispanic or Latino": 0.87,
        },
        "High School": {
            "White": 1.0,
            "Black or African American": 0.90,
            "Asian": 1.04,  # Reduced
            "Hispanic or Latino": 0.88,
        },
        "Some College/Associate": {
            "White": 1.0,
            "Black or African American": 0.90,
            "Asian": 1.04,  # Reduced
            "Hispanic or Latino": 0.89,
        },
        "Bachelor": {
            "White": 1.0,
            "Black or African American": 0.91,
            "Asian": 1.06,  # Reduced
            "Hispanic or Latino": 0.90,
        },
        "Advanced": {
            "White": 1.0,
            "Black or African American": 0.92,
            "Asian": 1.08,  # Reduced
            "Hispanic or Latino": 0.89,
        },
    }
}

# Validation Ranges
VALIDATION_RANGES: Dict[str, Tuple[float, float]] = {
    "age_mean": (35, 45),
    "employment_rate": (0.58, 0.65),
    "high_education_rate": (0.30, 0.40),
    "income_distribution_skew": (-0.5, 0.5),
    "education_rate_diff": (0.0, 0.05),
    "employment_rate_diff": (0.0, 0.05),
    "earnings_ratio_diff": (0.0, 0.10),
    "gender_ratio": (0.48, 0.52),
    "race_distribution": (0.0, 1.0),
    "ethnicity_ratio": (0.17, 0.22),
    "regional_distribution": (0.15, 0.40),
}

# [Consumer behavior categories remain unchanged]
PRODUCT_CATEGORIES = [
    "Frozen Foods",
    "Snacks",
    "Baking",
    "Condiments",
    "Ready Meals",
    "Vegetables",
    "Desserts",
    "Breakfast Foods",
    "Meat Products",
    "Plant-Based Foods",
    "Seasonings",
    "Popcorn",
    "Canned Goods",
    "Sauces",
    "Cooking Oils",
    "Hispanic Foods",
    "Spices",
]

COMMUNICATION_CHANNELS = [
    "Email",
    "SMS",
    "Social Media",
    "Direct Mail",
    "Mobile App Notifications",
    "Website",
    "TV Commercials",
    "Radio Ads",
    "In-Store Displays",
    "Print Ads",
    "Push Notifications",
    "Influencer Partnerships",
    "Video Streaming Platforms",
    "Podcast Sponsorships",
    "Interactive Voice Response (IVR)",
]

ENGAGEMENT_PREFERENCES = [
    "Coupons",
    "Newsletters",
    "Social Media Interactions",
    "Loyalty Programs",
    "Recipe Sharing",
    "Product Reviews",
    "Cooking Tutorials",
    "Virtual Tastings",
    "Personalized Product Recommendations",
    "Seasonal Promotions",
    "Meal Planning Tools",
    "Nutritional Information Access",
    "Sustainability Initiatives Participation",
    "Brand Ambassador Programs",
    "Interactive Packaging",
    "New Product Testing Opportunities",
    "Cooking Contests",
    "Community Forums",
    "Family-Oriented Activities",
    "Health and Wellness Challenges",
]


# [Validation functions remain unchanged]
@dataclass
class DemographicValidation:
    min_value: float
    max_value: float
    description: str


def validate_distribution_sums() -> None:
    distributions = {
        "AGE_DISTRIBUTION": AGE_DISTRIBUTION,
        "GENDER_DISTRIBUTION": GENDER_DISTRIBUTION,
        "RACE_DISTRIBUTION": RACE_DISTRIBUTION,
        "HISPANIC_DISTRIBUTION": HISPANIC_DISTRIBUTION,
        "EDUCATION_DISTRIBUTION": EDUCATION_DISTRIBUTION,
        "INCOME_DISTRIBUTION": INCOME_DISTRIBUTION,
        "REGION_WEIGHTS": REGION_WEIGHTS,
    }
    for name, dist in distributions.items():
        if not math.isclose(sum(dist.values()), 1.0, rel_tol=1e-9):
            raise ValidationError(
                f"{name} probabilities do not sum to 1.0: {sum(dist.values())}"
            )


def validate_education_by_race() -> None:
    """Validate education distribution sums for each race."""
    for race, dist in EDUCATION_BY_RACE.items():
        if not math.isclose(sum(dist.values()), 1.0, rel_tol=1e-9):
            raise ValidationError(
                f"Education distribution for {race} does not sum to 1.0: {sum(dist.values())}"
            )


def validate_income_ranges() -> None:
    """Validate income ranges are properly ordered."""
    prev_max = float("-inf")
    for level, (min_val, max_val) in INCOME_RANGES.items():
        if min_val >= max_val:
            raise ValidationError(
                f"Invalid income range for {level}: min ({min_val}) >= max ({max_val})"
            )
        if min_val <= prev_max:
            raise ValidationError(f"Income ranges overlap at {level}")
        prev_max = max_val


def validate_adjustment_factors() -> None:
    """Validate adjustment factors are positive."""
    for name, factors in [
        ("Education Employment", EDUCATION_EMPLOYMENT_ADJUSTMENT),
        ("Age Employment", AGE_EMPLOYMENT_ADJUSTMENT),
    ]:
        if any(v <= 0 for v in factors.values()):
            raise ValidationError(f"{name} adjustment factors must be positive")


def validate_states() -> None:
    """Validate state assignments are unique across regions."""
    all_states = set()
    for region, states in REGIONS.items():
        current_states = set(states)
        if current_states & all_states:
            raise ValidationError(f"Duplicate states found in {region}")
        all_states.update(current_states)


# Add validation for age ranges
def validate_age_ranges() -> None:
    """Validate age range formats."""
    for age_range in AGE_DISTRIBUTION.keys():
        if age_range == "65+":
            continue
        try:
            start, end = map(int, age_range.split("-"))
            if start >= end:
                raise ValidationError(f"Invalid age range: {age_range}")
        except ValueError as e:
            raise ValidationError(f"Invalid age range format: {age_range}")


# Update validate_config to include age range validation
def validate_config() -> None:
    """Run all configuration validations."""
    try:
        validate_distribution_sums()
        validate_education_by_race()
        validate_income_ranges()
        validate_adjustment_factors()
        validate_states()
        validate_age_ranges()  # Add this line
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {str(e)}")


# Validate configuration on import
validate_config()
