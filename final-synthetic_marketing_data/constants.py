"""Configuration constants for synthetic data generation"""

# ------------------------------------------------------------------------------
# Core Configuration
# ------------------------------------------------------------------------------

# Core PUMS Features to extract from demographic data
FEATURES = [
    "SERIALNO",  # Household ID
    "SPORDER",  # Person Number within Household
    "AGEP",  # Age
    "SEX",  # Gender
    "PINCP",  # Income
    "ADJINC",  # Income Adjustment Factor
    "PWGTP",  # Person Weight
    "STATE",  # Location
    "PUMA",  # Public Use Microdata Area
    "SCHL",  # Educational Attainment
]

# ACS Data Source Configuration
ACS_CONFIG = {
    "survey_year": "2023",
    "horizon": "1-Year",
    "survey": "person",
}

# State Selection Sets
STATES_TEST = [
    "NY",  # Large population, diverse demographics for testing
]

STATES_MEDIUM = [
    # One or two major states per region, balanced for demographic coverage
    "NY",  # Northeast: Finance, urban
    "CA",  # West: Tech, entertainment
    "TX",  # South: Energy, diverse economy
    "IL",  # Midwest: Urban/rural mix
    "FL",  # South: Tourism, retirement
    "WA",  # West: Tech, trade
]

STATES_FULL = [
    # Northeast (4 states)
    "NY",  # Large urban population, diverse economy
    "NJ",  # Dense suburbs, pharmaceuticals
    "MA",  # High education levels, tech sector
    "PA",  # Mix of urban/rural, manufacturing
    # South (6 states)
    "TX",  # Energy sector, diverse economy
    "FL",  # Retirement communities, tourism
    "GA",  # Transportation hub, growing tech
    "VA",  # Government, defense
    "NC",  # Banking, research triangle
    "TN",  # Music industry, automotive
    # Midwest (5 states)
    "IL",  # Mix of urban and rural
    "OH",  # Manufacturing, agriculture
    "MI",  # Manufacturing base
    "MN",  # Healthcare, agriculture
    "WI",  # Agriculture, manufacturing
    # West (5 states)
    "CA",  # Tech sector, entertainment
    "WA",  # Tech sector, trade
    "AZ",  # Retirement, growing tech
    "CO",  # Aerospace, outdoor recreation
    "OR",  # Tech, forestry
]

# ------------------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------------------

# Model Training Configuration
MODEL_PARAMETERS = {
    "epochs": 1,  # 1 for testing min. 100 for real
    "batch_size": 500,
    "log_frequency": True,
    "verbose": True,
    "discriminator_dim": (512, 256),
    "discriminator_lr": 1e-4,
    "discriminator_decay": 1e-6,
    "generator_dim": (512, 256),
    "generator_lr": 2e-4,
    "generator_decay": 1e-6,
    "embedding_dim": 128,
    "pac": 8,
}

# Validation Thresholds
VALIDATION_THRESHOLDS = {
    "numerical": {
        "mean_difference": 0.1,
        "std_difference": 0.15,
        "correlation_difference": 0.2,
        "tolerance": 0.1,
    },
    "categorical": {"distribution_difference": 0.15, "tolerance": 0.15},
    "engagement": {"tolerance": 0.02},  # 2% tolerance for engagement metrics
    "loyalty": {"tolerance": 0.05},  # 5% tolerance for loyalty metrics
    "campaign": {"tolerance": 0.05},  # 5% tolerance for campaign metrics
    "distribution": {"tolerance": 0.05},  # 5% tolerance for distributions
    "regional": {"tolerance": 0.10},  # 10% tolerance for regional variations
    "preference": {"tolerance": 0.05},  # 5% tolerance for preferences
    "adoption": {"tolerance": 0.08},  # 8% tolerance for adoption rates
    "device": {"tolerance": 0.10},  # 10% tolerance for device usage
    "transaction": {
        "value_difference": 0.15,  # 15% tolerance for transaction values
        "frequency_difference": 0.10,  # 10% tolerance for frequency patterns
        "distribution_difference": 0.10,  # 10% tolerance for distributions
    },
}

# ------------------------------------------------------------------------------
# Geographic and Demographic Parameters
# ------------------------------------------------------------------------------

# Regional Variations for Generation
REGIONAL_VARIATIONS = {
    "Northeast": {
        "baseline_multiplier": 1.18,  # Updated from NRF data
        "premium_brand_preference": 1.25,  # Higher due to urban concentration
        "seasonal_impact": 1.15,  # Stronger seasonal variation
        "engagement_multiplier": 1.2,  # Merkle Q4 2023 shows higher digital engagement
        "loyalty_rate": 1.25,  # Bond Brand Loyalty shows higher program participation
        "transaction_multiplier": 1.18,  # Matches NRF regional spending patterns
        "online_adoption_rate": 1.25,  # McKinsey shows higher digital adoption
        "digital_adoption": 1.15,  # From McKinsey Digital Consumer Survey
        "brand_affinity": 1.1,  # Deloitte Consumer Industry Report
    },
    "Midwest": {
        "baseline_multiplier": 0.92,  # Updated from NRF data
        "premium_brand_preference": 0.85,  # Lower premium brand penetration
        "seasonal_impact": 1.05,  # Moderate seasonal variation
        "engagement_multiplier": 0.90,  # Lower digital engagement rates
        "loyalty_rate": 1.05,  # Average loyalty program participation
        "transaction_multiplier": 0.92,  # Matches regional spending patterns
        "online_adoption_rate": 0.95,  # Slightly below average digital adoption
        "digital_adoption": 1.0,  # Average digital adoption rate
        "brand_affinity": 0.95,  # Average brand loyalty
    },
    "South": {
        "baseline_multiplier": 0.88,  # Updated from NRF data
        "premium_brand_preference": 0.80,  # Lower premium brand preference
        "seasonal_impact": 0.85,  # Less seasonal variation
        "engagement_multiplier": 0.85,  # Lower digital engagement
        "loyalty_rate": 0.85,  # Lower loyalty program participation
        "transaction_multiplier": 0.88,  # Matches regional spending patterns
        "online_adoption_rate": 0.80,  # Lower digital adoption rates
        "digital_adoption": 0.90,  # Below average digital adoption
        "brand_affinity": 0.90,  # Lower brand loyalty
    },
    "West": {
        "baseline_multiplier": 1.12,  # Updated from NRF data
        "premium_brand_preference": 1.15,  # Higher premium brand preference
        "seasonal_impact": 1.10,  # Moderate seasonal variation
        "engagement_multiplier": 1.20,  # Higher digital engagement
        "loyalty_rate": 1.15,  # Higher loyalty program participation
        "transaction_multiplier": 1.12,  # Matches regional spending patterns
        "online_adoption_rate": 1.20,  # Higher digital adoption rates
        "digital_adoption": 1.20,  # Highest digital adoption
        "brand_affinity": 1.15,  # Strong brand loyalty
    },
}

# State to Region Mapping
STATE_TO_REGION = {
    # Northeast
    "ME": "Northeast",
    "NH": "Northeast",
    "VT": "Northeast",
    "MA": "Northeast",
    "RI": "Northeast",
    "CT": "Northeast",
    "NY": "Northeast",
    "NJ": "Northeast",
    "PA": "Northeast",
    # Midwest
    "OH": "Midwest",
    "IN": "Midwest",
    "IL": "Midwest",
    "MI": "Midwest",
    "WI": "Midwest",
    "MN": "Midwest",
    "IA": "Midwest",
    "MO": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "NE": "Midwest",
    "KS": "Midwest",
    # South
    "DE": "South",
    "MD": "South",
    "DC": "South",
    "VA": "South",
    "WV": "South",
    "NC": "South",
    "SC": "South",
    "GA": "South",
    "FL": "South",
    "KY": "South",
    "TN": "South",
    "AL": "South",
    "MS": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    # West
    "MT": "West",
    "ID": "West",
    "WY": "West",
    "CO": "West",
    "NM": "West",
    "AZ": "West",
    "UT": "West",
    "NV": "West",
    "WA": "West",
    "OR": "West",
    "CA": "West",
    "AK": "West",
    "HI": "West",
}

# ------------------------------------------------------------------------------
# Consumer Behavior Parameters
# ------------------------------------------------------------------------------

# Transaction Generation Parameters
TRANSACTION_METRICS = {
    "average_value": {"mean": 45.99, "std_dev": 15.50},
    "items_per_transaction": {"mean": 3.2, "std_dev": 1.1},
}

# Buying Patterns for Generation
BUYING_PATTERNS = {
    "frequency": {
        "high": 0.2,  # Multiple times per week
        "medium": 0.5,  # Weekly to bi-weekly
        "low": 0.3,  # Monthly or less
    },
    "basket_size": {
        "small": 0.4,  # 1-2 items
        "medium": 0.45,  # 3-5 items
        "large": 0.15,  # 6+ items
    },
}

# Product Preferences for Generation
PRODUCT_PREFERENCES = {
    "ready_to_eat": {"base_price_range": (3.99, 7.99), "max_quantity": 5},
    "snacks": {"base_price_range": (2.99, 5.99), "max_quantity": 10},
    "sustainable": {"base_price_range": (4.99, 9.99), "max_quantity": 5},
    "family_size": {"base_price_range": (8.99, 15.99), "max_quantity": 3},
    "healthy_alternatives": {"base_price_range": (4.99, 8.99), "max_quantity": 5},
    "traditional": {"base_price_range": (5.99, 10.99), "max_quantity": 5},
}

# Device Preferences for Generation
DEVICE_PREFERENCES = {
    "18-29": {"mobile": 0.65, "desktop": 0.25, "tablet": 0.10},
    "30-49": {"mobile": 0.45, "desktop": 0.40, "tablet": 0.15},
    "50+": {"mobile": 0.25, "desktop": 0.45, "tablet": 0.30},
}

# Noise factors for synthetic data generation
NOISE_FACTORS = {
    "18-34": 0.1,  # Higher variability for younger group
    "35-54": 0.08,  # Moderate variability for middle group
    "55+": 0.05,  # Lower variability for older group
}

# ------------------------------------------------------------------------------
# Marketing and Campaign Parameters
# ------------------------------------------------------------------------------

# Creative Elements for Marketing Campaigns
CREATIVE_ELEMENTS = {
    "Static_Image": [
        "hero_image",
        "product_shot",
        "lifestyle_image",
        "brand_logo",
        "promotional_banner",
        "ingredient_closeup",
    ],
    "Carousel": [
        "product_gallery",
        "feature_slides",
        "testimonial_cards",
        "recipe_steps",
        "usage_instructions",
        "brand_story",
    ],
    "Video": [
        "product_demo",
        "customer_story",
        "brand_video",
        "recipe_tutorial",
        "lifestyle_content",
        "behind_scenes",
    ],
    "Interactive": [
        "quiz",
        "poll",
        "mini_game",
        "configurator",
        "product_finder",
        "nutrition_calculator",
    ],
    "Story": [
        "narrative_sequence",
        "behind_scenes",
        "day_in_life",
        "recipe_creation",
        "farm_to_table",
        "chef_spotlight",
    ],
}

# ------------------------------------------------------------------------------
# Metadata Configurations
# ------------------------------------------------------------------------------

METADATA_CONFIGURATIONS = {
    "demographics": {
        "numerical_columns": ["AGEP", "PINCP", "ADJINC", "PWGTP"],
        "categorical_columns": ["SEX", "STATE", "PUMA", "SCHL"],
        "id_column": "SERIALNO",
        "tolerance": 0.1,
        "cat_tolerance": 0.1,
    },
    "transactions": {
        "id_column": "transaction_id",
        "foreign_keys": ["consumer_id"],
        "datetime_columns": ["transaction_date"],
        "numerical_columns": ["transaction_value", "num_items"],
        "categorical_columns": [
            "channel",
            "state",
            "region",
            "age_group",
            "buying_frequency",
        ],
    },
    "transaction_details": {
        "id_column": "detail_id",
        "foreign_keys": ["transaction_id", "product_id"],
        "numerical_columns": ["quantity", "unit_price", "line_total"],
        "categorical_columns": ["category", "subcategory"],
    },
    "consumer": {
        "id_column": "consumer_id",
        "numerical_columns": [
            "monthly_purchases",
            "avg_basket_size",
            "social_media_engagement_rate",
            "online_shopping_rate",
            "loyalty_memberships",
        ],
        "categorical_columns": ["age_group", "buying_frequency", "brand_affinity"],
        "preference_columns": [
            "ready_to_eat",
            "snacks",
            "sustainable",
            "family_size",
            "healthy",
            "traditional",
            "health_conscious",
        ],
    },
    "campaigns": {
        "id_column": "campaign_id",
        "categorical_columns": ["campaign_type", "primary_channel", "creative_type"],
        "datetime_columns": ["start_date", "end_date"],
        "numerical_columns": [
            "target_age_min",
            "target_age_max",
            "target_income_min",
            "target_income_max",
            "base_engagement_rate",
            "budget",
        ],
        "text_columns": ["creative_elements", "target_regions"],
    },
    "engagements": {
        "id_column": "engagement_id",
        "foreign_keys": ["campaign_id"],
        "datetime_columns": ["date"],
        "numerical_columns": [
            "impressions",
            "clicks",
            "engagement_rate",
            "time_spent_minutes",
            "conversion_rate",
            "campaign_day",
            "campaign_progress",
        ],
        "categorical_columns": ["day_of_week"],
    },
    "loyalties": {
        "id_column": "consumer_id",
        "datetime_columns": ["enrollment_date"],
        "categorical_columns": ["status", "tier", "age_group"],
        "numerical_columns": ["points_balance", "lifetime_points", "redemption_rate"],
    },
}

# Add FIPS to state code mapping
FIPS_TO_STATE = {
    36: "NY",  # New York
    # Add other states as needed
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}

# Add to existing constants.py
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

# Data subdirectories
DATA_SUBDIRS = ["consumer", "marketing_engagement", "transactions"]

# Output file patterns
FILE_PATTERNS = {
    "demographic": "[a-z][a-z]_demographics.csv",
    "consumer": "consumer/*/consumer.csv",
    "marketing": "marketing_engagement/*/campaigns.csv",
    "transaction": "transactions/*/transactions.csv",
}

# Age group definitions
AGE_GROUPS = {
    "bins": [0, 18, 35, 55, float("inf")],
    "labels": ["Under 18", "18-34", "35-54", "55+"],
}

# Income group definitions (example values)
INCOME_GROUPS = {
    "bins": [0, 25000, 50000, 75000, 100000, float("inf")],
    "labels": ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
}

# Education level mapping
EDUCATION_GROUPS = {
    1: "No schooling completed",
    2: "Nursery school to 4th grade",
    3: "5th and 6th grade",
    4: "7th and 8th grade",
    5: "9th grade",
    6: "10th grade",
    7: "11th grade",
    8: "12th grade, no diploma",
    9: "High school graduate",
    10: "Some college, less than 1 year",
    11: "Some college, 1 or more years, no degree",
    12: "Associate's degree",
    13: "Bachelor's degree",
    14: "Master's degree",
    15: "Professional degree",
    16: "Doctorate degree",
}

# PUMS Field Mappings with friendly names
PUMS_FIELDS = {
    "SERIALNO": "Household Serial Number",
    "SPORDER": "Person Sequence Number within Household",
    "AGEP": "Age (AGEP)",
    "SEX": "Gender (SEX)",
    "PINCP": "Personal Income (PINCP)",
    "ADJINC": "Income Adjustment Factor (ADJINC)",
    "PWGTP": "Person Weight (PWGTP)",
    "STATE": "State Code (STATE)",
    "PUMA": "Public Use Microdata Area (PUMA)",
    "SCHL": "Educational Attainment (SCHL)",
}

# Gender mapping
GENDER_MAPPING = {1: "Male", 2: "Female"}
