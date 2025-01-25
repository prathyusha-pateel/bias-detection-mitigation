# transaction_constants.py

from typing import Optional, Dict
from datetime import datetime
import pandas as pd
import numpy as np


# Product categories and their typical price ranges
PRODUCT_CATEGORIES = {
    "Frozen Meals": {"min_price": 3.99, "max_price": 12.99},
    "Snacks": {"min_price": 1.99, "max_price": 7.99},
    "Condiments": {"min_price": 2.49, "max_price": 8.99},
    "Canned Goods": {"min_price": 0.99, "max_price": 5.99},
    "Ready Meals": {"min_price": 4.99, "max_price": 15.99},
}

# Store types and their characteristics
STORE_TYPES = {
    "Supermarket": {"weight": 0.5, "price_multiplier": 1.0},
    "Convenience Store": {"weight": 0.2, "price_multiplier": 1.3},
    "Wholesale Club": {"weight": 0.15, "price_multiplier": 0.8},
    "Online": {"weight": 0.15, "price_multiplier": 1.1},
}

# Payment methods and their typical usage weights
PAYMENT_METHODS = {
    "Credit Card": 0.4,
    "Debit Card": 0.35,
    "Cash": 0.15,
    "Digital Wallet": 0.1,
}

# Time of day weights for shopping
TIME_OF_DAY_WEIGHTS = {
    "Morning (6AM-11AM)": 0.25,
    "Afternoon (11AM-4PM)": 0.35,
    "Evening (4PM-9PM)": 0.30,
    "Night (9PM-6AM)": 0.10,
}

# Day of week weights for shopping
DAY_OF_WEEK_WEIGHTS = {
    "Monday": 0.12,
    "Tuesday": 0.13,
    "Wednesday": 0.14,
    "Thursday": 0.15,
    "Friday": 0.18,
    "Saturday": 0.20,
    "Sunday": 0.08,
}

# Seasonal factors affecting purchase patterns
SEASONAL_FACTORS = {
    "Spring": {"multiplier": 1.0, "categories": ["Condiments", "Snacks"]},
    "Summer": {"multiplier": 0.9, "categories": ["Frozen Meals", "Ready Meals"]},
    "Fall": {"multiplier": 1.1, "categories": ["Canned Goods", "Ready Meals"]},
    "Winter": {"multiplier": 1.2, "categories": ["Frozen Meals", "Canned Goods"]},
}

# Income level impacts on shopping patterns
INCOME_IMPACTS = {
    "Low": {"price_sensitivity": 1.5, "bulk_buying": 0.5},
    "Medium": {"price_sensitivity": 1.0, "bulk_buying": 1.0},
    "High": {"price_sensitivity": 0.7, "bulk_buying": 1.5},
}

# Age group shopping patterns
AGE_GROUP_PATTERNS = {
    "18-25": {"online_preference": 1.5, "health_conscious": 0.8},
    "26-35": {"online_preference": 1.3, "health_conscious": 1.0},
    "36-50": {"online_preference": 1.0, "health_conscious": 1.2},
    "51-65": {"online_preference": 0.7, "health_conscious": 1.3},
    "65+": {"online_preference": 0.5, "health_conscious": 1.4},
}

# Buying frequency patterns (transactions per month)
BUYING_FREQUENCY_PATTERNS = {
    "Weekly": {"min_transactions": 4, "max_transactions": 8},
    "Bi-Weekly": {"min_transactions": 2, "max_transactions": 4},
    "Monthly": {"min_transactions": 1, "max_transactions": 2},
}

# Special event multipliers
SPECIAL_EVENTS = {
    "Holiday Season": 1.5,
    "Back to School": 1.3,
    "Summer BBQ": 1.2,
    "Super Bowl": 1.4,
}

# Regional variations in shopping patterns
REGIONAL_VARIATIONS = {
    "Urban": {
        "store_type_weights": {
            "Supermarket": 0.4,
            "Convenience Store": 0.3,
            "Wholesale Club": 0.1,
            "Online": 0.2,
        },
        "online_multiplier": 1.2,
        "price_multiplier": 1.1,
    },
    "Suburban": {
        "store_type_weights": {
            "Supermarket": 0.5,
            "Convenience Store": 0.15,
            "Wholesale Club": 0.2,
            "Online": 0.15,
        },
        "online_multiplier": 1.0,
        "price_multiplier": 1.0,
    },
    "Rural": {
        "store_type_weights": {
            "Supermarket": 0.6,
            "Convenience Store": 0.2,
            "Wholesale Club": 0.1,
            "Online": 0.1,
        },
        "online_multiplier": 0.8,
        "price_multiplier": 0.9,
    },
}

# Special event date ranges
SPECIAL_EVENTS.update(
    {
        "black_friday": {
            "dates": ["2023-11-24", "2023-11-25", "2023-11-26"],
            "transaction_boost": 2.0,
            "value_boost": 1.8,
            "online_boost": 1.5,
        },
        "december_holiday": {
            "date_range": ["2023-12-01", "2023-12-31"],
            "transaction_boost": 1.5,
            "value_boost": 1.3,
            "online_boost": 1.4,
            "late_boost": 1.2,  # Additional boost for last-minute shopping
        },
        "back_to_school": {
            "date_range": ["2023-08-01", "2023-09-15"],
            "transaction_boost": 1.3,
            "value_boost": 1.2,
            "online_boost": 1.3,
        },
    }
)

# Category preferences by demographic
CATEGORY_PREFERENCES = {
    "Age": {
        "18-25": {
            "Frozen Meals": 1.2,
            "Snacks": 1.4,
            "Ready Meals": 1.3,
            "Condiments": 0.8,
            "Canned Goods": 0.7,
        },
        "26-35": {
            "Frozen Meals": 1.1,
            "Snacks": 1.2,
            "Ready Meals": 1.2,
            "Condiments": 1.0,
            "Canned Goods": 0.9,
        },
        "36-50": {
            "Frozen Meals": 1.0,
            "Snacks": 1.0,
            "Ready Meals": 1.1,
            "Condiments": 1.2,
            "Canned Goods": 1.1,
        },
        "51-65": {
            "Frozen Meals": 0.9,
            "Snacks": 0.8,
            "Ready Meals": 0.9,
            "Condiments": 1.3,
            "Canned Goods": 1.2,
        },
        "65+": {
            "Frozen Meals": 0.8,
            "Snacks": 0.7,
            "Ready Meals": 0.8,
            "Condiments": 1.4,
            "Canned Goods": 1.3,
        },
    },
    "Income": {
        "Low": {
            "Frozen Meals": 1.1,
            "Snacks": 0.9,
            "Ready Meals": 0.8,
            "Condiments": 1.0,
            "Canned Goods": 1.4,
        },
        "Medium": {
            "Frozen Meals": 1.0,
            "Snacks": 1.0,
            "Ready Meals": 1.0,
            "Condiments": 1.0,
            "Canned Goods": 1.0,
        },
        "High": {
            "Frozen Meals": 0.9,
            "Snacks": 1.2,
            "Ready Meals": 1.3,
            "Condiments": 1.1,
            "Canned Goods": 0.7,
        },
    },
    "Employment": {
        "Employed": {
            "Frozen Meals": 1.2,
            "Snacks": 1.1,
            "Ready Meals": 1.3,
            "Condiments": 0.9,
            "Canned Goods": 0.8,
        },
        "Unemployed": {
            "Frozen Meals": 0.9,
            "Snacks": 0.8,
            "Ready Meals": 0.7,
            "Condiments": 1.1,
            "Canned Goods": 1.3,
        },
        "Not in labor force": {
            "Frozen Meals": 1.0,
            "Snacks": 1.0,
            "Ready Meals": 0.9,
            "Condiments": 1.2,
            "Canned Goods": 1.2,
        },
    },
}

# Income brackets for categorization
INCOME_BRACKETS = {
    "Low": (0, 40000),
    "Medium": (40001, 100000),
    "High": (100001, float("inf")),
}

# Age brackets for categorization
AGE_BRACKETS = {
    "18-25": (18, 25),
    "26-35": (26, 35),
    "36-50": (36, 50),
    "51-65": (51, 65),
    "65+": (65, float("inf")),
}

# Employment status mapping
EMPLOYMENT_STATUS = {1: "Employed", 2: "Unemployed", 3: "Not in labor force"}

# Price effects by category and demographic factors
CATEGORY_PRICE_EFFECTS = {
    "Income": {
        "Low": {
            "Frozen Meals": 0.9,  # More likely to buy on sale/discount
            "Snacks": 0.85,
            "Ready Meals": 0.8,
            "Condiments": 0.9,
            "Canned Goods": 0.95,
        },
        "Medium": {
            "Frozen Meals": 1.0,
            "Snacks": 1.0,
            "Ready Meals": 1.0,
            "Condiments": 1.0,
            "Canned Goods": 1.0,
        },
        "High": {
            "Frozen Meals": 1.2,  # More likely to buy premium versions
            "Snacks": 1.3,
            "Ready Meals": 1.4,
            "Condiments": 1.2,
            "Canned Goods": 1.1,
        },
    },
    "Store_Type": {
        "Supermarket": {
            "Frozen Meals": 1.0,
            "Snacks": 1.0,
            "Ready Meals": 1.0,
            "Condiments": 1.0,
            "Canned Goods": 1.0,
        },
        "Convenience Store": {
            "Frozen Meals": 1.3,
            "Snacks": 1.4,
            "Ready Meals": 1.3,
            "Condiments": 1.2,
            "Canned Goods": 1.25,
        },
        "Wholesale Club": {
            "Frozen Meals": 0.8,
            "Snacks": 0.75,
            "Ready Meals": 0.85,
            "Condiments": 0.8,
            "Canned Goods": 0.7,
        },
        "Online": {
            "Frozen Meals": 1.1,
            "Snacks": 1.15,
            "Ready Meals": 1.2,
            "Condiments": 1.1,
            "Canned Goods": 1.05,
        },
    },
    "Season": {
        "Spring": {
            "Frozen Meals": 0.95,
            "Snacks": 1.1,
            "Ready Meals": 1.0,
            "Condiments": 1.2,
            "Canned Goods": 0.9,
        },
        "Summer": {
            "Frozen Meals": 0.9,
            "Snacks": 1.2,
            "Ready Meals": 0.95,
            "Condiments": 1.3,
            "Canned Goods": 0.85,
        },
        "Fall": {
            "Frozen Meals": 1.1,
            "Snacks": 1.0,
            "Ready Meals": 1.1,
            "Condiments": 0.9,
            "Canned Goods": 1.2,
        },
        "Winter": {
            "Frozen Meals": 1.2,
            "Snacks": 0.9,
            "Ready Meals": 1.15,
            "Condiments": 0.8,
            "Canned Goods": 1.1,
        },
    },
}

# Seasonal shopping patterns
SEASONAL_PATTERNS = {
    "Spring": {  # March-May
        "transaction_frequency_multiplier": 1.1,
        "basket_size_multiplier": 1.0,
        "preferred_categories": ["Condiments", "Snacks"],
        "store_type_weights": {
            "Supermarket": 0.5,
            "Convenience Store": 0.2,
            "Wholesale Club": 0.15,
            "Online": 0.15,
        },
        "time_of_day_weights": {
            "Morning (6AM-11AM)": 0.3,
            "Afternoon (11AM-4PM)": 0.35,
            "Evening (4PM-9PM)": 0.25,
            "Night (9PM-6AM)": 0.1,
        },
    },
    "Summer": {  # June-August
        "transaction_frequency_multiplier": 0.9,
        "basket_size_multiplier": 0.8,
        "preferred_categories": ["Frozen Meals", "Snacks"],
        "store_type_weights": {
            "Supermarket": 0.45,
            "Convenience Store": 0.25,
            "Wholesale Club": 0.15,
            "Online": 0.15,
        },
        "time_of_day_weights": {
            "Morning (6AM-11AM)": 0.25,
            "Afternoon (11AM-4PM)": 0.3,
            "Evening (4PM-9PM)": 0.35,
            "Night (9PM-6AM)": 0.1,
        },
    },
    "Fall": {  # September-November
        "transaction_frequency_multiplier": 1.2,
        "basket_size_multiplier": 1.1,
        "preferred_categories": ["Canned Goods", "Ready Meals"],
        "store_type_weights": {
            "Supermarket": 0.5,
            "Convenience Store": 0.15,
            "Wholesale Club": 0.2,
            "Online": 0.15,
        },
        "time_of_day_weights": {
            "Morning (6AM-11AM)": 0.25,
            "Afternoon (11AM-4PM)": 0.35,
            "Evening (4PM-9PM)": 0.3,
            "Night (9PM-6AM)": 0.1,
        },
    },
    "Winter": {  # December-February
        "transaction_frequency_multiplier": 1.3,
        "basket_size_multiplier": 1.2,
        "preferred_categories": ["Frozen Meals", "Canned Goods"],
        "store_type_weights": {
            "Supermarket": 0.45,
            "Convenience Store": 0.15,
            "Wholesale Club": 0.2,
            "Online": 0.2,
        },
        "time_of_day_weights": {
            "Morning (6AM-11AM)": 0.2,
            "Afternoon (11AM-4PM)": 0.3,
            "Evening (4PM-9PM)": 0.35,
            "Night (9PM-6AM)": 0.15,
        },
    },
}

# State codes and regions
STATE_CODE_TO_REGION = {
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "MD": "South",
    "NC": "South",
    "SC": "South",
    "VA": "South",
    "WV": "South",
    "AL": "South",
    "KY": "South",
    "MS": "South",
    "TN": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    "AZ": "West",
    "CO": "West",
    "ID": "West",
    "MT": "West",
    "NV": "West",
    "NM": "West",
    "UT": "West",
    "WY": "West",
    "AK": "West",
    "CA": "West",
    "HI": "West",
    "OR": "West",
    "WA": "West",
}

# Regional characteristics
REGION_CHARACTERISTICS = {
    "Midwest": {"urbanization": 0.7, "price_index": 0.95, "online_adoption": 0.8},
    "Northeast": {"urbanization": 0.9, "price_index": 1.15, "online_adoption": 1.1},
    "South": {"urbanization": 0.6, "price_index": 0.9, "online_adoption": 0.9},
    "West": {"urbanization": 0.8, "price_index": 1.1, "online_adoption": 1.2},
}


# Helper functions
def get_quarter(date: datetime) -> str:
    """Get quarter for a given date."""
    if isinstance(date, (np.datetime64, pd.Timestamp)):
        date = pd.Timestamp(date)
    return f"Q{(date.month - 1) // 3 + 1}"


def is_black_friday(date: datetime) -> bool:
    """Check if date is during Black Friday event."""
    date_str = date.strftime("%Y-%m-%d")
    return date_str in SPECIAL_EVENTS["black_friday"]["dates"]


def is_december_holiday(date: datetime) -> bool:
    """Check if date is during December holiday period."""
    date_str = date.strftime("%Y-%m-%d")
    start, end = SPECIAL_EVENTS["december_holiday"]["date_range"]
    return start <= date_str <= end


def is_back_to_school(date: datetime) -> bool:
    """Check if date is during back to school period."""
    date_str = date.strftime("%Y-%m-%d")
    start, end = SPECIAL_EVENTS["back_to_school"]["date_range"]
    return start <= date_str <= end


def is_special_event(date: datetime) -> Optional[str]:
    """Identify any special event for a given date."""
    if is_black_friday(date):
        return "black_friday"
    elif is_december_holiday(date):
        return "december_holiday"
    elif is_back_to_school(date):
        return "back_to_school"
    return None


def get_event_multipliers(date: datetime, base_value: float = 1.0) -> Dict[str, float]:
    """Get event-related multipliers for a given date."""
    event = is_special_event(date)
    if not event:
        return {"transaction": 1.0, "value": 1.0, "online": 1.0}

    event_info = SPECIAL_EVENTS[event]
    multipliers = {
        "transaction": event_info["transaction_boost"],
        "value": event_info.get("value_boost", 1.0),
        "online": event_info.get("online_boost", 1.0),
    }

    # Apply late December boost if applicable
    if event == "december_holiday" and date.strftime("%Y-%m-%d") >= "2023-12-20":
        multipliers["value"] *= event_info["late_boost"]

    return multipliers


def get_season(date: datetime) -> str:
    """Get season for a given date."""
    month = date.month
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:  # month in [12, 1, 2]
        return "Winter"
