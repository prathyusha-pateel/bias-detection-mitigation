"""Consumer preference benchmarks from market research."""

from constants import VALIDATION_THRESHOLDS  # Import from constants

# Social Media Adoption by Age (Pew Research, 2023)
SOCIAL_MEDIA_ADOPTION = {
    "18-34": 0.84,  # Updated: 84% social media usage
    "35-54": 0.81,  # Updated: 81% social media usage
    "55+": 0.66,  # 66% social media usage (unchanged for 65+)
}

# Social Media Platform Engagement (Merkle Q4 2023)
PLATFORM_ENGAGEMENT = {
    "instagram": 0.0215,  # 2.15% engagement rate
    "facebook": 0.0012,  # 0.12% engagement rate
    "twitter": 0.00045,  # 0.045% engagement rate
}

# Device Usage Patterns (DataReportal, 2023)
DEVICE_PATTERNS = {
    "mobile": 0.68,  # 68% of engagement
    "desktop": 0.27,  # 27% of engagement
    "tablet": 0.05,  # 5% of engagement
}

# Product Category Preferences (Deloitte, 2023)
PRODUCT_PREFERENCES = {
    "18-34": {
        "ready_to_eat": 0.38,
        "snacks": 0.42,
        "sustainable": 0.45,
        "health_conscious": 0.40,  # Added based on Deloitte 2023
    },
    "35-54": {
        "family_size": 0.45,
        "healthy_alternatives": 0.35,
        "sustainable": 0.38,
        "health_conscious": 0.45,  # Added based on Deloitte 2023
    },
    "55+": {"traditional": 0.52, "health_conscious": 0.48, "sustainable": 0.29},
}

# Online Shopping Adoption (McKinsey, 2023)
ONLINE_ADOPTION = {
    "18-34": 0.75,  # Updated: Gen Z (18-24): 73% + Millennials (25-40): 78%
    "35-54": 0.61,  # Gen X adoption rate
    "55+": 0.43,  # Boomers adoption rate
}

# Loyalty Program Behavior (Bond Brand Loyalty, 2023)
LOYALTY_METRICS = {
    "average_memberships": 16.7,
    "active_programs": 7.7,
    "program_engagement": {"retail": 0.65, "grocery": 0.55, "restaurants": 0.50},
    "redemption_rates": {
        "18-34": 0.70,  # Updated from Bond Brand Loyalty 2023
        "35-54": 0.54,  # Updated from Bond Brand Loyalty 2023
        "55+": 0.35,  # Updated from Bond Brand Loyalty 2023
    },
    "program_participation": {  # Added based on Bond Brand Loyalty 2023
        "active": 0.45,
        "semi_active": 0.35,
        "inactive": 0.20,
    },
}

# Income-Based Correlations (Kim & Kumar, 2023)
INCOME_CORRELATIONS = {
    "multi_channel": 0.31,
    "digital_channel": 0.28,
    "digital_service": 0.34,
    "mobile_app": 0.29,
    "premium_product": 0.35,
    "sustainable": 0.28,
    "health_conscious": 0.25,
    "ready_to_eat": -0.15,
    "avg_basket_size": 0.42,
    "monthly_purchases": 0.31,
}

# Research sources tracking
RESEARCH_SOURCES = {
    "social_media": "Pew Research, 2023",
    "platform_engagement": "Merkle Q4 2023",
    "device_usage": "DataReportal, 2023",
    "product_preferences": "Deloitte, 2023",
    "online_adoption": "McKinsey, 2023",
    "loyalty": "Bond Brand Loyalty, 2023",
    "income_correlations": "Kim & Kumar, 2023",
    "platform": "Merkle Q4 2023",
    "device": "DataReportal, 2023",
    "adoption": "Pew Research, 2023",
    "affinity": "Bond Brand Loyalty, 2023",
}

# Brand Affinity Distribution
BRAND_AFFINITY = {
    "loyal": 0.25,  # Consistently choose same brand
    "explorer": 0.35,  # Regularly try new brands
    "value": 0.30,  # Price/value-driven choices
    "unattached": 0.10,  # No strong brand preferences
}

# Time-of-Day Shopping Patterns
PEAK_ENGAGEMENT_TIMES = {
    "18-34": {"start": 20, "end": 23},  # 8 PM  # 11 PM
    "35-54": {"start": 18, "end": 21},  # 6 PM  # 9 PM
    "55+": {"start": 14, "end": 17},  # 2 PM  # 5 PM
}

# Device Preferences by Generation
DEVICE_PREFERENCES = {
    "Gen_Z": {"mobile": 0.72, "desktop": 0.28},  # Updated from research
    "Millennial": {"mobile": 0.65, "desktop": 0.35},
    "Gen_X": {"mobile": 0.48, "desktop": 0.52},
    "Boomer": {"mobile": 0.31, "desktop": 0.69},
}
