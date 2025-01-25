"""Benchmark definitions for marketing engagement"""

# Define benchmark dictionaries with research citations
ENGAGEMENT_METRICS = {
    "social_media": {
        "instagram": 0.0215,  # Instagram engagement rate
        "facebook": 0.0012,  # Facebook engagement rate
        "twitter": 0.00045,  # Twitter engagement rate
    },
    "email": {
        "open_rate": 0.1977,  # Email open rate
        "click_rate": 0.0201,  # Email click rate
    },
    "time_spent_minutes": {
        "social_content": 2.5,  # Default social content time
        "email_content": 1.5,  # Email content time
    },
    "temporal": {
        "day_multipliers": {
            0: 0.9,  # Monday
            1: 1.0,  # Tuesday
            2: 1.1,  # Wednesday
            3: 1.2,  # Thursday
            4: 1.1,  # Friday
            5: 0.8,  # Saturday
            6: 0.7,  # Sunday
        }
    },
}

LOYALTY_METRICS = {
    "program_enrollment": {
        "active": 0.45,  # Active members
        "semi_active": 0.35,  # Semi-active members
        "inactive": 0.20,  # Inactive members
    },
    "points": {
        "base_mean": 5000,  # Base points mean
        "base_std": 1500,  # Base points standard deviation
        "tiers": {
            "bronze": 0,  # 0-3000 points
            "silver": 3000,  # 3001-7500 points
            "gold": 7500,  # 7501+ points
        },
    },
    "redemption_rates": {
        "Gen_Z": 0.65,  # Higher engagement from younger generation
        "Millennial": 0.55,  # Strong digital adoption
        "Gen_X": 0.45,  # Moderate engagement
        "Boomer": 0.35,  # Lower digital engagement but high brand loyalty
    },
    "program_engagement": {
        "points_program": 0.75,  # 75% engagement with points program
        "rewards_program": 0.65,  # 65% engagement with rewards program
        "referral_program": 0.35,  # 35% engagement with referral program
        "vip_program": 0.25,  # 25% engagement with VIP program
        "birthday_program": 0.80,  # 80% engagement with birthday rewards
    },
}

CAMPAIGN_DISTRIBUTIONS = {
    "campaign_types": {
        "Brand_Awareness": 0.167,
        "Product_Launch": 0.167,
        "Seasonal_Promotion": 0.167,
        "Loyalty_Rewards": 0.167,
        "Re-engagement": 0.167,
        "Cross_sell": 0.167,
    },
    "channels": {
        "Email": 0.167,
        "Facebook": 0.167,
        "Instagram": 0.167,
        "Twitter": 0.167,
        "Display_Ads": 0.167,
        "Search_Ads": 0.167,
    },
    "creative_types": {
        "Static_Image": 0.20,
        "Carousel": 0.20,
        "Video": 0.20,
        "Interactive": 0.20,
        "Story": 0.20,
    },
}


# Research sources tracking
RESEARCH_SOURCES = {
    "engagement": "Merkle Q4 2023 Report, https://www.merkle.com/...",
    "loyalty": "Bond Brand Loyalty Report 2023, https://info.bondbrandloyalty.com/...",
    "digital_adoption": "McKinsey 2023, https://www.mckinsey.com/...",
    "product_preferences": "Deloitte 2023, https://www2.deloitte.com/...",
    "regional_variations": "Internal market analysis based on historical campaign performance",
}
