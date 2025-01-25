"""Benchmark definitions for transaction data validation"""

from constants import (
    VALIDATION_THRESHOLDS,
    TRANSACTION_METRICS,
)  # Import from constants

# Channel Distribution Benchmarks (McKinsey Digital Consumer Survey, 2023)
CHANNEL_DISTRIBUTION = {
    "mobile": 0.45,
    "desktop": 0.35,
    "in_store": 0.20,
    "by_age_group": {
        "18-29": {"mobile": 0.60, "desktop": 0.30, "in_store": 0.10},
        "30-49": {"mobile": 0.45, "desktop": 0.40, "in_store": 0.15},
        "50+": {"mobile": 0.30, "desktop": 0.35, "in_store": 0.35},
    },
}

# Product Category Benchmarks (Deloitte Consumer Industry Report, 2023)
CATEGORY_BENCHMARKS = {
    "distribution": {
        "ready_to_eat": 0.25,
        "snacks": 0.20,
        "sustainable": 0.15,
        "family_size": 0.15,
        "healthy_alternatives": 0.15,
        "traditional": 0.10,
    },
    "average_price": {
        "ready_to_eat": 8.99,
        "snacks": 4.99,
        "sustainable": 12.99,
        "family_size": 15.99,
        "healthy_alternatives": 9.99,
        "traditional": 7.99,
    },
}

# Research sources tracking
RESEARCH_SOURCES = {
    "transaction_values": "NRF Retail Library, 2023",
    "channel_distribution": "McKinsey Digital Consumer Survey, 2023",
    "category_benchmarks": "Deloitte Consumer Industry Report, 2023",
    "regional_variations": "FMI Grocery Shopper Trends, 2023",
}
