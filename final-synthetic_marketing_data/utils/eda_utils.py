"""
Utility functions and helper classes for EDA module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from json import JSONEncoder

__all__ = [
    "analyze_numerical_distribution",
    "analyze_categorical_distribution",
    "analyze_temporal_patterns",
    "create_distribution_plot",
    "create_correlation_heatmap",
    "create_time_series_plot",
    "calculate_summary_statistics",
    "analyze_missing_data",
    "format_statistics",
    "NumpyJSONEncoder",
]


class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def analyze_numerical_distribution(data: pd.Series) -> Dict:
    """Analyze distribution of numerical data"""
    return {
        "mean": float(data.mean()),
        "median": float(data.median()),
        "std": float(data.std()),
        "skew": float(data.skew()),
        "kurtosis": float(data.kurtosis()),
        "quartiles": data.quantile([0.25, 0.5, 0.75]).to_dict(),
        "range": (float(data.min()), float(data.max())),
    }


def analyze_categorical_distribution(data: pd.Series) -> Dict:
    """Analyze distribution of categorical data"""
    value_counts = data.value_counts()
    return {
        "unique_values": int(len(value_counts)),
        "mode": str(value_counts.index[0]),
        "frequency": value_counts.to_dict(),
        "missing_rate": float(data.isnull().mean()),
    }


def analyze_temporal_patterns(dates: pd.Series) -> Dict:
    """Analyze temporal patterns in datetime data"""
    return {
        "daily_pattern": dates.dt.day_name().value_counts().to_dict(),
        "hourly_pattern": dates.dt.hour.value_counts().to_dict(),
        "monthly_pattern": dates.dt.month.value_counts().to_dict(),
        "date_range": (dates.min().isoformat(), dates.max().isoformat()),
    }


def create_distribution_plot(
    data: pd.Series, title: str, plot_type: str = "histogram"
) -> go.Figure:
    """Create distribution plot using plotly"""
    if plot_type == "histogram":
        fig = px.histogram(data, title=title)
    elif plot_type == "box":
        fig = px.box(data, title=title)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    fig.update_layout(showlegend=True, template="plotly_white")
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame, numerical_cols: List[str]
) -> go.Figure:
    """Create correlation heatmap"""
    corr_matrix = df[numerical_cols].corr()

    fig = px.imshow(
        corr_matrix,
        title="Feature Correlations",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu",
    )
    return fig


def create_time_series_plot(data: pd.Series, dates: pd.Series, title: str) -> go.Figure:
    """Create time series plot"""
    fig = px.line(x=dates, y=data, title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title="Value", template="plotly_white")
    return fig


def calculate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive summary statistics"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    return {
        "numerical": {
            col: analyze_numerical_distribution(df[col]) for col in numerical_cols
        },
        "categorical": {
            col: analyze_categorical_distribution(df[col]) for col in categorical_cols
        },
    }


def detect_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect anomalies using z-score method"""
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


def analyze_missing_data(df: pd.DataFrame) -> Dict:
    """Analyze missing data patterns"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    return {
        "missing_counts": missing.to_dict(),
        "missing_percentages": missing_pct.to_dict(),
        "total_missing_rate": float(
            (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])).round(4)
        ),
        "complete_rows": int(df.dropna().shape[0]),
    }


def format_statistics(stats: Dict, precision: int = 2) -> Dict:
    """Format numerical statistics to specified precision"""
    formatted = {}
    for key, value in stats.items():
        if isinstance(value, float):
            formatted[key] = round(value, precision)
        elif isinstance(value, dict):
            formatted[key] = format_statistics(value, precision)
        else:
            formatted[key] = value
    return formatted
