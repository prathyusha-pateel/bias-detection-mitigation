"""
Custom JSON encoders for handling numpy and pandas types in validation results.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that properly handles numpy and pandas data types."""

    def default(self, obj: Any) -> Any:
        """
        Convert numpy and pandas types to JSON serializable types.

        Args:
            obj: Object to encode

        Returns:
            JSON serializable version of the object
        """
        # Handle numpy numeric types
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        # Handle numpy float types
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        # Handle numpy bool
        elif isinstance(obj, np.bool_):
            return bool(obj)

        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle pandas Timestamp
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        # Handle datetime
        elif isinstance(obj, datetime):
            return obj.isoformat()

        # Handle pandas Series
        elif isinstance(obj, pd.Series):
            return obj.to_dict()

        # Handle pandas Interval
        elif isinstance(obj, pd.Interval):
            return str(obj)

        # Handle pandas Categorical
        elif isinstance(obj, pd.Categorical):
            return obj.astype(str).tolist()

        # Default behavior
        return super().default(obj)
