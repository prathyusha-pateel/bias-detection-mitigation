from typing import Dict, Any, Sequence
import numpy as np
from scipy import stats


def calculate_metric_significance(
    observed_values: Sequence[float], expected_value: float, min_samples: int = 30
) -> Dict[str, Any]:
    """
    Calculate statistical significance metrics.

    Args:
        observed_values: Sequence of observed values
        expected_value: Expected/benchmark value
        min_samples: Minimum required samples (default: 30)

    Returns:
        Dict with p_value, effect_size, and confidence_interval
    """
    if not observed_values or len(observed_values) < min_samples:
        return {
            "p_value": None,
            "effect_size": None,
            "confidence_interval": (None, None),
            "warning": f"Insufficient samples (n={len(observed_values)})",
        }

    try:
        observed = np.array(observed_values)
        t_stat, p_value = stats.ttest_1samp(observed, expected_value)
        effect_size = (np.mean(observed) - expected_value) / np.std(observed)
        ci = stats.t.interval(
            0.95, len(observed) - 1, loc=np.mean(observed), scale=stats.sem(observed)
        )

        return {
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "confidence_interval": (float(ci[0]), float(ci[1])),
        }
    except Exception as e:
        return {
            "error": f"Statistical calculation failed: {str(e)}",
            "p_value": None,
            "effect_size": None,
            "confidence_interval": (None, None),
        }
