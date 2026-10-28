"""Input validation, healthy-reference estimates and descriptive statistics.

>>> import pandas as pd
>>> from data_simulator import simulate_process
>>> batch = simulate_process(240, 1)
>>> limits = fit_control_limits(batch.baseline)
>>> limits['baseline_n'].tolist()
[160, 160, 160]
>>> len(summarize_process(batch.monitoring))
3
"""

import numpy as np
import pandas as pd

from config import CONTROL_SIGMA, SENSORS


def validate_measurements(data, min_samples=1):
    """Reject missing, nonnumeric, nonfinite or ambiguously ordered samples."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Measurements must be a Pandas DataFrame.")
    required = {"timestamp", "tool_id", *SENSORS}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError("Measurements must be nonempty and contain no missing values.")
    if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
        raise ValueError("timestamp must contain parsed datetime values.")
    if not data["tool_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("tool_id must contain nonempty strings.")
    if data.duplicated(["tool_id", "timestamp"]).any():
        raise ValueError("Duplicate tool/timestamp pairs are not allowed.")
    for sensor in SENSORS:
        if not pd.api.types.is_numeric_dtype(data[sensor]) or pd.api.types.is_bool_dtype(data[sensor]):
            raise ValueError(f"{sensor} must be numeric.")
        if not np.isfinite(data[sensor].to_numpy(dtype=float)).all():
            raise ValueError(f"{sensor} contains nonfinite readings.")
    if (data.groupby("tool_id").size() < min_samples).any():
        raise ValueError(f"At least {min_samples} samples per tool are required.")


def fit_control_limits(baseline, sigma=CONTROL_SIGMA):
    """Fit mean ± sigma * sample SD on a separately supplied healthy baseline."""
    validate_measurements(baseline, min_samples=30)
    if not np.isfinite(sigma) or not 2.0 <= sigma <= 5.0:
        raise ValueError("sigma must be finite and between 2 and 5.")
    rows = []
    for tool_id, group in baseline.groupby("tool_id", sort=True):
        for sensor in SENSORS:
            mean, std = group[sensor].mean(), group[sensor].std(ddof=1)
            if std <= np.finfo(float).eps:
                raise ValueError(f"Healthy baseline has zero variance for {tool_id}/{sensor}.")
            rows.append({"tool_id": tool_id, "sensor": sensor, "mean": mean, "std": std,
                         "lcl": mean - sigma * std, "ucl": mean + sigma * std,
                         "sigma": sigma, "baseline_n": len(group),
                         "baseline_end": group["timestamp"].max()})
    return pd.DataFrame(rows)


def summarize_process(data):
    """Report specification compliance and Pp/Ppk using overall sample SD.

    Ppk is descriptive here: an unstable process is not qualified by this index.
    Specification limits are configuration assumptions, not SPC control limits.
    """
    validate_measurements(data, min_samples=2)
    rows = []
    for tool_id, group in data.groupby("tool_id", sort=True):
        for sensor, spec in SENSORS.items():
            values = group[sensor]
            mean, std = values.mean(), values.std(ddof=1)
            pp = (spec.upper_spec - spec.lower_spec) / (6 * std) if std > 0 else np.nan
            ppk = min(spec.upper_spec - mean, mean - spec.lower_spec) / (3 * std) if std > 0 else np.nan
            rows.append({"tool_id": tool_id, "sensor": sensor, "samples": len(values),
                         "mean": mean, "std": std, "minimum": values.min(), "maximum": values.max(),
                         "in_spec_pct": 100 * values.between(spec.lower_spec, spec.upper_spec).mean(),
                         "pp": pp, "ppk": ppk})
    return pd.DataFrame(rows)
