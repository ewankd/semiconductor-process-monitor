"""Causal SPC alerts: only current and past readings influence an alert.

>>> from data_simulator import simulate_process
>>> from data_analysis import fit_control_limits
>>> batch = simulate_process(240, 1, scenario='temperature_shift')
>>> limits = fit_control_limits(batch.baseline)
>>> detected = detect_faults(batch.monitoring, limits)
>>> bool(detected['alarm'].any())
True
>>> detect_faults(batch.monitoring.drop(columns=['fault_type', 'fault_sensor']), limits).equals(detected)
True
"""

import numpy as np
import pandas as pd

from config import RUN_LENGTH, SENSORS
from data_analysis import validate_measurements


def detect_faults(data, limits, use_run_rule=True):
    """Return one record per tool/sensor/sample, without using fault labels.

    Rules: outside fitted control limits; eight consecutive points strictly
    above or below the reference mean. Run state resets at a missing minute.
    The run rule can flag earlier than a three-sigma limit crossing.
    """
    validate_measurements(data)
    required = {"tool_id", "sensor", "mean", "std", "lcl", "ucl", "baseline_end"}
    if not isinstance(limits, pd.DataFrame) or not required.issubset(limits.columns):
        raise ValueError("Control-limit table is incomplete.")
    if limits.duplicated(["tool_id", "sensor"]).any():
        raise ValueError("Control-limit keys must be unique.")
    parameters = limits[["mean", "std", "lcl", "ucl"]].to_numpy(dtype=float)
    if not np.isfinite(parameters).all() or (limits["std"] <= 0).any():
        raise ValueError("Control parameters must be finite with positive standard deviations.")
    if ((limits["lcl"] >= limits["mean"]) | (limits["ucl"] <= limits["mean"])).any():
        raise ValueError("Control limits must bracket the baseline mean.")

    measured = data.melt(id_vars=["timestamp", "tool_id"], value_vars=list(SENSORS),
                         var_name="sensor", value_name="value")
    measured = measured.merge(limits, on=["tool_id", "sensor"], how="left", validate="many_to_one")
    if measured["std"].isna().any():
        raise ValueError("A monitoring tool/sensor has no fitted baseline.")
    if measured["baseline_end"].isna().any() or (measured["timestamp"] <= measured["baseline_end"]).any():
        raise ValueError("The healthy baseline must end before monitoring starts.")
    measured = measured.sort_values(["tool_id", "sensor", "timestamp"]).reset_index(drop=True)
    measured["z_score"] = (measured["value"] - measured["mean"]) / measured["std"]
    measured["outside_limit"] = (measured["value"] < measured["lcl"]) | (measured["value"] > measured["ucl"])
    measured["mean_run"] = False
    if use_run_rule:
        for _, indices in measured.groupby(["tool_id", "sensor"]).groups.items():
            group = measured.loc[indices]
            sides = np.sign(group["z_score"])
            gap = group["timestamp"].diff().ne(pd.Timedelta(minutes=1))
            blocks = (sides.ne(sides.shift()) | gap).cumsum()
            run_lengths = sides.groupby(blocks).cumcount() + 1
            measured.loc[indices, "mean_run"] = ((run_lengths >= RUN_LENGTH) & sides.ne(0)).to_numpy()
    measured["alarm"] = measured["outside_limit"] | measured["mean_run"]
    measured["rule"] = np.select(
        [measured["outside_limit"] & measured["mean_run"], measured["outside_limit"], measured["mean_run"]],
        ["Limit + mean run", "Outside control limits", f"{RUN_LENGTH}-point mean run"], default="Normal")
    return measured


def evaluate_detection(detections, ground_truth):
    """Evaluate sensor-sample predictions; no labels are passed to the detector.

    Fault windows include gradual drift before it is detectable; imperfect
    recall is expected. Metrics are synthetic, not evidence of field accuracy.
    """
    needed = {"timestamp", "tool_id", "fault_sensor", "fault_type"}
    if not needed.issubset(ground_truth.columns):
        raise ValueError("Ground truth must include tool, timestamp and fault labels.")
    if ground_truth.duplicated(["tool_id", "timestamp"]).any():
        raise ValueError("Ground-truth keys must be unique.")
    scored = detections.merge(ground_truth[list(needed)], on=["timestamp", "tool_id"],
                              how="left", validate="many_to_one")
    if scored["fault_type"].isna().any():
        raise ValueError("Every detection needs a matching ground-truth sample.")
    actual = scored["sensor"].eq(scored["fault_sensor"]) & scored["fault_type"].ne("none")
    predicted = scored["alarm"]
    tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
    fn, tn = int((~predicted & actual).sum()), int((~predicted & ~actual).sum())
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def event_detection_report(detections, events):
    """First alert within each injected event; delay is measured in minutes."""
    rows = []
    for event in events.to_dict("records"):
        hits = detections.loc[detections["tool_id"].eq(event["tool_id"])
                              & detections["sensor"].eq(event["sensor"])
                              & detections["timestamp"].between(event["start"], event["end"])
                              & detections["alarm"], "timestamp"]
        first = hits.min() if not hits.empty else pd.NaT
        rows.append({**event, "detected": not hits.empty, "first_alert": first,
                     "delay_minutes": (first - event["start"]).total_seconds() / 60 if not hits.empty else np.nan})
    return pd.DataFrame(rows, columns=[*events.columns, "detected", "first_alert", "delay_minutes"])
