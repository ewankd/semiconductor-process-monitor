"""Prioritized investigation suggestions based on measurements, not truth labels.

>>> from data_simulator import simulate_process
>>> from data_analysis import fit_control_limits, summarize_process
>>> from fault_detection import detect_faults
>>> batch = simulate_process(240, 1)
>>> alerts = detect_faults(batch.monitoring, fit_control_limits(batch.baseline))
>>> actions = recommend_actions(summarize_process(batch.monitoring), alerts)
>>> {'evidence', 'suggested_action', 'priority'}.issubset(actions.columns)
True
"""

import pandas as pd

from config import SENSORS


CHECKS = {
    "temperature": "Check temperature-sensor calibration, heater control and thermal contact.",
    "pressure": "Check pressure-sensor calibration, vacuum leaks, pump condition and throttle-valve control.",
    "rf_power": "Check RF readback calibration, generator stability and impedance matching.",
}


def recommend_actions(summary, detections):
    """Rank signals without claiming causation or changing equipment settings."""
    rows = []
    for (tool_id, sensor), group in detections.groupby(["tool_id", "sensor"], sort=True):
        stats = summary.loc[summary["tool_id"].eq(tool_id) & summary["sensor"].eq(sensor)]
        if len(stats) != 1:
            raise ValueError("Each detection group needs exactly one summary row.")
        stats = stats.iloc[0]
        alert_pct = 100 * group["alarm"].mean()
        if alert_pct == 0 and stats["in_spec_pct"] == 100:
            continue
        reference_std = float(group["std"].iloc[0])
        shift = float((stats["mean"] - group["mean"].iloc[0]) / reference_std)
        variance_ratio = float(stats["std"] / reference_std)
        if variance_ratio > 1.8:
            finding = "Excess variation or a changing operating level"
        elif abs(shift) > 1:
            finding = "Mean offset from the healthy reference"
        else:
            finding = "Intermittent excursions or a sustained mean run"
        severity = alert_pct + 2 * (100 - stats["in_spec_pct"])
        rows.append({"tool_id": tool_id, "sensor": SENSORS[sensor].label,
                     "priority": "High" if severity >= 20 else "Review",
                     "finding": finding,
                     "evidence": f"{alert_pct:.1f}% alert samples; {100 - stats['in_spec_pct']:.1f}% outside specification; "
                                 f"mean shift {shift:+.2f}σ; SD/reference SD {variance_ratio:.2f}×",
                     "suggested_action": CHECKS[sensor] + " Confirm with maintenance records and an independent measurement; "
                                         "then verify any approved change against a new healthy batch.",
                     "score": severity})
    columns = ["tool_id", "sensor", "priority", "finding", "evidence", "suggested_action", "score"]
    return pd.DataFrame(rows, columns=columns).sort_values("score", ascending=False).drop(columns="score").reset_index(drop=True)
