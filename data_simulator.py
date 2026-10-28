"""Reproducible healthy reference data and a separate monitoring batch.

>>> batch = simulate_process(n_samples=240, n_tools=1, seed=42)
>>> len(batch.monitoring), len(batch.events), batch.baseline.fault_type.unique().tolist()
(240, 3, ['none'])
>>> batch.monitoring.equals(simulate_process(240, 1, 42).monitoring)
True
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BASELINE_SAMPLES, SCENARIOS, SENSORS, START_TIME


@dataclass
class SimulationResult:
    baseline: pd.DataFrame
    monitoring: pd.DataFrame
    events: pd.DataFrame


def simulate_process(n_samples=480, n_tools=3, seed=42, scenario="mixed"):
    """Generate minute samples; fault labels are reserved for evaluation only.

    Baseline data ends before monitoring starts. Separate random streams make
    reference data independent of monitoring length and selected fault scenario.
    """
    for name, value, low, high in (("n_samples", n_samples, 120, 10080),
                                   ("n_tools", n_tools, 1, 12),
                                   ("seed", seed, 0, 2**32 - 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
    if scenario not in SCENARIOS.values():
        raise ValueError(f"Unknown scenario: {scenario}")

    baseline_parts, monitoring_parts, events = [], [], []
    start_time = pd.Timestamp(START_TIME)
    for tool_number in range(n_tools):
        tool_id = f"ETCH-{tool_number + 1:02d}"
        parts = []
        for phase, count in enumerate((BASELINE_SAMPLES, n_samples)):
            rng = np.random.default_rng(np.random.SeedSequence([seed, tool_number, phase]))
            start = start_time - pd.Timedelta(minutes=BASELINE_SAMPLES) if phase == 0 else start_time
            frame = pd.DataFrame({"timestamp": pd.date_range(start, periods=count, freq="min"),
                                  "tool_id": tool_id})
            for sensor, spec in SENSORS.items():
                # Small, fixed equipment offsets are learned per tool.
                offset = ((tool_number % 3) - 1) * 0.15 * spec.noise_std
                frame[sensor] = rng.normal(spec.target + offset, spec.noise_std, count)
            frame["fault_type"] = "none"
            frame["fault_sensor"] = "none"
            parts.append(frame)

        baseline, monitoring = parts
        injections = [
            ("temperature_shift", "temperature", 0.20, 0.36),
            ("pressure_drift", "pressure", 0.45, 0.65),
            ("rf_noise", "rf_power", 0.74, 0.90),
        ]
        fault_rng = np.random.default_rng(np.random.SeedSequence([seed, tool_number, 2]))
        for fault, sensor, start_fraction, stop_fraction in injections:
            if scenario not in ("mixed", fault):
                continue
            start, stop = int(n_samples * start_fraction), int(n_samples * stop_fraction)
            indices = monitoring.index[start:stop]
            sigma = SENSORS[sensor].noise_std
            if fault == "temperature_shift":
                disturbance = np.full(stop - start, 4.5 * sigma)
            elif fault == "pressure_drift":
                disturbance = np.linspace(0.0, 7.0 * sigma, stop - start)
            else:
                disturbance = fault_rng.normal(0.0, 5.0 * sigma, stop - start)
            monitoring.loc[indices, sensor] += disturbance
            monitoring.loc[indices, ["fault_type", "fault_sensor"]] = [fault, sensor]
            events.append({"tool_id": tool_id, "sensor": sensor, "fault_type": fault,
                           "start": monitoring.loc[start, "timestamp"],
                           "end": monitoring.loc[stop - 1, "timestamp"],
                           "samples": stop - start})
        baseline_parts.append(baseline)
        monitoring_parts.append(monitoring)

    event_columns = ["tool_id", "sensor", "fault_type", "start", "end", "samples"]
    return SimulationResult(pd.concat(baseline_parts, ignore_index=True),
                            pd.concat(monitoring_parts, ignore_index=True),
                            pd.DataFrame(events, columns=event_columns))
