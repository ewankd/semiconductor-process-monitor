"""Shared parameters for an educational, synthetic etch-tool simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorSpec:
    label: str
    unit: str
    target: float
    noise_std: float
    lower_spec: float
    upper_spec: float


# Illustrative engineering assumptions, not qualified manufacturing recipes.
SENSORS = {
    "temperature": SensorSpec("Chamber temperature", "°C", 400.0, 2.0, 390.0, 410.0),
    "pressure": SensorSpec("Chamber pressure", "mTorr", 30.0, 0.45, 27.0, 33.0),
    "rf_power": SensorSpec("RF power", "W", 1000.0, 12.0, 940.0, 1060.0),
}

SCENARIOS = {
    "Mixed faults": "mixed",
    "Healthy process": "healthy",
    "Temperature shift": "temperature_shift",
    "Pressure drift": "pressure_drift",
    "RF power instability": "rf_noise",
}

BASELINE_SAMPLES = 160
CONTROL_SIGMA = 3.0
RUN_LENGTH = 8
START_TIME = "2026-01-01T08:00:00Z"
TITLE = "Semiconductor Manufacturing Process Monitoring & Fault Detection"
