# Semiconductor Manufacturing Process Monitoring & Fault Detection

A Python portfolio project that simulates semiconductor etch-tool signals, detects abnormal equipment behavior with statistical process control (SPC), and presents evidence-based investigation suggestions in an interactive Streamlit dashboard.

**All data, equipment and process limits are synthetic.** This project does not connect to manufacturing equipment, represent a qualified recipe, demonstrate real-world fault detection accuracy, or establish wafer yield improvements.

## Run locally

Requires Python 3.11 or 3.12. From the project directory:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux (use instead of the Windows command):
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). No API keys, accounts, external data downloads or paid services are required. Stop the server with Ctrl+C.

## What you can explore

- Simulate 1–6 equipment units and 240–1,200 monitoring minutes per unit.
- Choose healthy operation, a temperature level shift, gradual pressure drift, RF power instability, or a mixture of all three.
- Change the seed, control-limit multiplier and mean-run rule; results are reproducible for the same settings.
- Inspect temperature, pressure and RF power charts with independently fitted reference limits and alert markers.
- Compare alerts with injected fault windows, including first-alert delays and a sensor-sample confusion matrix.
- Prioritize maintenance investigations using measured variation, offsets and specification compliance.
- Download healthy reference data, monitoring data, detections and the process review as CSV.

## Eight source files

| File | Responsibility |
| --- | --- |
| `config.py` | Sensor units, illustrative limits and scenario configuration |
| `data_simulator.py` | Independent healthy reference and monitoring batches; reproducible injected faults |
| `data_analysis.py` | Data validation, per-tool reference estimates, specification compliance, Pp/Ppk |
| `fault_detection.py` | Causal SPC rules, sample-level evaluation and event detection delays |
| `process_optimization.py` | Evidence-based investigation priorities and possible checks |
| `dashboard.py` | Interactive controls, Matplotlib charts, tables and CSV exports |
| `requirements.txt` | Runtime dependencies |
| `README.md` | Setup, methods, assumptions and verification |

## Statistical method

Each tool has its own **160-sample healthy reference batch**, collected before the monitoring batch. Limits are frozen at the reference mean ± *k* times the reference sample standard deviation; the default *k* is 3. Monitoring data never refits the reference, so a fault cannot widen its own control limits. The simulator has separate random streams for the reference, monitoring noise and injected disturbances.

The detector uses measurements only. For each tool and sensor it flags:

1. A reading strictly outside the fitted control limits.
2. Optionally, eight consecutive minute readings strictly on the same side of the reference mean. A timestamp gap resets the run. No future readings are used.

The simulator's fault labels are joined only during evaluation. Metrics count **sensor-minutes**: one equipment-minute yields three sensor predictions. Precision = TP/(TP+FP), recall = TP/(TP+FN), and healthy-sample false positive rate = FP/(FP+TN). A metric with a zero denominator is shown as N/A. A separate event table reports whether an alert occurred within each injected window and the first-alert delay. Persistent faults produce multiple alert samples, not multiple independent incidents.

**Specification limits differ from control limits.** Configured specification limits are illustrative acceptance bounds; statistical limits describe reference variation. In-spec samples can still trigger an SPC rule. The dashboard's all-sensors-in-spec percentage is not wafer yield, OEE or a count of good wafers.

Pp = (USL − LSL)/(6s) and Ppk = min(USL − mean, mean − LSL)/(3s), where *s* is the overall monitoring sample standard deviation. These are descriptive performance indices, not Cp/Cpk based on a within-subgroup estimate. Unstable data should be investigated before any capability claim.

## Reuse the analysis without the dashboard

```python
from data_simulator import simulate_process
from data_analysis import fit_control_limits, summarize_process
from fault_detection import detect_faults, evaluate_detection
from process_optimization import recommend_actions

batch = simulate_process(n_samples=480, n_tools=3, seed=42, scenario="mixed")
limits = fit_control_limits(batch.baseline)
detections = detect_faults(batch.monitoring, limits)
metrics = evaluate_detection(detections, batch.monitoring)
actions = recommend_actions(summarize_process(batch.monitoring), detections)
print(metrics)
print(actions.to_string(index=False))
```

For externally collected data, supply parsed datetimes, a nonempty string `tool_id`, and numeric `temperature`, `pressure`, `rf_power` columns in the configured units. Use a separately qualified healthy reference of at least 30 readings per tool ending before monitoring begins. The validator rejects missing/nonfinite measurements, duplicate tool/timestamp pairs, unknown baseline tools and zero reference variance. The run rule assumes one-minute sampling; this is not an arbitrary-rate production ingestion service. The dashboard currently generates synthetic data only.

## Verification

Run the executable examples from the project directory:

```bash
python -m doctest -v data_simulator.py data_analysis.py fault_detection.py process_optimization.py
```

The examples check reproducibility, separate reference data, fitted reference counts, fault alerts, detector independence from truth labels and investigation output. For an interactive smoke check, compare **Mixed faults** with **Healthy process**, turn off the run rule, and adjust sigma: the alert log, evaluation metrics and charts should update together. Healthy operation can still generate statistical false alarms; zero alerts are not guaranteed.

## Scope and limitations

- The simulator assumes independent Gaussian reference noise, stable healthy operation and three isolated fault windows. Actual tools can have autocorrelation, recipe transitions, missing data, interacting faults and non-Gaussian readings.
- Simple SPC rules are interpretable baselines, not trained machine learning models. Multiple sensors and added run rules increase false-alarm opportunities. Gradual drift may remain initially undetectable.
- RF noise injection may occasionally leave a reading close to normal; injected labels do not imply every affected sample is detectable.
- Investigation suggestions are hypotheses, not causal diagnoses. No recipe settings are changed automatically, and no experimentally measured improvement is claimed.
- Pp/Ppk and synthetic detection scores do not qualify a real process. Production use would require approved specifications, qualified references, robust ingestion, calibration, alert management and validation on representative equipment data.

## References

- [NIST/SEMATECH: What are control charts?](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc31.htm)
- [NIST/SEMATECH: Shewhart X-bar and R/S control charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm)
- [Streamlit documentation](https://docs.streamlit.io/)
- [NumPy random number generation](https://numpy.org/doc/stable/reference/random/generator.html)

Built with Python, Pandas, NumPy, Matplotlib and Streamlit.
