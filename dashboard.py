"""Interactive dashboard. Run with: python -m streamlit run dashboard.py"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import SCENARIOS, SENSORS, TITLE
from data_analysis import fit_control_limits, summarize_process
from data_simulator import simulate_process
from fault_detection import detect_faults, evaluate_detection, event_detection_report
from process_optimization import recommend_actions


@st.cache_data(show_spinner=False)
def run_analysis(n_samples, n_tools, seed, scenario, sigma, run_rule):
    batch = simulate_process(n_samples, n_tools, seed, scenario)
    limits = fit_control_limits(batch.baseline, sigma)
    detections = detect_faults(batch.monitoring, limits, run_rule)
    summary = summarize_process(batch.monitoring)
    return batch, limits, detections, summary


def percentage(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def control_chart(group, spec, show_specs):
    """Draw readings, frozen reference limits and alerts for one tool/sensor."""
    fig, ax = plt.subplots(figsize=(12, 3.3))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f9fc")
    ax.plot(group["timestamp"], group["value"], color="#2563eb", linewidth=1.0, label="Measurement")
    for name in ("lcl", "ucl"):
        ax.axhline(group[name].iloc[0], color="#d97706", linestyle="--", linewidth=1.1,
                   label="Control limits" if name == "ucl" else None)
    ax.axhline(group["mean"].iloc[0], color="#64748b", linestyle=":", linewidth=1, label="Baseline mean")
    if show_specs:
        for i, value in enumerate((spec.lower_spec, spec.upper_spec)):
            ax.axhline(value, color="#94a3b8", linestyle="-.", linewidth=1,
                       label="Specification limits" if i == 0 else None)
    hits = group.loc[group["alarm"]]
    ax.scatter(hits["timestamp"], hits["value"], color="#dc2626", s=13, zorder=3, label="SPC alert")
    ax.set_ylabel(f"{spec.label} ({spec.unit})", fontsize=9)
    ax.set_xlabel("Monitoring time (UTC)", fontsize=8)
    ax.grid(axis="y", alpha=0.15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=8, length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.15), ncol=5, fontsize=8, frameon=False)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Semiconductor Process Monitor", page_icon="🔬", layout="wide")
    st.markdown("""<style>
        .block-container {padding-top:2rem; max-width:1500px;}
        [data-testid="stMetric"] {background:#f5f7fb; border:1px solid #e2e8f0;
            border-radius:12px; padding:16px;}
        h1 {letter-spacing:-0.04em;}
        </style>""", unsafe_allow_html=True)
    st.caption("PROCESS ENGINEERING  /  SYNTHETIC EQUIPMENT MONITORING")
    st.title(TITLE)
    st.markdown("Track equipment signals, detect abnormal behavior, and prioritize investigations using statistical process control.")

    with st.sidebar:
        st.header("Simulation controls")
        scenario_label = st.selectbox("Scenario", list(SCENARIOS), key="scenario")
        n_tools = st.slider("Equipment count", 1, 6, 3, key="n_tools")
        n_samples = st.slider("Minutes per tool", 240, 1200, 480, step=60, key="n_samples")
        seed = int(st.number_input("Random seed", min_value=0, max_value=1000000, value=42, key="seed"))
        st.divider()
        st.subheader("Detection settings")
        sigma = st.slider("Control limit (σ)", 2.0, 5.0, 3.0, step=0.5, key="sigma")
        run_rule = st.checkbox("Enable 8-point mean-run rule", True, key="run_rule")
        show_specs = st.checkbox("Show specification limits", False, key="show_specs")
        st.caption("Control limits are fitted on 160 independent healthy samples per tool. Monitoring data never refits the baseline.")

    batch, limits, detections, summary = run_analysis(n_samples, n_tools, seed, SCENARIOS[scenario_label], sigma, run_rule)
    alarms = detections.loc[detections["alarm"]]
    sample_alerts = detections.groupby(["tool_id", "timestamp"])["alarm"].any()
    metrics = evaluate_detection(detections, batch.monitoring)
    within = batch.monitoring[list(SENSORS)].apply(
        lambda col: col.between(SENSORS[col.name].lower_spec, SENSORS[col.name].upper_spec))
    columns = st.columns(4)
    columns[0].metric("Equipment monitored", n_tools)
    columns[1].metric("Equipment-minute samples", f"{len(batch.monitoring):,}")
    columns[2].metric("Samples with any SPC alert", f"{100 * sample_alerts.mean():.1f}%")
    columns[3].metric("Samples within all specifications", f"{100 * within.all(axis=1).mean():.1f}%")
    st.caption("Specification compliance is a sensor-based proxy, not wafer yield or overall equipment effectiveness.")

    charts_tab, alerts_tab, actions_tab, evaluation_tab = st.tabs(
        ["Control charts", "Alert log", "Process review", "Simulation evaluation"])
    with charts_tab:
        tool_id = st.selectbox("Equipment", sorted(batch.monitoring["tool_id"].unique()), key="tool")
        st.caption("Blue: measurements · amber: frozen control limits · red: alerts. All times are UTC.")
        for sensor, spec in SENSORS.items():
            group = detections.loc[detections["tool_id"].eq(tool_id) & detections["sensor"].eq(sensor)]
            fig = control_chart(group, spec, show_specs)
            st.pyplot(fig)
            plt.close(fig)
        st.subheader("Observed process statistics")
        st.dataframe(summary.loc[summary["tool_id"].eq(tool_id)].round(3), hide_index=True, width="stretch")
        st.caption("Pp/Ppk use overall sample standard deviation. They describe this synthetic batch; unstable signals do not establish process capability.")

    with alerts_tab:
        st.subheader("SPC alert samples")
        st.caption("One row per flagged sensor-minute. A sustained event can produce multiple rows.")
        if alarms.empty:
            st.success("No SPC alerts for these settings.")
        else:
            alert_columns = ["timestamp", "tool_id", "sensor", "value", "z_score", "rule"]
            st.dataframe(alarms[alert_columns].sort_values("timestamp", ascending=False).round(3),
                         hide_index=True, width="stretch")
        st.download_button("Download all detections (CSV)", detections.to_csv(index=False),
                           file_name="spc_detections.csv", mime="text/csv")

    with actions_tab:
        st.subheader("Prioritized investigation plan")
        st.info("Suggestions are hypotheses to investigate. Sensor patterns cannot establish root cause or justify automatic recipe changes.")
        actions = recommend_actions(summary, detections)
        if actions.empty:
            st.success("No issues prioritized. Keep monitoring and periodically review calibration.")
        else:
            for row in actions.to_dict("records"):
                with st.expander(f"{row['priority']} · {row['tool_id']} · {row['sensor']}", expanded=True):
                    st.markdown(f"**{row['finding']}**")
                    st.write(row["evidence"])
                    st.write(row["suggested_action"])
        st.download_button("Download process review (CSV)", actions.to_csv(index=False),
                           file_name="process_review.csv", mime="text/csv")

    with evaluation_tab:
        st.subheader("How well were the injected faults detected?")
        st.caption("Fault labels are used only here, after detection. Precision and recall are measured per sensor-minute, not per event.")
        cols = st.columns(3)
        cols[0].metric("Alert precision", percentage(metrics["precision"]))
        cols[1].metric("Fault recall", percentage(metrics["recall"]))
        cols[2].metric("Healthy-sample false positive rate", percentage(metrics["false_positive_rate"]))
        confusion = pd.DataFrame(
            [[metrics["true_positives"], metrics["false_negatives"]],
             [metrics["false_positives"], metrics["true_negatives"]]],
            index=["Injected fault", "Healthy"], columns=["Alert", "No alert"])
        st.dataframe(confusion, width="stretch")
        event_report = event_detection_report(detections, batch.events)
        if event_report.empty:
            st.info("Healthy scenario: there are no injected fault events. Recall is undefined (N/A).")
        else:
            st.dataframe(event_report, hide_index=True, width="stretch")
        st.caption("Drift begins near normal and may initially be undetectable. Run rules can also produce false alarms after a fault ends. A first alert within a fault window is not proof of causation.")
        left, right = st.columns(2)
        left.download_button("Download monitoring data + truth", batch.monitoring.to_csv(index=False),
                             file_name="monitoring_with_truth.csv", mime="text/csv")
        right.download_button("Download healthy baseline", batch.baseline.to_csv(index=False),
                              file_name="healthy_baseline.csv", mime="text/csv")

    st.divider()
    st.caption("Educational portfolio project · Synthetic data only · No live fab connection or validated manufacturing recipes")


if __name__ == "__main__":
    main()
