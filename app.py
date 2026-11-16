"""
Petrol Pump Queue Simulation and Modelling
===========================================
A discrete-event simulation application built with Streamlit, SimPy, and Plotly.

Run with:
    streamlit run app.py
"""

import sys
import os

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd

from simulation.engine import PetrolPumpSimulation, SimulationConfig
from simulation.metrics import compute_metrics
from simulation.scenarios import ScenarioRunner, ScenarioConfig
from visualization.charts import (
    queue_length_chart,
    arrivals_over_time_chart,
    service_completions_chart,
    waiting_time_distribution_chart,
    dispenser_utilisation_chart,
    scenario_waiting_time_chart,
    scenario_queue_length_chart,
    scenario_utilisation_chart,
    scenario_throughput_chart,
)
from visualization.dashboard import render_kpi_cards
from utils.helpers import vehicles_to_dataframe, traffic_intensity

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Petrol Pump Queue Simulation",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for the dark petrol-station theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #16213e;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #16213e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #a0a0b0 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2ecc71 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #16213e;
        border-radius: 6px;
        color: #e0e0e0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2ecc71 !important;
        color: #1a1a2e !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #2ecc71 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #16213e;
        color: #e0e0e0;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #2a2a4a;
        border-radius: 8px;
    }

    /* Info / warning boxes */
    .stAlert {
        background-color: #16213e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar — Simulation Parameters
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛽ Simulation Parameters")
    st.markdown("---")

    num_dispensers = st.slider(
        "Number of Fuel Dispensers",
        min_value=1, max_value=10, value=3,
        help="How many fuel pumps are available at the station.",
    )
    arrival_rate = st.slider(
        "Vehicle Arrival Rate (vehicles/hour)",
        min_value=1.0, max_value=120.0, value=20.0, step=1.0,
        help="Average number of vehicles arriving per hour (λ).",
    )
    avg_service_time = st.slider(
        "Average Service Time (minutes)",
        min_value=1.0, max_value=20.0, value=5.0, step=0.5,
        help="Mean time to fuel a vehicle.",
    )
    service_time_std = st.slider(
        "Service Time Variability (std dev, min)",
        min_value=0.0, max_value=5.0, value=1.5, step=0.1,
        help="Standard deviation of fuel service time.",
    )
    sim_duration = st.slider(
        "Simulation Duration (minutes)",
        min_value=30, max_value=480, value=120, step=10,
        help="How long to run the simulation.",
    )
    random_seed = st.number_input(
        "Random Seed",
        min_value=0, max_value=99999, value=42,
        help="Seed for reproducible results.",
    )
    num_repetitions = st.slider(
        "Simulation Repetitions",
        min_value=1, max_value=20, value=1,
        help="Number of independent runs to average over.",
    )

    st.markdown("---")

    # Traffic intensity indicator
    service_rate = 60.0 / avg_service_time  # vehicles/hour per dispenser
    rho = traffic_intensity(arrival_rate, service_rate, num_dispensers)

    if rho < 0.7:
        st.success(f"Traffic Intensity ρ = {rho:.3f} — System is **under-loaded**.")
    elif rho < 1.0:
        st.warning(f"Traffic Intensity ρ = {rho:.3f} — System is **moderately loaded**.")
    else:
        st.error(
            f"Traffic Intensity ρ = {rho:.3f} — System is **unstable** (ρ ≥ 1). "
            "Queues will grow without bound. Add more dispensers or reduce demand."
        )

    run_button = st.button("🚀 Run Simulation", use_container_width=True, type="primary")

# ──────────────────────────────────────────────
# Build config and run simulation
# ──────────────────────────────────────────────
config = SimulationConfig(
    num_dispensers=num_dispensers,
    arrival_rate=arrival_rate,
    avg_service_time=avg_service_time,
    service_time_std=service_time_std,
    min_service_time=max(1.0, avg_service_time - 3 * service_time_std),
    max_service_time=avg_service_time + 3 * service_time_std,
    simulation_duration=float(sim_duration),
    random_seed=int(random_seed),
)


@st.cache_data
def run_simulation_cached(
    nd, ar, ast_, sts, mn, mx, sd, rs
):
    """Cache-friendly wrapper for the simulation."""
    cfg = SimulationConfig(
        num_dispensers=nd,
        arrival_rate=ar,
        avg_service_time=ast_,
        service_time_std=sts,
        min_service_time=mn,
        max_service_time=mx,
        simulation_duration=sd,
        random_seed=rs,
    )
    sim = PetrolPumpSimulation(cfg)
    result = sim.run()
    metrics = compute_metrics(result)
    return result, metrics, cfg


# Always run with current params (Streamlit re-renders on changes)
result, metrics, cfg = run_simulation_cached(
    config.num_dispensers,
    config.arrival_rate,
    config.avg_service_time,
    config.service_time_std,
    config.min_service_time,
    config.max_service_time,
    config.simulation_duration,
    config.random_seed,
)

# ──────────────────────────────────────────────
# Title
# ──────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>⛽ Petrol Pump Queue Simulation</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#a0a0b0; margin-top:0;'>"
    "Discrete-Event Simulation &amp; Queueing Theory Analysis</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ──────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🏗️ Live Simulation",
    "📈 Queue Analysis",
    "📐 Statistical Analysis",
    "🔀 Scenario Comparison",
    "🔮 What-If Analysis",
    "📖 Mathematical Model",
])

# ════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════
with tabs[0]:
    st.markdown("### Key Performance Indicators")
    render_kpi_cards(metrics)

    st.markdown("---")
    st.markdown("### Vehicle Log")
    df = vehicles_to_dataframe(result.vehicles)
    st.dataframe(df, use_container_width=True, height=400)

# ════════════════════════════════════════
# TAB 2 — Live Simulation Visualisation
# ════════════════════════════════════════
with tabs[1]:
    st.markdown("### Petrol Station Visualisation")

    if not result.vehicles:
        st.info("Run the simulation to see the visualisation.")
    else:
        # Time-stepped visualisation
        max_time = config.simulation_duration
        time_step = st.slider(
            "Simulation Time (minutes)", 0.0, max_time,
            value=max_time, step=1.0,
            key="time_slider",
        )

        # Categorise vehicles at selected time
        waiting = []
        fueling = []
        completed = []
        not_arrived = []

        for v in result.vehicles:
            if v.arrival_time > time_step:
                not_arrived.append(v)
            elif v.service_start_time == 0 or v.service_start_time > time_step:
                waiting.append(v)
            elif v.service_end_time == 0 or v.service_end_time > time_step:
                fueling.append(v)
            else:
                completed.append(v)

        # Display station state
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div style='background:#16213e;border:1px solid #2a2a4a;"
                f"border-radius:8px;padding:16px;text-align:center;'>"
                f"<h3 style='color:#f39c12 !important;'>🚗 Waiting</h3>"
                f"<p style='font-size:2.5em;color:#f39c12;margin:0;'>{len(waiting)}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div style='background:#16213e;border:1px solid #2a2a4a;"
                f"border-radius:8px;padding:16px;text-align:center;'>"
                f"<h3 style='color:#3498db !important;'>⛽ Fueling</h3>"
                f"<p style='font-size:2.5em;color:#3498db;margin:0;'>{len(fueling)}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div style='background:#16213e;border:1px solid #2a2a4a;"
                f"border-radius:8px;padding:16px;text-align:center;'>"
                f"<h3 style='color:#2ecc71 !important;'>✅ Completed</h3>"
                f"<p style='font-size:2.5em;color:#2ecc71;margin:0;'>{len(completed)}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Fuel Dispensers")
        disp_cols = st.columns(min(num_dispensers, 5))
        for i in range(num_dispensers):
            with disp_cols[i % len(disp_cols)]:
                # Check if dispenser i is currently serving
                currently_serving = None
                for v in fueling:
                    if v.dispenser_id == i:
                        currently_serving = v
                        break
                if currently_serving:
                    st.markdown(
                        f"<div style='background:#0a3d62;border:2px solid #3498db;"
                        f"border-radius:8px;padding:12px;text-align:center;'>"
                        f"<p style='margin:0;font-size:1.2em;'>⛽ Dispenser {i+1}</p>"
                        f"<p style='margin:4px 0 0;color:#3498db;'>🚗 Vehicle #{currently_serving.vehicle_id}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='background:#16213e;border:2px solid #2a2a4a;"
                        f"border-radius:8px;padding:12px;text-align:center;'>"
                        f"<p style='margin:0;font-size:1.2em;'>⛽ Dispenser {i+1}</p>"
                        f"<p style='margin:4px 0 0;color:#666;'>Idle</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        if waiting:
            st.markdown("#### Queue")
            queue_text = " → ".join([f"🚗 #{v.vehicle_id}" for v in waiting[:20]])
            if len(waiting) > 20:
                queue_text += f" ... (+{len(waiting)-20} more)"
            st.markdown(
                f"<div style='background:#16213e;border:1px solid #f39c12;"
                f"border-radius:8px;padding:12px;'>{queue_text}</div>",
                unsafe_allow_html=True,
            )

# ════════════════════════════════════════
# TAB 3 — Queue Analysis
# ════════════════════════════════════════
with tabs[2]:
    st.markdown("### Queue Analysis Charts")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            queue_length_chart(result.queue_length_over_time),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            arrivals_over_time_chart(result),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            service_completions_chart(result),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            waiting_time_distribution_chart(result),
            use_container_width=True,
        )

    st.markdown("### Dispenser Utilisation")
    st.plotly_chart(
        dispenser_utilisation_chart(metrics, num_dispensers),
        use_container_width=True,
    )

# ════════════════════════════════════════
# TAB 4 — Statistical Analysis
# ════════════════════════════════════════
with tabs[3]:
    st.markdown("### Detailed Statistical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Waiting Time Statistics")
        stats_data = {
            "Metric": [
                "Mean Waiting Time",
                "Median Waiting Time",
                "Std Dev of Waiting Time",
                "95th Percentile Waiting Time",
                "Maximum Waiting Time",
            ],
            "Value": [
                f"{metrics.avg_waiting_time:.3f} min",
                f"{metrics.median_waiting_time:.3f} min",
                f"{metrics.std_waiting_time:.3f} min",
                f"{metrics.p95_waiting_time:.3f} min",
                f"{metrics.max_waiting_time:.3f} min",
            ],
            "Explanation": [
                "Average time a vehicle waits before fueling begins.",
                "The middle value: half of vehicles wait less, half wait more.",
                "Measures how spread out waiting times are.",
                "95% of vehicles wait less than this time.",
                "The longest any single vehicle had to wait.",
            ],
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### System Performance")
        perf_data = {
            "Metric": [
                "Mean Time in System",
                "Average Queue Length",
                "Maximum Queue Length",
                "Average Dispenser Utilisation",
                "Throughput",
            ],
            "Value": [
                f"{metrics.avg_time_in_system:.3f} min",
                f"{metrics.avg_queue_length:.3f}",
                str(metrics.max_queue_length),
                f"{metrics.avg_utilisation:.1%}",
                f"{metrics.throughput_per_hour:.1f} veh/hr",
            ],
            "Explanation": [
                "Average total time from arrival to departure.",
                "Time-weighted average number of vehicles in the queue.",
                "Peak number of vehicles waiting simultaneously.",
                "Fraction of time dispensers are actively serving.",
                "Number of vehicles served per hour.",
            ],
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Little's Law Verification")
    L_sim = metrics.avg_queue_length
    lam_per_min = arrival_rate / 60.0
    W_sim = metrics.avg_waiting_time
    L_theory = lam_per_min * W_sim

    littles_data = {
        "Quantity": ["L (avg vehicles in queue)", "λ (arrival rate)", "W (avg waiting time)", "λ × W (predicted L)"],
        "Value": [f"{L_sim:.3f}", f"{lam_per_min:.4f} veh/min", f"{W_sim:.3f} min", f"{L_theory:.3f}"],
    }
    st.dataframe(pd.DataFrame(littles_data), use_container_width=True, hide_index=True)
    if L_sim > 0:
        error_pct = abs(L_sim - L_theory) / L_sim * 100
        st.info(
            f"Little's Law states L = λW. Simulated L = {L_sim:.3f}, "
            f"Predicted L = {L_theory:.3f} (difference: {error_pct:.1f}%). "
            "Small differences are expected due to transient warm-up effects and finite simulation duration."
        )

# ════════════════════════════════════════
# TAB 5 — Scenario Comparison
# ════════════════════════════════════════
with tabs[4]:
    st.markdown("### Scenario Comparison")
    st.markdown("Compare how different numbers of dispensers affect performance.")

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        sc_arrival = st.slider(
            "Arrival Rate for Comparison (veh/hr)",
            1.0, 120.0, arrival_rate, 1.0, key="sc_arrival",
        )
    with sc_col2:
        sc_reps = st.slider(
            "Repetitions per Scenario",
            1, 10, 3, key="sc_reps",
        )

    scenarios = []
    for nd in range(1, 5):
        sc_cfg = SimulationConfig(
            num_dispensers=nd,
            arrival_rate=sc_arrival,
            avg_service_time=avg_service_time,
            service_time_std=service_time_std,
            min_service_time=config.min_service_time,
            max_service_time=config.max_service_time,
            simulation_duration=config.simulation_duration,
            random_seed=config.random_seed,
        )
        scenarios.append(ScenarioConfig(name=f"Scenario {chr(65+nd-1)} ({nd} disp.)", config=sc_cfg))

    if st.button("🔄 Run Scenario Comparison", key="run_scenarios", type="primary"):
        with st.spinner("Running scenarios..."):
            sc_results = ScenarioRunner.run_multiple(scenarios, repetitions=sc_reps)
            st.session_state["sc_results"] = sc_results

    if "sc_results" in st.session_state:
        sc_results = st.session_state["sc_results"]

        # Summary table
        summary_rows = []
        for sr in sc_results:
            summary_rows.append({
                "Scenario": sr.name,
                "Dispensers": sr.config.num_dispensers,
                "Avg Wait (min)": f"{sr.metrics.avg_waiting_time:.2f}",
                "Max Queue": sr.metrics.max_queue_length,
                "Utilisation": f"{sr.metrics.avg_utilisation:.1%}",
                "Throughput (veh/hr)": f"{sr.metrics.throughput_per_hour:.1f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(scenario_waiting_time_chart(sc_results), use_container_width=True)
        with c2:
            st.plotly_chart(scenario_queue_length_chart(sc_results), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(scenario_utilisation_chart(sc_results), use_container_width=True)
        with c4:
            st.plotly_chart(scenario_throughput_chart(sc_results), use_container_width=True)

        # Auto-generated summary
        st.markdown("---")
        summary_text = ScenarioRunner.generate_comparison_summary(sc_results)
        st.markdown(summary_text)

# ════════════════════════════════════════
# TAB 6 — What-If Analysis
# ════════════════════════════════════════
with tabs[5]:
    st.markdown("### What-If Analysis")
    st.markdown("Experiment with parameters to see how they affect queue performance.")

    wi_col1, wi_col2 = st.columns(2)

    with wi_col1:
        wi_param = st.selectbox(
            "Parameter to vary",
            ["Arrival Rate (veh/hr)", "Service Time (min)", "Number of Dispensers"],
            key="wi_param",
        )

    with wi_col2:
        wi_steps = st.slider("Number of steps", 3, 10, 5, key="wi_steps")

    if st.button("🔬 Run What-If Analysis", key="run_whatif", type="primary"):
        with st.spinner("Running what-if simulations..."):
            wi_results = []

            if wi_param == "Arrival Rate (veh/hr)":
                values = np.linspace(5, 60, wi_steps)
                for val in values:
                    wi_cfg = SimulationConfig(
                        num_dispensers=num_dispensers,
                        arrival_rate=float(val),
                        avg_service_time=avg_service_time,
                        service_time_std=service_time_std,
                        min_service_time=config.min_service_time,
                        max_service_time=config.max_service_time,
                        simulation_duration=config.simulation_duration,
                        random_seed=config.random_seed,
                    )
                    sim = PetrolPumpSimulation(wi_cfg)
                    res = sim.run()
                    m = compute_metrics(res)
                    wi_results.append({"param": float(val), "metrics": m, "config": wi_cfg})
            elif wi_param == "Service Time (min)":
                values = np.linspace(2, 15, wi_steps)
                for val in values:
                    wi_cfg = SimulationConfig(
                        num_dispensers=num_dispensers,
                        arrival_rate=arrival_rate,
                        avg_service_time=float(val),
                        service_time_std=service_time_std,
                        min_service_time=max(1.0, float(val) - 3*service_time_std),
                        max_service_time=float(val) + 3*service_time_std,
                        simulation_duration=config.simulation_duration,
                        random_seed=config.random_seed,
                    )
                    sim = PetrolPumpSimulation(wi_cfg)
                    res = sim.run()
                    m = compute_metrics(res)
                    wi_results.append({"param": float(val), "metrics": m, "config": wi_cfg})
            else:  # Number of Dispensers
                values = list(range(1, wi_steps + 1))
                for val in values:
                    wi_cfg = SimulationConfig(
                        num_dispensers=int(val),
                        arrival_rate=arrival_rate,
                        avg_service_time=avg_service_time,
                        service_time_std=service_time_std,
                        min_service_time=config.min_service_time,
                        max_service_time=config.max_service_time,
                        simulation_duration=config.simulation_duration,
                        random_seed=config.random_seed,
                    )
                    sim = PetrolPumpSimulation(wi_cfg)
                    res = sim.run()
                    m = compute_metrics(res)
                    wi_results.append({"param": float(val), "metrics": m, "config": wi_cfg})

            st.session_state["wi_results"] = wi_results
            st.session_state["wi_param_name"] = wi_param

    if "wi_results" in st.session_state:
        wi_results = st.session_state["wi_results"]
        param_name = st.session_state["wi_param_name"]

        params = [w["param"] for w in wi_results]
        avg_waits = [w["metrics"].avg_waiting_time for w in wi_results]
        max_qlens = [w["metrics"].max_queue_length for w in wi_results]
        utils = [w["metrics"].avg_utilisation * 100 for w in wi_results]
        throughputs = [w["metrics"].throughput_per_hour for w in wi_results]

        import plotly.graph_objects as go
        from visualization.charts import ACCENT, BG_DARK, TEXT_COLOR, GRID_COLOR, COMMON_LAYOUT

        c1, c2 = st.columns(2)
        with c1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=params, y=avg_waits, mode="lines+markers",
                                      line=dict(color=ACCENT, width=2), marker=dict(size=8)))
            fig1.update_layout(title="Avg Waiting Time", xaxis_title=param_name,
                               yaxis_title="Minutes", **COMMON_LAYOUT)
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=params, y=max_qlens, mode="lines+markers",
                                      line=dict(color="#3498db", width=2), marker=dict(size=8)))
            fig2.update_layout(title="Max Queue Length", xaxis_title=param_name,
                               yaxis_title="Vehicles", **COMMON_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=params, y=utils, mode="lines+markers",
                                      line=dict(color="#f39c12", width=2), marker=dict(size=8)))
            fig3.update_layout(title="Dispenser Utilisation", xaxis_title=param_name,
                               yaxis_title="Utilisation (%)", **COMMON_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=params, y=throughputs, mode="lines+markers",
                                      line=dict(color="#e74c3c", width=2), marker=dict(size=8)))
            fig4.update_layout(title="Throughput", xaxis_title=param_name,
                               yaxis_title="Vehicles/Hour", **COMMON_LAYOUT)
            st.plotly_chart(fig4, use_container_width=True)

        # Stability warnings
        unstable_points = []
        for w in wi_results:
            c = w["config"]
            mu = 60.0 / c.avg_service_time
            rho = c.arrival_rate / (c.num_dispensers * mu)
            if rho >= 1:
                unstable_points.append(w["param"])

        if unstable_points:
            st.warning(
                f"⚠️ The system becomes unstable (ρ ≥ 1) at: "
                f"{', '.join(f'{p:.1f}' for p in unstable_points)}. "
                "At these points, queues grow without bound."
            )

# ════════════════════════════════════════
# TAB 7 — Mathematical Model
# ════════════════════════════════════════
with tabs[6]:
    st.markdown("### Mathematical Foundation")

    st.markdown("""
#### 1. Poisson Arrival Process

Vehicles arrive at the petrol station according to a **Poisson process** with rate λ (vehicles per hour).
The number of arrivals in any time interval of length *t* follows a Poisson distribution:

> P(N(t) = k) = (λt)ᵏ · e⁻λᵗ / k!

The **inter-arrival times** between consecutive vehicles are exponentially distributed:

> f(t) = λ · e⁻λᵗ, for t ≥ 0

We generate inter-arrival times using the **inverse transform method**:

> Inter-arrival time = −ln(U) / λ

where U is a uniform random number on (0, 1).

---

#### 2. Service Time Distribution

Each vehicle requires a random amount of time for fueling. We model this with a
**truncated normal distribution** bounded between a minimum and maximum service time:

> Service time ~ TruncatedNormal(μ, σ², a, b)

Where:
- μ = average service time
- σ = standard deviation
- a = minimum service time
- b = maximum service time

This ensures realistic service times that cannot be negative or unreasonably long.

---

#### 3. Multi-Server Queue Model (M/G/c)

The petrol station is modelled as an **M/G/c queue**:
- **M** — Markovian (Poisson) arrivals
- **G** — General (truncated normal) service times
- **c** — Number of fuel dispensers (servers)

Queue discipline: **First Come, First Served (FCFS)**.

---

#### 4. Traffic Intensity

The traffic intensity (server utilisation) is:

> ρ = λ / (c × μ)

Where:
- λ = arrival rate (vehicles per hour)
- c = number of dispensers
- μ = service rate per dispenser (vehicles per hour)

**Stability condition**: The queue is stable (does not grow without bound) only when ρ < 1.

When ρ ≥ 1, the arrival rate exceeds the system's service capacity, and queues will
grow indefinitely over time.

---

#### 5. Little's Law

**Little's Law** relates three fundamental quantities:

> **L = λ · W**

Where:
- L = average number of vehicles in the system (or queue)
- λ = average arrival rate
- W = average time spent in the system (or queue)

This relationship holds for **any** stable queueing system regardless of the arrival
distribution, service distribution, or number of servers.

---

#### 6. Discrete-Event Simulation

Rather than solving the queue mathematically (which is only tractable for simple models),
we use **discrete-event simulation (DES)**:

1. **Events** (arrivals, service starts, departures) are scheduled on a timeline.
2. The simulation clock jumps from one event to the next.
3. State variables (queue length, server status) are updated at each event.
4. Statistics are collected throughout the run.

We use the **SimPy** library, which provides a process-based DES framework in Python.

---

#### 7. Assumptions and Limitations

| Assumption | Justification |
|---|---|
| Poisson arrivals | Standard model for random, independent arrivals |
| Independent service times | Each vehicle's fueling is independent |
| Single queue | All vehicles share one FCFS queue |
| No vehicle abandonment | Vehicles never leave without fueling |
| No breakdowns | Dispensers do not fail during simulation |
| Infinite queue capacity | No limit on how many vehicles can wait |
| Steady-state analysis | Transient warm-up effects are not removed |

These simplifications make the model tractable while capturing the essential dynamics
of queue formation at a petrol station.
""")

