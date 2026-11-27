"""
Petrol Pump Queue Simulation and Modelling Studio
=================================================
A discrete-event simulation application built with Streamlit, SimPy, and Plotly.
"""

import sys
import os

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from simulation.engine import PetrolPumpSimulation, SimulationConfig
from simulation.metrics import compute_metrics
from simulation.scenarios import ScenarioRunner, ScenarioConfig
from visualization.charts import (
    queue_length_chart,
    arrivals_and_departures_chart,
    waiting_time_distribution_chart,
    dispenser_utilisation_chart,
    scenario_waiting_time_chart,
    scenario_queue_length_chart,
    scenario_utilisation_chart,
    scenario_throughput_chart,
    COMMON_LAYOUT,
    COLOR_EMERALD,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_AMBER,
    COLOR_ROSE,
)
from visualization.dashboard import render_kpi_cards, render_live_station
from utils.helpers import vehicles_to_dataframe, traffic_intensity

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PetroFlow • Petrol Pump Queue Simulation",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Ultra-Modern Executive Glassmorphic CSS Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #080c14;
        background-image: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.06) 0%, rgba(8, 12, 20, 1) 75%);
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d121f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Modern Glassmorphic Cards */
    .kpi-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        transform: translateY(-2px);
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .kpi-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        font-size: 1.1rem;
    }
    .icon-emerald { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    .icon-blue    { background: rgba(59, 130, 246, 0.15); color: #3b82f6; }
    .icon-amber   { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .icon-purple  { background: rgba(139, 92, 246, 0.15); color: #8b5cf6; }
    .kpi-tag {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .kpi-unit {
        font-size: 0.95rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        margin-top: 4px;
    }
    .kpi-footer {
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #cbd5e1;
    }

    /* Badges & Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-green { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-amber { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .pill-rose  { background: rgba(239, 68, 68, 0.18);  color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .pill-blue  { background: rgba(59, 130, 246, 0.18); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }

    /* Custom Header Bar */
    .hero-banner {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(15, 23, 42, 0.6) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-desc {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: #064e3b !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }

    /* Station Visualizer Components */
    .station-terminal-header {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .pulse-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .station-chip {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .chip-blue    { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .chip-amber   { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .chip-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

    .lane-container {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px;
        min-height: 440px;
    }
    .lane-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.04em;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .lane-badge {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
    }

    .queue-vehicle-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 9px 12px;
        margin-bottom: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .time-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: #fbbf24;
    }

    .dispenser-box {
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .bay-active {
        background: linear-gradient(145deg, rgba(14, 165, 233, 0.12), rgba(15, 23, 42, 0.8));
        border: 1.5px solid rgba(14, 165, 233, 0.45);
        box-shadow: 0 0 15px rgba(14, 165, 233, 0.15);
    }
    .bay-idle {
        background: rgba(30, 41, 59, 0.4);
        border: 1px dashed rgba(255, 255, 255, 0.1);
    }
    .bay-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .bay-num {
        font-size: 0.75rem;
        font-weight: 800;
        color: #94a3b8;
        letter-spacing: 0.05em;
    }
    .bay-status {
        font-size: 0.7rem;
        font-weight: 700;
    }
    .status-fueling { color: #38bdf8; }
    .status-idle    { color: #64748b; }
    .bay-vehicle {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-weight: 600;
        color: #ffffff;
    }
    .vehicle-id-pill {
        font-size: 0.7rem;
        padding: 2px 6px;
        border-radius: 4px;
        background: rgba(14, 165, 233, 0.2);
        color: #38bdf8;
    }
    .progress-container {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 6px;
    }
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #0ea5e9, #10b981);
        border-radius: 999px;
        transition: width 0.3s ease;
    }
    .bay-timer-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: #94a3b8;
    }
    .idle-graphic {
        text-align: center;
        font-size: 1.8rem;
        opacity: 0.35;
        padding: 6px 0;
    }

    .exit-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .empty-lane-state {
        text-align: center;
        padding: 60px 16px;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Quick Preset Configurations Handler
# ──────────────────────────────────────────────
PRESETS = {
    "Urban Commute (Balanced)": {
        "num_dispensers": 3,
        "arrival_rate": 20.0,
        "avg_service_time": 5.0,
        "service_time_std": 1.5,
        "sim_duration": 120,
    },
    "Morning Peak Rush": {
        "num_dispensers": 3,
        "arrival_rate": 34.0,
        "avg_service_time": 4.5,
        "service_time_std": 1.2,
        "sim_duration": 120,
    },
    "Highway Mega-Station": {
        "num_dispensers": 6,
        "arrival_rate": 65.0,
        "avg_service_time": 4.0,
        "service_time_std": 1.0,
        "sim_duration": 180,
    },
    "Severe Bottleneck (Stress Test)": {
        "num_dispensers": 1,
        "arrival_rate": 22.0,
        "avg_service_time": 5.5,
        "service_time_std": 1.5,
        "sim_duration": 90,
    },
}

# ──────────────────────────────────────────────
# Sidebar: Parameter Control Center
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 1.8rem;">⛽</span>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #ffffff; letter-spacing: -0.02em;">PETROFLOW</div>
            <div style="font-size: 0.72rem; color: #10b981; font-weight: 600;">SIMULATION ENGINE v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Quick Scenario Presets
    st.markdown("#### ⚡ Quick Presets")
    preset_choice = st.selectbox(
        "Load Preset Scenario",
        ["Select a preset...", *list(PRESETS.keys())],
        index=0,
        label_visibility="collapsed",
    )

    if preset_choice != "Select a preset...":
        p = PRESETS[preset_choice]
        st.session_state["num_dispensers"] = p["num_dispensers"]
        st.session_state["arrival_rate"] = p["arrival_rate"]
        st.session_state["avg_service_time"] = p["avg_service_time"]
        st.session_state["service_time_std"] = p["service_time_std"]
        st.session_state["sim_duration"] = p["sim_duration"]

    st.markdown("#### 🎛️ Station Parameters")

    num_dispensers = st.slider(
        "Fuel Dispensers (c)",
        min_value=1, max_value=10,
        value=st.session_state.get("num_dispensers", 3),
        help="Number of parallel fuel pumps available at the station.",
    )

    arrival_rate = st.slider(
        "Arrival Rate λ (veh/hr)",
        min_value=1.0, max_value=120.0,
        value=st.session_state.get("arrival_rate", 20.0),
        step=1.0,
        help="Average rate of incoming vehicles per hour (Poisson process).",
    )

    avg_service_time = st.slider(
        "Avg Service Time 1/μ (min)",
        min_value=1.0, max_value=20.0,
        value=st.session_state.get("avg_service_time", 5.0),
        step=0.5,
        help="Mean fueling and payment duration per vehicle.",
    )

    service_time_std = st.slider(
        "Service Std Dev σ (min)",
        min_value=0.0, max_value=5.0,
        value=st.session_state.get("service_time_std", 1.5),
        step=0.1,
        help="Variability in service times (truncated normal).",
    )

    st.markdown("#### ⏱️ Simulation Settings")
    sim_duration = st.slider(
        "Run Duration (min)",
        min_value=30, max_value=480,
        value=st.session_state.get("sim_duration", 120),
        step=10,
        help="Simulated time span in minutes.",
    )

    random_seed = st.number_input(
        "Stochastic Seed",
        min_value=0, max_value=99999,
        value=st.session_state.get("random_seed", 42),
        help="RNG seed for exact scientific reproducibility.",
    )

    st.markdown("---")

    # Real-Time Traffic Intensity Calculation
    service_rate_per_pump = 60.0 / avg_service_time  # veh/hr
    total_capacity = num_dispensers * service_rate_per_pump
    rho = traffic_intensity(arrival_rate, service_rate_per_pump, num_dispensers)

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; margin-bottom: 14px;">
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8; margin-bottom:4px;">
            <span>Traffic Intensity (ρ)</span>
            <span style="font-weight:700; color:{'#34d399' if rho < 0.7 else ('#fbbf24' if rho < 1.0 else '#f87171')}">{rho:.3f}</span>
        </div>
        <div style="width:100%; height:6px; background:rgba(255,255,255,0.1); border-radius:999px; overflow:hidden;">
            <div style="width:{min(100.0, rho * 100):.0f}%; height:100%; background:{'#10b981' if rho < 0.7 else ('#f59e0b' if rho < 1.0 else '#ef4444')};"></div>
        </div>
        <div style="font-size:0.72rem; color:#94a3b8; margin-top:6px;">
            Arrival Demand: <b>{arrival_rate:.0f}</b> / Capacity: <b>{total_capacity:.0f}</b> veh/hr
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Run Simulation Engine", type="primary", use_container_width=True):
        st.session_state["force_rerun"] = True

# ──────────────────────────────────────────────
# Execute Discrete-Event Simulation
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
def execute_simulation(nd, ar, ast_, sts, mn, mx, sd, rs):
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
    res = sim.run()
    met = compute_metrics(res)
    return res, met, cfg


result, metrics, _ = execute_simulation(
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
# Hero Banner
# ──────────────────────────────────────────────
if rho < 0.7:
    status_indicator = '<span class="status-pill pill-green">🟢 Normal Flow (ρ = {:.2f})</span>'.format(rho)
    status_desc = "Optimal operating state: Queue delays are minimal and customer turnaround is fast."
elif rho < 1.0:
    status_indicator = '<span class="status-pill pill-amber">🟡 Heavy Traffic (ρ = {:.2f})</span>'.format(rho)
    status_desc = "High utilization: Queues form intermittently during arrival bursts."
else:
    status_indicator = '<span class="status-pill pill-rose">🔴 System Saturated (ρ = {:.2f})</span>'.format(rho)
    status_desc = "Unstable condition: Demand exceeds service capacity; queues expand indefinitely."

st.markdown(f"""
<div class="hero-banner">
    <div>
        <h1 class="hero-title">
            <span>⛽ PetroFlow</span>
            <span style="font-weight: 400; color: #94a3b8; font-size: 1.25rem;">| Queue Simulation &amp; Modelling Studio</span>
        </h1>
        <p class="hero-desc">
            SimPy Discrete-Event Queue Engine • M/G/{num_dispensers} Multi-Server Stochastic Process • FCFS Service Discipline
        </p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
        {status_indicator}
        <span style="font-size: 0.75rem; color: #94a3b8;">{status_desc}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Interactive Tabs Architecture
# ──────────────────────────────────────────────
tabs = st.tabs([
    "📊 KPI Overview",
    "🏗️ Live Station Visualizer",
    "📈 Queue & Traffic Analytics",
    "📐 Statistical Deep-Dive",
    "🔀 Multi-Scenario Benchmark",
    "🔮 Sensitivity / What-If",
    "📖 Mathematical Model",
])

# ════════════════════════════════════════
# TAB 1 — KPI Overview & Vehicle Log
# ════════════════════════════════════════
with tabs[0]:
    render_kpi_cards(metrics, rho)

    # Operational Health Card
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    c_sum1, c_sum2 = st.columns([1.5, 1])
    with c_sum1:
        st.markdown(f"""
        <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px 22px;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                💡 Operational Diagnosis &amp; System Health
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                • <b>Arrival Rate:</b> {config.arrival_rate:.1f} vehicles/hour with average service duration of {config.avg_service_time:.1f} minutes.<br>
                • <b>Effective Service Capacity:</b> {total_capacity:.1f} vehicles/hour across {config.num_dispensers} dispenser(s).<br>
                • <b>Turnaround Efficiency:</b> Average customer stays in the station for <b>{metrics.avg_time_in_system:.2f} minutes</b> (waiting: {metrics.avg_waiting_time:.2f}m, fueling: {metrics.avg_time_in_system - metrics.avg_waiting_time:.2f}m).<br>
                • <b>Queue Congestion Risk:</b> {'Low — dispensers handle traffic smoothly.' if rho < 0.7 else ('Moderate — queues accumulate during micro-peaks.' if rho < 1.0 else 'Critical — arrival rate exceeds capacity. Adding another dispenser is strongly recommended.')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_sum2:
        st.markdown(f"""
        <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px 22px;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                ⚡ Fast Insights
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.55;">
                <span style="color:#10b981;">✔</span> <b>Completed Vehicles:</b> {metrics.total_served:,} of {metrics.total_arrived:,}<br>
                <span style="color:#38bdf8;">✔</span> <b>95% of Customers Wait Under:</b> {metrics.p95_waiting_time:.2f} min<br>
                <span style="color:#f59e0b;">✔</span> <b>Peak Queue Recorded:</b> {metrics.max_queue_length} vehicles<br>
                <span style="color:#8b5cf6;">✔</span> <b>Hourly Station Throughput:</b> {metrics.throughput_per_hour:.1f} veh/hr
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📋 Vehicle Event Ledger")
    
    df_vehicles = vehicles_to_dataframe(result.vehicles)
    
    # Filter controls
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search_status = st.multiselect(
            "Filter by Lifecycle Status",
            ["completed", "fueling", "waiting"],
            default=["completed", "fueling", "waiting"],
        )
    with col_f2:
        # CSV Export
        csv_data = df_vehicles.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Event Ledger (CSV)",
            data=csv_data,
            file_name="petrol_pump_simulation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if search_status:
        filtered_df = df_vehicles[df_vehicles["Status"].isin(search_status)]
    else:
        filtered_df = df_vehicles

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=340,
    )

# ════════════════════════════════════════
# TAB 2 — Live Station Layout Visualizer
# ════════════════════════════════════════
with tabs[1]:
    st.markdown("### 🏗️ Forecourt Live Simulation Visualizer")
    st.markdown(
        "Scrub through the simulation timeline to watch queue formation, "
        "dispenser allocation, and customer departure dynamics in real-time."
    )

    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        time_scrubber = st.slider(
            "Timeline Time (T in minutes)",
            min_value=0.0,
            max_value=float(config.simulation_duration),
            value=float(config.simulation_duration),
            step=1.0,
            key="forecourt_timeline_slider",
        )
    with t_col2:
        st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        if st.button("⏮ Jump to T=15m", use_container_width=True):
            st.session_state["forecourt_timeline_slider"] = 15.0

    render_live_station(result, time_scrubber, config.num_dispensers)

# ════════════════════════════════════════
# TAB 3 — Queue & Traffic Analytics
# ════════════════════════════════════════
with tabs[2]:
    st.markdown("### 📈 Forecourt Queue & Velocity Analytics")
    st.markdown("Interactive analytical charts examining stochastic queue build-up and service velocity.")

    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.plotly_chart(queue_length_chart(result.queue_length_over_time), use_container_width=True)
    with q_col2:
        st.plotly_chart(arrivals_and_departures_chart(result), use_container_width=True)

    q_col3, q_col4 = st.columns(2)
    with q_col3:
        st.plotly_chart(waiting_time_distribution_chart(result), use_container_width=True)
    with q_col4:
        st.plotly_chart(dispenser_utilisation_chart(metrics, config.num_dispensers), use_container_width=True)

# ════════════════════════════════════════
# TAB 4 — Statistical Deep-Dive
# ════════════════════════════════════════
with tabs[3]:
    st.markdown("### 📐 Statistical Deep-Dive & Queueing Theory Verification")

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.markdown("#### Waiting Time Dispersion Metrics")
        wait_df = pd.DataFrame({
            "Metric": [
                "Mean Waiting Time (Wq)",
                "Median Waiting Time",
                "Std Dev of Waiting Time",
                "95th Percentile Wait",
                "Maximum Wait Experienced",
            ],
            "Simulation Value": [
                f"{metrics.avg_waiting_time:.3f} min",
                f"{metrics.median_waiting_time:.3f} min",
                f"{metrics.std_waiting_time:.3f} min",
                f"{metrics.p95_waiting_time:.3f} min",
                f"{metrics.max_waiting_time:.3f} min",
            ],
            "Theoretical Significance": [
                "Expected time a vehicle remains in the queue prior to nozzle insertion.",
                "50% of vehicles experienced lower wait times than this value.",
                "Degree of dispersion and predictability in service turnaround.",
                "Service SLA benchmark: 95% of all drivers wait less than this.",
                "Worst-case bottleneck duration recorded in the run.",
            ]
        })
        st.dataframe(wait_df, use_container_width=True, hide_index=True)

    with stat_col2:
        st.markdown("#### Forecourt System Performance")
        sys_df = pd.DataFrame({
            "Metric": [
                "Mean Time in Station (W)",
                "Time-Weighted Queue Length (Lq)",
                "Peak Queue Depth Recorded",
                "Aggregate Dispenser Utilization",
                "Effective Service Throughput",
            ],
            "Simulation Value": [
                f"{metrics.avg_time_in_system:.3f} min",
                f"{metrics.avg_queue_length:.3f} vehicles",
                f"{metrics.max_queue_length} vehicles",
                f"{metrics.avg_utilisation:.1%}",
                f"{metrics.throughput_per_hour:.1f} veh/hr",
            ],
            "Theoretical Significance": [
                "Total dwell time from station entry to final departure.",
                "Integral of queue length over total simulated duration.",
                "Highest instantaneous backlog observed in the approach lane.",
                "Percentage of available pump capacity actively fueling vehicles.",
                "Completed vehicle processing rate per elapsed hour.",
            ]
        })
        st.dataframe(sys_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # Little's Law Empirical Proof
    st.markdown("#### ⚖️ Little's Law Empirical Validation: L = λ · W")
    
    lam_min = config.arrival_rate / 60.0
    L_sim = metrics.avg_queue_length
    W_sim = metrics.avg_waiting_time
    L_pred = lam_min * W_sim
    diff_pct = abs(L_sim - L_pred) / max(0.01, L_sim) * 100.0

    st.markdown(f"""
    <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">
                Formula: L<sub>q</sub> = λ · W<sub>q</sub>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                Where <b>λ</b> = {lam_min:.4f} veh/min, <b>W<sub>q</sub></b> = {W_sim:.3f} min
            </div>
        </div>
        <div style="display: flex; gap: 20px;">
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; color: #94a3b8;">Simulated L<sub>q</sub></div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #10b981;">{L_sim:.3f}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; color: #94a3b8;">Predicted λ · W<sub>q</sub></div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #38bdf8;">{L_pred:.3f}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; color: #94a3b8;">Discrepancy</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: {'#34d399' if diff_pct < 10 else '#fbbf24'};">{diff_pct:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Note: Residual divergence under 10% is expected in finite-horizon simulations due to transient initialization and discrete event boundaries.")

# ════════════════════════════════════════
# TAB 5 — Multi-Scenario Benchmark
# ════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🔀 Multi-Scenario Benchmark & Capacity Planning")
    st.markdown(
        "Evaluate station performance across different numbers of fuel dispensers (1 to 4) "
        "under identical arrival pressure to determine the optimal capital allocation."
    )

    sc_c1, sc_c2 = st.columns([2, 1])
    with sc_c1:
        benchmark_arrival = st.slider(
            "Benchmark Arrival Demand λ (veh/hr)",
            min_value=5.0, max_value=80.0,
            value=float(config.arrival_rate),
            step=1.0,
            key="benchmark_arr_rate",
        )
    with sc_c2:
        benchmark_reps = st.slider(
            "Replications per Configuration",
            min_value=1, max_value=5, value=2,
            help="Average over multiple stochastic runs to smooth variance.",
        )

    # Build 4 comparative scenarios
    scenarios = []
    for nd in [1, 2, 3, 4]:
        sc_cfg = SimulationConfig(
            num_dispensers=nd,
            arrival_rate=benchmark_arrival,
            avg_service_time=config.avg_service_time,
            service_time_std=config.service_time_std,
            min_service_time=config.min_service_time,
            max_service_time=config.max_service_time,
            simulation_duration=config.simulation_duration,
            random_seed=config.random_seed,
        )
        scenarios.append(ScenarioConfig(name=f"{nd} Pump{'s' if nd > 1 else ''}", config=sc_cfg))

    sc_results = ScenarioRunner.run_multiple(scenarios, repetitions=benchmark_reps)

    # Results Table
    sc_summary_data = []
    for sr in sc_results:
        rho_sc = traffic_intensity(sr.config.arrival_rate, 60.0 / sr.config.avg_service_time, sr.config.num_dispensers)
        sc_summary_data.append({
            "Configuration": sr.name,
            "Pumps": sr.config.num_dispensers,
            "Traffic Intensity (ρ)": f"{rho_sc:.3f}",
            "Status": "🟢 Stable" if rho_sc < 0.8 else ("🟡 Congested" if rho_sc < 1.0 else "🔴 Saturated"),
            "Avg Wait Time": f"{sr.metrics.avg_waiting_time:.2f} min",
            "Max Queue Depth": f"{sr.metrics.max_queue_length} veh",
            "Pump Utilization": f"{sr.metrics.avg_utilisation:.1%}",
            "Throughput": f"{sr.metrics.throughput_per_hour:.1f} veh/hr",
        })
    st.dataframe(pd.DataFrame(sc_summary_data), use_container_width=True, hide_index=True)

    # Comparison Grid Charts
    sc_g1, sc_g2 = st.columns(2)
    with sc_g1:
        st.plotly_chart(scenario_waiting_time_chart(sc_results), use_container_width=True)
    with sc_g2:
        st.plotly_chart(scenario_queue_length_chart(sc_results), use_container_width=True)

    sc_g3, sc_g4 = st.columns(2)
    with sc_g3:
        st.plotly_chart(scenario_utilisation_chart(sc_results), use_container_width=True)
    with sc_g4:
        st.plotly_chart(scenario_throughput_chart(sc_results), use_container_width=True)

    # Recommendation
    st.markdown("---")
    st.markdown(ScenarioRunner.generate_comparison_summary(sc_results))

# ════════════════════════════════════════
# TAB 6 — Sensitivity / What-If Studio
# ════════════════════════════════════════
with tabs[5]:
    st.markdown("### 🔮 Sensitivity & What-If Studio")
    st.markdown("Stress-test parameter variations to locate critical inflection points and system saturation boundaries.")

    wi_col1, wi_col2 = st.columns(2)
    with wi_col1:
        param_to_vary = st.selectbox(
            "Target Sensitivity Parameter",
            ["Arrival Rate λ (veh/hr)", "Service Time μ⁻¹ (min)", "Number of Dispensers (c)"],
        )
    with wi_col2:
        sweep_resolution = st.slider("Sweep Resolution Steps", min_value=4, max_value=8, value=6)

    # Run parameter sweep
    if param_to_vary == "Arrival Rate λ (veh/hr)":
        sweep_vals = np.linspace(10.0, 60.0, sweep_resolution)
    elif param_to_vary == "Service Time μ⁻¹ (min)":
        sweep_vals = np.linspace(2.0, 12.0, sweep_resolution)
    else:
        sweep_vals = np.arange(1, min(11, sweep_resolution + 2))

    wi_data = []
    for val in sweep_vals:
        if param_to_vary == "Arrival Rate λ (veh/hr)":
            cfg_s = SimulationConfig(
                num_dispensers=config.num_dispensers,
                arrival_rate=float(val),
                avg_service_time=config.avg_service_time,
                service_time_std=config.service_time_std,
                min_service_time=config.min_service_time,
                max_service_time=config.max_service_time,
                simulation_duration=config.simulation_duration,
                random_seed=config.random_seed,
            )
        elif param_to_vary == "Service Time μ⁻¹ (min)":
            cfg_s = SimulationConfig(
                num_dispensers=config.num_dispensers,
                arrival_rate=config.arrival_rate,
                avg_service_time=float(val),
                service_time_std=min(config.service_time_std, float(val) / 2.0),
                min_service_time=max(1.0, float(val) - 2.0),
                max_service_time=float(val) + 2.0,
                simulation_duration=config.simulation_duration,
                random_seed=config.random_seed,
            )
        else:
            cfg_s = SimulationConfig(
                num_dispensers=int(val),
                arrival_rate=config.arrival_rate,
                avg_service_time=config.avg_service_time,
                service_time_std=config.service_time_std,
                min_service_time=config.min_service_time,
                max_service_time=config.max_service_time,
                simulation_duration=config.simulation_duration,
                random_seed=config.random_seed,
            )

        sim_s = PetrolPumpSimulation(cfg_s)
        res_s = sim_s.run()
        met_s = compute_metrics(res_s)
        rho_s = traffic_intensity(cfg_s.arrival_rate, 60.0 / cfg_s.avg_service_time, cfg_s.num_dispensers)
        wi_data.append({
            "val": float(val),
            "wait": met_s.avg_waiting_time,
            "queue": met_s.max_queue_length,
            "util": met_s.avg_utilisation * 100.0,
            "throughput": met_s.throughput_per_hour,
            "rho": rho_s,
        })

    # Render Curves
    x_vals = [d["val"] for d in wi_data]
    
    wi_c1, wi_c2 = st.columns(2)
    with wi_c1:
        f1 = go.Figure()
        f1.add_trace(go.Scatter(
            x=x_vals, y=[d["wait"] for d in wi_data],
            mode="lines+markers", line=dict(color=COLOR_CYAN, width=3),
            marker=dict(size=8, color=COLOR_CYAN), name="Avg Wait Time",
        ))
        f1.update_layout(title="<b>Impact on Average Wait Time</b>", xaxis_title=param_to_vary, yaxis_title="Wait Time (min)", **COMMON_LAYOUT)
        st.plotly_chart(f1, use_container_width=True)

    with wi_c2:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(
            x=x_vals, y=[d["queue"] for d in wi_data],
            mode="lines+markers", line=dict(color=COLOR_AMBER, width=3),
            marker=dict(size=8, color=COLOR_AMBER), name="Peak Queue",
        ))
        f2.update_layout(title="<b>Impact on Peak Queue Depth</b>", xaxis_title=param_to_vary, yaxis_title="Vehicles", **COMMON_LAYOUT)
        st.plotly_chart(f2, use_container_width=True)

# ════════════════════════════════════════
# TAB 7 — Mathematical Model & Theory
# ════════════════════════════════════════
with tabs[6]:
    st.markdown("### 📖 Mathematical Foundations & Stochastic Model Formulation")
    
    st.markdown(r"""
#### 1. Stochastic Arrival Process: Poisson Process & Exponential Inter-arrivals
Incoming customer demand is represented as a homogeneous **Poisson Process** with mean arrival rate $\lambda$ (vehicles per hour):

$$P(N(t) = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}, \quad k = 0, 1, 2, \dots$$

The elapsed duration between successive vehicle arrivals (inter-arrival time $\Delta t$) follows an **Exponential Distribution**:

$$f(\Delta t) = \lambda e^{-\lambda \Delta t}, \quad \Delta t \ge 0$$

In our simulation engine, variates are synthesized via the **Inverse-Transform Sampling Method**:

$$\Delta t = -\frac{\ln(U)}{\lambda}, \quad U \sim \text{Uniform}(0, 1)$$

---

#### 2. Service Duration Model: Truncated Normal Distribution
Unlike simple analytical $M/M/c$ models that assume exponential service times, real-world fuel transactions have physical minimums (nozzle insertion, payment authentication) and maximums. We therefore formulate service duration $S$ using a **Truncated Normal Distribution**:

$$f(S; \mu, \sigma, a, b) = \frac{\frac{1}{\sigma}\phi\left(\frac{S - \mu}{\sigma}\right)}{\Phi\left(\frac{b - \mu}{\sigma}\right) - \Phi\left(\frac{a - \mu}{\sigma}\right)}, \quad a \le S \le b$$

Where:
- $\mu$: Configured average service duration
- $\sigma$: Standard deviation (variability)
- $a$: Physical minimum bound ($\ge 1.0$ min)
- $b$: Upper truncation cutoff

---

#### 3. Kendall Notation & Queue Classification
The station is mathematically designated as an **$M/G/c/\infty/\text{FCFS}$** system:
- **$M$**: Markovian (Poisson) arrival distribution
- **$G$**: General (Truncated Normal) service time distribution
- **$c$**: Number of identical fuel dispenser servers
- **$\infty$**: Infinite queue holding capacity
- **$\text{FCFS}$**: First-Come, First-Served queue discipline

---

#### 4. Traffic Intensity & Ergodicity Boundary
The stability parameter $\rho$ (traffic intensity or server utilization) is defined as:

$$\rho = \frac{\lambda}{c \cdot \mu}$$

Where $\mu = \frac{60}{\bar{S}}$ is the hourly service rate per dispenser.

> **Mathematical Stability Criterion:**
> - When **$\rho < 1$**: The queueing system is **ergodic (stable)**. A steady-state probability distribution exists, and queue lengths remain bounded.
> - When **$\rho \ge 1$**: The system is **supercritical (unstable)**. The arrival rate outpaces service capacity; queue length $L(t) \to \infty$ as $t \to \infty$.

---

#### 5. Little's Law
For any stationary queueing system in equilibrium:

$$L = \lambda W \quad \text{and} \quad L_q = \lambda W_q$$

Where:
- $L$: Average number of vehicles in the station
- $L_q$: Average number of vehicles waiting in queue
- $W$: Mean total time in station
- $W_q$: Mean waiting time prior to service start
""")
