"""Streamlit dashboard page renderers."""

import streamlit as st

from simulation.metrics import SimulationMetrics


def render_kpi_cards(metrics: SimulationMetrics) -> None:
    """Render the KPI overview cards."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🚗 Total Arrived", metrics.total_arrived)
    with c2:
        st.metric("✅ Total Served", metrics.total_served)
    with c3:
        st.metric("⏱️ Avg Wait (min)", f"{metrics.avg_waiting_time:.2f}")
    with c4:
        st.metric("⏱️ Max Wait (min)", f"{metrics.max_waiting_time:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("📊 Avg Queue Len", f"{metrics.avg_queue_length:.2f}")
    with c6:
        st.metric("📊 Max Queue Len", metrics.max_queue_length)
    with c7:
        st.metric("⛽ Avg Utilisation", f"{metrics.avg_utilisation:.1%}")
    with c8:
        st.metric("🕐 Avg Time in System", f"{metrics.avg_time_in_system:.2f} min")

