"""Streamlit modern dashboard widgets and live station layout visualizer."""

from typing import List
import streamlit as st
from simulation.metrics import SimulationMetrics
from simulation.engine import SimulationResult
from simulation.vehicle import Vehicle, VehicleStatus


def render_kpi_cards(metrics: SimulationMetrics, rho: float = 0.0) -> None:
    """Render executive glassmorphic KPI cards with badges, icons, and micro-metrics."""
    
    # Determine stability badge color
    if rho < 0.7:
        rho_badge = '<span class="status-pill pill-green">Stable (ρ < 0.7)</span>'
    elif rho < 1.0:
        rho_badge = '<span class="status-pill pill-amber">Moderate (ρ < 1.0)</span>'
    else:
        rho_badge = '<span class="status-pill pill-rose">Unstable (ρ ≥ 1.0)</span>'

    # Top Row Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-icon icon-blue">🚗</span>
                <span class="kpi-tag">Demand</span>
            </div>
            <div class="kpi-value">{metrics.total_arrived:,}</div>
            <div class="kpi-label">Total Vehicles Arrived</div>
            <div class="kpi-footer">
                <span class="kpi-sub">Served: <b>{metrics.total_served:,}</b> ({metrics.total_served / max(1, metrics.total_arrived):.1%})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        wait_color = "pill-green" if metrics.avg_waiting_time < 3 else ("pill-amber" if metrics.avg_waiting_time < 8 else "pill-rose")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-icon icon-emerald">⏱️</span>
                <span class="status-pill {wait_color}">{metrics.avg_waiting_time:.1f}m avg</span>
            </div>
            <div class="kpi-value">{metrics.avg_waiting_time:.2f} <span class="kpi-unit">min</span></div>
            <div class="kpi-label">Average Queue Wait</div>
            <div class="kpi-footer">
                <span class="kpi-sub">Worst Case: <b>{metrics.max_waiting_time:.1f}m</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        q_color = "pill-green" if metrics.avg_queue_length < 2 else ("pill-amber" if metrics.avg_queue_length < 5 else "pill-rose")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-icon icon-amber">📊</span>
                <span class="status-pill {q_color}">Peak: {metrics.max_queue_length}</span>
            </div>
            <div class="kpi-value">{metrics.avg_queue_length:.2f} <span class="kpi-unit">veh</span></div>
            <div class="kpi-label">Average Queue Depth</div>
            <div class="kpi-footer">
                <span class="kpi-sub">Peak Congestion: <b>{metrics.max_queue_length} vehicles</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        util_color = "pill-rose" if metrics.avg_utilisation > 0.9 else ("pill-green" if metrics.avg_utilisation >= 0.6 else "pill-blue")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-icon icon-purple">⛽</span>
                {rho_badge}
            </div>
            <div class="kpi-value">{metrics.avg_utilisation:.1%}</div>
            <div class="kpi-label">Dispenser Utilization</div>
            <div class="kpi-footer">
                <span class="kpi-sub">Throughput: <b>{metrics.throughput_per_hour:.1f} veh/hr</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_live_station(result: SimulationResult, time_step: float, num_dispensers: int) -> None:
    """Render an interactive, architectural petrol station diagram at time_step."""
    
    # Categorise vehicles at this specific timeline moment
    waiting: List[Vehicle] = []
    fueling: List[Vehicle] = []
    completed: List[Vehicle] = []

    for v in result.vehicles:
        if v.arrival_time <= time_step:
            if v.service_start_time == 0 or v.service_start_time > time_step:
                waiting.append(v)
            elif v.service_end_time == 0 or v.service_end_time > time_step:
                fueling.append(v)
            else:
                completed.append(v)

    # Sort waiting by arrival time (FCFS queue line)
    waiting.sort(key=lambda x: x.arrival_time)
    
    # Sort completed by departure time descending (most recently departed first)
    completed.sort(key=lambda x: x.departure_time, reverse=True)

    # Vehicle type icons generator based on ID
    def get_veh_badge(v_id: int):
        types = [
            ("🚗 Sedan", "#38bdf8"),
            ("🚙 SUV", "#34d399"),
            ("🏎️ Coupe", "#f43f5e"),
            ("🚐 Van", "#fbbf24"),
            ("🚕 Taxi", "#eab308"),
            ("🏍️ Moto", "#a855f7"),
        ]
        return types[v_id % len(types)]

    # 1. Station Control Room Header
    active_pumps = len(fueling)
    idle_pumps = max(0, num_dispensers - active_pumps)
    
    st.markdown(f"""
    <div class="station-terminal-header">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="pulse-indicator"></div>
                <div>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">
                        TERMINAL ACTIVE VIEW • T = {time_step:.1f} MIN
                    </span>
                    <div style="font-size: 0.8rem; color: #94a3b8;">
                        Interactive Snapshot of Forecourt, Queue Lanes & Dispenser Islands
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 10px; align-items: center;">
                <span class="station-chip chip-blue">⛽ {active_pumps} / {num_dispensers} Pumps Fueling</span>
                <span class="station-chip chip-amber">🚗 {len(waiting)} In Queue</span>
                <span class="station-chip chip-emerald">✅ {len(completed)} Departed</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Main Station Forecourt Visual Grid
    col_queue, col_pumps, col_exit = st.columns([1.1, 2.2, 0.9])

    # --- LEFT LANE: WAITING APPROACH LANE ---
    with col_queue:
        st.markdown("""
        <div class="lane-container">
            <div class="lane-title">
                <span>🛣️ QUEUE APPROACH LANE</span>
                <span class="lane-badge">FCFS Order</span>
            </div>
        """, unsafe_allow_html=True)

        if waiting:
            st.markdown(f'<div style="font-size:0.75rem; color:#94a3b8; margin-bottom:8px;">Head of Queue (First in Line):</div>', unsafe_allow_html=True)
            for idx, v in enumerate(waiting[:6]):  # Display top 6 in queue
                name, color = get_veh_badge(v.vehicle_id)
                current_wait = time_step - v.arrival_time
                st.markdown(f"""
                <div class="queue-vehicle-card" style="border-left: 3px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; color: #f8fafc; font-size: 0.85rem;">#{idx+1} {name} <span style="color:#94a3b8; font-size:0.75rem;">(ID #{v.vehicle_id})</span></span>
                        <span class="time-tag">⏱️ {current_wait:.1f}m wait</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">
                        Arrived at T = {v.arrival_time:.1f} min • Req Service: {v.service_duration:.1f} min
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if len(waiting) > 6:
                st.markdown(f"""
                <div style="text-align: center; padding: 8px; color: #f59e0b; font-size: 0.8rem; font-weight: 500;">
                    + {len(waiting) - 6} additional vehicles further back in line...
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-lane-state">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">✨</div>
                <div style="color: #10b981; font-weight: 600; font-size: 0.9rem;">Queue is Empty!</div>
                <div style="color: #64748b; font-size: 0.75rem;">All incoming vehicles have immediate pump access.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- CENTER: FUEL DISPENSER ISLANDS ---
    with col_pumps:
        st.markdown(f"""
        <div class="lane-container" style="background: rgba(15, 23, 42, 0.75);">
            <div class="lane-title">
                <span>⛽ FORECOURT DISPENSER BAYS</span>
                <span class="lane-badge" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">{num_dispensers} Total Islands</span>
            </div>
        """, unsafe_allow_html=True)

        # Render Dispenser bays in a responsive 2-column or 3-column subgrid
        p_cols = st.columns(2 if num_dispensers <= 4 else 3)

        for d_id in range(num_dispensers):
            with p_cols[d_id % len(p_cols)]:
                # Check if this dispenser is serving someone right now
                serving_v = None
                for v in fueling:
                    if v.dispenser_id == d_id:
                        serving_v = v
                        break

                if serving_v:
                    # Calculate progress
                    elapsed = max(0.0, time_step - serving_v.service_start_time)
                    total_dur = max(0.1, serving_v.service_duration)
                    progress_pct = min(100.0, (elapsed / total_dur) * 100.0)
                    rem_time = max(0.0, total_dur - elapsed)
                    v_name, v_color = get_veh_badge(serving_v.vehicle_id)

                    st.markdown(f"""
                    <div class="dispenser-box bay-active">
                        <div class="bay-header">
                            <span class="bay-num">PUMP #{d_id + 1}</span>
                            <span class="bay-status status-fueling">● FUELING</span>
                        </div>
                        <div class="bay-vehicle">
                            <span style="font-size: 1rem;">{v_name}</span>
                            <span class="vehicle-id-pill">ID #{serving_v.vehicle_id}</span>
                        </div>
                        <div class="progress-container">
                            <div class="progress-bar-fill" style="width: {progress_pct:.0f}%;"></div>
                        </div>
                        <div class="bay-timer-row">
                            <span>{progress_pct:.0f}% Done</span>
                            <span>⏳ {rem_time:.1f}m left</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Dispenser is Idle
                    st.markdown(f"""
                    <div class="dispenser-box bay-idle">
                        <div class="bay-header">
                            <span class="bay-num">PUMP #{d_id + 1}</span>
                            <span class="bay-status status-idle">○ IDLE</span>
                        </div>
                        <div class="idle-graphic">
                            <span>⛽</span>
                        </div>
                        <div style="font-size: 0.72rem; color: #64748b; text-align: center; margin-top: 4px;">
                            Ready for Next Vehicle
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- RIGHT LANE: DEPARTURE & RECENT EXITS ---
    with col_exit:
        st.markdown("""
        <div class="lane-container">
            <div class="lane-title">
                <span>🏁 DEPARTURE EXIT</span>
                <span class="lane-badge">Completed</span>
            </div>
        """, unsafe_allow_html=True)

        if completed:
            for v in completed[:5]:  # Show last 5 departures
                v_name, _ = get_veh_badge(v.vehicle_id)
                st.markdown(f"""
                <div class="exit-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; color:#f8fafc; font-size:0.8rem;">{v_name}</span>
                        <span style="color:#10b981; font-weight:700; font-size:0.75rem;">P#{v.dispenser_id+1}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#94a3b8; margin-top:3px;">
                        <span>Departed: T={v.departure_time:.1f}m</span>
                        <span>Total: {v.time_in_system:.1f}m</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if len(completed) > 5:
                st.markdown(f"""
                <div style="text-align: center; color: #64748b; font-size: 0.75rem; margin-top: 6px;">
                    + {len(completed) - 5} earlier completed vehicles
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-lane-state">
                <div style="font-size: 1.5rem; margin-bottom: 4px;">⏳</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">No Departures Yet</div>
                <div style="color: #64748b; font-size: 0.72rem;">First vehicles are currently being fueled.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
