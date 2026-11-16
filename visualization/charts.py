"""Plotly chart generators for the simulation dashboard."""

from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

from simulation.engine import SimulationResult
from simulation.metrics import SimulationMetrics
from simulation.vehicle import VehicleStatus
from simulation.scenarios import ScenarioResult


# ---- Colour palette ----
ACCENT = "#2ecc71"        # petrol-station green
BG_DARK = "#1a1a2e"       # dark charcoal
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"

COMMON_LAYOUT = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_DARK,
    font=dict(color=TEXT_COLOR, size=12),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    margin=dict(l=40, r=20, t=40, b=40),
)


def queue_length_chart(queue_ts: List[Tuple[float, int]]) -> go.Figure:
    """Area chart of queue length over simulation time."""
    if not queue_ts:
        fig = go.Figure()
        fig.update_layout(title="Queue Length Over Time", **COMMON_LAYOUT)
        return fig

    times = [t for t, _ in queue_ts]
    lengths = [q for _, q in queue_ts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=lengths, mode="lines",
        fill="tozeroy", fillcolor="rgba(46,204,113,0.2)",
        line=dict(color=ACCENT, width=2),
        name="Queue Length",
    ))
    fig.update_layout(
        title="Queue Length Over Time",
        xaxis_title="Time (minutes)",
        yaxis_title="Vehicles in Queue",
        **COMMON_LAYOUT,
    )
    return fig


def arrivals_over_time_chart(result: SimulationResult) -> go.Figure:
    """Cumulative arrivals chart."""
    if not result.vehicles:
        fig = go.Figure()
        fig.update_layout(title="Vehicle Arrivals Over Time", **COMMON_LAYOUT)
        return fig

    arrival_times = sorted(v.arrival_time for v in result.vehicles)
    cum_count = list(range(1, len(arrival_times) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=arrival_times, y=cum_count, mode="lines",
        line=dict(color="#3498db", width=2),
        name="Cumulative Arrivals",
    ))
    fig.update_layout(
        title="Vehicle Arrivals Over Time",
        xaxis_title="Time (minutes)",
        yaxis_title="Cumulative Arrivals",
        **COMMON_LAYOUT,
    )
    return fig


def service_completions_chart(result: SimulationResult) -> go.Figure:
    """Cumulative service completions chart."""
    completed = [
        v for v in result.vehicles
        if v.status == VehicleStatus.COMPLETED
    ]
    if not completed:
        fig = go.Figure()
        fig.update_layout(title="Service Completions Over Time", **COMMON_LAYOUT)
        return fig

    end_times = sorted(v.service_end_time for v in completed)
    cum_count = list(range(1, len(end_times) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=end_times, y=cum_count, mode="lines",
        line=dict(color=ACCENT, width=2),
        name="Cumulative Completions",
    ))
    fig.update_layout(
        title="Service Completions Over Time",
        xaxis_title="Time (minutes)",
        yaxis_title="Cumulative Completions",
        **COMMON_LAYOUT,
    )
    return fig


def waiting_time_distribution_chart(result: SimulationResult) -> go.Figure:
    """Histogram of vehicle waiting times."""
    completed = [
        v for v in result.vehicles
        if v.status == VehicleStatus.COMPLETED
    ]
    if not completed:
        fig = go.Figure()
        fig.update_layout(title="Waiting Time Distribution", **COMMON_LAYOUT)
        return fig

    waits = [v.waiting_time for v in completed]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=waits, nbinsx=30,
        marker=dict(color=ACCENT, line=dict(color=TEXT_COLOR, width=0.5)),
        name="Waiting Time",
    ))
    fig.update_layout(
        title="Waiting Time Distribution",
        xaxis_title="Waiting Time (minutes)",
        yaxis_title="Number of Vehicles",
        **COMMON_LAYOUT,
    )
    return fig


def dispenser_utilisation_chart(
    metrics: SimulationMetrics,
    num_dispensers: int,
) -> go.Figure:
    """Bar chart comparing utilisation across dispensers."""
    if not metrics.dispenser_utilisations:
        fig = go.Figure()
        fig.update_layout(title="Dispenser Utilisation", **COMMON_LAYOUT)
        return fig

    labels = [f"Dispenser {i+1}" for i in range(num_dispensers)]
    values = [u * 100 for u in metrics.dispenser_utilisations]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker=dict(
            color=values,
            colorscale=[[0, "#2ecc71"], [1, "#e74c3c"]],
            cmin=0, cmax=100,
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title="Dispenser Utilisation",
        xaxis_title="Dispenser",
        yaxis_title="Utilisation (%)",
        **COMMON_LAYOUT,
    )
    fig.update_yaxes(range=[0, 105])
    return fig


# ---- Scenario comparison charts ----

def scenario_waiting_time_chart(results: List[ScenarioResult]) -> go.Figure:
    """Bar chart comparing average waiting time across scenarios."""
    names = [r.name for r in results]
    waits = [r.metrics.avg_waiting_time for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=waits,
        marker_color=ACCENT,
        text=[f"{w:.2f}" for w in waits],
        textposition="auto",
    ))
    fig.update_layout(
        title="Average Waiting Time by Scenario",
        xaxis_title="Scenario",
        yaxis_title="Avg Waiting Time (min)",
        **COMMON_LAYOUT,
    )
    return fig


def scenario_queue_length_chart(results: List[ScenarioResult]) -> go.Figure:
    """Bar chart comparing max queue length across scenarios."""
    names = [r.name for r in results]
    qlens = [r.metrics.max_queue_length for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=qlens,
        marker_color="#3498db",
        text=[str(q) for q in qlens],
        textposition="auto",
    ))
    fig.update_layout(
        title="Maximum Queue Length by Scenario",
        xaxis_title="Scenario",
        yaxis_title="Max Queue Length",
        **COMMON_LAYOUT,
    )
    return fig


def scenario_utilisation_chart(results: List[ScenarioResult]) -> go.Figure:
    """Bar chart comparing average utilisation across scenarios."""
    names = [r.name for r in results]
    utils = [r.metrics.avg_utilisation * 100 for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=utils,
        marker=dict(
            color=utils,
            colorscale=[[0, "#2ecc71"], [1, "#e74c3c"]],
            cmin=0, cmax=100,
        ),
        text=[f"{u:.1f}%" for u in utils],
        textposition="auto",
    ))
    fig.update_layout(
        title="Average Dispenser Utilisation by Scenario",
        xaxis_title="Scenario",
        yaxis_title="Utilisation (%)",
        **COMMON_LAYOUT,
    )
    fig.update_yaxes(range=[0, 105])
    return fig


def scenario_throughput_chart(results: List[ScenarioResult]) -> go.Figure:
    """Bar chart comparing throughput across scenarios."""
    names = [r.name for r in results]
    tp = [r.metrics.throughput_per_hour for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=tp,
        marker_color="#f39c12",
        text=[f"{t:.1f}" for t in tp],
        textposition="auto",
    ))
    fig.update_layout(
        title="Throughput by Scenario (vehicles/hour)",
        xaxis_title="Scenario",
        yaxis_title="Vehicles / Hour",
        **COMMON_LAYOUT,
    )
    return fig

