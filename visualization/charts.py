"""Plotly chart generators with modern, executive-grade styling for the simulation dashboard."""

from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go
from simulation.engine import SimulationResult
from simulation.metrics import SimulationMetrics
from simulation.vehicle import VehicleStatus
from simulation.scenarios import ScenarioResult

# ---- Ultra-clean Obsidian / Dark Glass Theme ----
COLOR_EMERALD = "#10b981"   # Primary accent
COLOR_CYAN    = "#06b6d4"   # Secondary accent / arrivals
COLOR_BLUE    = "#3b82f6"   # Completed / through
COLOR_AMBER   = "#f59e0b"   # Queue / warning
COLOR_ROSE    = "#ef4444"   # Bottleneck / critical
COLOR_PURPLE  = "#8b5cf6"   # Theoretical / baseline

BG_CANVAS     = "#0b0f19"   # Deep obsidian
BG_SURFACE    = "#111827"   # Elevated card surface
BORDER_SUBTLE = "#1f2937"   # Fine border
TEXT_PRIMARY  = "#f9fafb"   # Crisp white
TEXT_MUTED    = "#9ca3af"   # Muted gray

COMMON_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17, 24, 39, 0.6)",
    font=dict(color=TEXT_PRIMARY, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", size=12),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_PRIMARY, size=12),
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_PRIMARY, size=12),
    ),
    margin=dict(l=48, r=24, t=48, b=40),
    hoverlabel=dict(
        bgcolor="#1f2937",
        font_color="#f9fafb",
        font_size=12,
        bordercolor="#374151",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color=TEXT_MUTED, size=11),
    ),
)


def queue_length_chart(queue_ts: List[Tuple[float, int]]) -> go.Figure:
    """Stepped area chart of queue length over simulation time with color thresholds."""
    fig = go.Figure()
    if not queue_ts:
        fig.update_layout(title="Queue Length Timeline", **COMMON_LAYOUT)
        return fig

    times = [t for t, _ in queue_ts]
    lengths = [q for _, q in queue_ts]
    max_q = max(lengths) if lengths else 0

    # Gradient area trace
    fig.add_trace(go.Scatter(
        x=times,
        y=lengths,
        mode="lines",
        line_shape="hv",  # Discrete event stepped transition
        line=dict(color=COLOR_AMBER, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(245, 158, 11, 0.15)",
        name="Queue Depth",
        hovertemplate="<b>Time:</b> %{x:.1f} min<br><b>Queue:</b> %{y} vehicles<extra></extra>",
    ))

    # Add warning reference line if queue gets high
    if max_q > 5:
        fig.add_hline(
            y=5,
            line_dash="dot",
            line_color="rgba(239, 68, 68, 0.5)",
            annotation_text="Congestion Threshold (5 veh)",
            annotation_position="bottom right",
            annotation_font=dict(color=COLOR_ROSE, size=10),
        )

    fig.update_layout(
        title=dict(text="<b>Live Queue Depth Over Time</b>", font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis_title="Simulation Time (minutes)",
        yaxis_title="Vehicles Waiting in Queue",
        **COMMON_LAYOUT,
    )
    return fig


def arrivals_and_departures_chart(result: SimulationResult) -> go.Figure:
    """Cumulative arrivals vs service completions showing the service backlog gap."""
    fig = go.Figure()
    if not result.vehicles:
        fig.update_layout(title="Arrivals vs Completions", **COMMON_LAYOUT)
        return fig

    # Cumulative arrivals
    arr_times = sorted(v.arrival_time for v in result.vehicles)
    arr_counts = list(range(1, len(arr_times) + 1))

    fig.add_trace(go.Scatter(
        x=arr_times,
        y=arr_counts,
        mode="lines",
        line=dict(color=COLOR_CYAN, width=2.5),
        name="Cumulative Arrivals",
        hovertemplate="<b>Arrivals:</b> %{y} at %{x:.1f} min<extra></extra>",
    ))

    # Cumulative completions
    completed = [v for v in result.vehicles if v.status == VehicleStatus.COMPLETED]
    if completed:
        end_times = sorted(v.service_end_time for v in completed)
        end_counts = list(range(1, len(end_times) + 1))
        fig.add_trace(go.Scatter(
            x=end_times,
            y=end_counts,
            mode="lines",
            line=dict(color=COLOR_EMERALD, width=2.5),
            name="Cumulative Served",
            hovertemplate="<b>Completed:</b> %{y} at %{x:.1f} min<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="<b>Cumulative Throughput Dynamics</b> (Arrivals vs Departures)", font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis_title="Simulation Time (minutes)",
        yaxis_title="Total Vehicles",
        **COMMON_LAYOUT,
    )
    return fig


def arrivals_over_time_chart(result: SimulationResult) -> go.Figure:
    """Standalone cumulative arrivals chart."""
    return arrivals_and_departures_chart(result)


def service_completions_chart(result: SimulationResult) -> go.Figure:
    """Hourly service completion velocity bar chart."""
    fig = go.Figure()
    completed = [v for v in result.vehicles if v.status == VehicleStatus.COMPLETED]
    if not completed:
        fig.update_layout(title="Service Completions Velocity", **COMMON_LAYOUT)
        return fig

    # Bin completions into 15-minute intervals
    dur = result.config.simulation_duration if result.config else 120.0
    bins = np.arange(0, dur + 15, 15)
    end_times = [v.service_end_time for v in completed]
    counts, bin_edges = np.histogram(end_times, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        marker=dict(
            color=COLOR_EMERALD,
            line=dict(color="rgba(16, 185, 129, 0.4)", width=1),
        ),
        name="Vehicles Served",
        hovertemplate="<b>Interval:</b> %{x:.0f}m ±7.5m<br><b>Served:</b> %{y} vehicles<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="<b>Service Velocity</b> (Vehicles Completed per 15-Min Window)", font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis_title="Simulation Timeline (minutes)",
        yaxis_title="Vehicles Completed",
        **COMMON_LAYOUT,
    )
    return fig


def waiting_time_distribution_chart(result: SimulationResult) -> go.Figure:
    """Polished distribution histogram of vehicle waiting times with mean & median indicators."""
    fig = go.Figure()
    completed = [v for v in result.vehicles if v.status == VehicleStatus.COMPLETED]
    if not completed:
        fig.update_layout(title="Waiting Time Distribution", **COMMON_LAYOUT)
        return fig

    waits = [v.waiting_time for v in completed]
    mean_w = float(np.mean(waits))
    med_w = float(np.median(waits))
    p95_w = float(np.percentile(waits, 95))

    fig.add_trace(go.Histogram(
        x=waits,
        nbinsx=24,
        marker=dict(
            color="rgba(16, 185, 129, 0.7)",
            line=dict(color="#10b981", width=1.5),
        ),
        name="Vehicle Count",
        hovertemplate="<b>Wait Range:</b> %{x:.1f} min<br><b>Vehicles:</b> %{y}<extra></extra>",
    ))

    # Mean line
    fig.add_vline(
        x=mean_w,
        line_dash="dash",
        line_color=COLOR_CYAN,
        annotation_text=f"Mean: {mean_w:.1f}m",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_CYAN, size=11),
    )

    # 95th Percentile line
    fig.add_vline(
        x=p95_w,
        line_dash="dot",
        line_color=COLOR_ROSE,
        annotation_text=f"95th %ile: {p95_w:.1f}m",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_ROSE, size=11),
    )

    fig.update_layout(
        title=dict(text="<b>Waiting Time Frequency Distribution</b>", font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis_title="Queue Wait Time (minutes)",
        yaxis_title="Number of Vehicles",
        **COMMON_LAYOUT,
    )
    return fig


def dispenser_utilisation_chart(metrics: SimulationMetrics, num_dispensers: int) -> go.Figure:
    """Individual dispenser workload comparison with utilization threshold benchmarks."""
    fig = go.Figure()
    if not metrics.dispenser_utilisations:
        fig.update_layout(title="Dispenser Utilisation", **COMMON_LAYOUT)
        return fig

    labels = [f"Pump #{i+1}" for i in range(num_dispensers)]
    values = [u * 100 for u in metrics.dispenser_utilisations]

    # Color code bars by load intensity
    bar_colors = [
        COLOR_ROSE if v >= 90 else (COLOR_AMBER if v >= 75 else COLOR_EMERALD)
        for v in values
    ]

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(255,255,255,0.15)", width=1),
        ),
        text=[f"<b>{v:.1f}%</b>" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=12),
        hovertemplate="<b>%{x}</b><br>Utilisation: %{y:.1f}%<extra></extra>",
    ))

    # Target optimal utilization zone (60-80%)
    fig.add_hrect(
        y0=60, y1=85,
        fillcolor="rgba(16, 185, 129, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Optimal Utilization Zone (60-85%)",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_EMERALD, size=10),
    )

    fig.update_layout(
        title=dict(text="<b>Fuel Dispenser Utilization Breakdown</b>", font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis_title="Fuel Pump Island",
        yaxis_title="Utilisation (%)",
        **COMMON_LAYOUT,
    )
    fig.update_yaxes(range=[0, 115])
    return fig


# ──────────────────────────────────────────────
# Scenario Comparison Charts
# ──────────────────────────────────────────────

def scenario_waiting_time_chart(results: List[ScenarioResult]) -> go.Figure:
    """Scenario comparison for average wait times."""
    fig = go.Figure()
    names = [r.name for r in results]
    waits = [r.metrics.avg_waiting_time for r in results]

    fig.add_trace(go.Bar(
        x=names,
        y=waits,
        marker=dict(
            color=COLOR_CYAN,
            line=dict(color="rgba(6, 182, 212, 0.4)", width=1),
        ),
        text=[f"{w:.2f} min" for w in waits],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hovertemplate="<b>%{x}</b><br>Avg Wait: %{y:.2f} min<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>Average Waiting Time Across Scenarios</b>", font=dict(size=13, color=TEXT_PRIMARY)),
        xaxis_title="Configuration",
        yaxis_title="Avg Waiting Time (min)",
        **COMMON_LAYOUT,
    )
    max_w = max(waits) if waits else 1
    fig.update_yaxes(range=[0, max_w * 1.25])
    return fig


def scenario_queue_length_chart(results: List[ScenarioResult]) -> go.Figure:
    """Scenario comparison for maximum queue depth."""
    fig = go.Figure()
    names = [r.name for r in results]
    qlens = [r.metrics.max_queue_length for r in results]

    fig.add_trace(go.Bar(
        x=names,
        y=qlens,
        marker=dict(
            color=COLOR_AMBER,
            line=dict(color="rgba(245, 158, 11, 0.4)", width=1),
        ),
        text=[f"{q} veh" for q in qlens],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hovertemplate="<b>%{x}</b><br>Max Queue: %{y} vehicles<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>Peak Queue Depth Across Scenarios</b>", font=dict(size=13, color=TEXT_PRIMARY)),
        xaxis_title="Configuration",
        yaxis_title="Peak Queue Length (vehicles)",
        **COMMON_LAYOUT,
    )
    max_q = max(qlens) if qlens else 1
    fig.update_yaxes(range=[0, max_q * 1.25])
    return fig


def scenario_utilisation_chart(results: List[ScenarioResult]) -> go.Figure:
    """Scenario comparison for average server utilization."""
    fig = go.Figure()
    names = [r.name for r in results]
    utils = [r.metrics.avg_utilisation * 100 for r in results]

    colors = [
        COLOR_ROSE if u >= 95 else (COLOR_AMBER if u >= 80 else COLOR_EMERALD)
        for u in utils
    ]

    fig.add_trace(go.Bar(
        x=names,
        y=utils,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
        text=[f"{u:.1f}%" for u in utils],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hovertemplate="<b>%{x}</b><br>Avg Utilization: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>Average Resource Utilization by Scenario</b>", font=dict(size=13, color=TEXT_PRIMARY)),
        xaxis_title="Configuration",
        yaxis_title="Utilisation (%)",
        **COMMON_LAYOUT,
    )
    fig.update_yaxes(range=[0, 115])
    return fig


def scenario_throughput_chart(results: List[ScenarioResult]) -> go.Figure:
    """Scenario comparison for hourly throughput."""
    fig = go.Figure()
    names = [r.name for r in results]
    tps = [r.metrics.throughput_per_hour for r in results]

    fig.add_trace(go.Bar(
        x=names,
        y=tps,
        marker=dict(
            color=COLOR_BLUE,
            line=dict(color="rgba(59, 130, 246, 0.4)", width=1),
        ),
        text=[f"{t:.1f} /hr" for t in tps],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hovertemplate="<b>%{x}</b><br>Throughput: %{y:.1f} veh/hr<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>Hourly Station Throughput</b>", font=dict(size=13, color=TEXT_PRIMARY)),
        xaxis_title="Configuration",
        yaxis_title="Vehicles Served / Hour",
        **COMMON_LAYOUT,
    )
    max_t = max(tps) if tps else 1
    fig.update_yaxes(range=[0, max_t * 1.25])
    return fig
