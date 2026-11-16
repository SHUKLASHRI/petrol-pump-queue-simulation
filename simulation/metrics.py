"""Statistical metrics computed from simulation results."""

from dataclasses import dataclass
from typing import List

import numpy as np

from simulation.vehicle import Vehicle, VehicleStatus
from simulation.dispenser import Dispenser
from simulation.engine import SimulationResult


@dataclass
class SimulationMetrics:
    """Aggregated metrics for a single simulation run."""
    total_arrived: int = 0
    total_served: int = 0
    avg_waiting_time: float = 0.0
    median_waiting_time: float = 0.0
    max_waiting_time: float = 0.0
    std_waiting_time: float = 0.0
    p95_waiting_time: float = 0.0
    avg_time_in_system: float = 0.0
    avg_queue_length: float = 0.0
    max_queue_length: int = 0
    avg_utilisation: float = 0.0
    throughput_per_hour: float = 0.0
    dispenser_utilisations: List[float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.dispenser_utilisations is None:
            self.dispenser_utilisations = []


def compute_metrics(result: SimulationResult) -> SimulationMetrics:
    """Compute all statistical metrics from simulation results.

    Args:
        result: The completed SimulationResult.

    Returns:
        Populated SimulationMetrics object.
    """
    config = result.config
    vehicles = result.vehicles
    dispensers = result.dispensers
    queue_ts = result.queue_length_over_time
    sim_duration = config.simulation_duration if config else 120.0

    metrics = SimulationMetrics()
    metrics.total_arrived = len(vehicles)

    completed = [v for v in vehicles if v.status == VehicleStatus.COMPLETED]
    metrics.total_served = len(completed)

    if completed:
        waiting_times = np.array([v.waiting_time for v in completed])
        system_times = np.array([v.time_in_system for v in completed])

        metrics.avg_waiting_time = float(np.mean(waiting_times))
        metrics.median_waiting_time = float(np.median(waiting_times))
        metrics.max_waiting_time = float(np.max(waiting_times))
        metrics.std_waiting_time = float(np.std(waiting_times))
        metrics.p95_waiting_time = float(np.percentile(waiting_times, 95))
        metrics.avg_time_in_system = float(np.mean(system_times))
    else:
        metrics.avg_waiting_time = 0.0
        metrics.median_waiting_time = 0.0
        metrics.max_waiting_time = 0.0
        metrics.std_waiting_time = 0.0
        metrics.p95_waiting_time = 0.0
        metrics.avg_time_in_system = 0.0

    # Queue length statistics (time-weighted average)
    if queue_ts:
        metrics.max_queue_length = max(ql for _, ql in queue_ts)
        # Compute time-weighted average queue length
        total_area = 0.0
        for i in range(len(queue_ts) - 1):
            t_curr, q_curr = queue_ts[i]
            t_next, _ = queue_ts[i + 1]
            total_area += q_curr * (t_next - t_curr)
        last_t, last_q = queue_ts[-1]
        if last_t < sim_duration:
            total_area += last_q * (sim_duration - last_t)
        metrics.avg_queue_length = total_area / sim_duration if sim_duration > 0 else 0.0
    else:
        metrics.avg_queue_length = 0.0
        metrics.max_queue_length = 0

    # Dispenser utilisation
    if dispensers:
        utilisations = [d.utilisation(sim_duration) for d in dispensers]
        metrics.dispenser_utilisations = utilisations
        metrics.avg_utilisation = float(np.mean(utilisations))
    else:
        metrics.dispenser_utilisations = []
        metrics.avg_utilisation = 0.0

    # Throughput
    sim_hours = sim_duration / 60.0
    metrics.throughput_per_hour = (
        metrics.total_served / sim_hours if sim_hours > 0 else 0.0
    )

    return metrics

