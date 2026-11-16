"""Miscellaneous helper functions."""

import pandas as pd
from typing import List

from simulation.vehicle import Vehicle, VehicleStatus


def vehicles_to_dataframe(vehicles: List[Vehicle]) -> pd.DataFrame:
    """Convert a list of Vehicle objects to a Pandas DataFrame."""
    records = []
    for v in vehicles:
        records.append({
            "Vehicle ID": v.vehicle_id,
            "Arrival Time": round(v.arrival_time, 2),
            "Queue Entry": round(v.queue_entry_time, 2),
            "Service Start": round(v.service_start_time, 2),
            "Service End": round(v.service_end_time, 2),
            "Departure": round(v.departure_time, 2),
            "Waiting Time": round(v.waiting_time, 2),
            "Service Duration": round(v.service_duration, 2),
            "Time in System": round(v.time_in_system, 2),
            "Dispenser": v.dispenser_id + 1 if v.dispenser_id >= 0 else "-",
            "Status": v.status.value,
        })
    return pd.DataFrame(records)


def traffic_intensity(arrival_rate: float, service_rate: float,
                      num_dispensers: int) -> float:
    """Compute traffic intensity rho = lambda / (c * mu).

    Args:
        arrival_rate: Vehicles per hour.
        service_rate: Vehicles served per hour per dispenser.
        num_dispensers: Number of dispensers.

    Returns:
        Traffic intensity rho.
    """
    if num_dispensers <= 0 or service_rate <= 0:
        return float("inf")
    return arrival_rate / (num_dispensers * service_rate)


def format_duration(minutes: float) -> str:
    """Format minutes into a human-readable string."""
    if minutes < 1:
        return f"{minutes * 60:.0f} seconds"
    elif minutes < 60:
        return f"{minutes:.1f} minutes"
    else:
        hrs = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hrs}h {mins}m"

