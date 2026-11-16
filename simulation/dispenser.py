"""Fuel dispenser resource for the petrol pump simulation."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DispenserRecord:
    """A single service record for a dispenser."""
    vehicle_id: int
    service_start: float
    service_end: float


@dataclass
class Dispenser:
    """Tracks utilisation data for a single fuel dispenser.

    Attributes:
        dispenser_id: Unique identifier.
        records: List of service records.
    """
    dispenser_id: int
    records: List[DispenserRecord] = field(default_factory=list)

    @property
    def total_busy_time(self) -> float:
        """Total time this dispenser spent actively fueling vehicles."""
        return sum(r.service_end - r.service_start for r in self.records)

    @property
    def vehicles_served(self) -> int:
        """Number of vehicles served by this dispenser."""
        return len(self.records)

    def utilisation(self, simulation_duration: float) -> float:
        """Fraction of time this dispenser was busy.

        Args:
            simulation_duration: Total simulation run time.

        Returns:
            Utilisation ratio between 0 and 1.
        """
        if simulation_duration <= 0:
            return 0.0
        return min(self.total_busy_time / simulation_duration, 1.0)

    def add_record(self, vehicle_id: int, start: float, end: float) -> None:
        """Log a completed service."""
        self.records.append(DispenserRecord(vehicle_id, start, end))

