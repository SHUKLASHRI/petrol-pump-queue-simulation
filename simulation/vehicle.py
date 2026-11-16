"""Vehicle entity for the petrol pump simulation."""

from dataclasses import dataclass, field
from enum import Enum


class VehicleStatus(Enum):
    """Possible states of a vehicle in the simulation."""
    ARRIVED = "arrived"
    WAITING = "waiting"
    FUELING = "fueling"
    COMPLETED = "completed"


@dataclass
class Vehicle:
    """Represents a single vehicle in the simulation.

    Attributes:
        vehicle_id: Unique identifier for the vehicle.
        arrival_time: Simulation time when the vehicle arrived at the station.
        queue_entry_time: Time when the vehicle entered the queue.
        service_start_time: Time when fueling began.
        service_end_time: Time when fueling completed.
        departure_time: Time when the vehicle left the station.
        service_duration: How long the fueling takes.
        dispenser_id: Which dispenser served this vehicle.
        status: Current state of the vehicle.
    """
    vehicle_id: int
    arrival_time: float
    queue_entry_time: float = 0.0
    service_start_time: float = 0.0
    service_end_time: float = 0.0
    departure_time: float = 0.0
    service_duration: float = 0.0
    dispenser_id: int = -1
    status: VehicleStatus = field(default=VehicleStatus.ARRIVED)

    @property
    def waiting_time(self) -> float:
        """Time spent waiting in the queue before service begins."""
        if self.service_start_time > 0:
            return self.service_start_time - self.arrival_time
        return 0.0

    @property
    def time_in_system(self) -> float:
        """Total time spent at the petrol station."""
        if self.departure_time > 0:
            return self.departure_time - self.arrival_time
        return 0.0

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.vehicle_id}, arrival={self.arrival_time:.2f}, "
            f"status={self.status.value})"
        )

