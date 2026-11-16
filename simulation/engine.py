"""Core discrete-event simulation engine using SimPy."""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import simpy
from scipy.stats import truncnorm

from simulation.vehicle import Vehicle, VehicleStatus
from simulation.dispenser import Dispenser


@dataclass
class SimulationConfig:
    """Configuration parameters for the petrol pump simulation.

    Attributes:
        num_dispensers: Number of fuel dispensers (1-10).
        arrival_rate: Average vehicles per hour (lambda).
        avg_service_time: Mean service time in minutes.
        service_time_std: Standard deviation of service time in minutes.
        min_service_time: Minimum service time in minutes.
        max_service_time: Maximum service time in minutes.
        simulation_duration: Total simulation time in minutes.
        random_seed: Seed for reproducibility.
    """
    num_dispensers: int = 3
    arrival_rate: float = 20.0           # vehicles per hour
    avg_service_time: float = 5.0        # minutes
    service_time_std: float = 1.5        # minutes
    min_service_time: float = 2.0        # minutes
    max_service_time: float = 12.0       # minutes
    simulation_duration: float = 120.0   # minutes (2 hours)
    random_seed: int = 42


@dataclass
class SimulationResult:
    """Container for all simulation output data."""
    vehicles: List[Vehicle] = field(default_factory=list)
    dispensers: List[Dispenser] = field(default_factory=list)
    queue_length_over_time: List[Tuple[float, int]] = field(default_factory=list)
    config: Optional[SimulationConfig] = None


class PetrolPumpSimulation:
    """Discrete-event simulation of a petrol station using SimPy.

    Models vehicle arrivals (Poisson process), queue waiting (FCFS),
    and fuel service (truncated-normal duration).
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.vehicles: List[Vehicle] = []
        self.dispensers: List[Dispenser] = []
        self.queue_length_over_time: List[Tuple[float, int]] = []
        self._vehicle_counter = 0
        self._current_queue_length = 0

    # ------------------------------------------------------------------
    # Random variate generators
    # ------------------------------------------------------------------

    def _generate_interarrival_time(self) -> float:
        """Generate inter-arrival time using the inverse-transform method.

        Inter-arrival time = -ln(U) / lambda
        where lambda is in vehicles per minute.
        """
        lam = self.config.arrival_rate / 60.0  # convert to per minute
        u = self.rng.random()
        # Avoid log(0)
        u = max(u, 1e-10)
        return -math.log(u) / lam

    def _generate_service_time(self) -> float:
        """Generate service time from a truncated normal distribution."""
        mu = self.config.avg_service_time
        sigma = self.config.service_time_std
        lo = self.config.min_service_time
        hi = self.config.max_service_time

        if sigma <= 0:
            return mu

        a = (lo - mu) / sigma
        b = (hi - mu) / sigma
        # Use scipy truncnorm with our RNG for reproducibility
        value = truncnorm.rvs(a, b, loc=mu, scale=sigma,
                              random_state=self.rng)
        return float(value)

    # ------------------------------------------------------------------
    # SimPy processes
    # ------------------------------------------------------------------

    def _vehicle_process(self, env: simpy.Environment,
                         dispenser_store: simpy.Store,
                         vehicle: Vehicle):
        """SimPy process for a single vehicle going through the station."""
        vehicle.queue_entry_time = env.now
        vehicle.status = VehicleStatus.WAITING
        self._current_queue_length += 1
        self.queue_length_over_time.append((env.now, self._current_queue_length))

        # Request an available dispenser (FCFS via Store.get())
        dispenser_idx = yield dispenser_store.get()

        self._current_queue_length -= 1
        self.queue_length_over_time.append(
            (env.now, self._current_queue_length)
        )

        vehicle.service_start_time = env.now
        vehicle.status = VehicleStatus.FUELING
        vehicle.dispenser_id = dispenser_idx

        service_time = vehicle.service_duration
        yield env.timeout(service_time)

        vehicle.service_end_time = env.now
        vehicle.departure_time = env.now
        vehicle.status = VehicleStatus.COMPLETED

        # Record in the dispenser tracker
        self.dispensers[dispenser_idx].add_record(
            vehicle.vehicle_id,
            vehicle.service_start_time,
            vehicle.service_end_time,
        )

        # Release dispenser back to the available pool
        yield dispenser_store.put(dispenser_idx)

    def _arrival_process(self, env: simpy.Environment,
                         dispenser_store: simpy.Store):
        """SimPy process that generates vehicle arrivals."""
        while True:
            iat = self._generate_interarrival_time()
            yield env.timeout(iat)

            if env.now > self.config.simulation_duration:
                break

            self._vehicle_counter += 1
            service_time = self._generate_service_time()

            vehicle = Vehicle(
                vehicle_id=self._vehicle_counter,
                arrival_time=env.now,
                service_duration=service_time,
            )
            self.vehicles.append(vehicle)
            env.process(self._vehicle_process(env, dispenser_store, vehicle))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Execute the simulation and return results."""
        self.vehicles.clear()
        self.queue_length_over_time.clear()
        self._vehicle_counter = 0
        self._current_queue_length = 0

        self.dispensers = [
            Dispenser(dispenser_id=i)
            for i in range(self.config.num_dispensers)
        ]

        env = simpy.Environment()
        dispenser_store = simpy.Store(env, capacity=self.config.num_dispensers)
        for i in range(self.config.num_dispensers):
            dispenser_store.put(i)

        env.process(self._arrival_process(env, dispenser_store))
        env.run(until=self.config.simulation_duration + 60)
        # Extra 60 minutes to let in-progress vehicles finish

        return SimulationResult(
            vehicles=self.vehicles,
            dispensers=self.dispensers,
            queue_length_over_time=self.queue_length_over_time,
            config=self.config,
        )

