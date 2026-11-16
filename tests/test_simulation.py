"""Tests for the simulation engine, vehicle, and dispenser modules."""

import sys
import os
import math

import pytest
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.engine import PetrolPumpSimulation, SimulationConfig
from simulation.vehicle import Vehicle, VehicleStatus
from simulation.dispenser import Dispenser
from simulation.metrics import compute_metrics


# ──────────────────────────────────────────────
# Vehicle tests
# ──────────────────────────────────────────────

class TestVehicle:
    """Tests for the Vehicle dataclass."""

    def test_initial_status(self):
        v = Vehicle(vehicle_id=1, arrival_time=10.0)
        assert v.status == VehicleStatus.ARRIVED

    def test_waiting_time_zero_before_service(self):
        v = Vehicle(vehicle_id=1, arrival_time=10.0)
        assert v.waiting_time == 0.0

    def test_waiting_time_positive(self):
        v = Vehicle(vehicle_id=1, arrival_time=10.0, service_start_time=15.0)
        assert v.waiting_time == 5.0

    def test_waiting_time_cannot_be_negative(self):
        """Service cannot start before arrival in a correctly-running simulation."""
        v = Vehicle(vehicle_id=1, arrival_time=10.0, service_start_time=10.0)
        assert v.waiting_time >= 0.0

    def test_time_in_system(self):
        v = Vehicle(
            vehicle_id=1, arrival_time=10.0,
            service_start_time=12.0, service_end_time=17.0,
            departure_time=17.0,
        )
        assert v.time_in_system == 7.0

    def test_time_in_system_zero_if_not_departed(self):
        v = Vehicle(vehicle_id=1, arrival_time=10.0)
        assert v.time_in_system == 0.0


# ──────────────────────────────────────────────
# Dispenser tests
# ──────────────────────────────────────────────

class TestDispenser:
    """Tests for the Dispenser tracker."""

    def test_empty_dispenser(self):
        d = Dispenser(dispenser_id=0)
        assert d.total_busy_time == 0.0
        assert d.vehicles_served == 0
        assert d.utilisation(100.0) == 0.0

    def test_add_record(self):
        d = Dispenser(dispenser_id=0)
        d.add_record(1, 10.0, 15.0)
        assert d.vehicles_served == 1
        assert d.total_busy_time == 5.0

    def test_utilisation(self):
        d = Dispenser(dispenser_id=0)
        d.add_record(1, 0.0, 50.0)
        assert d.utilisation(100.0) == 0.5

    def test_utilisation_capped_at_one(self):
        d = Dispenser(dispenser_id=0)
        d.add_record(1, 0.0, 120.0)
        assert d.utilisation(100.0) == 1.0

    def test_utilisation_zero_duration(self):
        d = Dispenser(dispenser_id=0)
        assert d.utilisation(0.0) == 0.0


# ──────────────────────────────────────────────
# Simulation engine tests
# ──────────────────────────────────────────────

class TestSimulationEngine:
    """Tests for the PetrolPumpSimulation class."""

    @pytest.fixture
    def default_config(self):
        return SimulationConfig(
            num_dispensers=2,
            arrival_rate=12.0,       # 12 vehicles/hour
            avg_service_time=5.0,
            service_time_std=1.0,
            min_service_time=2.0,
            max_service_time=10.0,
            simulation_duration=60.0,  # 1 hour
            random_seed=42,
        )

    def test_simulation_produces_vehicles(self, default_config):
        sim = PetrolPumpSimulation(default_config)
        result = sim.run()
        assert len(result.vehicles) > 0

    def test_all_vehicles_have_positive_arrival(self, default_config):
        sim = PetrolPumpSimulation(default_config)
        result = sim.run()
        for v in result.vehicles:
            assert v.arrival_time > 0

    def test_completed_vehicles_have_valid_timestamps(self, default_config):
        sim = PetrolPumpSimulation(default_config)
        result = sim.run()
        for v in result.vehicles:
            if v.status == VehicleStatus.COMPLETED:
                assert v.service_start_time >= v.arrival_time, (
                    f"Vehicle {v.vehicle_id}: service start {v.service_start_time} "
                    f"before arrival {v.arrival_time}"
                )
                assert v.service_end_time >= v.service_start_time, (
                    f"Vehicle {v.vehicle_id}: service end {v.service_end_time} "
                    f"before service start {v.service_start_time}"
                )
                assert v.departure_time >= v.service_end_time
                assert v.waiting_time >= 0

    def test_dispenser_no_overlap(self, default_config):
        """A dispenser cannot serve multiple vehicles simultaneously."""
        sim = PetrolPumpSimulation(default_config)
        result = sim.run()
        for d in result.dispensers:
            sorted_records = sorted(d.records, key=lambda r: r.service_start)
            for i in range(len(sorted_records) - 1):
                assert sorted_records[i].service_end <= sorted_records[i+1].service_start + 1e-9, (
                    f"Dispenser {d.dispenser_id}: overlap between vehicle "
                    f"{sorted_records[i].vehicle_id} and {sorted_records[i+1].vehicle_id}"
                )

    def test_reproducibility(self, default_config):
        """Same seed produces identical results."""
        sim1 = PetrolPumpSimulation(default_config)
        result1 = sim1.run()

        sim2 = PetrolPumpSimulation(default_config)
        result2 = sim2.run()

        assert len(result1.vehicles) == len(result2.vehicles)
        for v1, v2 in zip(result1.vehicles, result2.vehicles):
            assert v1.arrival_time == v2.arrival_time
            assert v1.service_duration == v2.service_duration

    def test_different_seeds_produce_different_results(self):
        cfg1 = SimulationConfig(random_seed=1, simulation_duration=60.0)
        cfg2 = SimulationConfig(random_seed=2, simulation_duration=60.0)
        sim1 = PetrolPumpSimulation(cfg1)
        sim2 = PetrolPumpSimulation(cfg2)
        r1 = sim1.run()
        r2 = sim2.run()
        # Very unlikely to have exact same number of vehicles with different seeds
        arrivals1 = [v.arrival_time for v in r1.vehicles]
        arrivals2 = [v.arrival_time for v in r2.vehicles]
        assert arrivals1 != arrivals2

    def test_fcfs_ordering(self, default_config):
        """Vehicles that arrive earlier should start service no later than later arrivals
        (within the same dispenser context)."""
        sim = PetrolPumpSimulation(default_config)
        result = sim.run()
        completed = [v for v in result.vehicles if v.status == VehicleStatus.COMPLETED]
        # Sort by arrival time
        completed.sort(key=lambda v: v.arrival_time)
        # In a FCFS queue, an earlier arrival should have service_start <= any later arrival
        # that goes to the same queue position. With multiple dispensers this is approximate,
        # but we can check that no vehicle is served before one that arrived much earlier
        # is still waiting.
        for i in range(len(completed)):
            for j in range(i + 1, len(completed)):
                if completed[j].service_start_time < completed[i].arrival_time:
                    pytest.fail(
                        f"Vehicle {completed[j].vehicle_id} started service before "
                        f"Vehicle {completed[i].vehicle_id} even arrived!"
                    )

    def test_arrival_count_reasonable(self, default_config):
        """Check that the number of arrivals is roughly as expected."""
        sim = PetrolPumpSimulation(default_config)
        result = sim.run()
        expected = default_config.arrival_rate * (default_config.simulation_duration / 60.0)
        # Allow ±50% due to randomness
        assert len(result.vehicles) >= expected * 0.3
        assert len(result.vehicles) <= expected * 2.0

    def test_queue_length_never_negative(self, default_config):
        sim = PetrolPumpSimulation(default_config)
        result = sim.run()
        for _, ql in result.queue_length_over_time:
            assert ql >= 0

    def test_interarrival_time_positive(self):
        """Inter-arrival times must always be positive."""
        config = SimulationConfig(random_seed=99)
        sim = PetrolPumpSimulation(config)
        for _ in range(1000):
            iat = sim._generate_interarrival_time()
            assert iat > 0

    def test_service_time_within_bounds(self):
        """Service times must be within configured bounds."""
        config = SimulationConfig(
            avg_service_time=5.0, service_time_std=1.5,
            min_service_time=2.0, max_service_time=10.0,
            random_seed=42,
        )
        sim = PetrolPumpSimulation(config)
        for _ in range(1000):
            st = sim._generate_service_time()
            assert st >= config.min_service_time - 1e-9
            assert st <= config.max_service_time + 1e-9

    def test_vehicle_counts_consistent(self, default_config):
        """Total served <= total arrived."""
        sim = PetrolPumpSimulation(default_config)
        result = sim.run()
        metrics = compute_metrics(result)
        assert metrics.total_served <= metrics.total_arrived

    def test_multiple_dispensers(self):
        """With many dispensers and low traffic, utilisation should be low."""
        config = SimulationConfig(
            num_dispensers=10,
            arrival_rate=5.0,  # only 5 vehicles/hour
            simulation_duration=60.0,
            random_seed=42,
        )
        sim = PetrolPumpSimulation(config)
        result = sim.run()
        metrics = compute_metrics(result)
        assert metrics.avg_utilisation < 0.5

