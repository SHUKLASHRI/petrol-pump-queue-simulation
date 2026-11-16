"""Tests for the metrics computation and scenario comparison modules."""

import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.engine import PetrolPumpSimulation, SimulationConfig, SimulationResult
from simulation.vehicle import Vehicle, VehicleStatus
from simulation.dispenser import Dispenser
from simulation.metrics import SimulationMetrics, compute_metrics
from simulation.scenarios import ScenarioRunner, ScenarioConfig


# ──────────────────────────────────────────────
# Metrics tests
# ──────────────────────────────────────────────

class TestMetrics:
    """Tests for the compute_metrics function."""

    @pytest.fixture
    def sample_result(self):
        """Create a small hand-crafted simulation result."""
        config = SimulationConfig(
            num_dispensers=2,
            simulation_duration=60.0,
        )

        vehicles = [
            Vehicle(
                vehicle_id=1, arrival_time=5.0,
                queue_entry_time=5.0, service_start_time=5.0,
                service_end_time=10.0, departure_time=10.0,
                service_duration=5.0, dispenser_id=0,
                status=VehicleStatus.COMPLETED,
            ),
            Vehicle(
                vehicle_id=2, arrival_time=6.0,
                queue_entry_time=6.0, service_start_time=6.0,
                service_end_time=12.0, departure_time=12.0,
                service_duration=6.0, dispenser_id=1,
                status=VehicleStatus.COMPLETED,
            ),
            Vehicle(
                vehicle_id=3, arrival_time=8.0,
                queue_entry_time=8.0, service_start_time=10.0,
                service_end_time=14.0, departure_time=14.0,
                service_duration=4.0, dispenser_id=0,
                status=VehicleStatus.COMPLETED,
            ),
        ]

        dispensers = [
            Dispenser(dispenser_id=0),
            Dispenser(dispenser_id=1),
        ]
        dispensers[0].add_record(1, 5.0, 10.0)
        dispensers[0].add_record(3, 10.0, 14.0)
        dispensers[1].add_record(2, 6.0, 12.0)

        queue_ts = [(5.0, 1), (5.0, 0), (6.0, 1), (6.0, 0),
                    (8.0, 1), (10.0, 0)]

        return SimulationResult(
            vehicles=vehicles,
            dispensers=dispensers,
            queue_length_over_time=queue_ts,
            config=config,
        )

    def test_total_counts(self, sample_result):
        m = compute_metrics(sample_result)
        assert m.total_arrived == 3
        assert m.total_served == 3

    def test_waiting_times_non_negative(self, sample_result):
        m = compute_metrics(sample_result)
        assert m.avg_waiting_time >= 0
        assert m.median_waiting_time >= 0
        assert m.max_waiting_time >= 0
        assert m.p95_waiting_time >= 0

    def test_utilisation_in_range(self, sample_result):
        m = compute_metrics(sample_result)
        assert 0 <= m.avg_utilisation <= 1
        for u in m.dispenser_utilisations:
            assert 0 <= u <= 1

    def test_throughput_positive(self, sample_result):
        m = compute_metrics(sample_result)
        assert m.throughput_per_hour > 0

    def test_avg_time_in_system(self, sample_result):
        m = compute_metrics(sample_result)
        # Vehicle 1: 10 - 5 = 5
        # Vehicle 2: 12 - 6 = 6
        # Vehicle 3: 14 - 8 = 6
        expected = (5.0 + 6.0 + 6.0) / 3.0
        assert abs(m.avg_time_in_system - expected) < 0.01

    def test_empty_result(self):
        config = SimulationConfig(simulation_duration=60.0)
        result = SimulationResult(
            vehicles=[], dispensers=[], queue_length_over_time=[], config=config,
        )
        m = compute_metrics(result)
        assert m.total_arrived == 0
        assert m.total_served == 0
        assert m.avg_waiting_time == 0.0
        assert m.throughput_per_hour == 0.0

    def test_max_queue_length(self, sample_result):
        m = compute_metrics(sample_result)
        assert m.max_queue_length >= 0


# ──────────────────────────────────────────────
# Scenario comparison tests
# ──────────────────────────────────────────────

class TestScenarios:
    """Tests for the scenario runner."""

    def test_single_scenario(self):
        cfg = SimulationConfig(
            num_dispensers=2,
            arrival_rate=12.0,
            simulation_duration=60.0,
            random_seed=42,
        )
        sc = ScenarioConfig(name="Test", config=cfg)
        result = ScenarioRunner.run_scenario(sc, repetitions=1)
        assert result.name == "Test"
        assert result.metrics.total_arrived > 0
        assert len(result.all_metrics) == 1

    def test_multiple_repetitions(self):
        cfg = SimulationConfig(
            num_dispensers=2,
            arrival_rate=12.0,
            simulation_duration=60.0,
            random_seed=42,
        )
        sc = ScenarioConfig(name="Test", config=cfg)
        result = ScenarioRunner.run_scenario(sc, repetitions=5)
        assert len(result.all_metrics) == 5

    def test_run_multiple_scenarios(self):
        scenarios = []
        for nd in [1, 2, 3]:
            cfg = SimulationConfig(
                num_dispensers=nd,
                arrival_rate=12.0,
                simulation_duration=60.0,
                random_seed=42,
            )
            scenarios.append(ScenarioConfig(name=f"{nd} dispensers", config=cfg))

        results = ScenarioRunner.run_multiple(scenarios, repetitions=1)
        assert len(results) == 3

    def test_more_dispensers_reduce_waiting(self):
        """Adding dispensers should generally reduce waiting time."""
        scenarios = []
        for nd in [1, 4]:
            cfg = SimulationConfig(
                num_dispensers=nd,
                arrival_rate=15.0,  # moderate load
                avg_service_time=5.0,
                simulation_duration=120.0,
                random_seed=42,
            )
            scenarios.append(ScenarioConfig(name=f"{nd}d", config=cfg))

        results = ScenarioRunner.run_multiple(scenarios, repetitions=3)
        # 4 dispensers should have less waiting than 1
        assert results[1].metrics.avg_waiting_time <= results[0].metrics.avg_waiting_time

    def test_comparison_summary_generated(self):
        scenarios = []
        for nd in [1, 2]:
            cfg = SimulationConfig(
                num_dispensers=nd,
                arrival_rate=12.0,
                simulation_duration=60.0,
                random_seed=42,
            )
            scenarios.append(ScenarioConfig(name=f"{nd}d", config=cfg))

        results = ScenarioRunner.run_multiple(scenarios, repetitions=1)
        summary = ScenarioRunner.generate_comparison_summary(results)
        assert "Scenario Comparison Summary" in summary
        assert "Recommendation" in summary

    def test_empty_comparison(self):
        summary = ScenarioRunner.generate_comparison_summary([])
        assert "No scenarios" in summary

