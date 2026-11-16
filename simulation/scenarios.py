"""Scenario comparison and what-if analysis runner."""

from dataclasses import dataclass
from typing import List

from simulation.engine import PetrolPumpSimulation, SimulationConfig
from simulation.metrics import SimulationMetrics, compute_metrics


@dataclass
class ScenarioConfig:
    """Configuration for a named scenario."""
    name: str
    config: SimulationConfig


@dataclass
class ScenarioResult:
    """Result of running a single scenario (possibly with repetitions)."""
    name: str
    config: SimulationConfig
    metrics: SimulationMetrics
    all_metrics: List[SimulationMetrics]


class ScenarioRunner:
    """Run and compare multiple simulation scenarios."""

    @staticmethod
    def run_scenario(scenario: ScenarioConfig,
                     repetitions: int = 1) -> ScenarioResult:
        """Run a single scenario with optional repetitions.

        Args:
            scenario: The scenario configuration.
            repetitions: Number of independent runs.

        Returns:
            ScenarioResult with averaged metrics.
        """
        all_metrics: List[SimulationMetrics] = []
        for rep in range(repetitions):
            cfg = SimulationConfig(
                num_dispensers=scenario.config.num_dispensers,
                arrival_rate=scenario.config.arrival_rate,
                avg_service_time=scenario.config.avg_service_time,
                service_time_std=scenario.config.service_time_std,
                min_service_time=scenario.config.min_service_time,
                max_service_time=scenario.config.max_service_time,
                simulation_duration=scenario.config.simulation_duration,
                random_seed=scenario.config.random_seed + rep,
            )
            sim = PetrolPumpSimulation(cfg)
            result = sim.run()
            metrics = compute_metrics(result)
            all_metrics.append(metrics)

        avg = ScenarioRunner._average_metrics(all_metrics)
        return ScenarioResult(
            name=scenario.name,
            config=scenario.config,
            metrics=avg,
            all_metrics=all_metrics,
        )

    @staticmethod
    def run_multiple(scenarios: List[ScenarioConfig],
                     repetitions: int = 1) -> List[ScenarioResult]:
        """Run multiple scenarios and return all results."""
        return [
            ScenarioRunner.run_scenario(s, repetitions)
            for s in scenarios
        ]

    @staticmethod
    def _average_metrics(all_m: List[SimulationMetrics]) -> SimulationMetrics:
        """Average metrics across multiple simulation runs."""
        if not all_m:
            return SimulationMetrics()
        n = len(all_m)
        avg = SimulationMetrics()
        avg.total_arrived = int(sum(m.total_arrived for m in all_m) / n)
        avg.total_served = int(sum(m.total_served for m in all_m) / n)
        avg.avg_waiting_time = sum(m.avg_waiting_time for m in all_m) / n
        avg.median_waiting_time = sum(m.median_waiting_time for m in all_m) / n
        avg.max_waiting_time = sum(m.max_waiting_time for m in all_m) / n
        avg.std_waiting_time = sum(m.std_waiting_time for m in all_m) / n
        avg.p95_waiting_time = sum(m.p95_waiting_time for m in all_m) / n
        avg.avg_time_in_system = sum(m.avg_time_in_system for m in all_m) / n
        avg.avg_queue_length = sum(m.avg_queue_length for m in all_m) / n
        avg.max_queue_length = int(sum(m.max_queue_length for m in all_m) / n)
        avg.avg_utilisation = sum(m.avg_utilisation for m in all_m) / n
        avg.throughput_per_hour = sum(m.throughput_per_hour for m in all_m) / n
        # Average per-dispenser utilisation
        if all_m[0].dispenser_utilisations:
            nd = len(all_m[0].dispenser_utilisations)
            avg.dispenser_utilisations = [
                sum(m.dispenser_utilisations[i] for m in all_m) / n
                for i in range(nd)
            ]
        return avg

    @staticmethod
    def generate_comparison_summary(results: List[ScenarioResult]) -> str:
        """Generate a textual comparison of scenario results."""
        if not results:
            return "No scenarios to compare."

        lines = []
        best_wait = min(results, key=lambda r: r.metrics.avg_waiting_time)
        best_util = max(results, key=lambda r: r.metrics.avg_utilisation)
        best_throughput = max(results, key=lambda r: r.metrics.throughput_per_hour)
        best_queue = min(results, key=lambda r: r.metrics.max_queue_length)

        lines.append("### Scenario Comparison Summary\n")
        lines.append(
            f"**Lowest Average Waiting Time**: {best_wait.name} "
            f"({best_wait.metrics.avg_waiting_time:.2f} min)"
        )
        lines.append(
            f"**Highest Dispenser Utilisation**: {best_util.name} "
            f"({best_util.metrics.avg_utilisation:.1%})"
        )
        lines.append(
            f"**Highest Throughput**: {best_throughput.name} "
            f"({best_throughput.metrics.throughput_per_hour:.1f} vehicles/hour)"
        )
        lines.append(
            f"**Smallest Maximum Queue**: {best_queue.name} "
            f"({best_queue.metrics.max_queue_length} vehicles)"
        )
        lines.append("")

        # Check stability for each scenario
        for r in results:
            lam = r.config.arrival_rate / 60.0  # per minute
            mu = 1.0 / r.config.avg_service_time  # per minute
            rho = lam / (r.config.num_dispensers * mu)
            status = "✅ Stable" if rho < 1 else "⚠️ Unstable (ρ ≥ 1)"
            lines.append(
                f"- **{r.name}**: ρ = {rho:.3f} — {status}"
            )

        lines.append("")
        lines.append(
            f"**Recommendation**: **{best_wait.name}** offers the best customer "
            f"experience with the lowest average wait, while **{best_util.name}** "
            f"maximises resource utilisation."
        )
        return "\n".join(lines)

