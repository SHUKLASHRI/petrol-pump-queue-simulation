"""
Lightweight Python Backend Server for PetroFlow
=============================================
Serves the pure HTML/CSS/JS frontend (docs/index.html) and exposes REST API
endpoints powered by the SimPy discrete-event simulation engine.

Run with:
    python server.py
"""

import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Ensure project root is importable
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from simulation.engine import PetrolPumpSimulation, SimulationConfig
from simulation.metrics import compute_metrics
from simulation.scenarios import ScenarioConfig, ScenarioRunner


DOCS_DIR = BASE_DIR / "docs"


class PetroFlowHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving static HTML/CSS/JS from docs/ and JSON APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS_DIR), **kwargs)

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json({"status": "ok", "engine": "simpy"})
            return
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        try:
            payload = self._read_json_body()
            if self.path == "/api/simulate":
                self._handle_simulate(payload)
            elif self.path == "/api/scenarios":
                self._handle_scenarios(payload)
            elif self.path == "/api/whatif":
                self._handle_whatif(payload)
            else:
                self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _build_config(self, p: dict) -> SimulationConfig:
        avg_s = float(p.get("avgServiceTime", 5.0))
        std_s = float(p.get("serviceTimeStd", 1.5))
        return SimulationConfig(
            num_dispensers=int(p.get("numDispensers", 3)),
            arrival_rate=float(p.get("arrivalRate", 20.0)),
            avg_service_time=avg_s,
            service_time_std=std_s,
            min_service_time=float(p.get("minServiceTime", max(1.0, avg_s - 3 * std_s))),
            max_service_time=float(p.get("maxServiceTime", avg_s + 3 * std_s)),
            simulation_duration=float(p.get("simulationDuration", 120.0)),
            random_seed=int(p.get("randomSeed", 42)),
        )

    def _handle_simulate(self, p: dict) -> None:
        cfg = self._build_config(p)
        sim = PetrolPumpSimulation(cfg)
        res = sim.run()
        met = compute_metrics(res)

        self._send_json({
            "vehicles": [
                {
                    "vehicleId": v.vehicle_id,
                    "arrivalTime": v.arrival_time,
                    "queueEntryTime": v.queue_entry_time,
                    "serviceStartTime": v.service_start_time,
                    "serviceEndTime": v.service_end_time,
                    "departureTime": v.departure_time,
                    "serviceDuration": v.service_duration,
                    "dispenserId": v.dispenser_id,
                    "status": v.status.value,
                }
                for v in res.vehicles
            ],
            "queueLengthOverTime": [[t, q] for t, q in res.queue_length_over_time],
            "metrics": {
                "totalArrived": met.total_arrived,
                "totalServed": met.total_served,
                "avgWaitingTime": met.avg_waiting_time,
                "medianWaitingTime": met.median_waiting_time,
                "maxWaitingTime": met.max_waiting_time,
                "stdWaitingTime": met.std_waiting_time,
                "p95WaitingTime": met.p95_waiting_time,
                "avgTimeInSystem": met.avg_time_in_system,
                "avgQueueLength": met.avg_queue_length,
                "maxQueueLength": met.max_queue_length,
                "dispenserUtils": met.dispenser_utilisations,
                "avgUtilisation": met.avg_utilisation,
                "throughputPerHour": met.throughput_per_hour,
            },
        })

    def _handle_scenarios(self, p: dict) -> None:
        base_cfg = self._build_config(p)
        arr = float(p.get("benchmarkArrival", base_cfg.arrival_rate))
        reps = int(p.get("repetitions", 3))

        scenarios = []
        for nd in [1, 2, 3, 4]:
            sc_cfg = SimulationConfig(
                num_dispensers=nd,
                arrival_rate=arr,
                avg_service_time=base_cfg.avg_service_time,
                service_time_std=base_cfg.service_time_std,
                min_service_time=base_cfg.min_service_time,
                max_service_time=base_cfg.max_service_time,
                simulation_duration=base_cfg.simulation_duration,
                random_seed=base_cfg.random_seed,
            )
            scenarios.append(ScenarioConfig(name=f"{nd} Pump{'s' if nd > 1 else ''}", config=sc_cfg))

        results = ScenarioRunner.run_multiple(scenarios, repetitions=reps)
        out = []
        for r in results:
            rho = r.config.arrival_rate / (r.config.num_dispensers * (60.0 / r.config.avg_service_time))
            out.append({
                "name": r.name,
                "nd": r.config.num_dispensers,
                "rho": rho,
                "avgWait": r.metrics.avg_waiting_time,
                "maxQ": r.metrics.max_queue_length,
                "util": r.metrics.avg_utilisation * 100.0,
                "tp": r.metrics.throughput_per_hour,
            })
        self._send_json({"scenarios": out})

    def _handle_whatif(self, p: dict) -> None:
        base_cfg = self._build_config(p)
        param = p.get("param", "arrival")
        steps = int(p.get("steps", 6))

        vals = []
        for i in range(steps):
            if param == "arrival":
                vals.append(10.0 + (50.0 * i) / max(1, steps - 1))
            elif param == "service":
                vals.append(2.0 + (10.0 * i) / max(1, steps - 1))
            else:
                vals.append(float(i + 1))

        waits, queues = [], []
        for v in vals:
            if param == "arrival":
                cfg = SimulationConfig(**{**base_cfg.__dict__, "arrival_rate": v})
            elif param == "service":
                cfg = SimulationConfig(
                    **{
                        **base_cfg.__dict__,
                        "avg_service_time": v,
                        "min_service_time": max(1.0, v - 2.0),
                        "max_service_time": v + 2.0,
                    }
                )
            else:
                cfg = SimulationConfig(**{**base_cfg.__dict__, "num_dispensers": int(v)})

            res = PetrolPumpSimulation(cfg).run()
            met = compute_metrics(res)
            waits.append(met.avg_waiting_time)
            queues.append(met.max_queue_length)

        self._send_json({"vals": vals, "waits": waits, "queues": queues})


def main(port: int = 8000) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), PetroFlowHandler)
    print(f"PetroFlow Python Backend running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    main(port)
