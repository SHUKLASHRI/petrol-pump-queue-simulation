# ⛽ Petrol Pump Queue Simulation and Modelling

A discrete-event simulation and modelling application that analyses petrol station queue performance using Python, SimPy, Streamlit, and Plotly.

**Subject**: Problem Solving Through Simulation and Modelling

---

## 📋 Project Overview

This project simulates how vehicles arrive at a petrol station, wait in queues, receive fuel, and depart. It demonstrates how discrete-event simulation and mathematical modelling can be used to analyse real-world queueing problems and help decision-makers optimise petrol station operations.

The interactive dashboard allows users to experiment with different configurations — number of fuel dispensers, arrival rates, service times — and instantly see the impact on queue length, waiting times, and resource utilisation.

## 🎯 Problem Statement

Petrol stations face a fundamental operational challenge: balancing the number of fuel dispensers against fluctuating vehicle demand. Too few dispensers lead to long queues and frustrated customers; too many result in wasted capital and idle resources.

This project uses simulation to answer questions such as:
- How many dispensers are needed to keep average waiting time below a target?
- What happens to queue length during peak hours?
- At what arrival rate does the system become unstable?
- What is the optimal balance between cost and customer satisfaction?

## 🎯 Objectives

1. Model vehicle arrivals using a **Poisson process** with exponential inter-arrival times.
2. Model fuel service times using a **truncated normal distribution**.
3. Implement a multi-server **FCFS (First Come, First Served)** queueing system.
4. Compute key performance metrics: waiting times, queue lengths, utilisation, throughput.
5. Validate simulation results against **queueing theory** (Little's Law, traffic intensity).
6. Provide interactive scenario comparison and what-if analysis.
7. Build a professional, presentation-ready dashboard.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.11+** | Core programming language |
| **SimPy** | Discrete-event simulation engine |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive charts and visualisations |
| **NumPy** | Numerical calculations |
| **Pandas** | Data processing and tabular display |
| **SciPy** | Truncated normal distribution for service times |
| **Pytest** | Automated testing |

## 📐 Mathematical Model

### Poisson Arrival Process

Vehicles arrive according to a Poisson process with rate λ (vehicles/hour). Inter-arrival times follow an exponential distribution:

```
Inter-arrival time = −ln(U) / λ
```

where U ~ Uniform(0, 1).

### Service Time Distribution

Fuel service times follow a truncated normal distribution:

```
Service time ~ TruncatedNormal(μ, σ², min, max)
```

### Traffic Intensity

```
ρ = λ / (c × μ)
```

- λ = arrival rate (vehicles/hour)
- c = number of dispensers
- μ = service rate per dispenser (vehicles/hour)

**Stability condition**: ρ < 1 (arrival rate must be less than total service capacity).

### Little's Law

```
L = λ × W
```

- L = average number of vehicles in the system
- λ = arrival rate
- W = average time in the system

## 🔬 Simulation Methodology

The simulation uses a **discrete-event simulation (DES)** approach:

1. **Vehicle Generator**: Creates vehicles at random intervals following the Poisson arrival process.
2. **Queue Manager**: Vehicles join a shared FCFS queue when all dispensers are busy.
3. **Fuel Dispensers**: SimPy `Resource` objects that serve one vehicle at a time.
4. **Data Collector**: Records timestamps for arrival, queue entry, service start, service end, and departure for every vehicle.
5. **Metrics Engine**: Computes aggregate statistics from the collected data.

## 🏗️ Project Architecture

```
petrol-pump-simulation/
│
├── app.py                    # Streamlit dashboard application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
│
├── simulation/              # Core simulation logic
│   ├── __init__.py
│   ├── engine.py            # SimPy-based simulation engine
│   ├── vehicle.py           # Vehicle entity and status
│   ├── dispenser.py         # Fuel dispenser tracking
│   ├── metrics.py           # Statistical metrics computation
│   └── scenarios.py         # Scenario comparison runner
│
├── visualization/           # Charts and dashboard components
│   ├── __init__.py
│   ├── charts.py            # Plotly chart generators
│   └── dashboard.py         # Streamlit dashboard widgets
│
├── utils/                   # Helper utilities
│   ├── __init__.py
│   └── helpers.py           # Data conversion and formatting
│
├── tests/                   # Automated tests
│   ├── test_simulation.py   # Tests for engine, vehicle, dispenser
│   └── test_metrics.py      # Tests for metrics and scenarios
│
└── assets/
    └── logo.svg             # Application logo
```

## 🚀 Installation

### Prerequisites

- Python 3.11 or later
- pip (Python package manager)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SHUKLASHRI/petrol-pump-queue-simulation.git
   cd petrol-pump-queue-simulation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Run the tests**:
   ```bash
   pytest tests/ -v
   ```

## 📊 Example Simulation Scenarios

### Scenario 1: Low Traffic
- Dispensers: 2
- Arrival rate: 10 vehicles/hour
- Avg service time: 5 minutes
- Expected: short queues, low utilisation (~42%)

### Scenario 2: Moderate Traffic
- Dispensers: 3
- Arrival rate: 30 vehicles/hour
- Avg service time: 5 minutes
- Expected: manageable queues, moderate utilisation (~83%)

### Scenario 3: Peak Hour
- Dispensers: 3
- Arrival rate: 40 vehicles/hour
- Avg service time: 5 minutes
- Expected: ρ > 1 (unstable), growing queues

### Scenario 4: Capacity Planning
- Compare 1, 2, 3, and 4 dispensers at 20 vehicles/hour to find the optimal number.

## 📈 Expected Outputs

The dashboard provides:

1. **KPI Cards**: Total arrivals, vehicles served, average/max waiting time, queue length, utilisation.
2. **Live Visualisation**: Time-stepped station view showing dispensers, queue, and vehicle states.
3. **Queue Analysis**: Interactive Plotly charts for queue length, arrivals, completions, waiting time distribution, and dispenser utilisation.
4. **Statistical Analysis**: Detailed metrics with explanations, plus Little's Law verification.
5. **Scenario Comparison**: Side-by-side comparison of different configurations with auto-generated recommendations.
6. **What-If Analysis**: Sensitivity analysis showing how key metrics change as parameters vary.
7. **Mathematical Model**: Full explanation of the underlying queueing theory.

## ⚠️ Limitations

1. **Poisson assumption**: Real arrival patterns may have time-of-day effects, batch arrivals, or dependencies.
2. **No vehicle abandonment**: The model assumes all vehicles wait regardless of queue length.
3. **Single queue**: All dispensers share one queue; in practice, drivers may choose lanes.
4. **No breakdowns**: Dispenser failures and maintenance are not modelled.
5. **Infinite queue capacity**: No physical limit on waiting space.
6. **No warm-up removal**: Transient effects at the start of the simulation are included in statistics.
7. **Homogeneous vehicles**: All vehicles are treated identically (no trucks, motorcycles, etc.).

## 🔮 Future Improvements

1. **Time-varying arrival rates** (rush hour patterns).
2. **Vehicle classes** (cars, trucks, motorcycles with different service times).
3. **Customer abandonment** (balking and reneging).
4. **Multiple queue lanes** with lane-choice behaviour.
5. **Cost modelling** (dispenser cost vs. customer waiting cost).
6. **Warm-up period detection and removal** for steady-state statistics.
7. **Confidence intervals** from multiple replications.
8. **Real-time animation** using WebSocket or alternative frameworks.

## 📸 Screenshots

*Screenshots can be added after running the application.*

## 🎓 Academic Relevance

This project demonstrates key concepts from **Problem Solving Through Simulation and Modelling**:

- **Discrete-Event Simulation**: Modelling a system by processing events in chronological order.
- **Stochastic Modelling**: Using probability distributions (Poisson, exponential, truncated normal) to model real-world randomness.
- **Queueing Theory**: Applying M/G/c queue models, traffic intensity, and Little's Law.
- **Statistical Analysis**: Computing and interpreting performance metrics from simulation data.
- **Scenario Analysis**: Comparing alternative system designs to support decision-making.
- **Validation**: Checking simulation output against theoretical predictions.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

