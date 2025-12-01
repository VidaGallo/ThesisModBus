# ThesisModBus
Python code for a Modular Bus Operational Model



## Requirements
Python 3.10
CPLEX ≥ 22.1
libraries in requirements.txt



## Code structure
ThesisModBus/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   ├── models/
│   │   ├── deterministic/
│   │   │   ├── taxi_like.mod
│   │   │   └── general.mod
│   │   │
│   │   └── stochastic/      
│   │       └──general_two_stage.mod
│   │
│   ├── solvers/
│   │   ├── solve_cplex_deterministic.py
│   │   └── solve_cplex_stochastic.py
│   │
│   ├── data_generation/
│   │   ├── generate_graph.py
│   │   ├── generate_demands.py
│   │   ├── assign_travel_times.py
│   │   └── format_instance.py
│   │
│   ├── utils/
│   │   ├── loaders.py
│   │   ├── time_expansion.py
│   │   ├── solver_utils.py
│   │   └── visualization.py
│   │
│   └── experiments/
│       ├── S/     # small experiments
│       │   ├── config.yaml
│       │   ├── run.py
│       │   ├── 01/
│       │   ├── 02/
│       │   └── ...
│       ├── M/     # medium experiments
│       │   ├── config.yaml
│       │   ├── run.py
│       │   ├── 01/
│       │   └── ...
│       └── L/     # large experiments
│           ├── config.yaml
│           ├── run.py
│           ├── 01/
│           └── ...
│
├── instances/
│   ├── raw/        # istanze generate da script
│   ├── formatted/  # istanze pronte per solver
│   ├── S/          # set S
│   ├── M/          # set M
│   └── L/          # set L
│
└── results/
    ├── S/
    │   ├── logs/
    │   ├── solutions/
    │   ├── figures/
    │   └── tables/
    ├── M/
    │   ├── logs/
    │   ├── solutions/
    │   ├── figures/
    │   └── tables/
    └── L/
        ├── logs/
        ├── solutions/
        ├── figures/
        └── tables/
