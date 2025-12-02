# ThesisModBus     
Python code for a Modular Bus Operational Model     
Copyright (c) 2025 Gabriele Pirilli, Vida Gallo    



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
|
├── src/  
│   ├── models/  
│   │   ├── deterministic/  
│   │   │   ├── taxi_like.mod  
│   │   │   └── general.mod  
│   │   │  
│   │   └── stochastic/        
│   │       └──general_two_stage.mod  
│   │  
│   │  
│   ├── solvers/  
│   │   ├── solve_cplex_deterministic.py   
│   │   └── solve_cplex_stochastic.py  
│   │  
│   │  
│   ├── data_generation/           &nbsp;&nbsp;# REMARK: continuous time   
│   │   ├── generate_network.py    &nbsp;&nbsp;# graph construction → grid_network.json, city_network.json   
│   │   ├── generate_bus_lines.py  &nbsp;&nbsp;# line construction    
│   │   └── generate_demands.py    &nbsp;&nbsp;# simulator  → taxi-like_requests.json, bus-like_requests.json    
│   │   
│   │  
│   ├── utils/   
│   │   ├── instance.py            &nbsp;&nbsp;# class Instance   
│   │   ├── loaders.py             &nbsp;&nbsp;# function to load the Parameters  
│   │   ├── print_data.py          &nbsp;&nbsp;# function to print info   
│   │   └── time_discretization.py &nbsp;&nbsp;# functions to discretize   
│   │   
│   │  
│   └── experiments/   
│       ├── GRID/       
│       │   ├── config.yaml   
│       │   ├── run.py   
│       │   ├── 01/   
│       │   ├── 02/   
│       │   └── ...   
│       └── CITY/       
│           ├── config.yaml    
│           ├── run.py   
│           ├── 01/  
│           └── ...      
│    
│     
└── results/  
    ├── GRID/   
    │   ├── logs/   
    │   ├── solutions/   
    │   ├── figures/    
    │   └── tables/    
    └── CITY/    
        ├── logs/   
        ├── solutions/     
        ├── figures/    
        └── tables/   
     