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
|  
├── instances/  
|   ├── GRID/  
│   |   ├── ...  
|   |  
│   └── CITY/  
│       ├── ...  
|  
|  
├── results/  
|   ├── GRID/  
│   |   ├── ...  
|   |  
│   └── CITY/  
│       ├── ...  
|  
|  
└── src/  
    ├── models/  
    │   ├── deterministic/  
    │   │   ├── ...
    │   │  
    │   └── stochastic/  
    │       ├── ..  
    │  
    │  
    ├── data_generation/  
    │   ├── generate_data.py       &nbsp;&nbsp;# data generation and discretization  
    │   ├── generate_network.py    &nbsp;&nbsp;# generate networks (continuous time)  
    │   ├── generate_demands.py    &nbsp;&nbsp;# generate demands (continuous time)  
    |   └── time_discretization.py &nbsp;&nbsp;# functions to discretize  
    │  
    │  
    ├── utils/  
    │   ├── MT       &nbsp;&nbsp;# Main-Trail model  
    │   │   ├── cplex_config.py       &nbsp;&nbsp;# CPLEX configurations for the solver     
    │   │   ├── instance_def.py       &nbsp;&nbsp;# class Instance   
    │   │   ├── loader_fun.py         &nbsp;&nbsp;# function to load the Parameters    
    |   │   ├── output_fun.py         &nbsp;&nbsp;# savign and managin outputs    
    │   │   ├── print_fun.py          &nbsp;&nbsp;# function to print info   
    │   │   └── runs_fun              &nbsp;&nbsp;# function to run the tests  
    │   └── Simple    &nbsp;&nbsp;# Simple model
    │       ├── cplex_config.py       &nbsp;&nbsp;# CPLEX configurations for the solver      
    │       ├── instance_def.py       &nbsp;&nbsp;# class Instance    
    │       ├── loader_fun.py         &nbsp;&nbsp;# function to load the Parameters     
    |       ├── output_fun.py         &nbsp;&nbsp;# savign and managin outputs     
    │       ├── print_fun.py          &nbsp;&nbsp;# function to print info   
    │       └── runs_fun              &nbsp;&nbsp;# function to run the test    
    │  
    │  
    │  
    ├── main_MT_grid.py               &nbsp;&nbsp;# main for the "Main-Trail" model on a grid   
    ├── main_simple_city.py           &nbsp;&nbsp;# main for the "Simple" model on a city   
    └── main_simple_grid.py           &nbsp;&nbsp;# main for the "Simple" model on a grid 
