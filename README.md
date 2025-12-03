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
    │   │   ├── taxi_like.mod   
    │   │   └── general.mod   
    │   │   
    │   └── stochastic/          
    │       └──general_two_stage.mod    
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
    │   ├── cplex_config.py        &nbsp;&nbsp;# CPLEX configurations for the solver of the model    
    │   ├── instance.py            &nbsp;&nbsp;# class Instance      
    │   ├── loaders.py             &nbsp;&nbsp;# function to load the Parameters     
    |   ├── outputs.py             &nbsp;&nbsp;# savign and managin outputs    
    │   └── print_data.py          &nbsp;&nbsp;# function to print info    
    │      
    │    
    │  
    ├── main_taxi_like_test.py     &nbsp;&nbsp;# multiple tests with multiple instances    
    └── main_taxi_like.py          &nbsp;&nbsp;# simple main