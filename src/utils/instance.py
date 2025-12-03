"""
Instance definition for the modular bus / taxi-like MILP.

This module defines the `Instance` dataclass, which aggregates all sets,
parameters and sparse data structures needed by the optimization model:
- network sets (nodes, arcs) and travel times,
- fleet data (modules, capacity, costs),
- request data (origins, destinations, time windows, demand),
- sparse boarding/alighting indicators d_in, d_out,
- basic discretization info (time step, time horizon).

The `Instance` object is the single container that is built by the loader
(from JSON files) and then passed to model builders / solvers (e.g. CPLEX/OPL
or docplex).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set


Node = int
Module = int
Request = int
Time = int
Arc = Tuple[Node, Node]


@dataclass   # __init__ created in auto
class Instance:
    ### --- sets ---
    N: Set[Node]              # nodes
    A: Set[Arc]               # directed arcs (i, j)
    M: Set[Module]            # available modules
    K: Set[Request]           # taxi-like requests
    T: List[Time]             # discrete time periods [1, ..., t_max]


    ### --- global parameters ---
    Q: int                    # module capacity
    c_km: float               # cost per km
    c_uns_taxi: float         # penalty cost for unserved taxi-like demand


    ### --- network-related parameters ---
    gamma: Dict[Arc, float]   # arc length
    tau_arc: Dict[Arc, int]   # discrete travel time on each arc


    ### --- request-related parameters ---
    q: Dict[Request, int]                  # requested quantity per request
    origin: Dict[Request, Node]            # origin node of each request
    dest: Dict[Request, Node]              # destination node of each request
    DeltaT: Dict[Request, List[Time]]      # overall feasible service window
    DeltaT_in: Dict[Request, List[Time]]   # feasible boarding times
    DeltaT_out: Dict[Request, List[Time]]  # feasible alighting times


    ### --- sparse boarding / alighting matrices ---
    # d_in[(k, i, t)] = 1 if request k can board at node i at time t, else 0
    # d_out[(k, i, t)] = 1 if request k can alight at node i at time t, else 0
    d_in: Dict[Tuple[Request, Node, Time], int] = field(default_factory=dict)
    d_out: Dict[Tuple[Request, Node, Time], int] = field(default_factory=dict)


    ### --- technical info ---
    dt: int = 1              # time discretization step (minutes)
    t_max: int = 0           # time horizon in number of discrete periods


    ### --- depot ---
    depot: int = 0

    ### --- internal nodes ---
    Nw: Set[Node] = field(init=False)

    def __post_init__(self):
        """Automatic computation of Nw"""
        degree = {i: 0 for i in self.N}
        for (i, j) in self.A:
            degree[i] += 1
            degree[j] += 1   # grado non orientato

        self.Nw = {i for i, deg in degree.items() if deg >= 3}
    
    @property  # attribute "read-only"
    def num_nodes(self) -> int:
        return len(self.N)


    @property
    def num_requests(self) -> int:
        return len(self.K)


    @property
    def num_modules(self) -> int:
        return len(self.M)
    

