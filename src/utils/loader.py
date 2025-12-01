
"""
Instance Loader for CPLEX
=========================

Utilities to build all sets and parameters needed by the MILP model,
starting from:
    - discretized network JSON (e.g. network_***maxmin_disc***max.json)
    - discretized requests JSON (e.g. taxi_like_requests_***maxmin_disc***max.json)

This module constructs:
    Sets:
        - N            : set of nodes
        - A            : set of arcs (i,j)
        - K            : set of requests
        - T            : set of time periods {1,...,t_max}
        - M            : set of modules {1,...,|M|}
    Parameters:
        - gamma[i,j]   : edge length (km)
        - tau[i,j]     : discrete travel time in periods (time_steps)
        - q[k]         : demand of request k
        - orig[k]      : origin node of request k
        - dest[k]      : destination node of request k
        - Tk[k]        : list of time indices ΔT_k
        - Tin[k]       : list of indices ΔT_k^{in}
        - Tout[k]      : list of indices ΔT_k^{out}
        - d_in[k,i,t]  : 1 if request k can board at node i at time t
        - d_out[k,i,t] : 1 if request k can alight at node i at time t

All time indices start from t = 1.
"""



from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import math




#################
# INSTANCE CLASS:
#################
 
@dataclass
class Instance:
    # set
    N: List[int]
    A: List[Tuple[int, int]]
    T: List[int]
    K: List[int]
    M: List[int]

    # parameters
    q: Dict[int, float]
    gamma: Dict[Tuple[int, int], float]
    tau: Dict[Tuple[int, int], int]

    DeltaT: Dict[int, List[int]]
    DeltaT_in: Dict[int, List[int]]
    DeltaT_out: Dict[int, List[int]]

    d_in: Dict[Tuple[int, int, int], int]
    d_out: Dict[Tuple[int, int, int], int]

    origin: Dict[int, int]
    destination: Dict[int, int]

    t_max: int
    Tmax_min: int       # orizzonte continuo (minuti)
    dt: int             # discretizzazione (minuti)
    time_step_min: Optional[float]

    # costi (se vuoi passarli)
    C_KM: Optional[float] = None
    C_SECOND: Optional[float] = None



