
"""
Instance Loader for CPLEX
=========================

Utilities to build all sets and parameters needed by the MILP model,
starting from:
    - discretized network JSON 
    - discretized requests JSON 

All time indices start from t = 1.
"""



import json
import math
from pathlib import Path
from typing import Tuple, Dict, List, Set

from .instance_def import *



##### Load from path
def load_discrete_network(network_path: str):
    """
    Load a DISCRETE network JSON (already discretized via time_discretization).

    Expected edge attributes:
        - length_km
        - time_steps   (integer discrete travel time)
    """
    path = Path(network_path)
    with path.open("r", encoding="utf-8") as f:
        net = json.load(f)

    N: Set[Node] = set()
    A: Set[Arc] = set()
    gamma: Dict[Arc, float] = {}
    tau_arc: Dict[Arc, int] = {}

    # Nodes
    for node in net["nodes"]:
        N.add(int(node["id"]))

    # Edges
    for e in net["edges"]:
        i = int(e["u"])
        j = int(e["v"])
        arc = (i, j)

        A.add(arc)
        gamma[arc] = float(e["length_km"])
        tau_arc[arc] = int(e["time_steps"])  # already discrete!

    return N, A, gamma, tau_arc




##### Load from a dict
Node = int
Arc = Tuple[int, int]
def load_discrete_network_from_dict(net: Dict) -> tuple[Set[Node], Set[Arc], Dict[Arc, float], Dict[Arc, int]]:
    N: Set[Node] = set()
    A: Set[Arc] = set()
    gamma: Dict[Arc, float] = {}
    tau_arc: Dict[Arc, int] = {}

    for node in net["nodes"]:
        N.add(int(node["id"]))

    for e in net["edges"]:
        i = int(e["u"])
        j = int(e["v"])
        arc = (i, j)
        A.add(arc)
        gamma[arc] = float(e["length_km"])
        tau_arc[arc] = int(e["time_steps"])  # già discreto

    return N, A, gamma, tau_arc




##### Load from path
def load_discrete_requests(requests_path: str, t_max: int):
    """
    Load DISCRETE taxi-like requests where each request already contains:
        - T_k_idx
        - T_in_idx
        - T_out_idx
    """
    path = Path(requests_path)
    with path.open("r", encoding="utf-8") as f:
        reqs = json.load(f)

    K: Set[Request] = set()
    q: Dict[Request, int] = {}
    origin: Dict[Request, Node] = {}
    dest: Dict[Request, Node] = {}
    DeltaT: Dict[Request, List[Time]] = {}
    DeltaT_in: Dict[Request, List[Time]] = {}
    DeltaT_out: Dict[Request, List[Time]] = {}

    d_in: Dict[Tuple[Request, Node, Time], int] = {}
    d_out: Dict[Tuple[Request, Node, Time], int] = {}

    for r in reqs:
        k = int(r["id"])
        K.add(k)

        origin[k] = int(r["origin"])
        dest[k] = int(r["destination"])
        q[k] = int(r["q_k"])

        # already discrete
        DeltaT[k] = r["T_k_idx"]
        DeltaT_in[k] = r["T_in_idx"]
        DeltaT_out[k] = r["T_out_idx"]

        # Build sparse d_in and d_out
        i_orig = origin[k]
        i_dest = dest[k]

        for t in DeltaT_in[k]:
            if 1 <= t <= t_max:
                d_in[(k, i_orig, t)] = 1

        for t in DeltaT_out[k]:
            if 1 <= t <= t_max:
                d_out[(k, i_dest, t)] = 1

    return K, q, origin, dest, DeltaT, DeltaT_in, DeltaT_out, d_in, d_out



##### Load from a list
Request = int
Time = int
def load_discrete_requests_from_list(reqs: List[Dict], t_max: int):
    K: Set[Request] = set()
    q: Dict[Request, int] = {}
    origin: Dict[Request, Node] = {}
    dest: Dict[Request, Node] = {}
    DeltaT: Dict[Request, List[Time]] = {}
    DeltaT_in: Dict[Request, List[Time]] = {}
    DeltaT_out: Dict[Request, List[Time]] = {}
    d_in: Dict[Tuple[Request, Node, Time], int] = {}
    d_out: Dict[Tuple[Request, Node, Time], int] = {}

    for r in reqs:
        k = int(r["id"])
        K.add(k)

        origin[k] = int(r["origin"])
        dest[k]   = int(r["destination"])

        # se q_k non è int, scegli regola:
        # q[k] = int(r["q_k"])                 # troncamento
        # q[k] = math.ceil(float(r["q_k"]))    # ceil
        q[k] = math.floor(float(r["q_k"]) + 0.5)  # 0.5 -> su (come volevi)

        DeltaT[k]     = list(r["T_k_idx"])
        DeltaT_in[k]  = list(r["T_in_idx"])
        DeltaT_out[k] = list(r["T_out_idx"])

        i_orig = origin[k]
        i_dest = dest[k]

        for t in DeltaT_in[k]:
            if 1 <= t <= t_max:
                d_in[(k, i_orig, int(t))] = 1

        for t in DeltaT_out[k]:
            if 1 <= t <= t_max:
                d_out[(k, i_dest, int(t))] = 1

    return K, q, origin, dest, DeltaT, DeltaT_in, DeltaT_out, d_in, d_out




##### Creation of the Instance Class with attributes inside, from path
def load_instance_discrete(
    network_path: str,
    requests_path: str,
    dt: int,
    t_max: int,
    num_modules: int,   # |M|
    num_trail: int,     # |P|
    Q: int,
    c_km: float,
    c_uns: float,
    depot: int,
    num_Nw: int,
    z_max: int | None = None
) -> Instance:
    """
    Build an Instance using already-discrete files.

    Parameters
    ----------
    network_path : str   # path to the network .json
    requests_path : str  # path to the requests .json
    dt : int             # discretization step (minutes)
    t_max : int          # number of time slots
    num_modules : int    # number of MAIN modules |M|
    num_trail : int      # number of TRAIL modules |P|
    z_max: int | None    # max number of trail modules per main module (if None, computed as |P|/|M|)
    Q : int              # module capacity
    c_km : float         # cost per km
    c_uns : float        # unserved demand penalty
    depot : int          # depot node
    num_Nw : int         # number of internal nodes where swaps are allowed
    """

    ### Discrete network
    N, A, gamma, tau_arc = load_discrete_network(network_path)

    ### Time set
    T = list(range(1, t_max + 1))     # {1,2,...,t_max}

    ### Modules
    M = set(range(1, num_modules + 1))    # {1,2,...,|M|}
    P = set(range(1, num_trail + 1))     # TRAIL modules {1,...,|P|}

    ### Discrete requests
    (
        K, q, origin, dest,
        DeltaT, DeltaT_in, DeltaT_out,
        d_in, d_out
    ) = load_discrete_requests(requests_path, t_max)


    ### Build Class Instance
    return Instance(
        N=N,
        A=A,
        M=M,
        P=P,
        K=K,
        T=T,
        Q=Q,
        c_km=c_km,
        c_uns=c_uns,
        gamma=gamma,
        tau_arc=tau_arc,
        q=q,
        origin=origin,
        dest=dest,
        DeltaT=DeltaT,
        DeltaT_in=DeltaT_in,
        DeltaT_out=DeltaT_out,
        d_in=d_in,
        d_out=d_out,
        depot=depot,
        dt=dt,
        t_max=t_max,
        num_Nw=num_Nw,
        z_max=z_max
    )




##### Creation of the Instance Class with attributes inside, from data
def load_instance_discrete_from_data(
    net_discrete: Dict,                 # dict rete DISCRETA (come json.load)
    reqs_discrete: List[Dict],          # lista richieste DISCRETE (come json.load)
    *,
    dt: int,
    t_max: int,
    num_modules: int,
    num_trail: int,
    Q: int,
    c_km: float,
    c_uns: float,
    depot: int,
    num_Nw: int,
    z_max: int | None = None,
) -> Instance:

    N, A, gamma, tau_arc = load_discrete_network_from_dict(net_discrete)
    T = list(range(1, t_max + 1))
    M = set(range(1, num_modules + 1))
    P = set(range(1, num_trail + 1))

    (K, q, origin, dest, DeltaT, DeltaT_in, DeltaT_out, d_in, d_out) = \
        load_discrete_requests_from_list(reqs_discrete, t_max)

    return Instance(
        N=N, A=A, M=M, P=P, K=K, T=T,
        Q=Q, c_km=c_km, c_uns=c_uns,
        gamma=gamma, tau_arc=tau_arc,
        q=q, origin=origin, dest=dest,
        DeltaT=DeltaT, DeltaT_in=DeltaT_in, DeltaT_out=DeltaT_out,
        d_in=d_in, d_out=d_out,
        depot=depot, dt=dt, t_max=t_max,
        num_Nw=num_Nw, z_max=z_max
    )
