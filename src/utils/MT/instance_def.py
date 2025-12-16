"""
INSTANCE
=========

Instance definition for the modular bus / taxi-like MILP.

The `Instance` dataclass aggregates all network, demand, fleet and
discretization data required by the optimization model, providing a
single structured input to the solver.
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
    M: Set[Module]            # available MAIN modules
    P: Set[Module]            # available TRAIL modules
    K: Set[Request]           # demand requests
    T: List[Time]             # discrete time periods [1, ..., t_max]


    ### --- global parameters ---
    Q: int                    # module capacity
    z_max: int                # max number of trail modules per main module (if None, computed as |P|/|M|)
    c_km: float               # cost per km
    c_uns: float              # penalty cost for unserved demand
    g_plat: float             # reward for having a platoon (to subtract to the c_op)


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
    num_Nw: int = 0                   # <-- To choose
    Nw: Set[Node] = field(init=False) # Defined after the __init__

    def __post_init__(self):
        """Compute Nw as the 'num_Nw' nodes with highest degree."""
        # Compute undirected degree
        degree = {i: 0 for i in self.N}
        for (i, j) in self.A:
            degree[i] += 1
            degree[j] += 1

        # Order nodes by degree (descending)
        nodes_sorted = sorted(degree.items(), key=lambda x: x[1], reverse=True)

        # Take first num_Nw nodes
        if self.num_Nw > 0:
            self.Nw = {node for node, deg in nodes_sorted[:self.num_Nw]}
        else:
            self.Nw = set()
    


    @property
    def num_nodes(self) -> int:
        return len(self.N)

    @property
    def num_requests(self) -> int:
        """|K| – number of requests."""
        return len(self.K)

    @property
    def num_modules(self) -> int:
        """|M| – numero di moduli MAIN."""
        return len(self.M)

    @property
    def num_trail_modules(self) -> int:
        """|P| – numero di moduli TRAIL."""
        return len(self.P)

    @property
    def Z_max(self) -> int:
        """
        Numero massimo di TRAIL agganciabili a un MAIN.
        """
        # ---- caso utente ----
        if self.z_max is not None:
            z = int(self.z_max)
            if z < 0:
                raise ValueError("z_max deve essere >= 0")
            if self.num_modules == 0:
                if z == 0 and self.num_trail_modules == 0:
                    return 0
                raise ValueError(
                    "z_max > 0 o TRAIL presenti senza MAIN disponibili"
                )
            if z > self.num_trail_modules:
                raise ValueError(
                    f"z_max={z} > |P|={self.num_trail_modules}"
                )
            return z

        # ---- fallback automatico ----
        if self.num_modules == 0:
            if self.num_trail_modules == 0:
                return 0
            raise ValueError("TRAIL presenti ma nessun MAIN disponibile")

        if self.num_trail_modules % self.num_modules != 0:
            raise ValueError(
                f"|P|={self.num_trail_modules} non multiplo di |M|={self.num_modules}"
            )

        return self.num_trail_modules // self.num_modules
    

