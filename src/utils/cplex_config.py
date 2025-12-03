from docplex.mp.model import Model

def configure_cplex(
    mdl: Model,
    time_limit: float = 600.0,      # seconds
    mip_gap: float = 1e-4,          # relative MIP gap
    threads: int = 0,               # max threads
    mip_display: int = 1,           # MIP log verbosity
    emphasis_mip: int = 2,          # 2 = optimality
    parallel: int = 0,              # 0 = auto
    max_nodes: int = None,
    max_sol: int = None,
    presolve_level: int = 1,        # preresolve
    aggregator_level: int = 0,      # aggregation of rows/columns
    cuts: int = -1                  # cutting planes
):
    """
    Configure CPLEX parameters for the given model.

    Parameters:
    ----------
    mdl : DOcplex model.

    time_limit : Time limit in seconds.

    mip_gap : Relative MIP gap tolerance (default: 1e-4).

    threads : Max number of threads 
                    0 = all
                    1 = 1 thread
                    ...

    mip_display : MIP log display level:
                    0 = no node log
                    1 = some
                    2 = default
                    3+ = more verbose

    emphasis_mip : MIP emphasis:
                    0 = balanced
                    1 = feasibility
                    2 = optimality
                    3 = best bound
                    4 = hidden feasible

    parallel : Controls the parallel search strategy.
                    0 = auto (CPLEX chooses the best strategy; recommended)
                    1 = opportunistic (maximum speed, but results are not reproducible)
                    2 = deterministic (reproducible results, but usually slower)

    max_nodes : Maximum number of branch-and-bound nodes to explore.

    max_sol : Maximum number of feasible solutions to keep/found.

    presolve_level : Presolve:
                    0 = off
                    1 = on

    aggregator_level : Row/column aggregation level:
                    0 = off
                    1 = conservative
                    2 = aggressive

    cuts_* : Cutting planes controls:
                    -1 = automatic (CPLEX decides)
                     0 = off
                     1/2/3 = increasing aggressiveness
    """


    # ------------------------
    # Basic global settings
    # ------------------------

    # Time limit
    mdl.parameters.timelimit = time_limit

    # Relative MIP gap
    mdl.parameters.mip.tolerances.mipgap = mip_gap

    # Threads
    mdl.parameters.threads = threads

    # Log verbosity on nodes
    mdl.parameters.mip.display = mip_display

    # Emphasis on feasibility vs optimality
    mdl.parameters.emphasis.mip = emphasis_mip

    # Parallel search mode:
    mdl.parameters.parallel = parallel

    # Limits for the MIP search:
    if max_nodes is not None:
        mdl.parameters.mip.limits.nodes = max_nodes
    if max_sol is not None:
        mdl.parameters.mip.limits.solutions = max_sol




    # ------------------------
    # Presolve & preprocessing
    # ------------------------
    mdl.parameters.preprocessing.presolve = presolve_level

    # Aggregator: row/column aggregation
    mdl.parameters.preprocessing.aggregator = aggregator_level




    # ------------------------
    # Cutting planes
    # ------------------------
    mdl.parameters.mip.cuts.mircut     = cuts        # Mixed-Integer Rounding cuts
    mdl.parameters.mip.cuts.implied    = cuts        # Implied bound cuts
    mdl.parameters.mip.cuts.gomory     = cuts        # Gomory fractional cuts
    mdl.parameters.mip.cuts.flowcovers = cuts        # Flow cover cuts (good for capacity)
    mdl.parameters.mip.cuts.pathcut    = cuts        # Path cuts (good for routing-like structure)