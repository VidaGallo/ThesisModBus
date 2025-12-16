from docplex.mp.model import Model


# seed
import random
import numpy as np
seed = 23
random.seed(seed)
np.random.seed(seed)



def configure_cplex(
    mdl: Model,
    time_limit: float = 36_000,     # seconds
    mip_gap: float = 0.01,          # relative MIP gap, 0.01 = 1%
    abs_mip_gap: float = 1e-6,      # absolute MIP gap, 1e-6
    threads: int = 0,               # max threads
    mip_display: int = 1,           # MIP log verbosity
    emphasis_mip: int = 2,          # 2 = optimality
    parallel: int = 0               # 0 = auto
):
    """
    Configure CPLEX parameters for the given model.

    Parameters:
    ----------
    mdl : DOcplex model.

    time_limit : Time limit in seconds.

    mip_gap : Relative MIP gap tolerance.
    abs_mip_gap : Absolute MIP gap tolerance.

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
    """


    # ------------------------
    # Basic global settings
    # ------------------------

    # Time limit
    mdl.parameters.timelimit = time_limit

    # Relative and absolute MIP gap
    mdl.parameters.mip.tolerances.mipgap = mip_gap
    mdl.parameters.mip.tolerances.absmipgap = abs_mip_gap


    # Threads
    mdl.parameters.threads = threads

    # Log verbosity on nodes
    mdl.parameters.mip.display = mip_display

    # Emphasis on feasibility vs optimality
    mdl.parameters.emphasis.mip = emphasis_mip

    # Parallel search mode:
    mdl.parameters.parallel = parallel
