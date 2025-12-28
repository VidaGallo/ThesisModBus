"""
CPLEX Configuration Module
==========================

Configure CPLEX solver parameters for MILP experiments.

This function sets time limits, optimality gaps, logging level,
parallelism, and search emphasis in a centralized and reproducible way.
"""


from docplex.mp.model import Model


from docplex.mp.model import Model


def configure_cplex(mdl: Model, cfg: dict | None = None):
    """
    Configure CPLEX parameters for the given model using a config dictionary.

    Expected keys in cfg (all optional):
    -----------------------------------
    time_limit     : seconds
    mip_gap        : relative MIP gap
    abs_mip_gap    : absolute MIP gap
    threads        : number of threads
    mip_display    : MIP log verbosity
    emphasis_mip   : MIP emphasis
    parallel       : parallel mode
    """

    if cfg is None:
        cfg = {}

    # ------------------------
    # Basic global settings
    # ------------------------

    # Time limit (seconds)
    mdl.parameters.timelimit = cfg.get(
        "time_limit",
        36_000      # seconds
    )

    # Relative and absolute MIP gap
    mdl.parameters.mip.tolerances.mipgap = cfg.get(
        "mip_gap",
        0.01        # relative MIP gap, 0.01 = 1%
    )

    mdl.parameters.mip.tolerances.absmipgap = cfg.get(
        "abs_mip_gap",
        1e-6        # absolute MIP gap
    )

    # Threads
    mdl.parameters.threads = cfg.get(
        "threads",
        0           # 0 = all available threads
    )

    # Log verbosity on nodes
    mdl.parameters.mip.display = cfg.get(
        "mip_display",
        1           # 0=no log, 1=some, 2=default, 3+=verbose
    )

    # Emphasis on feasibility vs optimality
    mdl.parameters.emphasis.mip = cfg.get(
        "emphasis_mip",
        2           # 2 = optimality
    )

    # Parallel search mode
    mdl.parameters.parallel = cfg.get(
        "parallel",
        0           # 0=auto, 1=opportunistic, 2=deterministic
    )
