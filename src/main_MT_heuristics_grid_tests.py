from utils.tests_fun import *

import random, numpy as np, pandas as pd
from pathlib import Path



### Set seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


### Solver configuration
CPLEX_CFG_HIGH_PRECISION = {
    "time_limit": 36_000,     # 1 day max
    "mip_gap": 0.1,           # relative MIP gap, 0.1 = 10%
    "abs_mip_gap": 0.1,       # absolute MIP gap, accetta differenza di ±0.1
    "threads": 0,             # 0 = all available threads
    "mip_display": 1,         # 0..5 (2 = default)
    "emphasis_mip": 2,        # 0 balanced, 1 feasibility, 2 optimality, ...
    "parallel": 0             # 0 auto, 1 opportunistic, 2 deterministic
}
CPLEX_CFG_LOW_PRECISION = {
    "time_limit": 60.0,      # 10min max
    "mip_gap": 0.5,           # relative MIP gap, 0.5 = 50%
    "abs_mip_gap": 1,         # absolute MIP gap, accetta differenza di ±1
    "threads": 0,             # 0 = all available threads
    "mip_display": 1,         # 0..5 (2 = default)
    "emphasis_mip": 1,        # 0 balanced, 1 feasibility, 2 optimality, ...
    "parallel": 0             # 0 auto, 1 opportunistic, 2 deterministic
}


### Probabilità per demand generation
PI_OD = {"PC": 0.35, "CP": 0.30, "CC": 0.20, "PP": 0.15}

PI_L_given_OD = {
    "PC": {"S": 0.15, "M": 0.45, "L": 0.40},
    "CP": {"S": 0.15, "M": 0.45, "L": 0.40},
    "CC": {"S": 0.45, "M": 0.45, "L": 0.1},   
    "PP": {"S": 0.25, "M": 0.45, "L": 0.30},
}




#### GLOBAL PARAMETERS
RUN_EXACT = True
RUN_HEUR  = False
WARM_START = True

CPLEX_CFG_EXACT = CPLEX_CFG_HIGH_PRECISION
CPLEX_CFG_HEURISTIC = CPLEX_CFG_LOW_PRECISION


heu_params = {
        "n_keep": 5,      # n. nodes that are fixed after each iteration (GRIGI)
        "it_in": 15,
        "n_clust": 5,
        "warm_start_bool": WARM_START    # Start minirouting with the best solution founded in the inner loop so far   
    }


if __name__ == "__main__":
    model_name = "w"

    grid_n = 2                 # (2x2 / 3x3 / ...)
    n_instances = 1           # How many different parameter-regimes to test
    repeat_seeds = [11, 15, 23, 123, 99]   # How many times to repeat each regime

    regimes = generate_instance_regimes(
        grid_n=grid_n,
        n_instances=n_instances,
        generator_seed=20250101,   # stable generator
    )



    df_all = run_generated_regimes(
        regimes=regimes,
        repeat_seeds=repeat_seeds,
        model_name=model_name,
        base_results_dir=Path("results") / "GRID_ASYM",
        run_exact=RUN_EXACT,
        run_heur=RUN_HEUR,
        cplex_cfg_exact=CPLEX_CFG_EXACT,
        cplex_cfg_heur=CPLEX_CFG_HEURISTIC,
        heu_params=heu_params,
    )
