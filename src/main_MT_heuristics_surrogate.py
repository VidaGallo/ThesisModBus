from utils.runs_fun import *
from utils.runs_heu_GP_fun import *

import random, numpy as np, pandas as pd
from pathlib import Path



### Set seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

TIME_LIMIT_TOT = 600    # Time limit for both exact and heuristic

### Solver configuration
CPLEX_CFG_HIGH_PRECISION = {
    "time_limit": TIME_LIMIT_TOT,     
    "mip_gap": 0.1,           # relative MIP gap, 0.1 = 10%
    "abs_mip_gap": 0.1,       # absolute MIP gap, accetta differenza di ±0.1
    "threads": 0,             # 0 = all available threads
    "mip_display": 1,         # 0..5 (2 = default)
    "emphasis_mip": 2,        # 0 balanced, 1 feasibility, 2 optimality, ...
    "parallel": 0             # 0 auto, 1 opportunistic, 2 deterministic
}
CPLEX_CFG_LOW_PRECISION = {
    "time_limit": 120.0,      # sec max
    "mip_gap": 0.5,           # relative MIP gap, 0.5 = 50%
    "abs_mip_gap": 1,         # absolute MIP gap, accetta differenza di ±1
    "threads": 0,             # 0 = all available threads
    "mip_display": 1,         # 0..5 (2 = default)
    "emphasis_mip": 1,        # 0 balanced, 1 feasibility, 2 optimality, ...
    "parallel": 0             # 0 auto, 1 opportunistic, 2 deterministic
}



#### GLOBAL PARAMETERS
RUN_EXACT = True
RUN_HEUR  = True

CPLEX_CFG_EXACT = CPLEX_CFG_HIGH_PRECISION
CPLEX_CFG_HEURISTIC = CPLEX_CFG_LOW_PRECISION




if __name__ == "__main__":
    seed = 123
    set_seed(seed)

    ### MODEL NAME
    model_name = "w"      # alternative "w_flow"

    inst_params = {
        # struttura rete
        "number": 3,    # side grid,
        "horizon": 90,
        "dt": 4,    # horizon/dt    
        "mean_edge_length_km": 2.0, #3.33
        "mean_speed_kmh": 40,
        "rel_std": 0.66,      # std for arch length

        # domanda
        "num_requests": 15,
        "q_min": 1,
        "q_max": 30,
        "slack_min": 30.0,
        "alpha":  0.123,     # parameter for the distribution of the demand (high => fast exp decay, low => almost uniform)

        # flotta / capacità
        "num_modules": 3,
        "num_trails": 4,
        "Q": 10,
        "z_max": 3,     # max n. trail for main

        # costi 
        "c_km": 1.0,
        "c_uns": 100.0,

        # topologia speciale
        "num_Nw": 4,      # n. nodes for module storage/excange
        "depot": 0,
    }

    heu_params = {
        "max_seconds": TIME_LIMIT_TOT,      # Max seconds for whole heuristics 
        "epsilon": 1.23,                    # Higher epsilone => more exploration 
        "n_init_GP": 2,                     # Number of initial observations for GP
        "n_iterations_GP": 3                # Number of observations for GP
    }                                       # tot GP evaluations = n_init_GP + n_iterations_GP



    exp_id = (
        f"GRID_n{inst_params['number']}_H{inst_params['horizon']}_dt{inst_params['dt']}"
        f"_M{inst_params['num_modules']}_P{inst_params['num_trails']}_Z{inst_params.get('z_max')}"
        f"_K{inst_params['num_requests']}_Nw{inst_params['num_Nw']}_seed{seed}"
    )

    base = Path("results") / "GRID_ASYM" / exp_id
    paths = {
        "base": base,
        "exact": base / "exact",
        "GP_Nw_heu": base / "heuristic_GP_Nw",
        "summary": base / "summary",
    }
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)

    instance, network_cont_path, requests_cont_path, network_disc_path, requests_disc_path, t_max = build_instance_and_paths(
        inst_params = inst_params,
        seed = seed
    )


    # EXACT
    if RUN_EXACT:
        exact_start = time.time()
        print("\n")
        print("*"*50)
        print("EXACT")
        print("\n")
        res_exact = run_single_model(
            instance = instance, 
            model_name = model_name, 
            inst_params = inst_params,
            seed=seed, 
            exp_id=exp_id,
            base_output_folder=paths["exact"],
            cplex_cfg=CPLEX_CFG_EXACT,
        )
        pd.DataFrame([res_exact]).to_csv(paths["summary"]/ "summary_exact.csv", index=False)
        exact_end = time.time()
        exact_time = exact_end - exact_start



    # HEURISTIC
    if RUN_HEUR:
        heu_start = time.time()
        print("\n")
        print("*"*50)
        print("HEURISTIC")
        print("\n")
        res_heu = run_GP_Nw_heu_model(
            instance = instance,
            model_name = model_name,
            inst_params = inst_params,          # instance parameters
            heu_params = heu_params,            # heuristic parameters
            seed=seed, 
            exp_id=exp_id,
            base_output_folder=paths["GP_Nw_heu"],
            cplex_cfg=CPLEX_CFG_HEURISTIC,
)
        pd.DataFrame([res_heu]).to_csv(paths["summary"]/ "summary_heur.csv", index=False)
        heu_end = time.time()
        heu_time = heu_end - heu_start

    # MERGE SUMMARY
    rows = []
    if RUN_EXACT: rows.append(res_exact)
    if RUN_HEUR:  rows.append(res_heu)
    pd.DataFrame(rows).to_csv(paths["summary"]/ "summary_all.csv", index=False)




### FINAL RESULTS
print("\n")
print("*"*50)
print("FINAL COMPARISON (EXACT vs HEURISTIC)")

diff = None
if RUN_EXACT and RUN_HEUR:
    totalK = inst_params["num_requests"]

    print(f"EXACT     | served = {res_exact.get('served')} / {totalK} ({100*res_exact.get('served_ratio',0):.1f}%)")
    print(f"HEURISTIC | served = {res_heu.get('served')} / {totalK} ({100*res_heu.get('served_ratio',0):.1f}%)")

    if res_exact.get("objective") is not None and res_heu.get("objective") is not None:
        diff = ((res_heu["objective"] - res_exact["objective"]) / abs(res_exact["objective"])) * 100.0
        print(f"EXACT = {res_exact['objective']:.3f}, HEUR = {res_heu['objective']:.3f}")
        print(f"RELATIVE GAP (HEUR vs EXACT): {diff:.2f}%")
    else:
        print("RELATIVE GAP (HEUR vs EXACT): n/a (missing objective)")

    dt_abs = abs(heu_time - exact_time)
    print(f"exact = {exact_time:.2f}s, heuristic = {heu_time:.2f}s")
    print(f"TIME DIFF (HEUR vs EXACT): {dt_abs:.2f}s")