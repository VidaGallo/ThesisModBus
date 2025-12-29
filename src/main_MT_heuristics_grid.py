from utils.runs_fun import *
from utils.runs_heu_random_fun import *

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
    "time_limit": 600,        # 10 min max
    "mip_gap": 0.5,           # relative MIP gap, 0.5 = 50%
    "abs_mip_gap": 1,         # absolute MIP gap, accetta differenza di ±1
    "threads": 0,             # 0 = all available threads
    "mip_display": 1,         # 0..5 (2 = default)
    "emphasis_mip": 2,        # 0 balanced, 1 feasibility, 2 optimality, ...
    "parallel": 0             # 0 auto, 1 opportunistic, 2 deterministic
}



#### GLOBAL PARAMETERS
RUN_EXACT = True
RUN_HEUR  = True
WARM_START = True

CPLEX_CFG_EXACT = CPLEX_CFG_HIGH_PRECISION
CPLEX_CFG_HEURISTIC = CPLEX_CFG_LOW_PRECISION




if __name__ == "__main__":
    seed = 23
    set_seed(seed)

    ### MODEL NAME
    model_name = "w"      # alternative "w_flow"

    inst_params = {
        # struttura rete
        "number": 3,    # side gridmber,
        "horizon": 60,
        "dt": 5,
        "mean_edge_length_km": 3.33,
        "mean_speed_kmh": 40,
        "rel_std": 0.66,      # std for arch length

        # domanda
        "num_requests": 5,
        "q_min": 1,
        "q_max": 10,
        "slack_min": 20.0,
        "alpha":  0.65,     # parameter for the distribution of the demand

        # flotta / capacità
        "num_modules": 3,
        "num_trails": 6,
        "Q": 10,
        "z_max": 3,     # max n. trail for main

        # costi 
        "c_km": 1.0,
        "c_uns": 100.0,

        # topologia speciale
        "num_Nw": 2,      # n. nodes for module storage/excange
        "depot": 0,
    }

    heu_params = {
        "n_keep": 4,      # n. nodes that are fixed after each iteration (GRIGI)
        "it_in": 20,
        "n_clust": 4,
        "warm_start_bool": WARM_START    # Start minirouting with the best solution founded in the inner loop so far   
    }


    exp_id = (
        f"GRID_n{inst_params['number']}_H{inst_params['horizon']}_dt{inst_params['dt']}"
        f"_M{inst_params['num_modules']}_P{inst_params['num_trails']}_Z{inst_params.get('z_max')}"
        f"_K{inst_params['num_requests']}_Nw{inst_params['num_Nw']}_seed{seed}"
    )

    base = Path("results") / "GRID_ASYM" / exp_id
    paths = {
        "base": base,
        "exact": base / "exact",
        "heur": base / "heuristic_prob",
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
        res_heu = run_heu_model(
            instance = instance,
            model_name = model_name,
            network_cont_path = network_cont_path,
            requests_cont_path = requests_cont_path,
            network_disc_path = network_disc_path,
            requests_disc_path = requests_disc_path,
            inst_params = inst_params,              # instance parameters
            heu_params = heu_params,    # heuristic parameters
            seed=seed, 
            exp_id=exp_id,
            base_output_folder=paths["heur"],
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

if RUN_EXACT and RUN_HEUR:
    print(
        f"EXACT     | served = {res_exact.get('served')} / {res_exact.get('num_requests')} "
        f"({100*res_exact.get('served_ratio',0):.1f}%)"
    )
    print(
        f"HEURISTIC | served = {res_heu.get('served')} / {res_heu.get('num_requests')} "
        f"({100*res_heu.get('served_ratio',0):.1f}%)"
    )
    print("\n")
    if res_exact.get("objective") is not None and res_heu.get("objective") is not None:
        gap = ((res_heu["objective"] - res_exact["objective"]) / abs(res_exact["objective"])) * 100.0
        print(
            f"EXACT = {res_exact['objective']:.3f}, "
            f"HEUR = {res_heu['objective']:.3f}"
        )
    print(f"RELATIVE GAP (HEUR vs EXACT): {gap:.2f}%")
    print("\n")
    dt_abs = abs(heu_time - exact_time)
    print(f"exact = {exact_time:.2f}s, heuristic = {heu_time:.2f}")
    print(f"TIME DIFF (HEUR vs EXACT): {dt_abs:.2f}%")
