from utils.MT.heuristic_prob_fun import *
from utils.MT.runs_fun import *


import random
import numpy as np
import pandas as pd
from pathlib import Path


# =========================
# FLAGS
# =========================
RUN_EXACT = True     # False → salta CPLEX
RUN_HEUR  = True     # False → salta HEURISTIC


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


CPLEX_CFG = {
    "time_limit": 36_000,     # seconds
    "mip_gap": 0.01,          # relative MIP gap, 0.05 = 5%
    "abs_mip_gap": 1e-6,      # absolute MIP gap
    "threads": 0,             # 0 = all available threads
    "mip_display": 1,         # 0..5 (2 = default)
    "emphasis_mip": 2,        # 0 balanced, 1 feasibility, 2 optimality, ...
    "parallel": 0             # 0 auto, 1 opportunistic, 2 deterministic
}


if __name__ == "__main__":

    # =========================
    # EXPERIMENT PARAMS
    # =========================

    ### SEED 
    seed = 23
    set_seed(seed)

    ### GRID (ASYM) PARAMS
    number = 2
    mean_speed_kmh = 40.0
    mean_edge_length_km = 3.33
    rel_std = 0.66
    num_Nw = 2     
    depot = 0      
    
    ### REQUEST PARAMS
    num_requests = 3        
    q_min, q_max = 1, 10
    alpha = 0.65
    slack_min = 20.0         

    ### TIME PARAMS
    horizon = 108         
    dt = 6
    
    ### MODULE PARAMS
    Q = 10
    c_km = 1.0
    c_uns = 100.0
    num_modules = 2
    num_trails = 6
    z_max = 3

    ### MODEL & EXPERIMENT NAME
    model_name = "w"  
    exp_id = (
        f"GRID_n{number}_H{horizon}_dt{dt}_"
        f"M{num_modules}_P{num_trails}_Z{z_max}_"
        f"K{num_requests}_Nw{num_Nw}_seed{seed}"
    )

    print("\n" + "=" * 80)
    print(f"EXPERIMENT {exp_id}")
    print("=" * 80)

    ### OUTPUT FOLDERS
    base = Path("results") / "GRID" / "MT" / exp_id
    paths = {
        "base": base,
        "exact": base / "exact",
        "heur": base / "heuristic_prob",
        "summary": base / "summary",
        "plots": base / "plots",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)



    # =========================
    # BUILD INSTANCE (ONCE)
    # =========================
    instance, network_path, requests_path, t_max = build_instance_and_paths(
        number=number,
        horizon=horizon,
        dt=dt,
        num_modules=num_modules,
        num_trails=num_trails,
        Q=Q,
        c_km=c_km,
        c_uns=c_uns,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        depot=depot,
        seed=seed,
        num_Nw=num_Nw,
        mean_edge_length_km=mean_edge_length_km,
        mean_speed_kmh=mean_speed_kmh,
        rel_std=rel_std,
        z_max=z_max,
        alpha=alpha,
    )

    print("\nData used:")
    print(" network :", network_path)
    print(" requests:", requests_path)




    # =========================
    # EXACT (CPLEX)
    # =========================
    exact_results = []

    if RUN_EXACT:
        print("\n" + "=" * 80)
        print("RUNNING EXACT (model = w)")
        print("=" * 80)

        res = run_single_model(
            instance=instance,
            model_name=model_name,
            network_path=network_path,
            requests_path=requests_path,
            number=number,
            horizon=horizon,
            q_min=q_min,
            q_max=q_max,
            alpha=alpha,
            slack_min=slack_min,
            seed=seed,
            exp_id=exp_id,
            mean_edge_length_km=mean_edge_length_km,
            mean_speed_kmh=mean_speed_kmh,
            rel_std=rel_std,
            base_output_folder=paths["exact"],
            cplex_cfg=CPLEX_CFG      # possibility of early stopping/feasable but not optimal
        )

        exact_results.append(res)
        df_exact = pd.DataFrame(exact_results)
        df_exact.to_csv(paths["summary"] / "summary_exact.csv", index=False)

    else:
        print("\n[SKIP] EXACT disabled")






    # =========================
    # HEURISTIC
    # =========================
    if RUN_HEUR:
        print("\n" + "=" * 80)
        print("RUNNING HEURISTIC")
        print("=" * 80)



    else:
        print("\n[SKIP] HEURISTIC disabled")

    print("\n" + "#" * 80)
    print("DONE")
    print("Outputs in:", paths["base"])
    print("#" * 80)
