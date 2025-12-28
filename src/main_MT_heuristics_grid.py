from utils.runs_fun import *
from utils.runs_heu_fun import *
import random, numpy as np, pandas as pd
from pathlib import Path

RUN_EXACT = True
RUN_HEUR  = True

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)


CPLEX_CFG = {
    "time_limit": 36_000,     # seconds
    "mip_gap": 0.1,          # relative MIP gap, 0.01 = 1%
    "abs_mip_gap": 1e-6,      # absolute MIP gap
    "threads": 0,             # 0 = all available threads
    "mip_display": 1,         # 0..5 (2 = default)
    "emphasis_mip": 2,        # 0 balanced, 1 feasibility, 2 optimality, ...
    "parallel": 0             # 0 auto, 1 opportunistic, 2 deterministic
}


if __name__ == "__main__":
    seed = 23
    set_seed(seed)

    # params...
    number = 3    # side grid
    horizon =120
    dt = 5
    depot = 0
    num_requests = 15
    q_min, q_max = 1, 10
    alpha = 0.65
    slack_min = 20.0
    Q = 10
    c_km = 1.0
    c_uns = 100.0
    num_modules = 3
    num_trails = 6
    z_max = 3
    num_Nw = 2
    mean_speed_kmh = 40.0
    mean_edge_length_km = 3.33
    rel_std = 0.66
    model_name = "w"

    exp_id = f"GRID_n{number}_H{horizon}_dt{dt}_M{num_modules}_P{num_trails}_Z{z_max}_K{num_requests}_Nw{num_Nw}_seed{seed}"

    base = Path("results") / "GRID_ASYM" / exp_id
    paths = {
        "base": base,
        "exact": base / "exact",
        "heur": base / "heuristic_prob",
        "summary": base / "summary",
    }
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)

    instance, network_cont_path, requests_cont_path, network_disc_path, requests_disc_path, t_max = build_instance_and_paths(
        number=number, horizon=horizon, dt=dt,
        num_modules=num_modules, num_trails=num_trails,
        Q=Q, c_km=c_km, c_uns=c_uns,
        num_requests=num_requests, q_min=q_min, q_max=q_max,
        slack_min=slack_min, depot=depot,
        seed=seed, num_Nw=num_Nw,
        mean_edge_length_km=mean_edge_length_km,
        mean_speed_kmh=mean_speed_kmh,
        rel_std=rel_std,
        z_max=z_max,
        alpha=alpha,
    )


    # EXACT
    if RUN_EXACT:
        exact_start = time.time()
        print("\n")
        print("*"*50)
        print("EXACT")
        print("\n")
        res_exact = run_single_model(
            instance=instance, model_name=model_name,
            network_path=network_disc_path,      
            requests_path=requests_disc_path,    
            number=number, horizon=horizon,
            q_min=q_min, q_max=q_max,
            alpha=alpha, slack_min=slack_min,
            seed=seed, exp_id=exp_id,
            mean_edge_length_km=mean_edge_length_km,
            mean_speed_kmh=mean_speed_kmh,
            rel_std=rel_std,
            base_output_folder=paths["exact"],
            cplex_cfg=CPLEX_CFG,
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
            instance=instance,
            model_name=model_name,
            network_cont_path=network_cont_path,
            requests_cont_path=requests_cont_path,
            network_disc_path=network_disc_path,
            requests_disc_path=requests_disc_path,
            number=number, horizon=horizon, dt=dt,
            num_modules=num_modules, num_trails=num_trails,
            c_km=c_km, c_uns=c_uns,
            depot=depot, num_Nw=num_Nw,
            z_max=z_max, t_max=t_max,
            q_min=q_min, q_max=q_max, Q=Q,
            alpha=alpha, slack_min=slack_min,
            seed=seed, exp_id=exp_id,
            mean_edge_length_km=mean_edge_length_km,
            mean_speed_kmh=mean_speed_kmh,
            rel_std=rel_std,
            base_output_folder=paths["heur"],
            cplex_cfg=CPLEX_CFG,
            n_keep = 4,
            it_out = 1000,   # <- usually stop earlier after fixing all the requests
            it_in = 20,
            time_out = 36_000,
            n_clust = 4
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
    print(f"RELATIVE GAP (HEUR vs EXACT): {gap:+.2f}%")
    print("\n")
    dt_abs = abs(heu_time - exact_time)
    print(f"exact = {exact_time:.2f}s, heuristic = {heu_time:.2f}")
    print(f"TIME DIFF (HEUR vs EXACT): {dt_abs:+.2f}%")
