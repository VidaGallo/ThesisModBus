from utils.runs_fun import *

import pandas as pd
from pathlib import Path
from collections import Counter


CPLEX_CFG = {
    "time_limit": 36_000,     # seconds
    "mip_gap": 0.01,          # relative MIP gap, 0.05 = 5%
    "abs_mip_gap": 1e-6,      # absolute MIP gap
    "threads": 0,             # 0 = all available threads
    "mip_display": 1,         # 0..5 (2 = default)
    "emphasis_mip": 2,        # 0 balanced, 1 feasibility, 2 optimality, ...
    "parallel": 0             # 0 auto, 1 opportunistic
}

if __name__ == "__main__":

    ### City
    city = "Torino, Italia"
    subdir = "TORINO_SUB"
    central_suburbs =   ["Centro", "Crocetta", "Santa Rita", "Aurora"]


    ### Seed
    seed = 23


    # ----------------
    # Base params 
    # ----------------
    horizon =108    # minuti
    dt = 6
    depot = 1198867366   # Centro Torino

    Q = 10
    mean_speed_kmh = 50.0
    c_km = 1.0
    c_uns = 100


    num_Nw = 1    # n°nodi che permettono lo scambio

    q_min = 1
    q_max = 10
    alpha = 0.65
    slack_min = 20.0

    # Parametri SPECIFICI
    num_modules   = 2
    num_trails    = 3
    z_max         = 3      # max trail per main
    num_requests  = 5
    

    # Nomi dei modelli da eseguire
    model_names = ["w"]

    all_results = []

    print("\n" + "="*80)
    print(f"  city        = {city}")
    print(f"  horizon       = {horizon}")
    print(f"  num_modules   = {num_modules}")
    print(f"  num_requests  = {num_requests}")
    print("="*80)

    # 1) genera dati + istanza UNA volta sola
    instance, _, _, network_path, requests_path, t_max = build_instance_and_paths_city(
        city=city,      # city name
        subdir=subdir,     # subdirectory name
        central_suburbs=central_suburbs,
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
        mean_speed_kmh=mean_speed_kmh,
        z_max=z_max,
        alpha=alpha
    )

    
    # --- Controllo: conteggio capacità delle richieste (q_k) ---
    cap_list = [instance.q[k] for k in instance.K]
    cap_counts = Counter(cap_list)
    print("\nRequest capacity counts:")
    for i in [1, 2, 3, 4]:
        print(f"  q={i}: {cap_counts.get(i, 0)}")
    gt4 = sum(v for k, v in cap_counts.items() if k > 4)
    print(f"  q>4: {gt4}")
    print(f"  full distribution: {dict(sorted(cap_counts.items()))}")


    # 2) cartella base per l'esperimento
    base_folder = build_output_folder(
        base_dir="results",
        network_path=network_path
    )
    base_folder = base_folder / subdir / f"{exp_id}"
    base_folder.mkdir(parents=True, exist_ok=True)


    # 3) Lanciare i modelli sulla stessa instance
    for model_name in model_names:
        print("\n")
        res = run_single_model_city(
            city=city,
            instance=instance,
            model_name=model_name,
            network_path=network_path,
            requests_path=requests_path,
            horizon=horizon,
            q_min=q_min,
            q_max=q_max,
            alpha=alpha,
            slack_min=slack_min,
            seed=seed,
            exp_id=exp_id,
            base_output_folder=base_folder,
            mean_speed_kmh=mean_speed_kmh,
            cplex_cfg=CPLEX_CFG
        )
        all_results.append(res)

    # ----------------------
    # Pandas DataFrame + CSV
    # ----------------------
    df_results = pd.DataFrame(all_results)

    # Nome del summary per QUESTA run
    summary_name = (
        f"summary_"
        f"{city}_"
        f"H{horizon}_"
        f"M{num_modules}_"
        f"P{num_trails}_"
        f"Z{z_max}_"
        f"K{num_requests}_"
        f"Nw{num_Nw}.csv"
    )
    summary_path = base_folder / model_name / summary_name
    df_results.to_csv(summary_path, index=False)

    #print(f"\nSummary saved to: {summary_path}")