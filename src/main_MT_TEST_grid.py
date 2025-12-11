from utils.MT.runs_fun import *

import pandas as pd
from pathlib import Path
from collections import Counter


# seed
import random
import numpy as np
seed = 23
random.seed(seed)
np.random.seed(seed)




if __name__ == "__main__":

    # ----------------
    # Base params 
    # ----------------
    horizon =108    # minuti
    dt = 6
    depot = 0

    Q = 10
    c_km = 1.0
    c_uns = 100
    g_plat = None    # or None

    num_Nw = 3    # n°nodi che permettono lo scambio

    q_min = 1
    q_max = 10
    slack_min = 20.0

    # Parametri SPECIFICI
    number        = 4      # lato griglia
    num_modules   = 3
    num_trails    = 6
    num_requests  = 30
    

    # I quattro modelli da confrontare
    model_names = ["ab"]    # ["base", "base_relax", "LR", "LR_relax" , "ab", "ab_relax", "ab_plat", "ab_LR", "ab_LR_relax", "ab_LR_plat"]

    all_results = []
    exp_id = f"n{number}_h{horizon}_m{num_modules}_r{num_requests}"

    print("\n" + "="*80)
    print(f"EXPERIMENT {exp_id}")
    print(f"  number        = {number}")
    print(f"  horizon       = {horizon}")
    print(f"  dt            = {dt}")
    print(f"  num_main   = {num_modules}")
    print(f"  num_trail   = {num_trails}")
    print(f"  num_requests  = {num_requests}")
    print("="*80)

    # 1) genera dati + istanza UNA volta sola
    instance, network_path, requests_path, t_max = build_instance_and_paths(
        number=number,
        horizon=horizon,
        dt=dt,
        num_modules=num_modules,
        num_trails=num_trails,
        Q=Q,
        c_km=c_km,
        c_uns=c_uns,
        g_plat=g_plat,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        depot=depot,
        seed=seed,
        num_Nw=num_Nw
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
        network_path=network_path,
        t_max=instance.t_max,
        dt=instance.dt,
    )
    base_folder = base_folder / f"{exp_id}"
    if not base_folder.exists():
        base_folder.mkdir(parents=True)

    # 3) lancia i modelli sulla stessa instance
    for model_name in model_names:
        print("\n")
        res = run_single_model(
            instance=instance,
            model_name=model_name,
            network_path=network_path,
            requests_path=requests_path,
            t_max=t_max,
            dt=dt,
            number=number,
            horizon=horizon,
            num_modules=num_modules,
            num_trails=num_trails,   # <<< aggiunto
            Q=Q,
            c_km=c_km,
            c_uns=c_uns,           # <<< nuovo nome
            g_plat=g_plat,
            num_requests=num_requests,
            q_min=q_min,
            q_max=q_max,
            slack_min=slack_min,
            depot=depot,
            seed=seed,
            exp_id=exp_id,
            base_output_folder=base_folder,
        )
        all_results.append(res)

    # ----------------------
    # Pandas DataFrame + CSV
    # ----------------------
    df_results = pd.DataFrame(all_results)

    # Nome del summary per QUESTA run
    summary_name = (
        f"summary_"
        f"{number}x{number}_"
        f"H{horizon}_"
        f"M{num_modules}_"
        f"K{num_requests}_"
        f"Nw{num_Nw}_models.csv"   # <<< cambiato il suffisso (non più 4models fisso)
    )
    summary_path = Path("results/GRID/summary") / summary_name

    df_results.to_csv(summary_path, index=False)

    #print(f"\nSummary saved to: {summary_path}")