from utils.MT.runs_fun import *

import pandas as pd
from pathlib import Path
from collections import Counter
from itertools import product  # per fare il prodotto cartesiano delle combinazioni


# seed base (lo useremo come riferimento, ma poi lo variamo)
import random
import numpy as np
BASE_SEED = 23


if __name__ == "__main__":

    ### City
    city = "Torino, Italia"
    subdir = "TORINO_SUB"
    central_suburbs = ["Centro", "Crocetta", "Santa Rita", "Aurora"]

    # ----------------
    # PARAMETRI FISSI (se vuoi, puoi spostarli nelle liste sotto)
    # ----------------
    depot = 1198867366   # Centro Torino

    Q = 10
    c_km = 1.0
    c_uns = 100
    g_plat = None

    q_min = 1
    q_max = 10
    slack_min = 30.0

    # Modelli da confrontare
    model_names = ["ab"]

    # =====================================================
    #  LISTE DI PARAMETRI CHE VUOI VARIARE NEGLI ESPERIMENTI
    #  (qui cambi tu i valori che vuoi testare)
    # =====================================================
    horizon_dt_pairs = [                 # minuti
        (108, 6),
        (180, 10)
    ]
    num_modules_list  = [3]               # n° MAIN
    num_trails_list   = [6]               # n° TRAIL
    num_Nw_list       = [3, 4]            # n° nodi scambio
    num_requests_list = [15, 25]          # numero richieste
    seed_list         = [23]              # seed

    # =====================================================
    #  PREPARAZIONE CARTELLE OUTPUT
    # =====================================================
    summary_dir = Path("results/CITY/MT/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    # file riassuntivo cumulativo di TUTTI gli esperimenti
    summary_name = "summary_CITY_MT_sweep.csv"
    summary_path = summary_dir / summary_name

    all_results = []

    # =====================================================
    #  LOOP SU TUTTE LE COMBINAZIONI DI PARAMETRI
    # =====================================================
    exp_counter = 0

    for (
        (horizon, dt),
        num_modules,
        num_trails,
        num_Nw,
        num_requests,
        seed
    ) in product(
        horizon_dt_pairs,
        num_modules_list,
        num_trails_list,
        num_Nw_list,
        num_requests_list,
        seed_list,
    ):
        
        exp_counter += 1

        # Fisso il seed per questo round (così Requests cambiano)
        random.seed(seed)
        np.random.seed(seed)

        exp_id = (
            f"{city}_H{horizon}_dt{dt}_M{num_modules}_P{num_trails}_"
            f"K{num_requests}_Nw{num_Nw}_seed{seed}"
        )

        print("\n" + "=" * 80)
        print(f"ROUND {exp_counter} - EXPERIMENT {exp_id}")
        print(f"  city          = {city}")
        print(f"  horizon       = {horizon}")
        print(f"  dt            = {dt}")
        print(f"  num_modules   = {num_modules}")
        print(f"  num_trails    = {num_trails}")
        print(f"  num_requests  = {num_requests}")
        print(f"  num_Nw        = {num_Nw}")
        print(f"  seed          = {seed}")
        print("=" * 80)

        # -------------------------------------------------
        # 1) genera dati + istanza per QUESTO ROUND
        # -------------------------------------------------
        instance, network_path, requests_path, t_max = build_instance_and_paths_city(
            city=city,
            subdir=subdir,
            central_suburbs=central_suburbs,
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
            num_Nw=num_Nw,
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

        # -------------------------------------------------
        # 2) cartella base per l'esperimento (specifica del round)
        # -------------------------------------------------
        base_folder = build_output_folder(
            base_dir="results",
            network_path=network_path,
            t_max=instance.t_max,
            dt=instance.dt,
        )
        base_folder = base_folder / f"{exp_id}"
        base_folder.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # 3) Lanciare i modelli sulla stessa instance
        # -------------------------------------------------
        round_results = []  # per stampare subito dopo
        for model_name in model_names:
            print(f"\n>> Running model: {model_name}")
            res = run_single_model_city(
                city=city,
                instance=instance,
                model_name=model_name,
                network_path=network_path,
                requests_path=requests_path,
                t_max=t_max,
                dt=dt,
                horizon=horizon,
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
                exp_id=exp_id,
                base_output_folder=base_folder,
            )

            # ATTACCO AI RISULTATI LE INFO DEI PARAMETRI DI QUESTO ROUND
            res["city"] = city
            res["horizon"] = horizon
            res["dt"] = dt
            res["num_modules"] = num_modules
            res["num_trails"] = num_trails
            res["num_Nw"] = num_Nw
            res["num_requests"] = num_requests
            res["seed"] = seed
            res["model_name"] = model_name

            all_results.append(res)
            round_results.append(res)

        # -------------------------------------------------
        # 4) STAMPA RIASSUNTO DEL ROUND (come volevi tu)
        # -------------------------------------------------
        print("\n--- ROUND FINITO, RISULTATI ---")
        df_round = pd.DataFrame(round_results)
        # qui decidi cosa mostrare: ad esempio tempo di CPU e obj
        cols_to_show = [c for c in df_round.columns
                        if c in ["model_name", "obj_value", "comp_time",
                                 "n_satisfied", "gap"] or c in
                        ["horizon", "dt", "num_modules", "num_requests", "num_Nw"]]
        print(df_round[cols_to_show])

        # -------------------------------------------------
        # 5) AGGIORNA / SALVA IL SUMMARY CUMULATIVO
        # -------------------------------------------------
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(summary_path, index=False)
        print(f"\n[Summary cumulativo aggiornato: {summary_path}]\n")

    print("\n" + "#" * 80)
    print(" TUTTI GLI ESPERIMENTI SONO FINITI ")
    print(f" Summary finale in: {summary_path}")
    print("#" * 80)
