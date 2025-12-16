from utils.MT.loader_fun import *
from utils.MT.instance_def import *
from utils.MT.print_fun import *
from utils.MT.cplex_config import *
from utils.MT.output_fun import *
from data_generation.generate_data import *

import time

from models.deterministic.model_MT_w import *


# seed
import random
import numpy as np
seed = 23
random.seed(seed)
np.random.seed(seed)


# ======================================================================
#  BUILD INSTANCE - GRID (ASYMMETRIC)
# ======================================================================
def build_instance_and_paths(
    number: int,
    horizon: int,
    dt: int,
    num_modules: int,
    num_trails: int,       # |P|
    Q: int,
    c_km: float,
    c_uns: float,         
    g_plat: float,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int,
    num_Nw: int,
    z_max: int | None = None
):
    """
    Generate the asymmetric GRID network and the requests, then build Instance.
    """
    random.seed(seed)
    np.random.seed(seed)

    t_max = horizon // dt

    # Generating asym grid
    network_path, requests_path = generate_all_data_asym(
        number=number,
        horizon=horizon,
        dt=dt,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        depot=depot
    )

    instance = load_instance_discrete(
        network_path=network_path,
        requests_path=requests_path,
        dt=dt,
        t_max=t_max,
        num_modules=num_modules,
        num_trail=num_trails,   # <-- |P|
        Q=Q,
        c_km=c_km,
        c_uns=c_uns,
        g_plat=g_plat,
        depot=depot,
        num_Nw=num_Nw,          # first N by degree
        z_max=z_max
    )

    return instance, network_path, requests_path, t_max


# ======================================================================
#  BUILD INSTANCE - CITY
# ======================================================================
def build_instance_and_paths_city(
    city: str,                    # city name
    subdir: str,                  # subdirectory name
    central_suburbs: list[str],
    horizon: int,
    dt: int,
    num_modules: int,
    num_trails: int,               # |P|
    Q: int,
    c_km: float,
    c_uns: float,
    g_plat: float,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int,
    num_Nw: int,
    z_max: int | None = None
):
    """
    Generate the CITY network and the requests, then build Instance.
    """
    random.seed(seed)
    np.random.seed(seed)

    t_max = horizon // dt

    # Generating network from a city
    network_path, requests_path = generate_all_data_city(
        city=city,
        subdir=subdir,
        central_suburbs=central_suburbs,
        horizon=horizon,
        dt=dt,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        depot=depot
    )

    instance = load_instance_discrete(
        network_path=network_path,
        requests_path=requests_path,
        dt=dt,
        t_max=t_max,
        num_modules=num_modules,
        num_trail=num_trails,   # <-- |P|
        Q=Q,
        c_km=c_km,
        c_uns=c_uns,
        g_plat=g_plat,
        depot=depot,
        num_Nw=num_Nw,          # first N by degree
        z_max=z_max
    )

    return instance, network_path, requests_path, t_max


# ======================================================================
#  RUN SINGLE MODEL - GRID
# ======================================================================
def run_single_model(
    instance: Instance,
    model_name: str,
    network_path,
    requests_path,
    t_max: int,
    dt: int,
    number: int,
    horizon: int,
    num_modules: int,
    num_trails: int,
    z_max: int | None,
    Q: int,
    c_km: float,
    c_uns: float,
    g_plat: float,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int,
    exp_id: int,
    base_output_folder,
    
) -> dict:
    """
    Costruisce e risolve UNO dei modelli su una stessa Instance (GRID).
    """

    # Sottocartella specifica per questo modello
    output_folder = base_output_folder / model_name
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    # ----------------
    # Costruzione modello
    # ----------------
    t_start_total = time.perf_counter()

    x = y = r = w = L = R = s = a = b = h = None
    D = U = z = kappa = None   # variabili TRAIL per i modelli con platoon

    if  model_name == "ab":
        model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_ab(instance)
    #elif model_name == "ab_LR":
    #    model, x, y, r, L, R, s, a, b = create_taxi_like_model_ab_LR(instance)
    #elif model_name == "ab_relax":
    #    model, x, y, r, w, s, a, b = create_taxi_like_model_ab_relax(instance)
    #elif model_name == "ab_LR_relax":
    #    model, x, y, r, L, R, s, a, b = create_taxi_like_model_ab_LR_relax(instance)
    #elif model_name == "ab_plat":
        # deve restituire anche D, U, z, kappa, h
    #    model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_taxi_like_model_ab_plat(instance)
    #elif model_name == "ab_LR_plat":
    #    # idem con L,R
    #    model, x, y, r, L, R, s, a, b, D, U, z, kappa, h = create_taxi_like_model_ab_LR_plat(instance)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    configure_cplex(model)

    # ----------------
    # Solve
    # ----------------
    t_start_solve = time.perf_counter()
    solution = model.solve(log_output=False)
    solve_time = time.perf_counter() - t_start_solve
    total_time = time.perf_counter() - t_start_total

    if solution:
        print(f"[EXP {exp_id} | model={model_name}] Status: {solution.solve_status}")
        print("Objective:", solution.objective_value)
    else:
        print(f"[EXP {exp_id} | model={model_name}] No solution found.")
    print("Solve time (sec):", solve_time)
    print("Total time (sec):", total_time)
    print("-" * 77)
    print(output_folder)

    # ----------------
    # Salvataggi
    # ----------------
    save_model_stats(model, output_folder)
    save_cplex_log(model, output_folder)

    if solution is not None:
        save_solution_summary(solution, output_folder)
        try:
            save_solution_variables_flex(
                solution=solution,
                output_folder=output_folder,
                x=x,
                y=y,
                r=r,
                w=w,
                s=s,
                L=L,
                R=R,
                a=a,
                b=b,
                h=h,
                D=D,
                U=U,
                z_main=z,    # z[m,t] numero TRAIL attaccati
                kappa=kappa, # κ[i,t] TRAIL parcheggiati
            )
        except TypeError:
            # per compatibilità con vecchie versioni
            pass

    # ----------------
    # Served summary
    # ----------------
    if solution is not None:
        served_requests = []
        for k in instance.K:
            val = solution.get_value(s[k])
            if val is not None and val > 0.5:
                served_requests.append(k)

        served = len(served_requests)
        total = len(instance.K)
        served_ratio = served / total if total > 0 else 0.0
    else:
        served_requests = []
        served = 0
        total = len(instance.K)
        served_ratio = 0.0

    print(f"[{model_name}] -> served: {served}/{total}  ({served_ratio*100:.1f}%)")
    print(f"[{model_name}] -> richieste servite (k): {served_requests}")

    # ----------------
    # Info solver
    # ----------------
    if solution is not None:
        status = str(solution.solve_status)
        objective = solution.objective_value
        try:
            mip_gap = model.solve_details.mip_relative_gap
        except Exception:
            mip_gap = None
    else:
        status = "NoSolution"
        objective = None
        mip_gap = None

    # ----------------
    # Dizionario risultato
    # ----------------
    result = {
        # identificazione esperimento + modello
        "exp_id": exp_id,
        "model_name": model_name,
        "num_Nw": instance.num_Nw,

        # --- experiment data ---
        "seed": seed,
        "number": number,
        "grid_nodes": number * number,
        "horizon": horizon,
        "dt": dt,
        "t_max": t_max,
        "num_modules": num_modules,
        "num_trails": num_trails,
        "z_max": z_max,
        "Q": Q,
        "c_km": c_km,
        "c_uns": c_uns,
        "g_plat": g_plat,
        "num_requests": num_requests,
        "served": served,
        "served_ratio": served_ratio,
        "q_min": q_min,
        "q_max": q_max,
        "slack_min": slack_min,
        "depot": depot,

        # --- instance sizes ---
        "N_size": len(instance.N),
        "A_size": len(instance.A),
        "K_size": len(instance.K),
        "M_size": len(instance.M),
        "P_size": len(instance.P),

        # --- solver output ---
        "status": status,
        "objective": objective,
        "mip_gap": mip_gap,
        "solve_time_sec": solve_time,
        "total_time_sec": total_time,

        # --- paths ---
        "output_folder": str(output_folder),
        "network_path": str(network_path),
        "requests_path": str(requests_path),
    }

    return result


# ======================================================================
#  RUN SINGLE MODEL - CITY
# ======================================================================

def run_single_model_city(
    city: str,
    instance: Instance,
    model_name: str,
    network_path,
    requests_path,
    t_max: int,
    dt: int,
    horizon: int,
    num_modules: int,
    num_trails: int,
    z_max: int | None,
    Q: int,
    c_km: float,
    c_uns: float,
    g_plat: float,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int,
    exp_id: int,
    base_output_folder,
) -> dict:
    """
    Costruisce e risolve UNO dei modelli su una stessa Instance (CITY).
    """

    # Sottocartella specifica per questo modello
    output_folder = base_output_folder / model_name
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    # ----------------
    # Costruzione modello
    # ----------------
    t_start_total = time.perf_counter()

    x = y = r = w = L = R = s = a = b = h = None
    D = U = z = kappa = None

    if model_name == "ab":
        # nuovo modello MT che restituisce anche D,U,z,kappa,h
        model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_ab(instance)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    configure_cplex(model)

    # ----------------
    # Solve
    # ----------------
    t_start_solve = time.perf_counter()
    solution = model.solve(log_output=False)
    solve_time = time.perf_counter() - t_start_solve
    total_time = time.perf_counter() - t_start_total

    if solution:
        print(f"[EXP {exp_id} | model={model_name}] Status: {solution.solve_status}")
        print("Objective:", solution.objective_value)
    else:
        print(f"[EXP {exp_id} | model={model_name}] No solution found.")
    print("Solve time (sec):", solve_time)
    print("Total time (sec):", total_time)
    print("-" * 77)
    print(output_folder)

    # ----------------
    # Salvataggi
    # ----------------
    save_model_stats(model, output_folder)
    save_cplex_log(model, output_folder)

    if solution is not None:
        save_solution_summary(solution, output_folder)
        try:
            save_solution_variables_flex(
                solution=solution,
                output_folder=output_folder,
                x=x,
                y=y,
                r=r,
                w=w,
                s=s,
                L=L,
                R=R,
                a=a,
                b=b,
                h=h,
                D=D,
                U=U,
                z_main=z,   # z[m,t]
                kappa=kappa,
            )
        except TypeError:
            # compatibilità se la versione di save_solution_variables_flex
            # non supporta ancora D,U,z_main,kappa
            pass

    # ----------------
    # Served summary
    # ----------------
    if solution is not None:
        served_requests = []
        for k in instance.K:
            val = solution.get_value(s[k])
            if val is not None and val > 0.5:
                served_requests.append(k)

        served = len(served_requests)
        total = len(instance.K)
        served_ratio = served / total if total > 0 else 0.0
    else:
        served_requests = []
        served = 0
        total = len(instance.K)
        served_ratio = 0.0

    print(f"[{model_name}] -> served: {served}/{total}  ({served_ratio*100:.1f}%)")
    print(f"[{model_name}] -> richieste servite (k): {served_requests}")

    # ----------------
    # Info solver
    # ----------------
    if solution is not None:
        status = str(solution.solve_status)
        objective = solution.objective_value
        try:
            mip_gap = model.solve_details.mip_relative_gap
        except Exception:
            mip_gap = None
    else:
        status = "NoSolution"
        objective = None
        mip_gap = None

    # ----------------
    # Dizionario risultato
    # ----------------
    result = {
        # identificazione esperimento + modello
        "exp_id": exp_id,
        "model_name": model_name,
        "num_Nw": instance.num_Nw,

        # --- experiment data ---
        "seed": seed,
        "city": city,
        "horizon": horizon,
        "dt": dt,
        "t_max": t_max,
        "num_modules": num_modules,
        "num_trails": num_trails,
        "z_max": z_max,
        "Q": Q,
        "c_km": c_km,
        "c_uns": c_uns,
        "g_plat": g_plat,
        "num_requests": num_requests,
        "served": served,
        "served_ratio": served_ratio,
        "q_min": q_min,
        "q_max": q_max,
        "slack_min": slack_min,
        "depot": depot,

        # --- instance sizes ---
        "N_size": len(instance.N),
        "A_size": len(instance.A),
        "K_size": len(instance.K),
        "M_size": len(instance.M),
        "P_size": len(instance.P),

        # --- solver output ---
        "status": status,
        "objective": objective,
        "mip_gap": mip_gap,
        "solve_time_sec": solve_time,
        "total_time_sec": total_time,

        # --- paths ---
        "output_folder": str(output_folder),
        "network_path": str(network_path),
        "requests_path": str(requests_path),
    }

    return result
