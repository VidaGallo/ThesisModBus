
from utils.loader_fun import *
from utils.instance_def import *
from utils.print_fun import *
from utils.cplex_config import *
from utils.output_fun import *
from data_generation.generate_data import *

import time


from models.deterministic.model_taxi_like_ab import *
from models.deterministic.model_taxi_like_ab_LR import *

from models.deterministic.model_taxi_like_ab_relax import *
from models.deterministic.model_taxi_like_ab_LR_relax import *

from models.deterministic.model_taxi_like_ab_plat import *
from models.deterministic.model_taxi_like_ab_LR_plat import *


# seed
import random
import numpy as np
seed = 23
random.seed(seed)
np.random.seed(seed)


### GRID
def build_instance_and_paths(
    number: int,
    horizon: int,
    dt: int,
    num_modules: int,
    Q: int,
    c_km: float,
    c_uns_taxi: float,
    g_plat: float,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int,
    num_Nw: int
):
    """
    Generate the network and the requests.
    """
    random.seed(seed)
    np.random.seed(seed)

    t_max = horizon // dt

    network_path, requests_path = generate_all_data_asym(    # Generating asym grid
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
        Q=Q,
        c_km=c_km,
        c_uns_taxi=c_uns_taxi,
        g_plat=g_plat,
        depot=depot,
        num_Nw = num_Nw     # First N by degree
    )

    return instance, network_path, requests_path, t_max


### CITY
def build_instance_and_paths_city(
    city: str,      # city name
    subdir: str,     # subdirectory name
    central_suburbs: list[str],
    horizon: int,
    dt: int,
    num_modules: int,
    Q: int,
    c_km: float,
    c_uns_taxi: float,
    g_plat: float,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int,
    num_Nw: int
):
    """
    Generate the network and the requests.
    """
    random.seed(seed)
    np.random.seed(seed)

    t_max = horizon // dt



    network_path, requests_path = generate_all_data_city(    # Generating network from a city
        city = city,
        subdir = subdir,
        central_suburbs = central_suburbs,
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
        Q=Q,
        c_km=c_km,
        c_uns_taxi=c_uns_taxi,
        g_plat=g_plat,
        depot=depot,
        num_Nw = num_Nw     # First N by degree
    )

    return instance, network_path, requests_path, t_max





### GRID
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
    Q: int,
    c_km: float,
    c_uns_taxi: float,
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
    Costruisce e risolve UNO dei modelli su una stessa Instance.
    """

    # Sottocartella specifica per questo modello
    output_folder = base_output_folder / model_name
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    #print("-" * 80)
    #print(f"[EXP {exp_id} | model={model_name}] Solving...")
    #print("-" * 80)

    # ----------------
    # Costruzione modello
    # ----------------
    t_start_total = time.perf_counter()

    x = y = r = w = L = R = s = a = b = h = None

    if model_name == "base":
        model, x, y, r, w, s = create_taxi_like_model(instance)
    elif model_name == "LR":
        model, x, y, r, L, R, s = create_taxi_like_model_LR(instance)
    elif model_name == "ab":
        model, x, y, r, w, s, a, b = create_taxi_like_model_ab(instance)
    elif model_name == "ab_LR":
        model, x, y, r, L, R, s, a, b = create_taxi_like_model_ab_LR(instance)
    elif model_name == "base_relax":
        model, x, y, r, w, s = create_taxi_like_model_relax(instance)
    elif model_name == "LR_relax":
        model, x, y, r, L, R, s = create_taxi_like_model_LR_relax(instance)
    elif model_name == "ab_relax":
        model, x, y, r, w, s, a, b = create_taxi_like_model_ab_relax(instance)
    elif model_name == "ab_LR_relax":
        model, x, y, r, L, R, s, a, b = create_taxi_like_model_ab_LR_relax(instance)
    elif model_name == "ab_plat":
        model, x, y, r, w, s, a, b, h = create_taxi_like_model_ab_plat(instance)
    elif model_name == "ab_LR_plat":
        model, x, y, r, L, R, s, a, b, h = create_taxi_like_model_ab_LR_plat(instance)
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

    #print("-" * 77)
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
            save_solution_variables_flex(solution=solution, output_folder=output_folder,
                x=x,
                y=y,
                r=r,
                w=w,   
                s=s,
                L=L,
                R=R,
                a=a,
                b=b,
                h=h
            )
        except TypeError:
            pass

    # ----------------
    # Served summary
    # ----------------
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
        "Q": Q,
        "c_km": c_km,
        "c_uns_taxi": c_uns_taxi,
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




### CITY
def run_single_model_city(
    city:str,
    instance: Instance,
    model_name: str,
    network_path,
    requests_path,
    t_max: int,
    dt: int,
    horizon: int,
    num_modules: int,
    Q: int,
    c_km: float,
    c_uns_taxi: float,
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
    Costruisce e risolve UNO dei modelli su una stessa Instance.
    """

    # Sottocartella specifica per questo modello
    output_folder = base_output_folder / model_name
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    #print("-" * 80)
    #print(f"[EXP {exp_id} | model={model_name}] Solving...")
    #print("-" * 80)

    # ----------------
    # Costruzione modello
    # ----------------
    t_start_total = time.perf_counter()

    x = y = r = w = L = R = s = a = b = h = None

    if model_name == "ab":
        model, x, y, r, w, s, a, b = create_taxi_like_model_ab(instance)
    elif model_name == "ab_LR":
        model, x, y, r, L, R, s, a, b = create_taxi_like_model_ab_LR(instance)
    elif model_name == "ab_relax":
        model, x, y, r, w, s, a, b = create_taxi_like_model_ab_relax(instance)
    elif model_name == "ab_LR_relax":
        model, x, y, r, L, R, s, a, b = create_taxi_like_model_ab_LR_relax(instance)
    elif model_name == "ab_plat":
        model, x, y, r, w, s, a, b, h = create_taxi_like_model_ab_plat(instance)
    elif model_name == "ab_LR_plat":
        model, x, y, r, L, R, s, a, b, h = create_taxi_like_model_ab_LR_plat(instance)
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

    #print("-" * 77)
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
            save_solution_variables_flex(solution=solution, output_folder=output_folder,
                x=x,
                y=y,
                r=r,
                w=w,   
                s=s,
                L=L,
                R=R,
                a=a,
                b=b,
                h=h
            )
        except TypeError:
            pass

    # ----------------
    # Served summary
    # ----------------
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
        "Q": Q,
        "c_km": c_km,
        "c_uns_taxi": c_uns_taxi,
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


