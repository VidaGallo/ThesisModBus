from utils.loader_fun import *
from utils.instance_def import *
from utils.cplex_config import *
from utils.output_fun import *
from utils.heuristic_prob_fun import *
from data_generation.generate_data import *
from models.model_MT_w import *

import time


import random
from math import inf
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter



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
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int,
    num_Nw: int,
    mean_edge_length_km: float,
    mean_speed_kmh: float,
    rel_std: float,
    z_max: int | None = None,
    alpha: float = 0.65,
):
    """
    Generate the asymmetric GRID network and the requests, then build Instance.
    """

    t_max = horizon // dt

    # Generating asym grid
    network_cont_path, requests_cont_path, network_path, requests_path = generate_all_data_asym(
        number=number,
        horizon=horizon,
        dt=dt,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        depot=depot,
        mean_edge_length_km=mean_edge_length_km,
        mean_speed_kmh=mean_speed_kmh,
        rel_std=rel_std,
        alpha=alpha,
        seed=seed
    )


    instance = load_instance_discrete(
        network_path=network_path,
        requests_path=requests_path,
        dt=dt,
        t_max=t_max,
        num_modules=num_modules,
        num_trail=num_trails,
        Q=Q,
        c_km=c_km,
        c_uns=c_uns,
        depot=depot,
        num_Nw=num_Nw,      # first N by degree
        z_max=z_max
    )

    return instance, network_cont_path, requests_cont_path, network_path, requests_path, t_max




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
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int,
    num_Nw: int,
    mean_speed_kmh: float,
    z_max: int | None = None,
    alpha: float = 0.65,
):
    """
    Generate the CITY network and the requests, then build Instance.
    """

    t_max = horizon // dt

    # Generating network from a city
    network_cont_path, requests_cont_path, network_path, requests_path = generate_all_data_city(
        city=city,
        subdir=subdir,
        central_suburbs=central_suburbs,
        horizon=horizon,
        dt=dt,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        mean_speed_kmh=mean_speed_kmh,
        depot=depot,
        seed=seed,
        alpha=alpha,
    )

    instance = load_instance_discrete(
        network_path=network_path,
        requests_path=requests_path,
        dt=dt,
        t_max=t_max,
        num_modules=num_modules,
        num_trail=num_trails,   
        Q=Q,
        c_km=c_km,
        c_uns=c_uns,
        depot=depot,
        num_Nw=num_Nw,          # first N by degree
        z_max=z_max
    )

    return instance, network_cont_path, requests_cont_path, network_path, requests_path, t_max




# ======================================================================
#  RUN SINGLE MODEL - GRID
# ======================================================================
def run_single_model(
    instance: Instance,
    model_name: str,
    network_path,
    requests_path,
    number: int,
    horizon: int,
    q_min: int,
    q_max: int,
    alpha: float,
    slack_min: float,
    seed: int,
    exp_id: str,
    mean_edge_length_km: float,
    mean_speed_kmh: float,
    rel_std: float,
    base_output_folder,   
    cplex_cfg: dict | None = None,

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

    if  model_name == "w":
        model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_w(instance)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


    configure_cplex(model, cplex_cfg)

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
        "mean_edge_length": mean_edge_length_km,
        "mean_speed": mean_speed_kmh,
        "std": rel_std,
        "horizon": horizon,
        "dt": instance.dt,
        "t_max": instance.t_max,
        "num_modules": instance.num_modules,
        "num_trails": instance.num_trail_modules,
        "z_max": instance.Z_max,
        "Q": instance.Q,
        "c_km": instance.c_km,
        "c_uns": instance.c_uns,
        "num_requests": instance.num_requests,
        "served": served,
        "served_ratio": served_ratio,
        "q_min": q_min,
        "q_max": q_max,
        "alpha": alpha,
        "slack_min": slack_min,
        "depot": instance.depot,


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
    horizon: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    seed: int,
    exp_id: str,
    mean_speed_kmh: float,
    base_output_folder,
    alpha: float,
    cplex_cfg: dict | None = None
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

    if model_name == "w":
        model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_w(instance)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    configure_cplex(model, cplex_cfg)

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
        "mean_speed": mean_speed_kmh,
        "dt": instance.dt,
        "t_max": instance.t_max,
        "num_modules": instance.num_modules,
        "num_trails": instance.num_trail_modules,
        "z_max": instance.Z_max,
        "Q": instance.Q,
        "c_km": instance.c_km,
        "c_uns": instance.c_uns,
        "num_requests": instance.num_requests,
        "served": served,
        "served_ratio": served_ratio,
        "q_min": q_min,
        "q_max": q_max,
        "alpha": alpha,
        "slack_min": slack_min,
        "depot": instance.depot,


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










def mini_routing(
    mini_requests_path: str | Path,
    network_path: str | Path,
    old_constraints_dict: Dict[str, Dict[tuple, float]],
    *,
    # --- instance/build params (pass explicitly, no globals) ---
    dt: int,
    t_max: int,
    num_modules: int,
    num_trails: int,
    Q: int,
    c_km: float,
    c_uns: float,
    depot: int,
    num_Nw: int,
    z_max: int,
    # --- run/config params ---
    base_output_folder: str | Path,
    model_name: str = "w",
    cplex_cfg: dict | None = None,
) -> Tuple[Any, Any]:
    """
    Build a mini instance from (network_path, mini_requests_path), create model,
    apply fix-and-optimize constraints from old_constraints_dict, configure cplex.
    Returns (model, instance).
    """

    mini_requests_path = Path(mini_requests_path)
    network_path = Path(network_path)
    base_output_folder = Path(base_output_folder)



    discretize_requests(
        input_path=requests_cont,
        output_path=requests_disc,
        time_step_min=float(dt),
    )

    # ----------------
    # Load instance
    # ----------------
    instance = load_instance_discrete(
        network_path=str(network_path),
        requests_path=str(mini_requests_path),
        dt=dt,
        t_max=t_max,
        num_modules=num_modules,
        num_trail=num_trails,  
        Q=Q,
        c_km=c_km,
        c_uns=c_uns,
        depot=depot,
        num_Nw=num_Nw,
        z_max=z_max,
    )

    # ----------------
    # Output folder
    # ----------------
    output_folder = base_output_folder / model_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # ----------------
    # Build model
    # ----------------
    t_start_total = time.perf_counter()

    if model_name == "w":
        model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_w(instance)

        # IMPORTANT: include here only actual docplex Var dicts
        var_dicts = {
            "x": x, "y": y,
            "r": r, "w": w, "s": s,
            "a": a, "b": b,
            "D": D, "U": U,
            "z": z, "kappa": kappa,
            "h": h,
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # -----------------
    # Configure solver
    # -----------------
    configure_cplex(model, cplex_cfg)

    # --------------------------
    # Apply previous constraints
    # --------------------------
    if old_constraints_dict:
        nfix = fix_constraints(model, var_dicts, old_constraints_dict)
        print("added fix constraints:", nfix)

    sol = model.solve(log_output=False)
    if sol is None:
        raise RuntimeError("mini_routing: infeasible or no solution found")
    
    # ----------------
    # Build NEW fix constraints only for selected_ids and only k-families
    # ----------------
    sol_vals = extract_solution_values(var_dicts, sol)
    new_fix = fix_only_k_families(sol_vals, selected_ids)

    # merge into old fixmap (deep)
    updated_fixmap = deep_merge_fixmaps(dict(old_constraints_dict), new_fix)

    elapsed = time.perf_counter() - t_start_total
    # print(f"mini_routing total time: {elapsed:.3f}s")

    return model, instance, updated_fixmap, sol





# ======================================================================
#  RUN HEURISTIC PROB MODEL - GRID
# ======================================================================
def run_heu_prob_model(
    instance: Instance,
    model_name: str,         ### Name of the heuristic
    network_cont_path, 
    requests_cont_path, 
    network_disc_path, 
    requests_disc_path,          
    number: int,
    horizon: int,
    q_min: int,
    q_max: int,
    alpha: float,
    slack_min: float,
    seed: int,
    exp_id: str,
    mean_edge_length_km: float,
    mean_speed_kmh: float,
    rel_std: float,
    base_output_folder,     ### Folder for results
    cplex_cfg: dict | None = None,
    keep: int = 4,
    fict: int = 3,
    it_out: int = 100,
    it_in: int = 50,
    time_out: float = 36_000,
    tol:float = 0.1
) -> dict:
    """
    Euristica basata su sentizzazione per prob. su una stessa Instance (GRID).
    """

    original_reqs = load_original_requests(str(requests_cont_path))
    original_ids = [r["id"] for r in original_reqs]

    ### Working with continuous requests and network
    G, req7d, node_xy = build_req7d_from_paths(
        network_path=network_cont_path,      
        requests_path=requests_cont_path,  
    )
    
    original_ids_old = original_ids.copy()
    original_ids_new = original_ids.copy()
    start_time = time.time()
    i = 0
    while (
          (i <= it_out) and 
          (time.time() - start_time < time_out) and 
          (len(original_ids_new) >= keep) and 
          original_ids_old != original_ids_new     
    ):
        selected_log, remaining_req = iterative_remove_by_centroid(
            req6d,
            n_remove=keep,
            use_capacity=True,     # use 7th dim
            capacity_key="q",
            standardize=True,
            mode="closest",
        )
        original_ids_old = original_ids_new

        selected_ids = [s["removed_k"] for s in selected_log] 
        selected_full = pick_original_by_ids(original_reqs, selected_ids)   # Selection from the original ones

        j = 0
        f_obj_approx_old = inf
        f_obj_approx_new = inf

        while (j < it_in) and (abs(f_obj_approx_old - f_obj_approx_new)) >= tol:
            f_obj_approx_old = f_obj_approx_new

            # Will be a GP in the future
            ##### QUIIIII PUPOOOOO """"
            fictitious_req, _, _ = make_fictitious_requests_from_remaining_list(
                remaining=remaining_req,
                n_fict=fict,
                use_capacity=True,
                capacity_key="q",
                standardize=True,
                random_state=seed,
            )

            #### FUNZIONE DA FARE ESTERNA ##################################################
            # Projection onto the original space graph
            fict_graph = [snap_fict_request_to_graph_nodes(f, node_xy) for f in fictitious_req]
            
            # Conversion to the original 2D format
            fict_full = to_demand_generator_format(
                G,
                fict_graph,
                slack_min=slack_min,
                force_arrival_eq_sp=False,  # oppure True se vuoi tD=tP+tau_sp
            )

            # Merge finale
            final_requests = selected_full + fict_full
            ############################################################################À


            # -----------------
            # MINI ROUTING
            # -----------------
            
            dict_of_fixed_variables
            ...
            original_ids_new.pop() di quelle che sono state soddsfatte


            # -----------------
            # EVALUATE MINIRUTING (relaxed)
            # -----------------
            ....
            mdl_lp = mdl.relax()
            sol_lp = mdl_lp.solve()
            f_obj_approx_new=....


            j += 1




        i += 1  # i++
        original_ids.pop("qualcosa") # <---



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
        "mean_edge_length": mean_edge_length_km,
        "mean_speed": mean_speed_kmh,
        "std": rel_std,
        "horizon": horizon,
        "dt": instance.dt,
        "t_max": instance.t_max,
        "num_modules": instance.num_modules,
        "num_trails": instance.num_trail_modules,
        "z_max": instance.Z_max,
        "Q": instance.Q,
        "c_km": instance.c_km,
        "c_uns": instance.c_uns,
        "num_requests": instance.num_requests,
        "served": served,
        "served_ratio": served_ratio,
        "q_min": q_min,
        "q_max": q_max,
        "alpha": alpha,
        "slack_min": slack_min,
        "depot": instance.depot,


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



