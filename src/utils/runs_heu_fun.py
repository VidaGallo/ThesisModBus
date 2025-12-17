from utils.loader_fun import *
from utils.instance_def import *
from utils.cplex_config import *
from utils.output_fun import *
from utils.heuristic_additional_fun import *
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
    instance,                               
    selected_ids: List[int],                # <-- richieste che vuoi fissare dopo questa solve
    fixed_constr: Dict[str, Dict[tuple, float]] | None,
    model_name: str = "w",
    cplex_cfg: dict | None = None,
) -> Tuple[Any, Dict[str, Dict[tuple, float]], Any]:
    """
    Crea e risolve il modello su 'instance' già pronta.
    Applica fix-and-optimize da old_constraints_dict.
    Dopo la solve, crea nuovi fix SOLO per le famiglie indicizzate da k (selected_ids),
    e fa merge con i fix esistenti.

    """
    # ----------------
    # Build model
    # ----------------
    if model_name == "w":
        model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_w(instance)
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
    if fixed_constr:
        fix_constraints(model, var_dicts, fixed_constr)    # Some constraints are already fixed
    else:
        fixed_constr = {}
    # ----------------
    # Solve
    # ----------------
    sol = model.solve(log_output=False)
    if sol is None:
        raise RuntimeError("mini_routing: infeasible or no solution found")

    # ----------------------------
    # Add NEW fixes for selected k
    # ----------------------------
    # Extract solution for k selected
    new_fixed_constr = extract_solution_values_only_selected_k(var_dicts, sol, selected_ids)
    # Merge old and new constraints
    updated_fixed_constr = deep_merge_fixmaps(fixed_constr, new_fixed_constr)

    return updated_fixed_constr




                


# ======================================================================
#  RUN HEURISTIC PROB MODEL - GRID
# ======================================================================
def run_heu_prob_model(
    instance: Instance,
    model_name: str,         # 'w'
    network_cont_path, 
    requests_cont_path, 
    network_disc_path, 
    requests_disc_path,          
    number: int,
    horizon: int,
    dt:int, 
    num_modules: int,
    num_trails: int,
    c_km: float,
    c_uns: float,
    depot: int,
    num_Nw: int,
    z_max: int,
    t_max: int,
    q_min: int,
    q_max: int,
    Q: int,
    alpha: float,
    slack_min: float,
    seed: int,
    exp_id: str,
    mean_edge_length_km: float,
    mean_speed_kmh: float,
    rel_std: float,
    base_output_folder,     ### Folder for results
    cplex_cfg: dict | None = None,
    n_keep: int = 4,
    n_fict: int = 3,
    it_out: int = 100,
    it_in: int = 50,
    time_out: float = 36_000,
    tol:float = 0.1
) -> dict:
    """
    Euristica basata su sentizzazione per prob. su una stessa Instance (GRID).
    """
    ### Loading of the discrete netowork and discrete request
    with network_disc_path.open("r", encoding="utf-8") as f:
        net_disc = json.load(f)
    with requests_disc_path.open("r", encoding="utf-8") as f:
        req_disc = json.load(f)

    ### Working with continuous requests and continuous network
    # Load original requests and project them in 7D (xo,yo,t0,xd,yd,td,q), all CONTINUOUS TIME
    G, req_original, req7d, node_xy = build_req7d_from_paths(
        network_path=network_cont_path,      
        requests_path=requests_cont_path,  
    )

    original_ids = [r["id"] for r in req_original]   
    

    ### Request to be selected
    remaining_ids = original_ids.copy()             # ID candidati GRIGI
    req7d_to_select = req7d.copy()                  # 7D candidati GRIGI
    req_original_to_select = req_original.copy()    # richieste candidati GRIGI
    
    ### Requests that were already selected in past
    fixed_requests = []        # List of dict       # richieste ROSA

    ###################
    ### OUTER WHILE ###
    ###################
    start_out_time = time.time()
    i = 0
    constraints_dict_old = {}       # constraints delle richieste ROSA
    while (
          (i <= it_out) and                                  # stop if too many iterations
          (time.time() - start_out_time < time_out) and      # stop if too much time
          (len(remaining_ids) >= n_keep)                     # stop if there are not enough requests remaining
          ###          # stop if the remaining requests don't change
        ):

        ### Select k = keep elements 
        selected_ids, remaining_ids = topk_ids_by_centroid_7d(req7d_to_select, n_keep)
        
        ### Pick the requests from original ones (in original format)
        selected_now = pick_original_by_ids(req_original, selected_ids)   # Selezione richieste GRIGIE 
        selected_full = selected_now + fixed_requests     # Unione con richieste già scelte nei giri precedenti (GRIGIE + ROSA)

        ### Pick the rest of the requests from req7d
        remaining_set = set(int(k) for k in remaining_ids)
        remaining_req7d = [r for r in req7d if int(r["k"]) in remaining_set]




        ###################
        ### OUTER WHILE ###
        ###################
        start_in_time = time.time()
        j = 0
        f_obj_approx_old = inf
        f_obj_approx_new = inf
        
        while (j < it_in) and (abs(f_obj_approx_old - f_obj_approx_new)) >= tol:
            # Reinitialization
            f_obj_approx_old = f_obj_approx_new.copy()
            constraints_dict_new = constraints_dict_old.copy()   # ritorno alle constraint senza i ROSSI (ovvero solo le ROSA)
            
            # Will be a GP in the future
            ##### QUIIIII PUPOOOOO """"
            fictitious_req, _, _ = make_fictitious_requests_from_remaining_list(
                remaining=remaining_req7d,
                n_fict=n_fict,
                use_capacity=True,
                capacity_key="q",
                standardize=True,
                random_state=seed,
            )

            # Return to the original format (CONTINUOUS TIME)
            # If capacity is float => ceil(q)
            fict_full = fict7d_to_requests_format(
                fictitious_req,
                node_xy=node_xy,
                G=G,
                slack_min=slack_min,
                top_snap=10,
                force_arrival_eq_sp=False,
            )

            # Final merge
            final_requests = selected_full + fict_full


            # Request dscretization
            final_requests_disc = discretize_requests_dict(final_requests, dt)



            # -----------------
            # MINI ROUTING
            # -----------------
            # Instance for current miniruting
            I_mr = load_instance_discrete_from_data(
                net_discrete = net_disc,
                reqs_discrete = final_requests_disc,          # lista richieste DISCRETE (come json.load)
                dt =dt,
                t_max=t_max, 
                num_modules=num_modules,
                num_trail=num_trails,
                Q=Q,
                c_km=c_km,
                c_uns=c_uns,
                depot=depot,
                num_Nw=num_Nw,
                z_max=z_max
                )
            
            # Run mini routing to obtain new fixed cosntraints
            constraints_dict_new = mini_routing(
                instance=I_mr,                 
                selected_ids=selected_ids,            # lista k (id richieste) che sono fissate
                fixed_constr=constraints_dict_old,    # constraints relativi alle richieste passate
                model_name=model_name,
                cplex_cfg=cplex_cfg,
            )



            # -----------------
            # EVALUATE MINIRUTING (relaxed)
            # -----------------
            #....
            #mdl_lp = mdl.relax()
            #sol_lp = mdl_lp.solve()
            #f_obj_approx_new=....


            j += 1
        

        fixed_ids_requests     # List of id
        fixed_requests     ### Si inseriranno richieste con .pop()
            = original_ids_new.pop() di quelle che sono state soddsfatte        
        constraints_dict_old = constraints_dict_new.copy()     ### Queste constriant saranno fissate in futuro
        
        i += 1  # i++
        stop_in_time =  time.time()
        print(f"Inner while time, j = {j}: ", stop_in_time - start_in_time)

    stop_out_time =  time.time()
    print(f"Outer while time, i = {i}: ", stop_out_time - start_out_time)

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



