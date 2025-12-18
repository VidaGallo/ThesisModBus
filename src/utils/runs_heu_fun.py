from utils.loader_fun import *
from utils.instance_def import *
from utils.cplex_config import *
from utils.output_fun import *
from utils.heuristic_additional_fun import *
from data_generation.generate_data import *
from models.model_MT_w import *

import time

import copy
from math import inf
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter



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

    ### se infeasable fare obective value = 1e100
    return updated_fixed_constr
"""

def mini_routing(
    instance,
    selected_ids: List[int],                                  # GRIGI: richieste da fissare dopo la solve
    fixed_constr: Optional[Dict[str, Dict[tuple, float]]],     # ROSA: fix già accumulati
    *,
    labels_event: Optional[Dict[tuple[int, str], int]] = None, # {(k,"P"/"D"): cluster or -1}
    cluster_ks: Optional[Iterable[int]] = None,               # NERI: k su cui applicare labels_event
    model_name: str = "w",
    cplex_cfg: dict | None = None,
    infeas_value: float = 1e100,
) -> Tuple[Dict[str, Dict[tuple, float]], float]:
    """
    Build+solve model with:
      1) old fixed constraints (ROSA)
      2) event-wise clustering constraints on cluster_ks (NERI):
         - if k has P or D in -1 => force entire request k to 0 (s,r,a,b,w)
         - same module within pickup clusters (a)
         - same module within dropoff clusters (b)

    Returns:
      (updated_fixed_constr, objective_value_or_infeas_value)
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

    configure_cplex(model, cplex_cfg)

    # --------------------------
    # Apply previous constraints
    # --------------------------
    fixed_constr = {} if fixed_constr is None else fixed_constr
    if fixed_constr:
        fix_constraints(model, var_dicts, fixed_constr)

    # --------------------------
    # Apply event-wise clustering constraints
    # --------------------------
    if labels_event is not None:
        if cluster_ks is None:
            raise ValueError("mini_routing: if labels_event is provided, you must also provide cluster_ks")

        cluster_ks = [int(k) for k in cluster_ks]

        ignored_ks, clusters_P, clusters_D = split_event_labels(labels_event, cluster_ks)

        # -1 => force whole request to 0
        if ignored_ks:
            add_ignored_request_zero_constraints(model, instance, r, a, b, w, s, ignored_ks, name="ign")

        # same module within pickup/drop clusters
        if clusters_P or clusters_D:
            add_cluster_same_module_constraints_eventwise(
                model, instance, a, b, clusters_P, clusters_D, name="clE"
            )

    # ----------------
    # Solve
    # ----------------
    sol = model.solve(log_output=False)
    if sol is None:
        return fixed_constr, infeas_value

    obj = float(sol.objective_value)

    # ----------------------------
    # Add NEW fixes for selected k
    # ----------------------------
    new_fixed_constr = extract_solution_values_only_selected_k(var_dicts, sol, selected_ids)
    updated_fixed_constr = deep_merge_fixmaps(fixed_constr, new_fixed_constr)

    return updated_fixed_constr, obj

                


# ======================================================================
#  RUN HEURISTIC PROB MODEL - GRID
# ======================================================================
def run_heu_model(
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
    it_out: int = 100,
    it_in: int = 50,
    time_out: float = 36_000,
    tol:float = 0.1,
    k_clust: int = 4,
) -> dict:
    """
    Euristica
    """
    start_heu_time = time.time()

    ### Loading of the discrete netowork and discrete request
    with network_disc_path.open("r", encoding="utf-8") as f:
        net_disc = json.load(f)
    with requests_disc_path.open("r", encoding="utf-8") as f:
        req_disc = json.load(f)

    # Full instance (with original discrete data)
    I_full = load_instance_discrete_from_data(
            net_discrete=net_disc,
            reqs_discrete=req_disc,        # tutte le richieste originali
            dt=dt,
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

    ### Working with continuous requests and continuous network
    # Load original requests and project them in 7D (xo,yo,t0,xd,yd,td,q), all CONTINUOUS TIME
    G, req_original, req7d, node_xy = build_req7d_from_paths(
        network_path=network_cont_path,      
        requests_path=requests_cont_path,  
    )


    original_ids = [r["id"] for r in req_original]   
    

    ### Request to be selected
    remaining_ids = original_ids.copy()                     # ID candidati GRIGI
    req7d_to_select = copy.deepcopy(req7d)                  # 7D candidati GRIGI
    req_original_to_select = copy.deepcopy(req_original)    # richieste candidati GRIGI
    
    ### Requests that were already selected in past
    fixed_requests = []        # List of dict       # richieste ROSA
    fixed_constraints = {}     # Dictionary of fixed variables     # richieste ROSA


    ###################
    ### OUTER WHILE ###
    ###################
    start_out_time = time.time()
    i = 0
    while (
          (i < it_out) and                                   # stop if too many iterations
          (time.time() - start_out_time < time_out) and      # stop if too much time
          (len(remaining_ids) >= n_keep)                     # stop if there are not enough requests remaining
        ):
        print(f"Nel out while i: {i}")

        ### Select k = keep elements 
        # id GRIGI + id NERI
        selected_ids, remaining_ids = topk_ids_by_centroid_7d(req7d_to_select, n_keep)   # Selezione richieste GRIGIE 
        
        ### Pick the requests (in the original format)
        remaining_req7d = pick_by_ids(req7d_to_select, remaining_ids)

        events_req4d = build_events_4d_from_req7d(remaining_req7d)    # Separation of O and D



        ######################################
        ### INNER WHILE - GAUSSIAN PROCESS ###
        ######################################
        start_in_time = time.time()
        j = 0
        obj_f_best = 1e100
        old_f_best_old = 1e100
        diff = 1e100
        best_candidate_constraints = None     # Dictionary with old + new constraints     # ROSA + GRIGI
        
        theta = Theta(p_noise=0.2)    # Theta start for this round of GP
        
        while (j < it_in) and (diff) >= tol:
            print(f"Nel in while j: {j}")

            ### 4D, CLUSTERING of the remaining requests (dei NERI)
            labels_event =cluster_events_random_eventwise(
                events4d = events_req4d,
                n_clusters = k_clust,
                p_noise = theta.p_noise,
                seed = seed,
                enforce_pair_noise = True   # Se O o D stanno in -1, entrambi in -1
            ) 

            ### Mini Routing to obtain new candidate cosntraints + obj_f
            candidate_constraints, candidate_obj_f = mini_routing(
                instance=I_full,
                selected_ids=selected_ids,
                fixed_constr=fixed_constraints,  # cosntraints già decise in passato (richieste ROSA)
                model_name=model_name,
                cplex_cfg=cplex_cfg,
                labels=labels_event,
                remaining_ids=remaining_ids,   # serve per sapere su quali k applicare vincoli
            )

            if candidate_obj_f < obj_f_best:
                diff = abs(old_f_best_old - candidate_obj_f)   # We have an improvement

                old_f_best_old = obj_f_best   # It's the new old
                obj_f_best = candidate_obj_f
                best_candidate_constraints = copy.deepcopy(candidate_constraints)   # Save the constraints (they are good)
                
            j += 1   # j++
            
        stop_in_time =  time.time()
        print(f"End in while , j = {j}: ", stop_in_time - start_in_time)

        
        ### Selected and now fixed requests (GRIGI diventati ROSA)
        selected_set = set(int(k) for k in selected_ids)

        # Estrazione nuovi fixed 
        new_fixed_requests = [r for r in req_original_to_select if int(r["id"]) in selected_set]
        fixed_requests.extend(new_fixed_requests)   # I GRIGI diventano ROSA

        # Rimozione dei nuovi fixed (dei GRIGI diventati ROSA)
        req7d_to_select = [r for r in req7d_to_select if int(r["k"]) not in selected_set]
        req_original_to_select = [r for r in req_original_to_select if int(r["id"]) not in selected_set]

        if best_candidate_constraints is None:
            best_candidate_constraints = copy.deepcopy(fixed_constraints)
        fixed_constraints = copy.deepcopy(best_candidate_constraints)     ### Queste constriant saranno fissate in futuro (ROSA)
        
        i += 1  # i++

    stop_out_time =  time.time()
    print(f"End out while, i = {i}: ", stop_out_time - start_out_time)



    # ----------------------------------------------
    # Solve complete model with complete constriants
    # ----------------------------------------------

    ### Folder for the results
    output_folder = Path(base_output_folder) / f"{model_name}_HEU"
    output_folder.mkdir(parents=True, exist_ok=True)

    t_start_total = time.perf_counter()
    if model_name == "w":
        model_final, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_w(I_full)
        var_dicts = {
            "x": x, "y": y,
            "r": r, "w": w, "s": s,
            "a": a, "b": b,
            "D": D, "U": U,
            "z": z, "kappa": kappa,
            "h": h,
        }
        L = R = None
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    configure_cplex(model_final, cplex_cfg)

    ### Apply fixed constraints
    if fixed_constraints:   
        fix_constraints(model_final, var_dicts, fixed_constraints)

    # Solve the full model 
    t_start_solve = time.perf_counter()
    solution = model_final.solve(log_output=False)
    solve_time = time.perf_counter() - t_start_solve
    total_time = time.perf_counter() - t_start_total


    ### Time
    end_heu_time = time.time()
    print(f"Tot heuristic time, j = {j}: ", end_heu_time - start_heu_time)  


    ### Prints solution
    if solution:
        print(f"[EXP {exp_id} | FINAL FULL | model={model_name}] Status: {solution.solve_status}")
        print("Objective:", solution.objective_value)
    else:
        print(f"[EXP {exp_id} | FINAL FULL | model={model_name}] No solution found.")
    print("Solve time (sec):", solve_time)
    print("Total time (sec):", total_time)
    print("-" * 77)
    print(output_folder)

    
    ### Saves solution
    save_model_stats(model_final, output_folder)
    save_cplex_log(model_final, output_folder)

    if solution is not None:
        save_solution_summary(solution, output_folder)
        try:
            save_solution_variables_flex(
                solution=solution,
                output_folder=output_folder,
                x=x, y=y, r=r, w=w, s=s,
                L=L, R=R,
                a=a, b=b, h=h,
                D=D, U=U,
                z_main=z,
                kappa=kappa,
            )
        except TypeError:
            pass


    ### Served requests summary
    if solution is not None:
        served_requests = []
        for k in I_full.K:
            val = solution.get_value(s[k])
            if val is not None and val > 0.5:
                served_requests.append(k)

        served = len(served_requests)
        total = len(I_full.K)
        served_ratio = served / total if total > 0 else 0.0
    else:
        served_requests = []
        served = 0
        total = len(I_full.K)
        served_ratio = 0.0

    print(f"[FINAL FULL {model_name}] -> served: {served}/{total}  ({served_ratio*100:.1f}%)")
    print(f"[FINAL FULL {model_name}] -> richieste servite (k): {served_requests}")


    ### Solver info
    if solution is not None:
        status = str(solution.solve_status)
        objective = float(solution.objective_value)
        try:
            mip_gap = model_final.solve_details.mip_relative_gap
        except Exception:
            mip_gap = None
    else:
        status = "NoSolution"
        objective = None
        mip_gap = None


    ### Results
    result = {
        "exp_id": exp_id,
        "model_name": f"{model_name}_HEU_FINAL_FULL",
        "num_Nw": I_full.num_Nw,
        "seed": seed,
        "number": number,
        "grid_nodes": number * number,
        "mean_edge_length": mean_edge_length_km,
        "mean_speed": mean_speed_kmh,
        "std": rel_std,
        "horizon": horizon,
        "dt": I_full.dt,
        "t_max": I_full.t_max,
        "num_modules": I_full.num_modules,
        "num_trails": I_full.num_trail_modules,
        "z_max": I_full.Z_max,
        "Q": I_full.Q,
        "c_km": I_full.c_km,
        "c_uns": I_full.c_uns,
        "num_requests": I_full.num_requests,
        "served": served,
        "served_ratio": served_ratio,
        "q_min": q_min,
        "q_max": q_max,
        "alpha": alpha,
        "slack_min": slack_min,
        "depot": I_full.depot,
        "N_size": len(I_full.N),
        "A_size": len(I_full.A),
        "K_size": len(I_full.K),
        "M_size": len(I_full.M),
        "P_size": len(I_full.P),
        "status": status,
        "objective": objective,
        "mip_gap": mip_gap,
        "solve_time_sec": solve_time,
        "total_time_sec": total_time,
        "output_folder": str(output_folder),
        "network_path": str(network_disc_path),
        "requests_path": str(requests_disc_path),
    }

    return result



