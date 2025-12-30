from utils.loader_fun import *
from utils.instance_def import *
from utils.cplex_config import *
from utils.output_fun import *
from utils.heuristic_additional_fun import *
from data_generation.generate_data import *
from models.model_MT_w import *

import time
import copy

from pathlib import Path




def mini_routing(
    instance,
    future_fixed_ids: List[int],       # GRIGI: richieste da fissare dopo la solve
    clustered_ids: List[int],           # NERI: richieste clsuterizzate    
    fixed_constr: Optional[Dict[str, Dict[tuple, float]]],        # ROSA: fix già accumulati
    labels_PD: Optional[Dict[tuple[int, str], int]],              # {(k,"P"/"D"): cluster or -1}
    base_model,
    base_var_dicts,
    warm_start_bool: bool = False,
    mip_start: Optional[Dict[str, float]] = None,   # Partial solution proposal (for warm start)
    infeas_value: float = 1e100,
    
) -> Tuple[Dict[str, Dict[tuple, float]], float, Optional[Dict[str, float]]]:
    """
    Build+solve model with:
      1) old fixed constraints (ROSA)
      2) new clustering constraints (NERI)

    Returns:
      (new_candidate_fixed_constr, objective_value_or_infeas_value)
    """
    # -----------
    # Build model
    # -----------
    model = base_model.clone()
    var_dicts = map_vars_by_name(model, base_var_dicts)



    a = var_dicts["a"]
    b = var_dicts["b"]

    # apply mip start (warm start)
    if mip_start and warm_start_bool:    # If we have a warmup solution and we want to use it
        n = add_mip_start_by_name(model, mip_start)  
        #print(n)



    ### BASELINE: 
    sol_base = model.solve(log_output=False)
    if sol_base is None:
        print(" Base model infeasible")
        obj_base = None
    else:
        obj_base = float(sol_base.objective_value)
        sd = model.solve_details
        print(f"\n   [BASE] status={sd.status} time={sd.time:.2f} obj={obj_base}")


    # -------------------------
    # Apply cluster constraints
    # -------------------------
    n_ct_before = model.number_of_constraints
    n_var_before = model.number_of_variables

    ### Constrainints for the not served requests + same module for each cluster
    if len(clustered_ids)>0:
        if clustered_ids and labels_PD:
            ignored_ks, active_ks = split_ignored_and_active_ks(labels_PD, clustered_ids)

            # Non-served requests
            n_ign = add_ignored_request_zero_constraints_ab(mdl=model, I=instance,
                                                a=a, b=b,
                                                ignored_ks=ignored_ks,
                                                name="ign"
                                                )
            # Fixed modules cluster-wise (con introduzione nuova variabile al modello)
            u = add_cluster_same_module_constraints_ab(mdl=model, I=instance, a=a, b=b,
                                                  labels_PD=labels_PD,
                                                  active_ids=active_ks,
                                                  name="cl_mod",
                                                  )
            

    # -------
    # Solve
    # -------
    ### INTEGER
    sol = model.solve(log_output=False)
    if sol is None:   # Soluzione infeasable
        print("Clustering integer infeasable")
        #return fixed_constr, infeas_value, None    

    obj = float(sol.objective_value)
    sd = model.solve_details
    print(f"   [MINI MIP] status={sd.status} time={sd.time:.2f} obj={obj}")

    ### Check
    print(len(u))
    n_ct_after = model.number_of_constraints
    n_var_after = model.number_of_variables

    print(f"[DBG] Δconstraints = {n_ct_after - n_ct_before}, Δvars = {n_var_after - n_var_before}")
    print(f"[DBG] ignored={len(ignored_ks)} active={len(active_ks)} u_is_None={u is None}")

    def count_active_ab(sol, a, b, thr=0.5):
        na = sum(1 for v in a.values() if sol.get_value(v) is not None and sol.get_value(v) >= thr)
        nb = sum(1 for v in b.values() if sol.get_value(v) is not None and sol.get_value(v) >= thr)
        return na, nb

    na0, nb0 = count_active_ab(sol_base, a, b)
    na1, nb1 = count_active_ab(sol, a, b)
    print(f"[DBG] active a,b: base=({na0},{nb0}) clustered=({na1},{nb1})")

    # ----------------------------------
    # Add NEW constraints for selected k
    # ----------------------------------
    new_fixed_constr = extract_solution_values_only_selected_k_ab(var_dicts, sol, future_fixed_ids)
    updated_fixed_constr = merge_constraints_ab(fixed_constr, new_fixed_constr, check_conflict=True)

    if warm_start_bool:
        start_for_next = extract_mip_start_by_name(sol, var_dicts)
        return updated_fixed_constr, obj, start_for_next
    else:   # No saving of the solution required
        return updated_fixed_constr, obj, None

    

                





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
    inst_params: dict,
    heu_params:dict,
    seed: int,
    exp_id: str,
    base_output_folder,   # Folder for results
    cplex_cfg: dict | None = None,

) -> dict:
    """
    Euristica
    """
    ### Parametri
    n_keep = heu_params["n_keep"]    # n. nodes that are fixed after each iteration (GRIGI)
    it_in =  heu_params["it_in"]
    n_clust = heu_params["n_clust"]
    warm_start_bool = heu_params["warm_start_bool"]    # Start minirouting with the best solution founded in the inner loop so far   

    start_heu_time = time.time()

    # Full instance (with original discrete data)
    I_full = instance

    ### Working with continuous requests and continuous network
    # Load original requests and project them in 7D (xo,yo,t0,xd,yd,td,q), all CONTINUOUS TIME
    G, req_original, req7d, node_xy = build_req7d_from_paths(
        network_path=network_cont_path,      
        requests_path=requests_cont_path,  
    )



    ### Request to be selected
    req7d_remaining = copy.deepcopy(req7d)                  # 7D candidati GRIGI
    req_original_remaining= copy.deepcopy(req_original)    # richieste candidati GRIGI
    
    ### Constraints that will be fixed
    fixed_constraints = {}     # Dictionary of fixed variables     # richieste ROSA

    ### Best global solution (for warm start)
    best_global_warmup_sol = None   
    best_global_warmup_obj = 1e100

    ###################
    ### OUTER WHILE ###
    ###################
    start_out_time = time.time()
    i = 0
    while (len(req_original_remaining) >= n_keep):            # stop if there are not enough requests remaining
        #print(f"Nel out while i: {i}")

        ### Select k = keep elements 
        # id GRIGI + id NERI
        selected_ids, remaining_ids = topk_ids_by_centroid_7d(req7d_remaining, n_keep)   # Selezione richieste GRIGIE 
        

        ### Remaining requests (NERE)
        req7d_remaining = pick_by_ids(req7d_remaining, remaining_ids, key="k")       # new req7d_remaining
        req4d_PD_remaining = build_events_4d_from_req7d(req7d_remaining)      # Separation of PICKUP and DELIVERY
        req_original_remaining = [r for r in req_original_remaining if int(r["id"]) in remaining_ids]  

            


        ######################################
        ### INNER WHILE - GAUSSIAN PROCESS ###
        ######################################
        start_in_time = time.time()
        j = 0
        obj_f_best = 1e100    # Best obj.f. up to now
        best_candidate_constraints = None     # Dictionary with old + new constraints     # ROSA + GRIGI
        
        ### Soluzione greedy per warmup iniziale
        if warm_start_bool:
            greedy_warmup_sol = build_greedy_warmup_sol_unique_modules(
                instance=I_full,
                model_name=model_name,
                base_fixed_constr=fixed_constraints,
                n_warm=5,
                seed=seed,
                cplex_cfg=cplex_cfg,
            )
            best_candidate_warmup_sol = greedy_warmup_sol
        else:
            best_candidate_warmup_sol = None

        
        no_improve = 0   # Number of no improvements in f.obj
        patience = 5

        ### Creazione del base model con fixed constraints (senza clustering)
        base_model, base_var_dicts = build_base_model_with_fixed_constraints(
            instance=I_full,
            model_name=model_name,
            fixed_constr=fixed_constraints,
            cplex_cfg=cplex_cfg,
        )
        seen_signatures: set[int] = set()    # Hash clustering generati e studiati
        while (j < it_in) and no_improve < patience:
            #print(f"Nel in while j: {j}")
            now = time.time()
            
            ### 4D, CLUSTERING of the remaining requests (dei NERI)
            p_insodd = 0    # <- temporaneo
            labels_PD_events =cluster_PD_events_random(
                events4d = req4d_PD_remaining,
                n_clusters = n_clust,
                p_noise = p_insodd,
                seed = seed + j    #cambiare seed per cambiare clustering (cmq riproducibile)
            ) 
            
            ### Verifica clustering già studiato
            sig = tuple(sorted(labels_PD_events.items()))
            sig_hash = hash(sig)
            if sig_hash in seen_signatures:   # clustering già studiato
                j += 1   # j++
                continue
            seen_signatures.add(sig_hash)


            ### Mini Routing to obtain new candidate cosntraints + obj_f

            candidate_constraints, candidate_obj_f, candidate_solution = mini_routing(
                instance=I_full,
                future_fixed_ids=selected_ids,         # requests that will have fixed constraints in the future
                clustered_ids=remaining_ids,           # ID of the requests that were clsutered
                fixed_constr=fixed_constraints,        # cosntraints già decise in passato (richieste ROSA)
                mip_start=best_candidate_warmup_sol,   # best partial solution so far
                warm_start_bool = warm_start_bool,      # use or not warm start
                labels_PD=labels_PD_events,            # CLUSTERIZZAZIONE
                base_model=base_model,                 # base model with FIXED CONSTRAINTS
                base_var_dicts=base_var_dicts,
            )
              

            print(f"   Inner while, j={j}, f.obj={candidate_obj_f}, time = {time.time()-now}")

            ### Inner while best
            if candidate_obj_f < obj_f_best:   # We have an improvement
                obj_f_best = candidate_obj_f    # A new better solution
                best_candidate_constraints = copy.deepcopy(candidate_constraints)    # Save the new better constraints
                best_candidate_warmup_sol = copy.deepcopy(candidate_solution)    # Solution for warmup
                no_improve = 0
            else:
                no_improve += 1
                
            j += 1   # j++
        
    

        stop_in_time =  time.time()
        print(f"\nEND Inner while, j = {j}, f_obj = {obj_f_best}, time = {stop_in_time - start_in_time}")


        if best_candidate_constraints is None:
            best_candidate_constraints = copy.deepcopy(fixed_constraints)
        fixed_constraints = copy.deepcopy(best_candidate_constraints)     ### Queste constriant saranno fissate in futuro (ROSA)
        
        ### Global best (for warm start)
        if obj_f_best < best_global_warmup_obj:
            best_global_warmup_obj = obj_f_best
            best_global_warmup_sol = best_candidate_warmup_sol   

        i += 1  # i++
        print(f"Outer while, i={i}, f.obj={obj_f_best}\n")

    stop_out_time =  time.time()
    print(f"END Outter while, i = {i}, time = {stop_out_time - start_out_time}")
    print("\n")



    # ----------------------------------------------------------
    # Solve complete model considering all the fixed constriants
    # ----------------------------------------------------------

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
        fix_constraints_ab(model_final, var_dicts, fixed_constraints)

    ### Apply warm start
    if best_global_warmup_sol and warm_start_bool:    # We have a warmup solution and we want to use it
        n = add_mip_start_by_name(model_final, best_global_warmup_sol)  
        #print(n)

    ### Solve the full model 
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
    inst_block = dict(inst_params)
    inst_block["grid_nodes"] = inst_params["number"] ** 2

    result = {
        "exp_id": exp_id,
        "model_name": f"{model_name}_HEU_FINAL_FULL",
        "seed": seed,

        # parametri istanza
        **inst_block,

        # parametri euristica (se li vuoi nel CSV)
        "heu_n_keep": heu_params["n_keep"],
        "heu_it_in": heu_params["it_in"],
        "heu_n_clust": heu_params["n_clust"],
        "heu_warm_start": bool(heu_params["warm_start_bool"]),

        # risultati modello finale
        "status": status,
        "objective": objective,
        "mip_gap": mip_gap,
        "solve_time_sec": solve_time,
        "total_time_sec": total_time,
        "served": served,
        "served_ratio": served_ratio,

        # tempo totale euristica (utile)
        "heu_total_time_sec": end_heu_time - start_heu_time,

        # output folder/path (minimo indispensabile)
        "output_folder": str(output_folder),
        "network_path": str(network_disc_path),
        "requests_path": str(requests_disc_path),
    }

    return result



