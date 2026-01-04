from utils.loader_fun import *
from utils.instance_def import *
from utils.cplex_config import *
from utils.output_fun import *
from utils.heuristic_additional_fun import *
from data_generation.generate_data import *
from models.model_MT_w import *

import time
import copy
import random
import numpy as np
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern, 
    RationalQuadratic, 
    WhiteKernel,
    ConstantKernel as C
)





def extract_solution_dict(
    solution,
    x=None, y=None, r=None, w=None, z=None,
    s=None, L=None, R=None, a=None, b=None,
    h=None, D=None, U=None, z_main=None, kappa=None,
    thr: float = 0.5,
) -> dict:
    """
    Estrae la soluzione come dizionario:
    sol["x"][(m,i,t)] = 1
    sol["r"][(k,t,m)] = 1
    ecc.
    """
    sol = {}

    def extract(var_dict, thr_local):
        if var_dict is None:
            return None
        out = {}
        for key, var in var_dict.items():
            val = solution.get_value(var)
            if val is None:
                continue
            if val > thr_local:
                out[key] = val
        return out

    sol["x"] = extract(x, thr)
    sol["y"] = extract(y, thr)
    sol["r"] = extract(r, thr)
    sol["w"] = extract(w, thr)
    sol["z_old"] = extract(z, thr)
    sol["s"] = extract(s, 0.0)      # s always present
    sol["L"] = extract(L, thr)
    sol["R"] = extract(R, thr)
    sol["a"] = extract(a, thr)
    sol["b"] = extract(b, thr)
    sol["h"] = extract(h, thr)
    sol["D"] = extract(D, thr)
    sol["U"] = extract(U, thr)
    sol["z_main"] = extract(z_main, thr)
    sol["kappa"] = extract(kappa, thr)

    # rimuove voci None o vuote
    sol = {k: v for k, v in sol.items() if v}

    return sol



### Solution for NO Nw using CPLEX
def first_solution_initialization_solver(
    instance_no_Nw: Instance,
    cplex_cfg: dict | None = None,
) -> dict:
    """
    Risoluzione modello exact
    """

    # Initialization
    x = y = r = w = L = R = s = a = b = h = None
    D = U = z = kappa = None  

    # NO Nw
    instance_no_Nw.num_Nw = 0
    instance_no_Nw.__post_init__()   # to recalculate Nw = set()
    
    model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_w(instance_no_Nw)
    configure_cplex(model, cplex_cfg)

    # Solve
    solution = model.solve(log_output=False)
    if solution:
        print(f"[First solution Nw=0 (solver)] Objective: {solution.objective_value}")
    else:
        print(f"[First solution Nw=0 (solver)] No solution found.")
        return None, None


    # Extract solution
    sol_dict = extract_solution_dict(
        solution,
        x=x, y=y, r=r, w=w, z=z,
        s=s, a=a, b=b, h=h,
        D=D, U=U, z_main=z, kappa=kappa,
        thr=0.5
    )
    obj = solution.objective_value
    return sol_dict, obj



### Evalute candidate using CPLEX
def evaluate_candidate(
    instance_base,
    cplex_cfg,
    is_Nw_used_cand: dict,
    nw: int,
    t: int,
):
    """
    Valuta il candidato (nw, t)
    """
    # Initialization
    x = y = r = w = L = R = s = a = b = h = None
    D = U = z = kappa = None  
    
    model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_w_Nw(instance_base, is_Nw_used_cand)
    configure_cplex(model, cplex_cfg)

    # Solve
    solution = model.solve(log_output=False)
    if solution is None:
        print(f"   For {(nw,t)} NO solution found.")
        return None, None


    # Extract solution
    sol_dict = extract_solution_dict(
        solution,
        x=x, y=y, r=r, w=w, z=z,
        s=s, a=a, b=b, h=h,
        D=D, U=U, z_main=z, kappa=kappa,
        thr=0.5
    )
    obj = solution.objective_value
    return sol_dict, obj






def GP_window_selection(
    instance_base, 
    time_windows,
    is_Nw_used: dict,
    current_solution: dict,
    value_current_solution: float,
    epsilon: float,
    n_iterations: int,
    n_init:int,    # n° of solutions as initialization for each GP round
    seed: int,
    cplex_cfg: dict | None = None
):
    """
    Function that for each Nw selects a t where Nw will be active
    """
    new_sol = copy.deepcopy(current_solution)
    new_obj_val = value_current_solution
    initial_obj_val = value_current_solution
    gap = 0.0   # Gap to track the improvement
    new_is_Nw_used = copy.deepcopy(is_Nw_used)


    rng = random.Random(seed)
    Nw_nodes_shuffled = list(instance_base.Nw)    # Nw set in a list and then shuffle
    rng.shuffle(Nw_nodes_shuffled)

    # For each node Nw we try to insert a new active time
    # The previus insertion will be present in new_is_Nw_used
    for nw in Nw_nodes_shuffled:
        print(f"   nw = {nw}")
        candidate_times = [    # [t to try]
            t for t in time_windows
            if new_is_Nw_used[(nw, t)] == 0
        ]
        # Model without new time windows for the current nw
        observed_t = []    # no new time windows observed for the current nw
        observed_y = []    # obj_value without trying a new time windows
        observed_sol = []  # solution without trying a new time windows

        # ---------------
        # Initialization (evaluate y for some t)
        # ---------------
        n_init0 = min(n_init, len(candidate_times))
        init = rng.sample(candidate_times, k=n_init0)     # Sample some candidate
        j = 0
        for t in init:
            is_Nw_used_candidate = copy.deepcopy(new_is_Nw_used)
            is_Nw_used_candidate[(nw, t)] = 1   # Activate the time windos for (nw,t)
            sol_dict, obj = evaluate_candidate(
                instance_base = instance_base,
                is_Nw_used_cand = is_Nw_used_candidate, 
                cplex_cfg = cplex_cfg,
                nw=nw,
                t=t
            )
            if obj is None:   # Small check
                candidate_times.remove(t)
                continue
            observed_t.append(t)
            candidate_times.remove(t)    # remove the tested t from candidate list
            observed_y.append(obj)
            observed_sol.append(sol_dict)
            print(f"      GP init {j}: nw={nw}, t_init={t}, obj={obj}")
            j += 1
        
        # --------------
        # Loop GP / UCB
        # --------------
        ### GP definition
        ### REMARK: "fixed" if small number of evaluations
        Tmax = instance_base.t_max
        ell0 = 3.0  # distanza in slot temporali oltre cui i valori non sono più correlati
        kernel = (   
            # non smooth, ma localmente correlato
            C(1.0, constant_value_bounds="fixed") *    # valore iniziale e bound di variazione per σ^2 fisso
            (Matern(length_scale=ell0, length_scale_bounds="fixed",  nu=1.5)   
            + RationalQuadratic(length_scale=ell0, alpha=0.5))           # alpha = 0.5 significa tante scale diverse
            + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=None,  # NO ottimizzazione iperparametri
            alpha=1e-10,    # jitter numerico per piccola quantità aggiunta alla diagonale della matrice di covarianza per poter invertire la matrice
            normalize_y=True,
        )
        ### GP iterations
        for it in range(n_iterations):
            if len(candidate_times) == 0:
                break

            ### GP Fit on observed data
            X_obs = np.array(observed_t, dtype=float).reshape(-1, 1)    # column vector
            y_obs = np.array(observed_y, dtype=float)                      
            gp.fit(X_obs, y_obs)

            ### GP predict on remaining candidates
            X_cand = np.array(candidate_times, dtype=float).reshape(-1, 1)
            y_mean, y_std = gp.predict(X_cand, return_std=True)

            ### MINIMIZATION: use LCB
            lcb = y_mean - epsilon * y_std
            best_idx = int(np.argmin(lcb))
            t_next = candidate_times[best_idx]

            # Extract best:
            y_mean_best = float(y_mean[best_idx])
            y_std_best  = float(y_std[best_idx])
            lcb_best    = float(lcb[best_idx])

            ### Evaluation of the best candidate
            is_Nw_used_candidate = copy.deepcopy(new_is_Nw_used)
            is_Nw_used_candidate[(nw, t_next)] = 1

            sol_next, y_next = evaluate_candidate(
                instance_base=instance_base,
                is_Nw_used_cand=is_Nw_used_candidate,
                cplex_cfg=cplex_cfg,
                nw=nw,
                t=t_next
            )
            if y_next is None:
                # remove infeasible / failed candidate and continue
                candidate_times.pop(best_idx)
                continue

            print(f"      GP it {it}: nw={nw}, t_next={t_next}, obj={y_next}")
            print(f"            LCB=y_mean-ε*y_std: {lcb_best:.2f} = {y_mean_best:.2f} - {epsilon:.1f} * {y_std_best:.2f}")

            ### Update observed data
            observed_t.append(t_next)
            observed_y.append(y_next)
            observed_sol.append(sol_next)

            ### Remove observed from list of candidates
            candidate_times.pop(best_idx)

        # ----------------------
        # Best t for current nw
        # ----------------------
        if len(observed_y) == 0:
            continue

        best_local_idx = int(np.argmin(observed_y))
        best_local_t = observed_t[best_local_idx]
        best_local_obj = observed_y[best_local_idx]
        best_local_sol = observed_sol[best_local_idx]

        # accept only if improves global
        if best_local_obj < new_obj_val:
            new_obj_val = best_local_obj
            new_sol = best_local_sol
            new_is_Nw_used[(nw, best_local_t)] = 1    # Activation of a new (nw,t)

    gap = initial_obj_val - new_obj_val   # Total improvement
    return new_sol, new_obj_val, new_is_Nw_used, gap
    










# ======================================================================
#  RUN HEURISTIC SURROGATE - GRID
# ======================================================================
def run_GP_Nw_heu_model(
    instance: Instance, 
    inst_params:dict,  
    model_name:str,     
    heu_params:dict,
    seed: int,
    base_output_folder,   # Folder for results
    exp_id:str,
    cplex_cfg: dict | None = None,
):
    """
    Meta-heuristic loop semplificato:
    - inizializza soluzione
    - chiama GP (per trovare t migliore per Nw)
    - aggiorna soluzione globale solo se GP restituisce una nuova soluzione valida
    """
    start_heu_time = time.time()

    Nw_nodes=instance.Nw
    Tmax=instance.t_max
    time_windows = list(range(1, Tmax + 1))    # {1,2, ..., t_max}

    max_seconds = heu_params["max_seconds"]    # Max seconds for whole heuristics
    epsilon = heu_params["epsilon"]            # How much exploration for UCB
    n_iterations_GP = heu_params["n_iterations_GP"]  # Number of it for GP
    n_init_GP = heu_params["n_init_GP"]        # Number of solution points to initialize the GP
    if n_init_GP < 2:
        print("GP requires at least 2 init points!")
        return None

    # ----------------------------
    # Solution initialization
    # ----------------------------
    current_sol, current_obj_val = first_solution_initialization_solver(
            instance_no_Nw = copy.deepcopy(instance),
            cplex_cfg = cplex_cfg
            )
    start_time = time.time()
    i = 0
    running_time = 0.0
    improvement = True
    is_Nw_used = {       # When Nw is active (value 1) or inactive (value 0)
            (nw, t): 0   # At the beginning all 0
            for nw in Nw_nodes
            for t in time_windows
        }
    while (running_time < max_seconds) and (improvement):  # Stop se fuori tempo max e non si hanno più miglioramenti
        print(f"\n--- OUTER ITERATION {i} ---")
        # ------------------------------------------
        # GP to activate a time windows for each Nw
        # ------------------------------------------
        # Return the same solution or a best one
        current_sol, current_obj_val, is_Nw_used, gap = GP_window_selection(      
            instance_base = copy.deepcopy(instance), 
            time_windows=time_windows,
            is_Nw_used=is_Nw_used,
            current_solution=current_sol,
            value_current_solution=current_obj_val,
            epsilon=epsilon,
            n_iterations=n_iterations_GP,
            n_init=n_init_GP,
            seed=seed+1     # To have a different shuffle of the Nw points to try first
        )
        print(f"Gap for {i}:", gap)

        if gap < 1e-6:   # no real improvements
            improvement = False

        i += 1
        running_time = time.time() - start_time


    # --- Time ---
    end_heu_time = time.time()
    tot_heu_time = end_heu_time - start_heu_time
    print(f"Tot heuristic time, i = {i}: {tot_heu_time:.2f}s")

    # --- Output folder ---
    output_folder = Path(base_output_folder) / f"{model_name}_GP_Nw_HEU"
    output_folder.mkdir(parents=True, exist_ok=True)
    print(output_folder)

    # --- Save solution (DICT) ---
    if current_sol is not None:
        save_solution_dict_summary(
            sol_dict=current_sol,
            objective=current_obj_val,
            output_folder=output_folder,
            extra={
                "heuristic": "GP_Nw",
                "total_time_sec": tot_heu_time,
                "epsilon": heu_params["epsilon"],
                "max_seconds": heu_params["max_seconds"],
                "n_init_GP": heu_params["n_init_GP"],
                "n_iterations_GP": heu_params["n_iterations_GP"],
            }
        )
        save_solution_dict_variables_flex(current_sol, output_folder, thr=0.5)

    # --- Served requests summary (DICT) ---
    if current_sol is not None:
        s_dict = current_sol.get("s", {})

        served_requests = [
            k for k in instance.K
            if (k in s_dict) and (s_dict[k] is not None) and (s_dict[k] > 0.5)
        ]

        served = len(served_requests)
        total = len(instance.K)
        served_ratio = served / total if total > 0 else 0.0

        status = "HeuristicDict"
        objective = float(current_obj_val) if current_obj_val is not None else None
    else:
        served_requests = []
        served = 0
        total = len(instance.K)
        served_ratio = 0.0
        status = "NoSolution"
        objective = None

    print(f"[FINAL HEU {model_name}] -> served: {served}/{total}  ({served_ratio*100:.1f}%)")
    print(f"[FINAL HEU {model_name}] -> richieste servite (k): {served_requests}")

    # --- Results ---
    inst_block = dict(inst_params)
    inst_block["grid_nodes"] = inst_params["number"] ** 2

    result = {
        "exp_id": exp_id,
        "model_name": f"{model_name}_GP_Nw_HEU",
        "seed": seed,

        **inst_block,

        # parametri euristica GP
        "heu_max_seconds": heu_params["max_seconds"],
        "heu_epsilon": heu_params["epsilon"],
        "heu_n_init_GP": heu_params["n_init_GP"],
        "heu_n_iterations_GP": heu_params["n_iterations_GP"],

        # risultati
        "status": status,
        "objective": objective,
        "total_time_sec": tot_heu_time,
        "served": served,
        "served_ratio": served_ratio,

        "output_folder": str(output_folder),
    }

    return result