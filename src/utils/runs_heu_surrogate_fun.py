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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    RationalQuadratic,
    ConstantKernel as C
)





def extract_solution_dict(
    solution,
    *,
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
        print(f"[First solution Nw=0 (solver)] Status: {solution.solve_status}")
        print("Objective:", solution.objective_value)
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
        print(f"[First solution Nw=0 (solver)] Status: {solution.solve_status}")
        print("Objective:", solution.objective_value)
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
    
    model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_w_with_Nw(instance_base, is_Nw_used_cand)
    configure_cplex(model, cplex_cfg)

    # Solve
    solution = model.solve(log_output=False)
    if solution:
        print(f"   Objective function for {(nw,t)}:", solution.objective_value)
    else:
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
    Nw_nodes,
    dt: int,
    Tmax: int,
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
    new_is_Nw_used = copy.deepcopy(is_Nw_used)
    
    # For each node Nw we try to insert a new active time
    rng = random.Random(seed)
    Nw_nodes_shuffled = list(Nw_nodes)
    rng.shuffle(Nw_nodes_shuffled)
    for nw in Nw_nodes_shuffled:
        candidate_times = [    # [t to try]
            t for t in time_windows
            if is_Nw_used[(nw, t)] == 0
        ]
        # Model without new time windows for the current nw
        observed_t = []    # no new time windows observed for the current nw
        observed_y = []    # obj_value without trying a new time windows
        observed_sol = []  # solution without trying a new time windows

        ### Initialization (evaluate y for some t)
        init = rng.sample(candidate_times, k=n_init)     # Sample some candidate
        for t in init:
            is_Nw_used_candidate = copy.deepcopy(is_Nw_used)
            is_Nw_used_candidate[(nw, t)] = 1   # Activate the time windos for (nw,t)
            sol, obj = evaluate_candidate(
                instance_base = instance_base,
                is_Nw_used_cand = is_Nw_used_candidate, 
                cplex_cfg = cplex_cfg
            )
            observed_t.append(t)
            candidate_times.remove(t)    # remove the tested t
            observed_y.append(obj)
            observed_sol.append(sol)
        
        ### Loop GP / UCB
        for it in range(n_iterations):
            if len(candidate_times) == 0:
                break

            kernel = C(1.0, (1e-3, 1e3)) * (
                RBF(length_scale=dt, length_scale_bounds=(dt * 0.1, Tmax)) +
                RationalQuadratic(length_scale=dt, alpha=1.0)
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )

            # ---- fit  ----
            X_obs = np.array(observed_t, dtype=float).reshape(-1, 1)    # column vector
            y_obs = np.array(observed_y, dtype=float)                      
            gp.fit(X_obs, y_obs)

            # ---- predict on remaining candidates ----
            X_cand = np.array(candidate_times, dtype=float).reshape(-1, 1)
            y_mean, y_std = gp.predict(X_cand, return_std=True)

            # ---- MINIMIZATION: use LCB ----
            lcb = y_mean - epsilon * y_std
            best_idx = int(np.argmin(lcb))
            t_next = candidate_times[best_idx]

            # ---- evaluate real candidate ----
            is_Nw_used_candidate = copy.deepcopy(new_is_Nw_used)
            is_Nw_used_candidate[(nw, t_next)] = 1

            sol_next, y_next = evaluate_candidate(
                instance_base=instance_base,
                is_Nw_used_cand=is_Nw_used_candidate,
                cplex_cfg=cplex_cfg,
                nw=nw,
                t=t_next,
            )
            if y_next is None:
                # remove infeasible / failed candidate and continue
                candidate_times.pop(best_idx)
                continue

            print(f"      GP it {it+1}: nw={nw}, t={t_next}, obj={y_next}, LCB={lcb[best_idx]}")

            # ---- update observed dataset ----
            observed_t.append(t_next)
            observed_y.append(y_next)
            observed_sol.append(sol_next)

            # ---- remove used candidate ----
            candidate_times.pop(best_idx)

        # ----------------------------
        # FINAL DECISION FOR THIS nw
        # ----------------------------
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
            new_is_Nw_used[(nw, best_local_t)] = 1









# ======================================================================
#  RUN HEURISTIC SURROGATE - GRID
# ======================================================================
def run_main_heuristic(
    instance: Instance,        
    heu_params:dict,
    seed: int,
    base_output_folder,   # Folder for results
    cplex_cfg: dict | None = None,
):
    """
    Meta-heuristic loop semplificato:
    - inizializza soluzione
    - chiama GP (per trovare t migliore per Nw)
    - aggiorna soluzione globale solo se GP restituisce una nuova soluzione valida
    """

    Nw_nodes=instance.Nw
    dt=instance.dt
    Tmax=instance.t_max
    time_windows = list(range(1, Tmax + 1))    # {1,2, ..., t_max}

    max_seconds = heu_params["max_seconds"]    # Max seconds for whole heuristics
    epsilon = heu_params["epsilon"]            # How much exploration for UCB
    n_iterations_GP = heu_params["n_iterations_GP"]  # Number of it for GP
    n_init_GP = heu_params["n_init_GP"]        # Number of solution points to initialize the GP

    # ----------------------------
    # Solution initialization
    # ----------------------------
    current_sol, current_obj_val = first_solution_initialization_solver(
            instance_no_Nw = copy.deepcopy(instance),
            cplex_cfg = cplex_cfg
            )
    start_time = time.time()
    i = 0
    best_obj_val = 1e100
    best_solution = None
    running_time = 0.0
    improvement = True
    is_Nw_used = {   # When Nw is active (value 1) or inactive (value 0)
            (nw, t): 0
            for nw in Nw_nodes
            for t in time_windows
        }
    while (running_time < max_seconds) and (improvement):  # Stop se fuori tempo max e non si hanno più miglioramenti
        print(f"\n   ---ITERAZIONE EURISTICA {i} ---")

        # ------------------------------------------
        # GP to activate a time windows for each Nw
        # ------------------------------------------
        # New current solutiona + new current objective value + updated Nw list
        current_sol, current_obj_val, is_Nw_used = GP_window_selection(    
            Nw_nodes=Nw_nodes,
            dt=dt,
            Tmax=Tmax,
            time_window=time_windows,
            is_Nw_used=is_Nw_used,
            current_solution=current_sol,
            value_current_solution=current_obj_val,
            epsilon=epsilon,
            n_iterations=n_iterations_GP,
            n_init=n_init_GP,
            seed=seed
        )

        if current_obj_val > best_obj_val:
            best_obj_val = current_obj_val
            best_solution = current_sol
        else:  # no improvements
            improvement = False
        
        running_time = time.time() - start_time

    return ...