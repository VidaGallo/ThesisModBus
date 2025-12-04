from utils.loader import *
from utils.instance import *
from utils.print_data import *
from models.deterministic.model_taxi_like_asym_new import *
from utils.cplex_config import *
from utils.outputs import *
from data_generation.generate_data import *

import time
import random
import numpy as np
import pandas as pd
import os



def run_single_experiment(
    number: int,
    horizon: int,
    dt: int,
    num_modules: int,
    Q: int,
    c_km: float,
    c_uns_taxi: float,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int,
    run: int 
) -> dict:
    """
    Run ONE experiment with given parameters and return a dictionary with summary results.
    """

    # -------------------
    ### Fixed seed     ###
    # -------------------
    random.seed(seed)
    np.random.seed(seed)

    # ----------------
    ### Parameters ###
    # ----------------
    t_max = horizon // dt   # number of discrete time slots

    # ---------------------
    ### Data generation ###
    # ---------------------
    t_start_total = time.perf_counter()

    network_path, requests_path = generate_all_data_asym(
        number=number,
        horizon=horizon,
        dt=dt,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
    )


    # -------------------
    ### Load Instance ###
    # -------------------
    instance = load_instance_discrete(
        network_path=network_path,
        requests_path=requests_path,
        dt=dt,
        t_max=t_max,
        num_modules=num_modules,
        Q=Q,
        c_km=c_km,
        c_uns_taxi=c_uns_taxi,
        depot=depot,
    )

    # -----------
    ### MODEL ###
    # -----------
    model, x, y, r, s, z = create_taxi_like_model_asym_new(instance)   # model construction
    

    configure_cplex(model)                                    # model configuration

    t_start_solve = time.perf_counter()
    solution = model.solve(log_output=False)                  # model solution
    solve_time = time.perf_counter() - t_start_solve
    total_time = time.perf_counter() - t_start_total

    # ------------
    ### OUTPUT ###
    # ------------
    print("-"*77)
    if solution:
        print(f"[EXP seed={seed}] Status: {solution.solve_status}")
        print("Objective:", solution.objective_value)
    else:
        print(f"[EXP seed={seed}] No solution found.")
    print("Solve time (sec):", solve_time)
    print("Total time (sec):", total_time)
    print("-"*77)

    # Creation of the output folder
    output_folder = build_output_folder(
        base_dir="results",
        network_path=network_path,
        t_max=instance.t_max,
        dt=instance.dt,
    )
    output_folder = output_folder / f"run{run}"
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    print(output_folder)

    ### Save logs, stats, summary
    save_model_stats(model, output_folder)

    if solution is None:
        save_cplex_log(model, output_folder)
    else:
        save_cplex_log(model, output_folder)
        save_solution_summary(solution, output_folder)
        save_solution_variables_new(solution, x, y, r, s, z, output_folder)



    # -----------------------
    ### SERVED SUMMARY   ###
    # -----------------------
    if solution is not None:
        served = sum(int(solution.get_value(s[k]) > 0.5) for k in instance.K)
        total  = len(instance.K)
        served_ratio = served / total if total > 0 else 0.0
    else:
        served = 0
        total  = len(instance.K)
        served_ratio = 0.0

    print(f"  -> served: {served}/{total}  ({served_ratio*100:.1f}%)")





    # -------------------
    ### Build summary ###
    # -------------------
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

    result = {
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






if __name__ == "__main__":
    
    # ----------------
    ### Base params ###
    # ----------------
    dt = 5                     # minutes per slot
    depot = 0

    Q = 10
    c_km = 1.0
    c_uns_taxi = 100

    q_min = 1                  # min q_k
    q_max = 3                  # max q_k
    slack_min = 20.0           # minutes of flexibility

    # --------------------------
    ### Grid of experiments  ###
    # --------------------------
    grid_numbers      = [3, 5, 8]        # grid side (number x number)
    horizons          = [600]            # time horizon in minutes (continuous)
    num_modules_list  = [3, 5]           # number of modules
    num_requests_list = [20]             # how many taxi-like requests
    seeds             = [23]             # for reproducibility

    all_results = []
    exp_id = 0

    for number in grid_numbers:
        for horizon in horizons:
            for num_modules in num_modules_list:
                for num_requests in num_requests_list:
                    for seed in seeds:

                        exp_id += 1
                        print("\n" + "="*80)
                        print(f"EXPERIMENT {exp_id}")
                        print(f"  number        = {number}")
                        print(f"  horizon       = {horizon}")
                        print(f"  num_modules   = {num_modules}")
                        print(f"  num_requests  = {num_requests}")
                        print(f"  seed          = {seed}")
                        print("="*80)

                        res = run_single_experiment(
                            number=number,
                            horizon=horizon,
                            dt=dt,
                            num_modules=num_modules,
                            Q=Q,
                            c_km=c_km,
                            c_uns_taxi=c_uns_taxi,
                            num_requests=num_requests,
                            q_min=q_min,
                            q_max=q_max,
                            slack_min=slack_min,
                            depot=depot,
                            seed=seed,
                            run=exp_id
                        )

                        # add experiment id in the result
                        res["exp_id"] = exp_id
                        all_results.append(res)

    # ----------------------
    ### Pandas DataFrame ###
    # ----------------------
    df_results = pd.DataFrame(all_results)

    # Where to save the summary (inside results/)
    summary_path = f"results/summary_experiments_{number}x{number}_h{horizon}_m{num_modules}_r{num_requests}_asym_new.csv"
    df_results.to_csv(summary_path, index=False)

    print("\n\n\n" + "#"*80)
    print(f"\nSummary saved to: {summary_path}")
