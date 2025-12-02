from utils.loader import *
from utils.instance import *
from utils.print_save_data import *
from models.deterministic.model_taxi_like import *
from utils.cplex_config import *
from utils.outputs import *





FLAG_VERBOSE = 1  # 1 for displaying everything, 0 for displaying nothing





if __name__ == "__main__":
    
    ### Parameters 
    dt = 5              # minutes per slot
    horizon = 100
    t_max = horizon // dt   # e.g., time horizon 100 minutes → 20 slots
    number = 5      # side of the grid
    num_modules = 3
    Q = 10
    c_km = 1.0
    c_uns_taxi = 100     # <------

    depot = 0


    ### File paths (already existing and discretized)
    base = f"instances/GRID/{number}x{number}"    # Folder with Data
    network_path = f"{base}/network_disc{dt}min.json"
    requests_path = f"{base}/taxi_like_requests_{horizon}maxmin_disc{dt}min.json"


    ### Load instance
    instance = load_instance_discrete(
        network_path=network_path,
        requests_path=requests_path,
        dt=dt,
        t_max=t_max,
        num_modules=num_modules,
        Q=Q,
        c_km=c_km,
        c_uns_taxi=c_uns_taxi,
        depot=depot
    )

    if FLAG_VERBOSE:
        ### Check
        print_instance_summary(instance)



    ### Model construction
    model, x, y, r, w, s = create_taxi_like_model(instance)


    ### Model configuration
    configure_cplex(model)


    ### Model solution
    solution = model.solve(log_output=True)

    print("-"*77)
    if solution:
        print("Status:", solution.solve_status)
        print("Objective:", solution.objective_value)
    else:
        print("No solution found.")
    print("-"*77)


    ### Creation of the output folder
    output_folder = build_output_folder(
        base_dir="results",
        network_path=network_path,
        t_max=instance.t_max,
        dt=instance.dt,
    )


    ### Save logs, stats, summary
    save_model_stats(model, output_folder)

    if solution is None:
        save_cplex_log(model, output_folder)
    else:
        save_cplex_log(model, output_folder)
        save_solution_summary(solution, output_folder)

        if FLAG_VERBOSE:
            print_solution_movements_and_positions(instance, solution, x, y)



    initial_snapshot_path = output_folder / "initial_grid_t0.png"

    t0 = min(instance.T)
    plot_initial_grid_with_modules(
        I=instance,
        solution=solution,
        x=x,
        network_path=network_path,
        t0=t0,
        output_path=initial_snapshot_path,
    )


    debug_full_system_timeline(
        K = instance.K,
        M = instance.M,
        N = instance.N,
        T = instance.T,
        DeltaT = instance.DeltaT,
        t0 = t0,
        solution = solution,
        x = x,
        y = y,
        r = r,
        w = w,
        s = s,
    )

    debug_requests_details(
    K       = instance.K,
    M       = instance.M,
    N       = instance.N,
    T       = instance.T,      # oppure instance.T se usi quello
    DeltaT  = instance.DeltaT,
    t0      = t0,
    solution= solution,
    r       = r,
    d_in    = instance.d_in,
    d_out   = instance.d_out,
    q       = getattr(instance, "q", None),   # se q sta dentro instance
    s       = s,
    )


