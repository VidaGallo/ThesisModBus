from utils.loader import *
from utils.instance import *
from utils.print_data import *
from models.deterministic.model_taxi_like import *
from utils.cplex_config import *
from utils.outputs import *
from data_generation.generate_data import *




FLAG_VERBOSE = 1  # 1 for displaying everything, 0 for displaying nothing





if __name__ == "__main__":
    
    # ----------------
    ### Parameters ###
    # ----------------
    dt = 5                    # minutes per slot
    horizon = 100             # time horizon in minutes (continuous)
    t_max = horizon // dt     # number of discrete time slots
    number = 5                # grid side (number x number)
    depot = 0

    num_modules = 2
    Q = 10
    c_km = 1.0
    c_uns_taxi = 100

    num_requests = 10         # how many taxi-like requests to generate
    q_min = 1                 # min q_k
    q_max = 3                 # max q_k
    slack_min = 10.0          # minutes of flexibility






    # ---------------------
    ### Data generation ###
    # ---------------------
    network_path, requests_path = generate_all_data(
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
        depot=depot
    )

    if FLAG_VERBOSE:
        print_instance_summary(instance)





    # -----------
    ### MODEL ###
    # -----------
    model, x, y, r, w, s = create_taxi_like_model(instance)   # model construction

    configure_cplex(model)                                    # model configuration

    solution = model.solve(log_output=True)                   # model solution



    # ------------
    ### OUTPUT ###
    # ------------

    # Solution print
    print("-"*77)
    if solution:
        print("Status:", solution.solve_status)
        print("Objective:", solution.objective_value)
    else:
        print("No solution found.")
    print("-"*77)


    # Creation of the output folder
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
        save_solution_variables(solution, x, y, r, w, s, output_folder)


    
    if FLAG_VERBOSE:
        full_report(
            instance, solution, x, y, r, w, s,
            network_path=network_path,
            output_folder=output_folder,
            display_instance_summary=True,
            display_movements=True,
            display_initial_grid=True,
            display_timeline=True,
            display_request_details=True,
        )



