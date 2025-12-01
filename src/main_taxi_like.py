
from utils import loader






if __name__ == "__main__":
    # Example usage for GRID 5x5 with 3 modules:
    network_path = "instances/GRID/5x5/network_disc5min.json"
    requests_path = "instances/GRID/5x5/taxi_like_requests_100maxmin_disc5min.json"

    ### --- Costs ---
    C_KM = 1
    C_SECOND = 1

    ### --- Number of modules max |M| ---
    NUM_MODULES = 3
    CAP_MODULES = 10
    print("*" * 50)
    print("MODULES:")
    print("-n° of modules =", NUM_MODULES)
    print("cap. of modules =", CAP_MODULES)
    print("\n")



    ### --- Max Time Horizon t_max ---
    filename = requests_path.split("/")[-1]       # take only the file part
    parts = filename.split("_")
    Tmax_min_str = parts[3]             # "***maxmin"
    disc_str = parts[4]                 # "disc***min.json"

    Tmax_min = int(Tmax_min_str.replace("maxmin", ""))
    dt = int(disc_str.replace("disc", "").replace("min.json", ""))
    t_max = Tmax_min // dt
    print("*" * 50)
    print("TIME HORIZON and DISCRETIZATION:")
    print("-Tmax_min =", Tmax_min)
    print("-dt =", dt)
    print("-t_max =", t_max)
    print("\n")




    ###
    data = build_cplex_instance(
        network_path=network_path,
        requests_path=requests_path,
        num_modules=num_modules,
    )

    print("[INFO] Instance built:")
    print(f"  |N| = {len(data['N'])}")
    print(f"  |A| = {len(data['A'])}")
    print(f"  |K| = {len(data['K'])}")
    print(f"  |T| = {len(data['T'])}")
    print(f"  |M| = {len(data['M'])}")
    print(f"  Number of d entries      = {len(data['d'])}")
    print(f"  Number of d_tilde entries= {len(data['d_tilde'])}")

