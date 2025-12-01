
from utils.loader import *



if __name__ == "__main__":
    
    ### Parameters 
    dt = 5              # minutes per slot
    t_max = 100 // dt   # e.g., time horizon 100 minutes → 20 slots
    num_modules = 3
    Q = 10
    c_km = 1.0
    c_uns_taxi = 1     # ???


    # --- File paths (already existing and discretized) ---
    base = "instances/GRID/5x5"    # Folder with the current Data
    network_path = f"{base}/network_disc{dt}min.json"
    requests_path = f"{base}/taxi_like_requests_100maxmin_disc{dt}min.json"

    print("Fino a qui vaaaaaa")


    # --- Load instance ---
    instance = load_instance_discrete(
        network_path=network_path,
        requests_path=requests_path,
        dt=dt,
        t_max=t_max,
        num_modules=num_modules,
        Q=Q,
        c_km=c_km,
        c_uns_taxi=c_uns_taxi
    )

    # --- Quick validation ---
    print("[INFO] Instance loaded:")
    print(f"  |N| = {instance.num_nodes}")
    print(f"  |A| = {len(instance.A)}")
    print(f"  |K| = {instance.num_requests}")
    print(f"  |M| = {instance.num_modules}")
    print(f"  |T| = {len(instance.T)}")