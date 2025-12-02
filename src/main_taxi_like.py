from utils.loader import *
from utils.instance import *
from utils.print_data import *
from models.deterministic.model_taxi_like import *
from docplex.mp.model import Model

FLAG_VERBOSE = 1  # 1 for displaying everything, 0 for displaying nothing


if __name__ == "__main__":
    
    ### Parameters 
    dt = 5              # minutes per slot
    t_max = 100 // dt   # e.g., time horizon 100 minutes → 20 slots
    num_modules = 3
    Q = 10
    c_km = 1.0
    c_uns_taxi = 1     # <------


    ### File paths (already existing and discretized)
    base = "instances/GRID/5x5"    # Folder with Data
    network_path = f"{base}/network_disc{dt}min.json"
    requests_path = f"{base}/taxi_like_requests_100maxmin_disc{dt}min.json"

    ### Load instance
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

    if FLAG_VERBOSE:
        ### Check
        print_instance_summary(instance)



    ### Model construction
    mdl = Model(name="taxi_like")

    add_taxi_like_constraints(model, I, x, y, r, w, s)
    add_taxi_like_objective(model, I, y, s)
