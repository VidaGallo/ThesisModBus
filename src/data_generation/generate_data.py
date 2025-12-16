"""
Data Generation Pipeline
========================

This module builds the full input dataset for the MILP experiments.

It provides high-level functions that:
1) generate a continuous network (GRID / asymmetric GRID / CITY),
2) generate continuous taxi-like requests on that network,
3) discretize both network travel times and request time windows,
4) save all files in a consistent folder structure.

Each function returns the paths of the DISCRETIZED JSON files
(network + requests), ready to be loaded into the Instance class.
"""



from pathlib import Path
from .generate_network import *
from .generate_demands import *
from .time_discretization import *

import random
import numpy as np

def set_seed(seed: int = 23):
    random.seed(seed)
    np.random.seed(seed)





def generate_all_data(
    number: int,
    horizon: int,
    dt: int,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    seed: int = 23,
    mean_edge_length_km: float = 3.0,
    mean_speed_kmh: float = 40.0
    ):
    """
    Generate continuous network + continuous requests,
    then discretize both.

    Returns:
      network_disc_path, requests_disc_path
    """
    set_seed(seed)

    ### BUILD FOLDER
    base = f"instances/GRID/{number}x{number}"
    Path(base).mkdir(parents=True, exist_ok=True)

    network_cont = f"{base}/network.json"
    requests_cont = f"{base}/requests_{horizon}maxmin.json"

    network_disc = f"{base}/network_disc{dt}min.json"
    requests_disc = f"{base}/requests_{horizon}maxmin_disc{dt}min.json"


    ### GENERATE CONTINUOUS NETWORK
    network = generate_grid_network(
        side=number,
        edge_length_km=mean_edge_length_km,
        speed_kmh=mean_speed_kmh
    )
    save_network_json(network, base, filename="network.json")


    ### GENERATE CONTINUOUS REQUESTS
    G = load_network_as_graph(network_cont)

    taxi_requests = generate_requests(
        G=G,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        time_horizon_max=float(horizon)
    )

    save_requests(requests_cont, taxi_requests)



    ### DISCRETIZE REQUESTS & NETWORK
    discretize_taxi_requests(
        input_path=requests_cont,
        output_path=requests_disc,
        time_step_min=float(dt),
    )

    discretize_network_travel_times(
        input_path=network_cont,
        output_path=network_disc,
        time_step_min=float(dt),
    )

    #print("\n[DATA GENERATION COMPLETE]")
    #print("Continuous and discrete data created in:", base)

    # RETURN PATHS FOR THE MAIN
    return network_disc, requests_disc





def generate_all_data_asym(
    number: int,
    horizon: int,
    dt: int,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    mean_edge_length_km: float = 3.33,
    mean_speed_kmh: float = 40.0,
    length_std: float = 0.66
):
    """
    Generate continuous network + continuous requests,
    then discretize both.

    Returns:
      network_disc_path, requests_disc_path
    """
    set_seed(seed)

    ### BUILD FOLDER
    base = f"instances/GRID/{number}x{number}"
    Path(base).mkdir(parents=True, exist_ok=True)

    network_cont = f"{base}/network.json"
    requests_cont = f"{base}/requests_{horizon}maxmin.json"

    network_disc = f"{base}/network_disc{dt}min.json"
    requests_disc = f"{base}/requests_{horizon}maxmin_disc{dt}min.json"


    ### GENERATE CONTINUOUS NETWORK
    network = generate_grid_network_asym(
        side=number,
        edge_length_km=mean_edge_length_km,
        speed_kmh=mean_speed_kmh,
        rel_std = length_std
    )
    save_network_json(network, base, filename="network.json")


    ### GENERATE CONTINUOUS REQUESTS
    G = load_network_as_graph(network_cont)

    taxi_requests = generate_requests(
        G=G,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        time_horizon_max=float(horizon),
        depot=depot
    )

    save_requests(requests_cont, taxi_requests)



    ### DISCRETIZE REQUESTS & NETWORK
    discretize_taxi_requests(
        input_path=requests_cont,
        output_path=requests_disc,
        time_step_min=float(dt),
    )

    discretize_network_travel_times(
        input_path=network_cont,
        output_path=network_disc,
        time_step_min=float(dt),
    )

    #print("\n[DATA GENERATION COMPLETE]")
    #print("Continuous and discrete data created in:", base)



    # RETURN PATHS FOR THE MAIN
    return network_disc, requests_disc








def generate_all_data_city(
    city: str,      # city name
    subdir: str,     # subdirectory name
    central_suburbs: list[str],
    horizon: int,
    dt: int,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int = 23,
    mean_speed_kmh: float = 40.0,
):
    """
    Generate continuous network + continuous requests,
    then discretize both.

    Returns:
      network_disc_path, requests_disc_path
    """

    ### BUILD FOLDER
    base = f"instances/CITY/{subdir}"
    Path(base).mkdir(parents=True, exist_ok=True)

    network_cont = f"{base}/network.json"
    requests_cont = f"{base}/requests_{horizon}maxmin.json"

    network_disc = f"{base}/network_disc{dt}min.json"
    requests_disc = f"{base}/requests_{horizon}maxmin_disc{dt}min.json"


    ### GENERATE CONTINUOUS NETWORK
    network = generate_grid_network_city(
        place = city,
        central_suburbs = central_suburbs,
        speed_kmh = mean_speed_kmh
    )
    save_network_json(network, base, filename="network.json")


    ### GENERATE CONTINUOUS REQUESTS
    G = load_network_as_graph(network_cont)

    taxi_requests = generate_requests(
        G=G,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        time_horizon_max=float(horizon),
        depot=depot
    )

    save_requests(requests_cont, taxi_requests)



    ### DISCRETIZE REQUESTS & NETWORK
    discretize_taxi_requests(
        input_path=requests_cont,
        output_path=requests_disc,
        time_step_min=float(dt),
    )

    discretize_network_travel_times(
        input_path=network_cont,
        output_path=network_disc,
        time_step_min=float(dt),
    )

    #print("\n[DATA GENERATION COMPLETE]")
    #print("Continuous and discrete data created in:", base)



    # RETURN PATHS FOR THE MAIN
    return network_disc, requests_disc








