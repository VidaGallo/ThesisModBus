"""
Data Generation Pipeline
========================

This module provides a single high-level function, `generate_all_data()`,
which creates the full dataset needed for the MILP model.

The pipeline performs:
  1. Continuous GRID network generation.
  2. Continuous taxi-like request generation.
  3. Time discretization of both network and requests.
  4. Automatic creation of the output folder structure.

It returns the paths to the **discretized** network and request files,
ready to be loaded into the Instance class.

Usage (in main):
    network_path, requests_path = generate_all_data(...)
"""



from pathlib import Path
from .generate_network import *
from .generate_demands import *
from .time_discretization import *



def generate_all_data(
    number: int,
    horizon: int,
    dt: int,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
):
    """
    Generate continuous network + continuous requests,
    then discretize both.

    Returns:
      network_disc_path, requests_disc_path
    """

    ### BUILD FOLDER
    base = f"instances/GRID/{number}x{number}"
    Path(base).mkdir(parents=True, exist_ok=True)

    network_cont = f"{base}/network.json"
    requests_cont = f"{base}/taxi_like_requests_{horizon}maxmin.json"

    network_disc = f"{base}/network_disc{dt}min.json"
    requests_disc = f"{base}/taxi_like_requests_{horizon}maxmin_disc{dt}min.json"


    ### GENERATE CONTINUOUS NETWORK
    network = generate_grid_network(
        side=number,
        edge_length_km=1.0,
        speed_kmh=40.0,
    )
    save_network_json(network, base, filename="network.json")


    ### GENERATE CONTINUOUS REQUESTS
    G = load_network_as_graph(network_cont)

    taxi_requests = generate_taxi_requests(
        G=G,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        time_horizon_max=float(horizon),
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
