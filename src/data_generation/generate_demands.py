"""
Demand Generator
================

Generates travel requests in continuous time for use in the optimization
experiments. The generator works with any network produced by
`generate_network.py` (GRID or CITY).

The output requests are *not discretized*.

Each request includes:
    - origin and destination node
    - demand q_k (passengers)
    - desired departure / arrival time (continuous, minutes)
    - slack window ȳΔ_k (flexibility, e.g. 30 min)
    - shortest-path travel time τ_sp (minutes)
    - continuous service window ΔT_k_min = [t_start, t_end]
    - continuous boarding window ΔT_k_in_min
    - continuous alighting window ΔT_k_out_min

Outputs (continuous-time):
    - taxi_like_requests.json
    - bus_like_requests.json 
"""

import json
import random
import networkx as nx
from pathlib import Path
import math



############################
# SHORTEST PATH (with time):
############################
def compute_shortest_path_time(G: nx.Graph, origin: int, dest: int) -> float:
    """
    Compute shortest-path travel time in minutes using edge attribute 'time_min'.
    """
    path = nx.shortest_path(G, origin, dest, weight="time_min")
    time_min = sum(G[u][v]["time_min"] for u, v in zip(path[:-1], path[1:]))
    return time_min





############################################
#  Generate taxi-like requests (continuous):
############################################
def generate_taxi_requests(
    G: nx.Graph,
    num_requests: int,
    q_min: int = 1,
    q_max: int = 3,
    slack_min: float = 30.0,
    time_horizon_max: float = 600.0,   # 10h
) -> list:
    """
    Generate taxi-like requests with continuous time fields.
    """
    requests = []
    nodes = list(G.nodes())

    for k in range(num_requests):
        valid = False    # valid if the Time window stays inside t_max
        attempts = 0

        while not valid:
            attempts += 1


            ##### Origin and Destionation: #####
            origin = random.choice(nodes)
            dest = random.choice(nodes)
            while dest == origin:
                dest = random.choice(nodes)



            ##### Demand generation: #####
            # (1) random
            # (2) exponential demand, with q_k = 1 highest probability

            # --- (1) random ---
            #demand = random.randint(q_min, q_max)
            
            # --- (2) exponential ---
            q_values = list(range(q_min, q_max + 1))

            # exponential decay: weight(q) = exp(-alpha * (q-1))
            alpha = 1.0
            q_weights = [math.exp(-alpha * (q - q_min)) for q in q_values]

            # normalize
            total = sum(q_weights)
            q_weights = [w / total for w in q_weights]

            demand = random.choices(q_values, weights=q_weights, k=1)[0]




            ##### Shortest-path travel time (minutes): #####
            try:
                tau_sp = compute_shortest_path_time(G, origin, dest)
            except nx.NetworkXNoPath:
                # if doesn't exist try again with the next while
                continue



            ##### Departure time #####
            #  desired_dep + tau_sp + slack_min <= time_horizon_max
            latest_departure = time_horizon_max - (tau_sp + slack_min)

            if latest_departure <= 0:
                # if the time windows are > t_max, continue with the next while
                continue

            desired_dep = random.uniform(0.0, latest_departure)
            desired_arr = desired_dep + tau_sp


            ##### Time windows #####
            delta = slack_min
            T_k_min = [desired_dep, desired_arr + delta]
            T_in_min = [desired_dep, desired_dep + delta / 2.0]
            T_out_min = [
                desired_dep + tau_sp,
                desired_dep + tau_sp + delta,   # for sure ≤ time_horizon_max
                ]

            
            ##### Request #####
            requests.append({
                # info
                "id": k,
                "origin": origin,
                "destination": dest,
                "q_k": demand,

                # continuous time fields
                "desired_departure_min": desired_dep,
                "desired_arrival_min": desired_arr,
                "slack_min": delta,
                "tau_sp_min": tau_sp,

                # continuous windows
                "T_k_min": T_k_min,
                "T_in_min": T_in_min,
                "T_out_min": T_out_min,
                })

            valid = True
    return requests






#######
# MAIN:
#######
def load_network_as_graph(network_path: str) -> nx.DiGraph:
    """
    Load a network (GRID or CITY) from JSON and build a NetworkX DiGraph
    with edge attribute 'time_min' (and 'length_km').
    """
    with open(network_path, "r", encoding="utf-8") as f:
        net = json.load(f)

    G = nx.DiGraph()

    # add nodes
    for node in net["nodes"]:
        G.add_node(node["id"])

    # add edges
    for edge in net["edges"]:
        G.add_edge(
            edge["u"],
            edge["v"],
            time_min=edge["time_min"],
            length_km=edge["length_km"],
        )

    return G

def save_requests(path: str, requests: list) -> None:
    """
    Save list of requests (dicts) to JSON.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)

    #print(f"[INFO] Saved {len(requests)} requests to: {out_path}")





### TEST MAIN ###
if __name__ == "__main__":

    network_path = "instances/GRID/3x3/network.json"   # Path to an already generated network

    G = load_network_as_graph(network_path)  


    # Parameters:
    num_requests = 50
    q_min = 1
    q_max = 3
    slack_min = 10         # in minutes

    time_horizon_max = 60  # in minutes
    number = 3 # side of the grid

    # Generation of requests
    taxi_requests = generate_taxi_requests(
        G=G,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        time_horizon_max=time_horizon_max,
    )

    # Save in the same directory
    output_path = f"instances/GRID/{number}x{number}/taxi_like_requests_{int(time_horizon_max)}maxmin.json"
    save_requests(output_path, taxi_requests)

