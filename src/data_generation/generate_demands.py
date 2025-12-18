"""
Demand Generator
================

This module generates travel requests in continuous time
for the optimization experiments.

It works with any network produced by `generate_network.py`
(e.g., GRID or CITY networks).

Each generated request includes:
- origin and destination nodes,
- passenger demand q_k,
- desired departure and arrival times (continuous, in minutes),
- slack time (flexibility),
- shortest-path travel time,
- continuous service, boarding, and alighting time windows.

The output requests are saved in JSON format and are NOT discretized.
"""

import json
import networkx as nx
from pathlib import Path
import math
import random




def compute_shortest_path_time(G: nx.Graph, origin: int, dest: int) -> float:
    """
    Compute shortest-path travel time in minutes using edge attribute 'time_min'.
    """
    path = nx.shortest_path(G, origin, dest, weight="time_min")
    time_min = sum(G[u][v]["time_min"] for u, v in zip(path[:-1], path[1:]))
    return time_min



###################################
#  Generate requests (continuous):
###################################
def generate_requests(
    output_path,
    network_path: str,
    num_requests: int,
    q_min: int = 1,
    q_max: int = 3,
    slack_min: float = 20.0,
    time_horizon_max: float = 600.0,
    depot: int = 0,
    alpha: float = 0.65,
    rng=None,
):
    """
    Generate requests with continuous time fields.
    """

    # build graph internally
    G = load_network_as_graph(network_path)

    # reproducible RNG
    rng = rng or random 

    requests = []
    nodes = list(G.nodes())

    for k in range(num_requests):
        valid = False    # valid if the Time window stays inside t_max
        attempts = 0

        while not valid:
            attempts += 1


            ##### Origin and Destionation: #####
            origin = rng.choice(nodes)
            dest = rng.choice(nodes)
            while dest == origin:
                dest = rng.choice(nodes)



            ##### Demand generation: #####
            # (1) random
            # (2) exponential demand, with q_k = 1 highest probability

            # --- (1) random ---
            #demand = rng.randint(q_min, q_max)
            
            # --- (2) exponential ---
            q_values = list(range(q_min, q_max + 1))

            # exponential decay: weight(q) = exp(-alpha * (q-1))
            q_weights = [math.exp(-alpha * (q - q_min)) for q in q_values]

            # normalize
            total = sum(q_weights)
            q_weights = [w / total for w in q_weights]

            demand = rng.choices(q_values, weights=q_weights, k=1)[0]




            ##### Shortest-path travel times (minutes): #####
            try:
                # tempo tra origin e destination (tau_sp)
                tau_sp = compute_shortest_path_time(G, origin, dest)
                
                # tempo dal depot all'origine (tau_depot_origin)
                tau_depot_origin = compute_shortest_path_time(G, depot, origin)
            except nx.NetworkXNoPath:
                # se non esiste percorso, riprova con un'altra coppia
                continue



            ##### Departure time #####
            #  desired_dep + tau_sp + slack_min <= time_horizon_max
            latest_departure = time_horizon_max - (tau_sp + slack_min)

            # Si vuole desired_dep >= tau_depot_origin
            if latest_departure <= tau_depot_origin:
                # non esiste un intervallo valido per desired_dep
                continue

            # Start sampling from first feasable moment
            desired_dep = rng.uniform(tau_depot_origin, latest_departure)

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

    ### Save requests continuous
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)
        
    return 



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
