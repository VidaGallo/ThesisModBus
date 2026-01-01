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
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict




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




########################################
# GENERATE REQUESTS (with probabilities)
########################################
### Plot requests
def _get_node_positions(G):
    """
    Try to read positions from node attributes.
    Falls back to spring_layout if not available.
    """
    # OSMnx-style: nodes have 'x' (lon) and 'y' (lat)
    xs = nx.get_node_attributes(G, "x")
    ys = nx.get_node_attributes(G, "y")
    if xs and ys and len(xs) == G.number_of_nodes() and len(ys) == G.number_of_nodes():
        return {n: (xs[n], ys[n]) for n in G.nodes()}

    # Common style: nodes have 'pos' = (x,y)
    pos_attr = nx.get_node_attributes(G, "pos")
    if pos_attr and len(pos_attr) == G.number_of_nodes():
        return pos_attr

    # Fallback: compute layout
    return nx.spring_layout(G, seed=1)

def _jittered_point(base_xy, idx, jitter=0.0008):
    """
    Small deterministic offsets so points at same node don't overlap.
    idx = 0,1,2,... for events at same node
    """
    x, y = base_xy
    ring = idx // 8 + 1
    k = idx % 8
    # 8 directions around the node
    dirs = [
        (1, 0), (0, 1), (-1, 0), (0, -1),
        (1, 1), (-1, 1), (-1, -1), (1, -1)
    ]
    dx, dy = dirs[k]
    return (x + dx * jitter * ring, y + dy * jitter * ring)

def plot_graph_and_requests(
    G,
    requests,
    pos=None,
    node_size=8,
    edge_width=0.6,
    pickup_size=30,
    dropoff_size=30,
    jitter=0.0008,
    title="Graph + Requests",
):
    """
    Draw graph + pickup/dropoff points.
    - Pickups: green
    - Dropoffs: blue
    Adds jitter to avoid overlap when multiple events share the same node.
    """
    if pos is None:
        pos = _get_node_positions(G)

    # Draw base graph
    plt.figure()
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.6)

    # Build jittered pickup/dropoff coordinates
    pick_count = defaultdict(int)
    drop_count = defaultdict(int)

    px, py = [], []
    dx, dy = [], []

    for r in requests:
        o = r["origin"]
        d = r["destination"]

        # pickup jitter
        i = pick_count[o]
        pick_count[o] += 1
        xo, yo = _jittered_point(pos[o], i, jitter=jitter)
        px.append(xo); py.append(yo)

        # dropoff jitter
        j = drop_count[d]
        drop_count[d] += 1
        xd, yd = _jittered_point(pos[d], j, jitter=jitter)
        dx.append(xd); dy.append(yd)

    # Scatter events
    plt.scatter(px, py, s=pickup_size, marker="o", alpha=0.9, label="Pickup")
    plt.scatter(dx, dy, s=dropoff_size, marker="x", alpha=0.9, label="Dropoff")

    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()




### sampling + labeling
def _check_prob_sum(d, name, tol=1e-9):
    s = float(sum(d.values()))
    if abs(s - 1.0) > tol:
        raise ValueError(f"{name} must sum to 1.0, got {s}")
    
def _sample_from_probs(rng, prob_dict):
    keys = list(prob_dict.keys())
    weights = [prob_dict[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]

def _od_sets(od_group, C, P):
    if od_group == "PC":
        return P, C
    if od_group == "CP":
        return C, P
    if od_group == "CC":
        return C, C
    if od_group == "PP":
        return P, P
    raise ValueError(f"Unknown od_group={od_group}")

def _classify_bucket(tau_sp, thresholds):
    # thresholds: {"S": q33, "M": q66}
    if tau_sp <= thresholds["S"]:
        return "S"
    if tau_sp <= thresholds["M"]:
        return "M"
    return "L"

def split_center_periphery_by_centrality(
    G,
    center_top_frac: float = 0.30,
    method: str = "closeness",
    weight_attr: str | None = None,
):
    if not (0.0 < center_top_frac < 1.0):
        raise ValueError("center_top_frac must be in (0,1)")

    if method == "closeness":
        centrality = nx.closeness_centrality(G, distance=weight_attr)
    elif method == "betweenness":
        centrality = nx.betweenness_centrality(G, weight=weight_attr, normalized=True)
    else:
        raise ValueError("method must be 'closeness' or 'betweenness'")

    nodes_sorted = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    n_center = max(1, int(round(center_top_frac * len(nodes_sorted))))

    C = [n for n, _ in nodes_sorted[:n_center]]
    P = [n for n, _ in nodes_sorted[n_center:]]
    return C, P, centrality

def compute_bucket_percentiles_by_od(
    G,
    C, P,
    n_samples_per_group: int = 2000,
    rng=None,
):
    """
    Compute q33/q66 per OD-group using sampled OD pairs.
    Returns dict: {"PC":{"S":q33,"M":q66}, ...}
    """
    rng = rng or random
    out = {}

    for odg in ["PC", "CP", "CC", "PP"]:
        Oset, Dset = _od_sets(odg, C, P)
        Oset = list(Oset); Dset = list(Dset)

        taus = []
        tries = 0
        max_tries = n_samples_per_group * 30

        while len(taus) < n_samples_per_group and tries < max_tries:
            tries += 1
            o = rng.choice(Oset)
            d = rng.choice(Dset)
            if d == o:
                continue
            try:
                tau = compute_shortest_path_time(G, o, d)
            except nx.NetworkXNoPath:
                continue
            taus.append(tau)

        if len(taus) < max(200, n_samples_per_group // 10):
            raise RuntimeError(f"Too few OD samples to compute percentiles for {odg}.")

        out[odg] = {
            "S": float(np.percentile(taus, 33)),
            "M": float(np.percentile(taus, 66)),
        }

    return out



DISPLAY = True
### Request generation with probs
def generate_requests_with_prob(
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

    # Parameters for centrality
    center_top_frac: float = 0.30,
    centrality_method: str = "closeness",
    centrality_weight_attr: str | None = None,

    # pi probabilities
    pi_od=None,                   # {"PC":..., "CP":..., "CC":..., "PP":...}
    pi_len_given_od=None,         # {"PC":{"S":..,"M":..,"L":..}, ...}

    # Stampa warning
    warn: bool = True,
):
    """
    Generate requests with:
      - OD-group control (PC/CP/CC/PP)
      - bucket control (S/M/L) based on percentiles computed from the graph (per OD-group)
    """
    # N° campionamenti per calcolare i percentili (ok approssimati)
    percentile_samples_per_group = num_requests*2     # Per non calcolare tutte le combinazioni 


    # Controllo somma prob. = 1
    _check_prob_sum(pi_od, "pi_od")


    # Build graph + RNG
    G = load_network_as_graph(network_path)
    rng = rng or random

    # Compute center/periphery from centrality
    C, P, _ = split_center_periphery_by_centrality(
        G,
        center_top_frac=center_top_frac,
        method=centrality_method,
        weight_attr=centrality_weight_attr,
    )
    if len(C) == 0 or len(P) == 0:
        raise ValueError("Centrality split produced empty C or P. Change center_top_frac!")

    # Compute percentiles (q33/q66) per OD-group
    bucket_thresholds = compute_bucket_percentiles_by_od(
        G, C, P,
        n_samples_per_group=percentile_samples_per_group,
        rng=rng,
    )

    # Demand distribution (exponential discrete)
    q_values = list(range(q_min, q_max + 1))
    q_weights = [math.exp(-alpha * (q - q_min)) for q in q_values]
    total = sum(q_weights)
    q_weights = [w / total for w in q_weights]

    requests = []
    for k in range(num_requests):
        made = False
        tries_req = 0

        # sampling limits
        max_tries_request = 100
        max_tries_bucket = 10
        while not made:
            tries_req += 1
            if tries_req > max_tries_request:
                raise RuntimeError(
                    f"Failed to generate request {k} after {max_tries_request} tries. "
                )

            # Sample OD group
            od_group = _sample_from_probs(rng, pi_od)
            origin_set, dest_set = _od_sets(od_group, C, P)
            origin_set = list(origin_set)
            dest_set = list(dest_set)

            # thresholds for this OD group (percentile-based)
            thresholds = bucket_thresholds[od_group]

            # Try buckets with resampling (NO fallback)
            chosen = None
            bucket_resample_limit = 3   ### Limite resamplings
            for rep in range(bucket_resample_limit):
                target_bucket = _sample_from_probs(rng, pi_len_given_od[od_group])

                # Try to realize the sampled bucket by sampling (o,d)
                for _ in range(max_tries_bucket):
                    origin = rng.choice(origin_set)
                    dest = rng.choice(dest_set)
                    if dest == origin:
                        continue

                    try:
                        tau_sp = compute_shortest_path_time(G, origin, dest)
                        tau_depot_origin = compute_shortest_path_time(G, depot, origin)
                    except nx.NetworkXNoPath:
                        continue

                    dist_bucket = _classify_bucket(tau_sp, thresholds)
                    if dist_bucket != target_bucket:
                        continue

                    chosen = (origin, dest, tau_sp, tau_depot_origin, dist_bucket, target_bucket)
                    break

                if chosen is not None:
                    break

                if warn:
                    print(f"[WARN] k={k}: OD={od_group}, target bucket={target_bucket} not found "
                          f"after {max_tries_bucket} tries. Resampling bucket...")

            if chosen is None:
                # Could not realize any sampled bucket for this OD group -> resample OD group
                if warn:
                    print(f"[WARN] k={k}: OD={od_group} failed after {bucket_resample_limit} bucket resamples. "
                          f"Resampling OD group...")
                continue

            origin, dest, tau_sp, tau_depot_origin, dist_bucket, target_bucket = chosen

            # Sample demand
            demand = rng.choices(q_values, weights=q_weights, k=1)[0]

            # Departure time feasibility
            latest_departure = time_horizon_max - (tau_sp + slack_min)
            if latest_departure <= tau_depot_origin:
                # resample OD/bucket again
                continue

            desired_dep = rng.uniform(tau_depot_origin, latest_departure)
            desired_arr = desired_dep + tau_sp

            # Time windows
            delta = slack_min
            T_k_min = [desired_dep, desired_arr + delta]
            T_in_min = [desired_dep, desired_dep + delta / 2.0]
            T_out_min = [desired_dep + tau_sp, desired_dep + tau_sp + delta]

            class_tag = f"{od_group}_{dist_bucket}"

            requests.append({
                "id": k,
                "origin": origin,
                "destination": dest,
                "q_k": demand,

                "desired_departure_min": desired_dep,
                "desired_arrival_min": desired_arr,
                "slack_min": delta,
                "tau_sp_min": tau_sp,

                "T_k_min": T_k_min,
                "T_in_min": T_in_min,
                "T_out_min": T_out_min,

                "od_group": od_group,
                "dist_bucket": dist_bucket,       # actual bucket
                "class_tag": class_tag,
                "target_bucket": target_bucket,   # requested bucket
            })

            made = True

    # Save to JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)

    # Plot
    if DISPLAY:
        plot_graph_and_requests(G, requests, title=f"Requests: {len(requests)}")

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
