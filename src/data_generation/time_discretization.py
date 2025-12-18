"""
Time Discretization Utilities
=============================

Tools for converting all continuous-time inputs (requests and network travel
times) into discrete time indices.

1. Request Discretization:
   - Convert continuous service windows (ΔT_k), boarding windows (ΔT_k^{in}),
     and alighting windows (ΔT_k^{out}) into integer time indices
     t = 1, 2, ..., t_max, using a fixed time step Δt (e.g., 5 minutes).
   - Each interval [start_min, end_min] is mapped to:
         t_start = ceil(start_min / Δt) + 1
         t_end   = floor(end_min  / Δt) + 1
     producing index sets:
         T_k_idx, T_in_idx, T_out_idx.

2. Network Discretization:
   - Convert each continuous edge travel time τ(i,j) (in minutes)
     into a discrete number of periods:
         time_steps = max(1, ceil(time_min / Δt)).
   - This yields integer arc travel times consistent with the request windows.

Outputs:
    - Discretized request files (taxi-like or bus-like), with discrete
      time-index sets added to each request.
    - Discretized network files, where each edge contains both:
          - time_min      (continuous travel time, minutes)
          - time_steps    (discretized travel time, periods)

All discrete indices start at t = 1,
"""


import json
import math
from pathlib import Path
import networkx as nx



def interval_to_indices(start_min: float, end_min: float, time_step_min: float):
    """
    Map a continuous interval [start_min, end_min] (in minutes)
    to discrete time indices t = 1,2,... using step = time_step_min.
    """
    if end_min < start_min:
        return []

    # +1 so indices start from 1 
    t_start = math.ceil(start_min / time_step_min) + 1
    t_end   = math.floor(end_min  / time_step_min) + 1

    if t_start > t_end:
        return []

    return list(range(int(t_start), int(t_end) + 1))





def discretize_requests(
    input_path: str,
    output_path: str,
    time_step_min: float,
    network_disc_path: str | None = None,
    depot: int = 0,
):
    with open(input_path, "r", encoding="utf-8") as f:
        requests = json.load(f)

    # Discretization of the shortest path
    Gd = None
    if network_disc_path is not None:
        with open(network_disc_path, "r", encoding="utf-8") as f:
            net = json.load(f)
        Gd = nx.DiGraph()
        for n in net["nodes"]:
            Gd.add_node(int(n["id"]))
        for e in net["edges"]:
            Gd.add_edge(int(e["u"]), int(e["v"]), time_steps=int(e["time_steps"]))

        def sp_steps(u: int, v: int) -> int:
            path = nx.shortest_path(Gd, u, v, weight="time_steps")
            return int(sum(Gd[a][b]["time_steps"] for a, b in zip(path[:-1], path[1:])))

    for req in requests:
        # --- finestre discrete ---
        T_k_min  = req["T_k_min"]
        T_in_min = req["T_in_min"]
        T_out_min= req["T_out_min"]

        req["T_k_idx"]  = interval_to_indices(T_k_min[0],  T_k_min[1],  time_step_min)
        req["T_in_idx"] = interval_to_indices(T_in_min[0], T_in_min[1], time_step_min)
        req["T_out_idx"]= interval_to_indices(T_out_min[0],T_out_min[1],time_step_min)

        # --- shortest path discreto ---
        if Gd is not None:
            o = int(req["origin"])
            d = int(req["destination"])

            tau_sp_steps = sp_steps(o, d)
            tau_dep_steps = sp_steps(depot, o)

            req["tau_sp_steps"] = tau_sp_steps
            req["tau_sp_min_disc"] = float(tau_sp_steps) * float(time_step_min)

            req["tau_depot_origin_steps"] = tau_dep_steps
            req["tau_depot_origin_min_disc"] = float(tau_dep_steps) * float(time_step_min)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)

    #print(f"[INFO] Discretized requests saved to: {out_path}")







def discretize_requests_dict(
    requests: list[dict],
    time_step_min: float,
    network_disc_dict: dict | None = None, 
    depot: int = 0,
) -> list[dict]:
    """
    Discretizza richieste CONTINUE (lista di dict) in memoria.
    - aggiunge T_k_idx, T_in_idx, T_out_idx
    - se network_disc è fornito, aggiunge anche shortest path discreti:
        tau_sp_steps, tau_sp_min_disc,
        tau_depot_origin_steps, tau_depot_origin_min_disc
    Ritorna: lista di dict (requests discretizzate).
    """

    # Build discrete graph from dict (if provided)
    Gd = None
    if network_disc_dict is not None:
        Gd = nx.DiGraph()
        for n in network_disc_dict.get("nodes", []):
            Gd.add_node(int(n["id"]))
        for e in network_disc_dict.get("edges", []):
            Gd.add_edge(
                int(e["u"]), int(e["v"]),
                time_steps=int(e["time_steps"])
            )

        def sp_steps(u: int, v: int) -> int:
            path = nx.shortest_path(Gd, u, v, weight="time_steps")
            return int(sum(Gd[a][b]["time_steps"] for a, b in zip(path[:-1], path[1:])))

    req_out: list[dict] = []

    for req in requests:
        r = dict(req)  # copia

        # --- finestre discrete ---
        T_k_min   = r["T_k_min"]
        T_in_min  = r["T_in_min"]
        T_out_min = r["T_out_min"]

        r["T_k_idx"]   = interval_to_indices(T_k_min[0],   T_k_min[1],   time_step_min)
        r["T_in_idx"]  = interval_to_indices(T_in_min[0],  T_in_min[1],  time_step_min)
        r["T_out_idx"] = interval_to_indices(T_out_min[0], T_out_min[1], time_step_min)

        # --- shortest path discreto ---
        if Gd is not None:
            o = int(r["origin"])
            d = int(r["destination"])

            tau_sp = sp_steps(o, d)
            tau_dep = sp_steps(int(depot), o)

            r["tau_sp_steps"] = int(tau_sp)
            r["tau_sp_min_disc"] = float(tau_sp) * float(time_step_min)

            r["tau_depot_origin_steps"] = int(tau_dep)
            r["tau_depot_origin_min_disc"] = float(tau_dep) * float(time_step_min)

        req_out.append(r)

    return req_out





def discretize_network_travel_times(
    input_path: str,
    output_path: str,
    time_step_min: float,
):
    """
    Load a continuous-time network (GRID or CITY) from JSON and discretize
    edge travel times into integer time steps.

    For each edge:
        - time_min: continuous travel time in minutes (from generator)
        - time_steps: discrete travel time in number of periods (>= 1)

    We use:
        time_steps = ceil(time_min / time_step_min)
    """
    with open(input_path, "r", encoding="utf-8") as f:
        net = json.load(f)

    edges = net.get("edges", [])

    for edge in edges:
        time_min = edge.get("time_min", 0.0)
        # discrete travel time (at least 1 step)
        time_steps = max(1, math.ceil(time_min / time_step_min))
        edge["time_steps"] = int(time_steps)

    # opzionale: salva anche il passo temporale usato
    net["time_step_min"] = float(time_step_min)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(net, f, indent=2)

    #print(f"[INFO] Discretized network saved to: {out_path}")
