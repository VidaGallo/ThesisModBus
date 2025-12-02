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



def discretize_taxi_requests(
    input_path: str,
    output_path: str,
    time_step_min: float,
):
    """
    Load continuous-time taxi-like requests from JSON, discretize their
    time windows, and save a JSON with discrete time indices.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        requests = json.load(f)

    for req in requests:
        T_k_min = req["T_k_min"]
        T_in_min = req["T_in_min"]
        T_out_min = req["T_out_min"]

        # build discrete sets of indices
        T_k_idx = interval_to_indices(T_k_min[0],  T_k_min[1],  time_step_min)
        T_in_idx = interval_to_indices(T_in_min[0], T_in_min[1], time_step_min)
        T_out_idx = interval_to_indices(T_out_min[0], T_out_min[1], time_step_min)

        req["T_k_idx"] = T_k_idx
        req["T_in_idx"] = T_in_idx
        req["T_out_idx"] = T_out_idx

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)

    print(f"[INFO] Discretized requests saved to: {out_path}")




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

    print(f"[INFO] Discretized network saved to: {out_path}")



if __name__ == "__main__":

    time_step_min = 5.0
    number = 3      # side of the grid
    horizon = 60   # in minutes

    # 1) Discretization of the request
    input_req = f"instances/GRID/{number}x{number}/taxi_like_requests_{horizon}maxmin.json"
    output_req = f"instances/GRID/{number}x{number}/taxi_like_requests_{horizon}maxmin_disc{int(time_step_min)}min.json"

    discretize_taxi_requests(
        input_path=input_req,
        output_path=output_req,
        time_step_min=time_step_min,
    )

    # 2) Discretization of the network
    input_net = f"instances/GRID/{number}x{number}/network.json"
    output_net = f"instances/GRID/{number}x{number}/network_disc{int(time_step_min)}min.json"

    discretize_network_travel_times(
        input_path=input_net,
        output_path=output_net,
        time_step_min=time_step_min,
    )
