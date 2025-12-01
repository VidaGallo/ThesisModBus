"""
Demand Generator
================

Generates travel requests for the experiments, using the network created
in `generate_network.py`.

Produces:
- Taxi-like requests
- Bus-like requests linked to a fixed line/path

Each request is stored with:
    - origin/destination
    - demand q_k
    - service window
    - feasible boarding/alighting times (d and d_tilde)

Outputs:
    - `taxi_like_requests.json`
    - `bus_like_requests.json`
"""
