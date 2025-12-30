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
import hashlib
import json
from datetime import datetime



def set_seed(seed: int) -> random.Random:
    if seed is None:
        raise ValueError(
            "seed=None is not allowed. "
        )

    random.seed(seed)
    np.random.seed(seed)
    return random.Random(seed)



### To create a HASH of the instance (in order to not repeat instances generation unecessarly)

GENERATOR_VERSION = "v1"   # To change if there are some changes made in the functions!!!

def canonical_dumps(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def make_hash(params: dict, n: int = 12) -> str:
    s = canonical_dumps(params)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def write_meta(base: Path, meta: dict) -> None:
    meta = dict(meta)
    meta["created_at"] = datetime.now().isoformat(timespec="seconds")
    (base / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

def maybe_skip_existing(base: Path) -> bool:  # If there a file already exists no need to do it again
    return (base / "meta.json").exists()





### SYMMETRIC GRID
def generate_all_data(
    number: int,
    horizon: int,
    dt: int,
    num_requests: int,
    q_min: int,
    q_max: int,
    slack_min: float,
    depot: int,
    seed: int = 23,
    mean_edge_length_km: float = 3.0,
    mean_speed_kmh: float = 40.0,
    alpha: float = 0.65,
    ):
    """
    Generate continuous network + continuous requests,
    then discretize both.

    Returns:
      network_disc_path, requests_disc_path
    """
    rng = set_seed(seed)

    ### HASH GENERATION
    params = dict(
        dataset="GRID",
        seed=seed,
        number=number,
        horizon=horizon,
        dt=dt,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        depot=depot,
        mean_edge_length_km=mean_edge_length_km,
        mean_speed_kmh=mean_speed_kmh,
        alpha=alpha,
        generator_version=GENERATOR_VERSION,
    )
    h = make_hash(params)


    ### BUILD FOLDER
    base = (
        Path("instances")
        / "GRID"
        / f"{number}x{number}"
        / f"{h}_seed{seed}_H{horizon}_dt{dt}_K{num_requests}"
    )
    base.mkdir(parents=True, exist_ok=True)

    network_cont  = base / "network.json"
    requests_cont = base / "requests.json"
    network_disc  = base / f"network_disc_dt={dt}.json"
    requests_disc = base / f"requests_disc_dt={dt}.json"



    # Se l'istanza è già stata generata in passato
    if maybe_skip_existing(base):
        return base, network_cont, requests_cont, network_disc, requests_disc

    
    ### GENERATE CONTINUOUS NETWORK + SAVING
    generate_grid_network(
        output_path=network_cont, 
        side=number,
        edge_length_km=mean_edge_length_km,
        speed_kmh=mean_speed_kmh,
    )


    ### GENERATE CONTINUOUS REQUESTS + SAVING
    generate_requests(
        output_path=requests_cont,
        network_path=network_cont,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        time_horizon_max=float(horizon),
        depot=depot,
        alpha=alpha,
        rng=rng
    )

    ### DISCRETIZE REQUESTS & NETWORK
    discretize_network_travel_times(
        input_path=network_cont,
        output_path=network_disc,
        time_step_min=float(dt),
    )

    # DISCRETIZE REQUESTS (shortest path on the discrete netowork)
    discretize_requests(
        input_path=requests_cont,
        output_path=requests_disc,
        time_step_min=float(dt),
        network_disc_path=network_disc,   # for discrete shortest path
        depot=depot,                      # to check τ depot→origin discrete
    )

    write_meta(base, {"hash": h, "params": params})   # Si segna la generazione dell'istanza

    # Return paths
    return base, network_cont, requests_cont, network_disc, requests_disc



### ASSYMETRIC GRID
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
    rel_std: float = 0.66,
    seed: int = 23,
    alpha: float = 0.65,
):
    """
    Generate continuous network + continuous requests,
    then discretize both.

    Returns:
      network_disc_path, requests_disc_path
    """
    rng = set_seed(seed)

    ### HASH GENERATION
    params = dict(
        dataset="GRID_ASYM",
        seed=seed,
        number=number,
        horizon=horizon,
        dt=dt,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        depot=depot,
        mean_edge_length_km=mean_edge_length_km,
        mean_speed_kmh=mean_speed_kmh,
        rel_std=rel_std,
        alpha=alpha,
        generator_version=GENERATOR_VERSION,
    )
    h = make_hash(params)

    ### BUILD FOLDER
    base = (
        Path("instances")
        / "GRID_ASYM"
        / f"{number}x{number}"
        / f"{h}_seed{seed}_H{horizon}_dt{dt}_K{num_requests}"
    )
    base.mkdir(parents=True, exist_ok=True)

    network_cont  = base / "network.json"
    requests_cont = base / "requests.json"
    network_disc  = base / f"network_disc_dt={dt}.json"
    requests_disc = base / f"requests_disc_dt={dt}.json"


    # Se l'istanza è già stata generata in passato non serve ricrearla
    if maybe_skip_existing(base):
        return base, network_cont, requests_cont, network_disc, requests_disc


    ### GENERATE CONTINUOUS NETWORK + SAVING
    generate_grid_network_asym(
        output_path=network_cont,
        side=number,
        edge_length_km=mean_edge_length_km,
        speed_kmh=mean_speed_kmh,
        rel_std = rel_std,
        rng = rng

    )


    ### GENERATE CONTINUOUS REQUESTS + SAVING
    generate_requests(
        output_path=requests_cont,
        network_path=network_cont,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        time_horizon_max=float(horizon),
        depot=depot,
        alpha=alpha,
        rng=rng
    )

    ### DISCRETIZE REQUESTS & NETWORK
    discretize_network_travel_times(
        input_path=network_cont,
        output_path=network_disc,
        time_step_min=float(dt),
    )

    # DISCRETIZE REQUESTS (shortest path on the discrete netowork)
    discretize_requests(
        input_path=requests_cont,
        output_path=requests_disc,
        time_step_min=float(dt),
        network_disc_path=network_disc,   # for discrete shortest path
        depot=depot,                      # to check τ depot→origin discrete
    )

    write_meta(base, {"hash": h, "params": params})   # Si segna la generazione dell'istanza

    # Return paths
    return base, network_cont, requests_cont, network_disc, requests_disc



### CITY GRAPH
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
    alpha: float = 0.65,
):
    """
    Generate continuous network + continuous requests,
    then discretize both.

    Returns:
      network_disc_path, requests_disc_path
    """
    rng = set_seed(seed)

    ### HASH GENERATION
    params = dict(
        dataset="CITY",
        city=city,
        central_suburbs=sorted(list(central_suburbs)),  # ordine deterministico
        seed=seed,
        horizon=horizon,
        dt=dt,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        depot=depot,
        mean_speed_kmh=mean_speed_kmh,
        alpha=alpha,
        generator_version=GENERATOR_VERSION,
    )
    h = make_hash(params)

    ### BUILD FOLDER
    city_slug = (
        city.lower()
        .replace(",","")
        .replace(" ", "_")
    )

    base = (
        Path("instances")
        / "CITY"
        / subdir
        / city_slug
        / f"{h}_seed{seed}_H{horizon}_dt{dt}_K{num_requests}"
    )
    base.mkdir(parents=True, exist_ok=True)

    network_cont  = base / "network.json"
    requests_cont = base / "requests.json"
    network_disc  = base / f"network_disc_dt={dt}.json"
    requests_disc = base / f"requests_disc_dt={dt}.json"

    # Se l'istanza è già stata generata in passato
    if maybe_skip_existing(base):
        return base, network_cont, requests_cont, network_disc, requests_disc


    ### GENERATE CONTINUOUS NETWORK + SAVING
    generate_grid_network_city(
        output_path=network_cont,
        place = city,
        central_suburbs = central_suburbs,
        speed_kmh = mean_speed_kmh
    )


    ### GENERATE CONTINUOUS REQUESTS + SAVING
    generate_requests(
        output_path=requests_cont,
        network_path=network_cont,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        time_horizon_max=float(horizon),
        depot=depot,
        alpha=alpha,
        rng=rng
    )

    ### DISCRETIZE REQUESTS & NETWORK
    discretize_network_travel_times(
        input_path=network_cont,
        output_path=network_disc,
        time_step_min=float(dt),
    )

    # DISCRETIZE REQUESTS (shortest path on the discrete netowork)
    discretize_requests(
        input_path=requests_cont,
        output_path=requests_disc,
        time_step_min=float(dt),
        network_disc_path=network_disc,   # for discrete shortest path
        depot=depot,                      # to check τ depot→origin discrete
    )

    write_meta(base, {"hash": h, "params": params})   # Si segna la generazione dell'istanza

    # Return paths
    return base, network_cont, requests_cont, network_disc, requests_disc








