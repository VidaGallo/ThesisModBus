from utils.MT.heuristic_fun import *
from utils.MT.runs_fun import *

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path


# =========================
# FLAGS
# =========================
RUN_EXACT = True     # False → salta CPLEX
RUN_HEUR  = True     # False → salta HEURISTIC


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


CPLEX_CFG = {
    "time_limit": 36_000,     # seconds
    "mip_gap": 0.01,          # relative MIP gap, 0.05 = 5%
    "abs_mip_gap": 1e-6,      # absolute MIP gap
    "threads": 0,             # 0 = all available threads
    "mip_display": 1,         # 0..5 (2 = default)
    "emphasis_mip": 2,        # 0 balanced, 1 feasibility, 2 optimality, ...
    "parallel": 2             # 0 auto, 1 opportunistic, 2 deterministic
}


if __name__ == "__main__":

    # =========================
    # SEED
    # =========================
    seed = 23
    set_seed(seed)

    # =========================
    # BASE PARAMS
    # =========================
    horizon = 108
    dt = 6
    depot = 0

    Q = 10
    mean_speed_kmh = 40.0
    mean_edge_length_km = 3.33
    rel_std = 0.66
    c_km = 1.0
    c_uns = 100.0

    num_Nw = 2
    q_min, q_max = 1, 10
    alpha = 0.65
    slack_min = 20.0

    # =========================
    # GRID PARAMS
    # =========================
    number = 2
    num_modules = 2
    num_trails = 6
    z_max = 3
    num_requests = 30

    model_name = "w"   # <<< MODEL FISSO

    exp_id = (
        f"GRID_n{number}_H{horizon}_dt{dt}_"
        f"M{num_modules}_P{num_trails}_Z{z_max}_"
        f"K{num_requests}_Nw{num_Nw}_seed{seed}"
    )

    print("\n" + "=" * 80)
    print(f"EXPERIMENT {exp_id}")
    print("=" * 80)

    # =========================
    # OUTPUT FOLDERS
    # =========================
    base = Path("results") / "GRID" / "MT" / exp_id
    paths = {
        "base": base,
        "exact": base / "01_exact",
        "heur": base / "02_heuristic",
        "summary": base / "summary",
        "plots": base / "plots",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # =========================
    # BUILD INSTANCE (ONCE)
    # =========================
    instance, network_path, requests_path, t_max = build_instance_and_paths(
        number=number,
        horizon=horizon,
        dt=dt,
        num_modules=num_modules,
        num_trails=num_trails,
        Q=Q,
        c_km=c_km,
        c_uns=c_uns,
        num_requests=num_requests,
        q_min=q_min,
        q_max=q_max,
        slack_min=slack_min,
        depot=depot,
        seed=seed,
        num_Nw=num_Nw,
        mean_edge_length_km=mean_edge_length_km,
        mean_speed_kmh=mean_speed_kmh,
        rel_std=rel_std,
        z_max=z_max,
        alpha=alpha,
    )

    print("\nData used:")
    print(" network :", network_path)
    print(" requests:", requests_path)

    # =========================
    # EXACT (CPLEX)
    # =========================
    exact_results = []

    if RUN_EXACT:
        print("\n" + "=" * 80)
        print("RUNNING EXACT (model = w)")
        print("=" * 80)

        res = run_single_model(
            instance=instance,
            model_name=model_name,
            network_path=network_path,
            requests_path=requests_path,
            number=number,
            horizon=horizon,
            q_min=q_min,
            q_max=q_max,
            alpha=alpha,
            slack_min=slack_min,
            seed=seed,
            exp_id=exp_id,
            mean_edge_length_km=mean_edge_length_km,
            mean_speed_kmh=mean_speed_kmh,
            rel_std=rel_std,
            base_output_folder=paths["exact"],
            cplex_cfg=CPLEX_CFG      # possibe early stopping/feasable but not optimal
        )

        exact_results.append(res)
        df_exact = pd.DataFrame(exact_results)
        df_exact.to_csv(paths["summary"] / "summary_exact.csv", index=False)

    else:
        print("\n[SKIP] EXACT disabled")

    # =========================
    # HEURISTIC
    # =========================
    if RUN_HEUR:
        print("\n" + "=" * 80)
        print("RUNNING HEURISTIC")
        print("=" * 80)

        # usa ESATTAMENTE gli stessi dati
        G = load_network_continuous_as_graph(str(network_path))
        req4d = build_requests_4d_from_file(str(requests_path))

        node_xy = mds_embed_nodes_from_sp(G, weight="time_min", symmetrize="avg", dim=2)
        req6d = build_requests_6d_from_4d(req4d, node_xy)

        KEEP = 4
        FICT = 5

        steps7, remaining7 = iterative_remove_by_centroid(
            req6d,
            n_remove=KEEP,
            use_capacity=True,
            capacity_key="q",
            standardize=True,
            mode="closest",
        )

        fixed_k = [s["removed_k"] for s in steps7]

        fict6d, _, _ = make_fictitious_requests_from_remaining(
            req6d_all=req6d,
            fixed_k=fixed_k,
            n_fict=FICT,
            use_capacity=True,
            standardize=True,
            random_state=seed,
        )

        fict_graph = [snap_fict_request_to_graph_nodes(f, node_xy) for f in fict6d]

        fixed_real = [r for r in req4d if r["k"] in set(fixed_k)]

        fict_4d = [{
            "k": f["k"],
            "o": f["o"],
            "tP": f["tP"],
            "d": f["d"],
            "tD": f["tD"],
            "q": f["q"],
        } for f in fict_graph]

        final_requests = fixed_real + fict_4d

        out_path = paths["heur"] / "requests_REDUCED.json"
        out_json = [{
            "id": int(r["k"]),
            "origin": int(r["o"]),
            "destination": int(r["d"]),
            "q_k": int(r["q"]),
            "desired_departure_min": float(r["tP"]),
            "desired_arrival_min": float(r["tD"]),
        } for r in final_requests]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2)

        df_heur = pd.DataFrame([{
            "exp_id": exp_id,
            "seed": seed,
            "K_original": len(req4d),
            "K_reduced": len(final_requests),
            "KEEP": KEEP,
            "FICT": FICT,
            "fixed_k": fixed_k,
        }])

        df_heur.to_csv(paths["summary"] / "summary_heuristic.csv", index=False)

    else:
        print("\n[SKIP] HEURISTIC disabled")

    print("\n" + "#" * 80)
    print("DONE")
    print("Outputs in:", paths["base"])
    print("#" * 80)
