import json
from pathlib import Path
from utils.MT.runs_fun import *
from typing import List, Dict, Tuple
import math
import numpy as np
import networkx as nx

def load_network_continuous_as_graph(network_path: str) -> nx.DiGraph:
    with open(network_path, "r", encoding="utf-8") as f:
        net = json.load(f)

    G = nx.DiGraph()
    for n in net["nodes"]:
        G.add_node(int(n["id"]))

    for e in net["edges"]:
        G.add_edge(
            int(e["u"]), int(e["v"]),
            time_min=float(e["time_min"]),
            length_km=float(e.get("length_km", 0.0)),
        )
    return G


def shortest_path_time_min(G: nx.DiGraph, u: int, v: int) -> float:
    return nx.shortest_path_length(G, source=u, target=v, weight="time_min")


def build_requests_4d_from_file(requests_path: str) -> List[Dict]:
    """
    One point per request k.
    Returns list of dict:
      {k, o, tP, d, tD, q}
    """
    with open(requests_path, "r", encoding="utf-8") as f:
        reqs = json.load(f)

    req4d = []
    for r in reqs:
        k = int(r["id"])
        req4d.append({
            "k": k,
            "o": int(r["origin"]),
            "tP": float(r["desired_departure_min"]),
            "d": int(r["destination"]),
            "tD": float(r["desired_arrival_min"]),
            "q": int(r["q_k"]),
        })
    # (opzionale) ordina per k per essere sicuri che indice = k
    req4d.sort(key=lambda x: x["k"])
    return req4d


def build_fully_connected_request_4d_matrix(
    G: nx.DiGraph,
    req4d: List[Dict],
    alpha_O: float = 1.0,   # peso spazio origine (min su grafo)
    beta_P: float  = 1.0,   # peso tempo pickup
    alpha_D: float = 1.0,   # peso spazio destinazione (min su grafo)
    beta_D: float  = 1.0,   # peso tempo delivery
    symmetrize_space: bool = False,
) -> np.ndarray:
    """
    W is KxK where K = number of requests.
    If symmetrize_space=True we average SP(u->v) and SP(v->u) to get an undirected-like distance.
    """

    K = len(req4d)
    W = np.zeros((K, K), dtype=float)

    for i in range(K):
        oi, tPi, di, tDi = req4d[i]["o"], req4d[i]["tP"], req4d[i]["d"], req4d[i]["tD"]

        for j in range(i + 1, K):
            oj, tPj, dj, tDj = req4d[j]["o"], req4d[j]["tP"], req4d[j]["d"], req4d[j]["tD"]

            # tempo
            d_tP = abs(tPi - tPj)
            d_tD = abs(tDi - tDj)

            # spazio su grafo (in minuti)
            dO_ij = shortest_path_time_min(G, oi, oj)
            dD_ij = shortest_path_time_min(G, di, dj)

            # simmetrizzazione distanze
            if symmetrize_space:
                dO_ji = shortest_path_time_min(G, oj, oi)
                dD_ji = shortest_path_time_min(G, dj, di)
                d_O = 0.5 * (dO_ij + dO_ji)
                d_D = 0.5 * (dD_ij + dD_ji)
            else:
                d_O = dO_ij
                d_D = dD_ij


            # standardizzazione (z-score)
            # To avoid scale bias between spatial and temporal components, each distance term is standardized to zero mean and unit variance across all request pairs before aggregation.
            D_O = []
            D_tP = []
            D_D = []
            D_tD = []

            D_O.append(d_O)
            D_tP.append(d_tP)
            D_D.append(d_D)
            D_tD.append(d_tD)

            mu_O,  std_O  = np.mean(D_O),  np.std(D_O)
            mu_tP, std_tP = np.mean(D_tP), np.std(D_tP)
            mu_D,  std_D  = np.mean(D_D),  np.std(D_D)
            mu_tD, std_tD = np.mean(D_tD), np.std(D_tD)

            # avoid division by zero
            std_O  = max(std_O,  1e-8)
            std_tP = max(std_tP, 1e-8)
            std_D  = max(std_D,  1e-8)
            std_tD = max(std_tD, 1e-8)

            d_O_n  = (d_O  - mu_O)  / std_O
            d_tP_n = (d_tP - mu_tP) / std_tP
            d_D_n  = (d_D  - mu_D)  / std_D
            d_tD_n = (d_tD - mu_tD) / std_tD

            # calcolo distanza
            dist = math.sqrt(
                (alpha_O * d_O_n)  ** 2 +
                (beta_P  * d_tP_n) ** 2 +
                (alpha_D * d_D_n)  ** 2 +
                (beta_D  * d_tD_n) ** 2
            )

            W[i, j] = dist
            W[j, i] = dist

    return W





if __name__ == "__main__":

    # -----------------------------
    # CHOOSE MODE
    # -----------------------------
    MODE = "GRID"   # "GRID" or "CITY"

    # -----------------------------
    # COMMON PARAMS
    # -----------------------------
    seed = 23
    horizon = 120          # minutes
    dt = 3
    num_requests = 30
    q_min, q_max = 1, 6
    slack_min = 20.0
    depot = 0

    number = 3
    num_modules = 3
    num_trails  = 6
    Q = 10
    c_km = 1.0
    c_uns = 100.0
    g_plat = None
    num_Nw = 3

    # output base
    base_output_folder = Path("results") / f"{MODE}_h{horizon}_dt{dt}_K{num_requests}_seed{seed}"
    base_output_folder.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # DATA GENERATION + INSTANCE
    # -----------------------------
    if MODE == "GRID":

        # 1) generate network + requests JSON (your existing function)
        network_path, requests_path = generate_all_data_asym(
            number=number,
            horizon=horizon,
            dt=dt,
            num_requests=num_requests,
            q_min=q_min,
            q_max=q_max,
            slack_min=slack_min,
            depot=depot
        )
        network_cont_path  = f"instances/GRID/{number}x{number}/network.json"
        requests_cont_path = f"instances/GRID/{number}x{number}/taxi_like_requests_{horizon}maxmin.json"

        G = load_network_continuous_as_graph(network_cont_path)
        req4d = build_requests_4d_from_file(requests_cont_path)

        W = build_fully_connected_request_4d_matrix(
            G, req4d,
            alpha_O=1.0, beta_P=1.0,
            alpha_D=1.0, beta_D=1.0,
            symmetrize_space=True
        )

        print("K =", len(req4d))
        print("W shape =", W.shape)
        print("W[0,1] =", f"{W[0,1]:.4f}")
        print("\n--- REQUEST NODES (4D representation) ---")
        for r in req4d:
            print(
                f"k={r['k']:2d} | "
                f"O={r['o']}, tP={r['tP']:.2f} -> "
                f"D={r['d']}, tD={r['tD']:.2f} | "
                f"q={r['q']}"
            )





    elif MODE == "CITY":
        city = "Torino, Italia"
        subdir = "TORINO_SUB"
        central_suburbs = ["Centro", "Crocetta", "Santa Rita", "Aurora"]
        depot = 1198867366

        # 1) generate network + requests JSON (your existing function)
        network_path, requests_path = generate_all_data_city(
            city=city,
            subdir=subdir,
            central_suburbs=central_suburbs,
            horizon=horizon,
            dt=dt,
            num_requests=num_requests,
            q_min=q_min,
            q_max=q_max,
            slack_min=slack_min,
            depot=depot
        )

        # 2) build instance
        t_max = horizon // dt
        instance = load_instance_discrete(
            network_path=network_path,
            requests_path=requests_path,
            dt=dt,
            t_max=t_max,
            num_modules=num_modules,
            num_trail=num_trails,
            Q=Q,
            c_km=c_km,
            c_uns=c_uns,
            g_plat=g_plat,
            depot=depot,
            num_Nw=num_Nw
        )



    else:
        raise ValueError(f"Unknown MODE: {MODE}")
