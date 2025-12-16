import json
from pathlib import Path
from utils.MT.runs_fun import *
from typing import List, Dict, Tuple
import math
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
from typing import Literal
from sklearn.cluster import KMeans







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

def build_fully_connected_request_6d_matrix(
    req6d: List[Dict],
    alpha_O: float = 1.0,
    beta_P: float  = 1.0,
    alpha_D: float = 1.0,
    beta_D: float  = 1.0,
    standardize: bool = True,
) -> np.ndarray:

    K = len(req6d)
    W = np.zeros((K, K), dtype=float)

    # --- 1) precompute component distances for all pairs (i<j) ---
    pairs = []
    DO, DtP, DD, DtD = [], [], [], []

    for i in range(K):
        ri = req6d[i]
        for j in range(i + 1, K):
            rj = req6d[j]

            d_xyO = math.hypot(ri["xo"] - rj["xo"], ri["yo"] - rj["yo"])
            d_xyD = math.hypot(ri["xd"] - rj["xd"], ri["yd"] - rj["yd"])
            d_tP  = abs(ri["tP"] - rj["tP"])
            d_tD  = abs(ri["tD"] - rj["tD"])

            pairs.append((i, j, d_xyO, d_tP, d_xyD, d_tD))
            DO.append(d_xyO); DtP.append(d_tP); DD.append(d_xyD); DtD.append(d_tD)

    # --- 2) compute z-score params once ---
    if standardize:
        mu_O,  std_O  = float(np.mean(DO)),  float(np.std(DO))
        mu_tP, std_tP = float(np.mean(DtP)), float(np.std(DtP))
        mu_D,  std_D  = float(np.mean(DD)),  float(np.std(DD))
        mu_tD, std_tD = float(np.mean(DtD)), float(np.std(DtD))

        std_O  = max(std_O,  1e-8)
        std_tP = max(std_tP, 1e-8)
        std_D  = max(std_D,  1e-8)
        std_tD = max(std_tD, 1e-8)
    else:
        mu_O = mu_tP = mu_D = mu_tD = 0.0
        std_O = std_tP = std_D = std_tD = 1.0

    # --- 3) fill W ---
    for (i, j, d_xyO, d_tP, d_xyD, d_tD) in pairs:
        d_O_n  = (d_xyO - mu_O)  / std_O
        d_tP_n = (d_tP  - mu_tP) / std_tP
        d_D_n  = (d_xyD - mu_D)  / std_D
        d_tD_n = (d_tD  - mu_tD) / std_tD

        dist = math.sqrt(
            (alpha_O * d_O_n)  ** 2 +
            (beta_P  * d_tP_n) ** 2 +
            (alpha_D * d_D_n)  ** 2 +
            (beta_D  * d_tD_n) ** 2
        )

        W[i, j] = dist
        W[j, i] = dist

    return W






def all_pairs_shortest_path_time_matrix(
    G: nx.DiGraph,
    nodes: List[int],
    weight: str = "time_min",
    symmetrize: str = "avg",   # "avg" | "min" | "max" | "none"
    unreachable_value: float | None = None,
) -> np.ndarray:
    """
    Build NxN shortest-path time matrix (minutes) over given node ordering.

    symmetrize:
      - "avg": D = (D + D^T)/2
      - "min": D = min(D, D^T)
      - "max": D = max(D, D^T)
      - "none": keep directed distances (NOT ok for standard MDS)

    unreachable_value:
      if None -> raise if any pair unreachable
      else fill with that value (e.g., big number)
    """
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    D = np.full((N, N), np.inf, dtype=float)
    np.fill_diagonal(D, 0.0)

    # single-source shortest paths for each source
    for s in nodes:
        i = idx[s]
        dist = nx.single_source_dijkstra_path_length(G, s, weight=weight)
        for t, d in dist.items():
            if t in idx:
                j = idx[t]
                D[i, j] = float(d)

    if np.isinf(D).any():
        if unreachable_value is None:
            # mostra un esempio utile
            bad = np.argwhere(np.isinf(D))
            a, b = bad[0]
            raise ValueError(
                f"Graph has unreachable pairs for MDS. Example: {nodes[a]} -> {nodes[b]} has no path."
            )
        else:
            D[np.isinf(D)] = float(unreachable_value)

    if symmetrize == "avg":
        D = 0.5 * (D + D.T)
    elif symmetrize == "min":
        D = np.minimum(D, D.T)
    elif symmetrize == "max":
        D = np.maximum(D, D.T)
    elif symmetrize == "none":
        # ok solo se poi NON fai MDS standard (o fai metodi per directed)
        pass
    else:
        raise ValueError("symmetrize must be one of: avg|min|max|none")

    return D


def mds_embed_nodes_from_sp(
    G: nx.DiGraph,
    weight: str = "time_min",
    dim: int = 2,
    symmetrize: str = "avg",
    random_state: int = 23,
    n_init: int = 4,
    max_iter: int = 300,
    normalized_stress: str = "auto",
) -> Dict[int, np.ndarray]:
    """
    Returns: dict node_id -> np.array([x,y]) (or dim-D).
    Embedding tries to preserve shortest-path travel times as euclidean distances.
    """
    nodes = list(G.nodes())
    D = all_pairs_shortest_path_time_matrix(
        G, nodes, weight=weight, symmetrize=symmetrize, unreachable_value=None
    )

    mds = MDS(
        n_components=dim,
        dissimilarity="precomputed",
        metric=True,                 # metric MDS
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        normalized_stress=normalized_stress,
    )

    X = mds.fit_transform(D)  # shape (N, dim)

    coord = {nodes[i]: X[i, :] for i in range(len(nodes))}
    return coord


def build_requests_6d_from_4d(
    req4d: List[Dict],
    node_xy: Dict[int, np.ndarray],
) -> List[Dict]:
    """
    From your req4d [{k,o,tP,d,tD,q}] build req6d:
      {k, xo, yo, tP, xd, yd, tD, q}
    """
    req6d = []
    for r in req4d:
        o = r["o"]
        d = r["d"]
        if o not in node_xy or d not in node_xy:
            raise KeyError(f"Missing node coords for o={o} or d={d}")

        xo, yo = node_xy[o][0], node_xy[o][1]
        xd, yd = node_xy[d][0], node_xy[d][1]

        req6d.append({
            "k": r["k"],
            "xo": float(xo),
            "yo": float(yo),
            "tP": float(r["tP"]),
            "xd": float(xd),
            "yd": float(yd),
            "tD": float(r["tD"]),
            "q": int(r["q"]),
        })
    return req6d



def compute_centroid_distances(
    req6d: List[Dict],
    use_capacity: bool = False,          # -> 7D se True
    capacity_key: str = "q",             # nome del campo capacità/domanda
    standardize: bool = True,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X   : (K,D) raw features
      Xn  : (K,D) standardized features (z-score), if standardize=True else X
      c   : (D,) centroid in standardized space (mean of Xn)
      dist: (K,) Euclidean distance of each request to centroid

    D = 6 or 7 depending on use_capacity.
    Feature order:
      6D: [xo, yo, tP, xd, yd, tD]
      7D: [xo, yo, tP, xd, yd, tD, q]
    """

    def req_to_vec(r: Dict) -> np.ndarray:
        base = [r["xo"], r["yo"], r["tP"], r["xd"], r["yd"], r["tD"]]
        if use_capacity:
            base.append(r[capacity_key])
        return np.array(base, dtype=float)

    X = np.vstack([req_to_vec(r) for r in req6d])  # (K,D)

    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.maximum(sd, eps)
        Xn = (X - mu) / sd
    else:
        mu = np.zeros(X.shape[1], dtype=float)
        sd = np.ones(X.shape[1], dtype=float)
        Xn = X

    c = Xn.mean(axis=0)  # centroide (in spazio standardizzato)
    dist = np.linalg.norm(Xn - c, axis=1)

    return X, Xn, c, dist, mu, sd


def print_req_centroid_debug(req6d, Xn, dist, use_capacity=False):
    D = Xn.shape[1]
    print(f"\n--- Standardized requests ({D}D) + dist to centroid ---")
    for i, r in enumerate(req6d):
        if use_capacity:
            print(
                f"k={r['k']:2d} | "
                f"[xo,yo,tP,xd,yd,tD,q]_z = "
                f"({Xn[i,0]:+.3f},{Xn[i,1]:+.3f},{Xn[i,2]:+.3f},"
                f"{Xn[i,3]:+.3f},{Xn[i,4]:+.3f},{Xn[i,5]:+.3f},{Xn[i,6]:+.3f}) "
                f"| dist={dist[i]:.4f}"
            )
        else:
            print(
                f"k={r['k']:2d} | "
                f"[xo,yo,tP,xd,yd,tD]_z = "
                f"({Xn[i,0]:+.3f},{Xn[i,1]:+.3f},{Xn[i,2]:+.3f},"
                f"{Xn[i,3]:+.3f},{Xn[i,4]:+.3f},{Xn[i,5]:+.3f}) "
                f"| dist={dist[i]:.4f}"
            )



def topk_closest_to_centroid(
    req6d: List[Dict],
    k: int = 5,
    use_capacity: bool = False,      # False -> 6D, True -> 7D
    capacity_key: str = "q",
    standardize: bool = True,
):
    X, Xn, c, dist, mu, sd = compute_centroid_distances(
        req6d,
        use_capacity=use_capacity,
        capacity_key=capacity_key,
        standardize=standardize,
    )

    k = min(k, len(req6d))
    idx_sorted = np.argsort(dist)          # crescente = più vicino
    idx_top = idx_sorted[:k]

    top_reqs = [req6d[i] for i in idx_top]
    top_dist = dist[idx_top]
    return idx_top, top_reqs, top_dist, c


def print_topk(reqs: List[Dict], dists: np.ndarray, use_capacity: bool = False):
    tag = "7D" if use_capacity else "6D"
    print(f"\n--- Top closest to centroid ({tag}) ---")
    for r, d in zip(reqs, dists):
        if use_capacity:
            print(
                f"k={r['k']:2d} | "
                f"O=({r['xo']:.3f},{r['yo']:.3f}) "
                f"D=({r['xd']:.3f},{r['yd']:.3f}) "
                f"tP={r['tP']:.2f} tD={r['tD']:.2f} q={r['q']} "
                f"| dist={d:.4f}"
            )
        else:
            print(
                f"k={r['k']:2d} | "
                f"O=({r['xo']:.3f},{r['yo']:.3f}) "
                f"D=({r['xd']:.3f},{r['yd']:.3f}) "
                f"tP={r['tP']:.2f} tD={r['tD']:.2f} "
                f"| dist={d:.4f}"
            )





def iterative_remove_by_centroid(
    req6d: List[Dict],
    n_remove: int,
    use_capacity: bool = False,
    capacity_key: str = "q",
    standardize: bool = True,
    mode: Literal["closest", "farthest"] = "closest",
):
    """
    Iterativamente:
      1) calcola distanze dal centroide (su dataset corrente)
      2) rimuove 1 richiesta (closest o farthest)
      3) ripete

    Returns:
      removed_steps: lista di dict con info step-by-step
      remaining: lista req rimaste
    """
    remaining = list(req6d)
    removed_steps = []

    n_remove = min(n_remove, len(remaining))

    for step in range(1, n_remove + 1):
        X, Xn, c, dist, mu, sd = compute_centroid_distances(
            remaining,
            use_capacity=use_capacity,
            capacity_key=capacity_key,
            standardize=standardize,
        )

        if mode == "closest":
            idx_local = int(np.argmin(dist))
        elif mode == "farthest":
            idx_local = int(np.argmax(dist))
        else:
            raise ValueError("mode must be 'closest' or 'farthest'")

        r = remaining.pop(idx_local)

        removed_steps.append({
            "step": step,
            "removed_k": r["k"],
            "removed_req": r,
            "removed_dist": float(dist[idx_local]),
            "centroid": c.copy(),
            "n_remaining": len(remaining),
        })

    return removed_steps, remaining


def print_removed_steps(steps: List[Dict], use_capacity: bool = False):
    tag = "7D" if use_capacity else "6D"
    print(f"\n--- Iterative removals ({tag}) ---")
    for s in steps:
        r = s["removed_req"]
        if use_capacity:
            print(
                f"step={s['step']:2d} | removed k={s['removed_k']:2d} | dist={s['removed_dist']:.4f} | "
                f"O=({r['xo']:.3f},{r['yo']:.3f}) D=({r['xd']:.3f},{r['yd']:.3f}) "
                f"tP={r['tP']:.2f} tD={r['tD']:.2f} q={r['q']}"
            )
        else:
            print(
                f"step={s['step']:2d} | removed k={s['removed_k']:2d} | dist={s['removed_dist']:.4f} | "
                f"O=({r['xo']:.3f},{r['yo']:.3f}) D=({r['xd']:.3f},{r['yd']:.3f}) "
                f"tP={r['tP']:.2f} tD={r['tD']:.2f}"
            )





# Per ora k-means, in futuro Bayes
def make_fictitious_requests_from_remaining(
    req6d_all: List[Dict],
    fixed_k: List[int],
    n_fict: int = 4,
    use_capacity: bool = True,      # True = 7D clustering, False = 6D
    standardize: bool = True,
    random_state: int = 23,
):
    """
    Crea n_fict richieste fittizie dalle richieste NON fissate.

    Return:
      fict_reqs: lista di dict (6D o 7D) con k fittizio negativo
      remaining: lista richieste originali usate per creare i fittizi
      labels: cluster id per ogni remaining
    """
    # 1) split fixed vs remaining
    fixed_set = set(fixed_k)
    remaining = [r for r in req6d_all if r["k"] not in fixed_set]
    if len(remaining) < n_fict:
        raise ValueError(f"Too few remaining requests ({len(remaining)}) to make {n_fict} fictitious.")

    # 2) build feature matrix X
    # 6D: [xo,yo,tP,xd,yd,tD]
    # 7D: add q
    def vec(r):
        base = [r["xo"], r["yo"], r["tP"], r["xd"], r["yd"], r["tD"]]
        if use_capacity:
            base.append(r["q"])
        return np.array(base, dtype=float)

    X = np.vstack([vec(r) for r in remaining])

    # 3) standardize (important!)
    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.maximum(sd, 1e-8)
        Xn = (X - mu) / sd
    else:
        mu = np.zeros(X.shape[1], dtype=float)
        sd = np.ones(X.shape[1], dtype=float)
        Xn = X

    # 4) clustering
    kmeans = KMeans(n_clusters=n_fict, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(Xn)

    # 5) build fictitious requests (mean in RAW space, q sum)
    fict_reqs = []
    for c in range(n_fict):
        idx = np.where(labels == c)[0]
        cluster = [remaining[i] for i in idx]

        xo = float(np.mean([r["xo"] for r in cluster]))
        yo = float(np.mean([r["yo"] for r in cluster]))
        tP = float(np.mean([r["tP"] for r in cluster]))
        xd = float(np.mean([r["xd"] for r in cluster]))
        yd = float(np.mean([r["yd"] for r in cluster]))
        tD = float(np.mean([r["tD"] for r in cluster]))

        if use_capacity:
            q = int(np.sum([r["q"] for r in cluster]))
        else:
            q = int(np.sum([r["q"] for r in cluster]))  # comunque somma capacità totale

        fict_reqs.append({
            "k": -(c + 1),      # id fittizi: -1,-2,-3,-4
            "xo": xo, "yo": yo, "tP": tP,
            "xd": xd, "yd": yd, "tD": tD,
            "q": q,
            "n_agg": int(len(cluster)),   # quante richieste aggregate
        })

    return fict_reqs, remaining, labels



def nearest_nodes_mds(x: float, y: float, node_xy: Dict[int, np.ndarray], top: int = 10):
    arr = []
    for nid, xy in node_xy.items():
        dx = float(xy[0]) - x
        dy = float(xy[1]) - y
        arr.append((dx*dx + dy*dy, int(nid)))
    arr.sort(key=lambda t: t[0])
    return arr[:top]


# Prendere (xo,yo) e (xd,yd) nello spazio MDS e trovare il nodo reale del grafo che, 
# NELLO STESSO SPAZIO MDS, è più vicino.
def snap_fict_request_to_graph_nodes(f: Dict, node_xy: Dict[int, np.ndarray], top: int = 10) -> Dict:
    cand_o = nearest_nodes_mds(f["xo"], f["yo"], node_xy, top=top)
    cand_d = nearest_nodes_mds(f["xd"], f["yd"], node_xy, top=top)

    o = cand_o[0][1]
    d = cand_d[0][1]

    if o == d:
        # scegli il primo candidato D diverso da o
        for _, nid in cand_d[1:]:
            if nid != o:
                d = nid
                break

    return {
        "k": int(f["k"]),
        "o": int(o),
        "d": int(d),
        "tP": float(f["tP"]),
        "tD": float(f["tD"]),
        "q": int(f["q"]),
        "n_agg": int(f.get("n_agg", 0)),
        "snap_o_d2": float(cand_o[0][0]),
        "snap_d_d2": float(cand_d[0][0]),
    }




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

        print("\nK =", len(req4d))
        print("\n--- REQUEST NODES (4D representation) ---")
        for r in req4d:
            print(
                f"k={r['k']:2d} | "
                f"O={r['o']}, tP={r['tP']:.2f} -> "
                f"D={r['d']}, tD={r['tD']:.2f} | "
                f"q={r['q']}"
            )


        # 1) embedding nodi (x,y) che “imitano” shortest path time_min
        node_xy = mds_embed_nodes_from_sp(G, weight="time_min", symmetrize="avg", dim=2)

        # 2) richieste 6D
        req6d = build_requests_6d_from_4d(req4d, node_xy)


        print("\n\nK =", len(req6d))
        print("\n--- REQUEST NODES (6D representation) ---")
        for i in range(len(req4d)):
            r4 = req4d[i]
            r6 = req6d[i]
            print(
                f"k={r4['k']:2d} | "
                f"O={r4['o']} -> ({r6['xo']:.3f},{r6['yo']:.3f}), "
                f"D={r4['d']} -> ({r6['xd']:.3f},{r6['yd']:.3f}), "
                f"tP={r4['tP']:.2f}, tD={r4['tD']:.2f}, q={r4['q']}"
            )

        W6 = build_fully_connected_request_6d_matrix(
            req6d,
            alpha_O=1.0, beta_P=1.0,
            alpha_D=1.0, beta_D=1.0,
            standardize=True
        )

        print("\nW6 shape =", W6.shape)





        # ---- 6D ----
        X, Xn, c, dist, mu, sd = compute_centroid_distances(req6d, use_capacity=False, standardize=True)
        print_req_centroid_debug(req6d, Xn, dist, use_capacity=False)

        # richiesta “media fittizia” (centroide) in spazio standardizzato
        print("\nCentroid (6D, standardized):", np.round(c, 4))


        # ---- 7D (aggiungo capacità q) ----
        X7, Xn7, c7, dist7, mu7, sd7 = compute_centroid_distances(req6d, use_capacity=True, capacity_key="q", standardize=True)
        print_req_centroid_debug(req6d, Xn7, dist7, use_capacity=True)
        print("\nCentroid (7D, standardized):", np.round(c7, 4))



        KEEP = 4  # quante “più rappresentative” vuoi, con minor varianza, che se rimosse cambiano meno la distribuzione

        ### NON ITERATIVE VERSION ###
        # ---- 6D ----
        idx6, top6, dist6, c6 = topk_closest_to_centroid(req6d, k=KEEP, use_capacity=False, standardize=True)
        print_topk(top6, dist6, use_capacity=False)
        print("\nIndices 6D:", idx6.tolist())

        # ---- 7D ----
        idx7, top7, dist7, c7 = topk_closest_to_centroid(req6d, k=KEEP, use_capacity=True, capacity_key="q", standardize=True)
        print_topk(top7, dist7, use_capacity=True)      
        print("Indices 7D:", idx7.tolist())



        ### ITERATIVE VERSION ###
        # -------- 6D --------
        steps6, remaining6 = iterative_remove_by_centroid(req6d, n_remove=KEEP, use_capacity=False, standardize=True, mode="closest")
        print_removed_steps(steps6, use_capacity=False)
        print("Selected (iterative 6D):", [s["removed_k"] for s in steps6])

        # -------- 7D --------
        steps7, remaining7 = iterative_remove_by_centroid(req6d, n_remove=KEEP, use_capacity=True, capacity_key="q", standardize=True, mode="closest")
        print_removed_steps(steps7, use_capacity=True)
        print("Selected (iterative 7D):", [s["removed_k"] for s in steps7])





        #### Crea richieste fittizie da remaining (usando 7D clustering) ####

        # --- 1) richieste fisse ---
        fixed_k = [s["removed_k"] for s in steps7]
        print("Fixed k:", fixed_k)

        FICT = 5
        # --- 2) crea 4 fittizie dalle altre 26 ---
        fict6d, remaining, labels = make_fictitious_requests_from_remaining(
            req6d_all=req6d,
            fixed_k=fixed_k,
            n_fict=FICT,
            use_capacity=True,      # clustering in 7D
            standardize=True,
            random_state=23,
        )

        print("\n--- FICTITIOUS (in embedding coords) ---")
        for f in fict6d:
            print(f"k={f['k']} | n_agg={f['n_agg']} | "
                f"O=({f['xo']:.3f},{f['yo']:.3f}) -> D=({f['xd']:.3f},{f['yd']:.3f}) | "
                f"tP={f['tP']:.2f} tD={f['tD']:.2f} q={f['q']}")

        # --- 3) snap su nodi reali ---
        fict_graph = [snap_fict_request_to_graph_nodes(f, node_xy) for f in fict6d]

        print("\n--- FICTITIOUS (snapped to graph node IDs) ---")
        for f in fict_graph:
            print(f"k={f['k']} | n_agg={f['n_agg']} | "
                f"o={f['o']} d={f['d']} | tP={f['tP']:.2f} tD={f['tD']:.2f} q={f['q']}")
            


        fixed_reqs_real = [r for r in req4d if r["k"] in set(fixed_k)]

        # converti fict_graph nel formato req4d
        fict_reqs_4d = [{
            "k": f["k"],
            "o": f["o"],
            "tP": f["tP"],
            "d": f["d"],
            "tD": f["tD"],
            "q": f["q"],
        } for f in fict_graph]

        final_requests = fixed_reqs_real + fict_reqs_4d
        print("\nFINAL K =", len(final_requests))


        # salva json nel formato identico a quello originale
        out_path = base_output_folder / "requests_REDUCED.json"
        out_json = []
        for r in final_requests:
            out_json.append({
                "id": int(r["k"]),  # ok anche negativo se il loader lo accetta
                "origin": int(r["o"]),
                "destination": int(r["d"]),
                "q_k": int(r["q"]),
                "desired_departure_min": float(r["tP"]),
                "desired_arrival_min": float(r["tD"]),
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2)

        print("Saved reduced requests to:", out_path)

        










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
