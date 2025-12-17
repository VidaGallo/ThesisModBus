import json
from typing import Any, Dict, List, Tuple
import math
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
from typing import Literal
from sklearn.cluster import KMeans
import numpy as np



def load_original_requests(requests_path: str) -> List[Dict]:
    with open(requests_path, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_original_by_ids(original_reqs: List[Dict], selected_ids: List[int]) -> List[Dict]:
    sel = set(int(k) for k in selected_ids)
    return [r for r in original_reqs if int(r["id"]) in sel]



### Load the original continuous graph
def load_network_continuous_as_graph(network_cont_path: str) -> nx.DiGraph:
    with open(network_cont_path, "r", encoding="utf-8") as f:
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




def build_requests_5d_from_file(requests_path: str) -> List[Dict]:
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



Request = Dict[str, Any]
def build_req7d_from_paths(
    network_path,
    requests_path,
    *,
    sp_weight: str = "time_min",
    symmetrize: str = "avg",
    dim: int = 2,
) -> Tuple[nx.DiGraph, List[Request], Dict[int, Tuple[float, float]]]:
    """
    Carica rete+richieste, crea embedding MDS, costruisce req6d e aggiunge la 7a dimensione 'cap'.

    Returns:
        G       : grafo nx.DiGraph
        req7d   : lista richieste con campi 6d + cap
        node_xy : embedding {node_id: (x,y)}
    """
    # loading
    G = load_network_continuous_as_graph(str(network_path))
    req4d = build_requests_5d_from_file(str(requests_path))

    # embedding
    node_xy = mds_embed_nodes_from_sp(
        G, weight=sp_weight, symmetrize=symmetrize, dim=dim
    )
    req6d = build_requests_7d_from_5d(req4d, node_xy)


    return G, req7d, node_xy








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


def build_requests_5d_from_4d(
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
def make_fictitious_requests_from_remaining_list(
    remaining: List[Dict],
    n_fict: int = 4,
    use_capacity: bool = True,      # True = 7D (aggiunge q)
    capacity_key: str = "q",
    standardize: bool = True,
    random_state: int = 23,
) -> Tuple[List[Dict], List[Dict], np.ndarray]:
    """
    Crea n_fict richieste fittizie usando SOLO 'remaining' (già filtrate).

    Return:
      fict_reqs: lista dict (6D/7D) con k fittizio negativo
      remaining: stessa lista remaining (ritornata per comodità)
      labels: cluster id per ogni elemento di remaining
    """
    if len(remaining) < n_fict:
        raise ValueError(f"Too few remaining requests ({len(remaining)}) to make {n_fict} fictitious.")

    def vec(r):
        base = [r["xo"], r["yo"], r["tP"], r["xd"], r["yd"], r["tD"]]
        if use_capacity:
            base.append(r[capacity_key])
        return np.array(base, dtype=float)

    X = np.vstack([vec(r) for r in remaining])

    # standardize
    if standardize:
        mu = X.mean(axis=0)
        sd = np.std(X, axis=0)
        sd = np.maximum(sd, 1e-8)
        Xn = (X - mu) / sd
    else:
        Xn = X

    # clustering
    kmeans = KMeans(n_clusters=n_fict, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(Xn)

    # build fictitious requests (mean in raw space, capacity sum)
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
        q  = int(np.sum([r.get(capacity_key, 0) for r in cluster])) if use_capacity else int(np.sum([r.get("q", 0) for r in cluster]))

        fict_reqs.append({
            "k": -(c + 1),
            "xo": xo, "yo": yo, "tP": tP,
            "xd": xd, "yd": yd, "tD": tD,
            "q": q,
            "n_agg": int(len(cluster)),
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





def to_demand_generator_format(
    G: nx.DiGraph,
    reqs: List[Dict],
    *,
    slack_min: float,
    force_arrival_eq_sp: bool = False,
):
    out = []

    for r in reqs:
        k = int(r["k"])
        o = int(r["o"])
        d = int(r["d"])
        q = int(r["q"])
        tP = float(r["tP"])

        # shortest path time
        tau_sp = float(nx.shortest_path_length(G, o, d, weight="time_min"))

        if force_arrival_eq_sp:
            tD = tP + tau_sp
        else:
            tD = float(r["tD"])

        delta = float(slack_min)

        out.append({
            "id": k,
            "origin": o,
            "destination": d,
            "q_k": q,
            "desired_departure_min": tP,
            "desired_arrival_min": tD,
            "slack_min": delta,
            "tau_sp_min": tau_sp,
            "T_k_min":  [tP, tD + delta],
            "T_in_min": [tP, tP + delta / 2.0],
            "T_out_min": [tP + tau_sp, tP + tau_sp + delta],
        })

    return out








def fix_constraints(
    model,
    var_dicts: Dict[str, Dict[Tuple, Any]],
    fix_map: Dict[str, Dict[Tuple, float]],
    *,
    tol: float = 1e-9,
    name_prefix: str = "fix",
) -> int:
    """
    model      : docplex.mp.model.Model
    var_dicts  : {"x": x_vars, "a": a_vars, ...} dove x_vars[(...)] è una Var docplex
    fix_map    : {"x": {(i,j,t):1, ...}, "a": {...}, ...}

    Aggiunge vincoli var == value.
    Return: numero vincoli aggiunti.
    """
    n_added = 0

    for fam, fixes in fix_map.items():
        if fam not in var_dicts:
            raise KeyError(f"fix_map chiede famiglia '{fam}' ma non esiste in var_dicts")

        V = var_dicts[fam]  # dict indicizzato -> Var

        for key, val in fixes.items():
            if key not in V:
                raise KeyError(f"Variabile {fam}{key} non trovata nel modello (key={key})")

            v = V[key]
            value = float(val)

            # evita vincoli inutili se già fissata (opzionale)
            lb = getattr(v, "lb", None)
            ub = getattr(v, "ub", None)
            if lb is not None and ub is not None and abs(lb - value) <= tol and abs(ub - value) <= tol:
                continue

            ct_name = f"{name_prefix}_{fam}_{'_'.join(map(str, key))}"
            model.add_constraint(v == value, ctname=ct_name)
            n_added += 1

    return n_added









def fix_only_k_families(sol_vals, selected_ids):
    sel = set(int(k) for k in selected_ids)
    out = {}

    # famiglie dove la key inizia con k
    k_first_fams = {"r", "w", "a", "b"}
    for fam in k_first_fams:
        if fam in sol_vals:
            out[fam] = {key: val for key, val in sol_vals[fam].items() if int(key[0]) in sel}

    # s ha key = k
    if "s" in sol_vals:
        out["s"] = {k: val for k, val in sol_vals["s"].items() if int(k) in sel}

    return out
