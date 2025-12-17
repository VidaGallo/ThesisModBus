import json
from typing import Any, Dict, List, Tuple, Iterable
import math
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
from typing import Literal
from sklearn.cluster import KMeans
import numpy as np
from copy import deepcopy






### Load original CONTINUOUS TIME requests
def load_original_requests(requests_path: str) -> List[Dict]:
    with open(requests_path, "r", encoding="utf-8") as f:
        return json.load(f)





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




### Build the DISTANCE MATRIX between nodes as shortest path on a Graph
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




### Place the requests in a 2D space, 
# such that the travel time (shortest path) is preserved as a distance between the points
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
    Returns: dict node_id -> np.array([x,y]).
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





### Take the requests (in continuous time) and project them in 7D
Request = Dict[str, Any]
def build_req7d_from_paths(
    network_path,     # continuous
    requests_path,    # continuous
    sp_weight: str = "time_min",     # shortest path on times
    symmetrize: str = "avg",         # to simmetrize the distances
    dim: int = 2,                    # dim of the embedding (x,y)
) -> Tuple[nx.DiGraph, List[Request], Dict[int, Tuple[float, float]]]:
    """
    Carica rete+richieste, crea embedding MDS, costruisce req6d e aggiunge la 7a dimensione 'cap'.
    """
    ### Load graph
    G = load_network_continuous_as_graph(str(network_path))

    ### MDS embedding for nodes
    node_xy = mds_embed_nodes_from_sp(
        G, weight=sp_weight, symmetrize=symmetrize, dim=dim
    )

    ### Load requests 
    with open(requests_path, "r", encoding="utf-8") as f:
        reqs = json.load(f)

    ### Build req7d
    req7d: List[Request] = []
    for r in reqs:
        k = int(r["id"])
        o = int(r["origin"])
        d = int(r["destination"])

        # coords from embedding
        xo, yo = float(node_xy[o][0]), float(node_xy[o][1])
        xd, yd = float(node_xy[d][0]), float(node_xy[d][1])

        req7d.append({
            "k": k,
            "xo": xo, "yo": yo,
            "tP": float(r["desired_departure_min"]),
            "xd": xd, "yd": yd,
            "tD": float(r["desired_arrival_min"]),
            "q": int(r["q_k"]),
        })

    req7d.sort(key=lambda x: x["k"])
    return G, reqs, req7d, node_xy






### Select the k nearest to the centroid, with standardization
def topk_ids_by_centroid_7d(
        req7d: list[dict], 
        k: int, 
        standardize: bool = True, 
        eps: float = 1e-8
    ):
    X = np.array([[r["xo"], r["yo"], r["tP"], r["xd"], r["yd"], r["tD"], r["q"]] for r in req7d], dtype=float)

    if standardize:
        mu = X.mean(axis=0)
        sd = np.maximum(X.std(axis=0), eps)
        Xn = (X - mu) / sd
    else:
        Xn = X

    c = Xn.mean(axis=0)
    dist = np.linalg.norm(Xn - c, axis=1)

    k = min(k, len(req7d))
    idx_sel = np.argsort(dist)[:k]

    selected_ids = [int(req7d[i]["k"]) for i in idx_sel]
    sel = set(selected_ids)
    remaining_ids = [int(r["k"]) for r in req7d if int(r["k"]) not in sel]
    return selected_ids, remaining_ids






### Pick the original requests using the orignal id
def pick_original_by_ids(original_reqs: List[Dict], selected_ids: List[int]) -> List[Dict]:
    sel = set(int(k) for k in selected_ids)
    return [r for r in original_reqs if int(r["id"]) in sel]



### Prende il nearest node
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



### Conversion to the "original" request format
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
        q = math.floor(r["q"] + 0.5)
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





### Considering the fictitius points in 7d, they need to be brough back to the original space
def fict7d_to_requests_format(
    fict7d: List[Dict],
    *,
    node_xy: Dict[int, "np.ndarray"],
    G: nx.DiGraph,
    slack_min: float,
    top_snap: int = 10,
    force_arrival_eq_sp: bool = False,
) -> List[Dict]:
    """
    Pipeline esterna:
    fict7d (xo,yo,tP,xd,yd,tD,q, k<0)  ->
      1) snap su nodi reali (o,d) usando node_xy (MDS)
      2) conversione in formato richieste originale (id, origin, destination, T_k...)
    """
    # 1) snap: 7D -> (o,d,tP,tD,q)
    fict_graph = [
        snap_fict_request_to_graph_nodes(f, node_xy, top=top_snap)
        for f in fict7d
    ]

    # 2) (o,d,...) -> formato richieste dictionary
    fict_full = to_demand_generator_format(
        G,
        fict_graph,
        slack_min=slack_min,
        force_arrival_eq_sp=force_arrival_eq_sp,
    )

    return fict_full















# Per ora k-means, in futuro Bayes
Request7D = Dict[str, float | int]
def make_fictitious_requests_kmeans_7d(
    remaining: List[Request7D],
    n_fict: int,
    *,
    standardize: bool = True,
    random_state: int = 23,
    n_init: int = 10,
) -> Tuple[List[Request7D], np.ndarray]:
    """
    Crea n_fict richieste fittizie da remaining (SEMPRE 7D: xo,yo,tP,xd,yd,tD,q)

    Return:
      fict_reqs : lista dict con k negativo, + n_agg
      labels    : cluster label per ogni elemento di remaining
    """
    if len(remaining) < n_fict:
        raise ValueError(f"Too few remaining requests ({len(remaining)}) to make {n_fict} fictitious.")

    # matrice (K,7)
    X = np.array(
        [[r["xo"], r["yo"], r["tP"], r["xd"], r["yd"], r["tD"], r["q"]] for r in remaining],
        dtype=float,
    )

    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.maximum(sd, 1e-8)
        Xn = (X - mu) / sd
    else:
        Xn = X

    kmeans = KMeans(n_clusters=n_fict, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(Xn)

    fict_reqs: List[Request7D] = []
    for c in range(n_fict):
        idx = np.where(labels == c)[0]
        cluster = [remaining[i] for i in idx]
        if not cluster:
            continue

        fict_reqs.append({
            "k": -(c + 1),
            "xo": float(np.mean([r["xo"] for r in cluster])),
            "yo": float(np.mean([r["yo"] for r in cluster])),
            "tP": float(np.mean([r["tP"] for r in cluster])),
            "xd": float(np.mean([r["xd"] for r in cluster])),
            "yd": float(np.mean([r["yd"] for r in cluster])),
            "tD": float(np.mean([r["tD"] for r in cluster])),
            "q":  int(np.sum([r["q"] for r in cluster])),   # aggrego capacità
            "n_agg": int(len(cluster)),
        })

    return fict_reqs, labels








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





def extract_solution_values_only_selected_k(
    var_dicts: Dict[str, Dict[Any, Any]],
    sol: Any,
    selected_ids: Iterable[int],
    *,
    keep_fams: set[str] | None = None,
    binary_round: bool = True,
    binary_tol: float = 1e-6,
    include_zeros: bool = True,
) -> Dict[str, Dict[Any, float]]:
    """
    Estrae SOLO famiglie legate a k (r,w,a,b) + s per k selezionati.
    include_zeros=True => salva anche 0 (così puoi fissare a 0).
    """
    if sol is None:
        raise ValueError("sol is None")

    sel = set(int(k) for k in selected_ids)

    # default: quelle che userai per fixing sulle richieste
    if keep_fams is None:
        keep_fams = {"r", "w", "a", "b", "s"}

    out: Dict[str, Dict[Any, float]] = {}

    for fam in keep_fams:
        if fam not in var_dicts:
            continue

        V = var_dicts[fam]
        fam_vals: Dict[Any, float] = {}

        # s ha chiave = k
        if fam == "s":
            for k, var in V.items():
                if int(k) not in sel:
                    continue
                val = sol.get_value(var)
                if val is None:
                    continue
                val = float(val)

                # binaria: arrotonda (opzionale)
                if binary_round:
                    fam_vals[int(k)] = 1.0 if val >= 0.5 else 0.0
                else:
                    fam_vals[int(k)] = val

            out[fam] = fam_vals
            continue

        # famiglie k-first: key[0] = k
        for key, var in V.items():
            if not isinstance(key, tuple) or len(key) == 0:
                continue
            if int(key[0]) not in sel:
                continue

            val = sol.get_value(var)
            if val is None:
                continue
            val = float(val)

            vt = getattr(getattr(var, "vartype", None), "short_name", None)
            if vt == "B" and binary_round:
                val = 1.0 if val >= 0.5 else 0.0
            else:
                # per intere/continue: se vuoi proprio fissare anche “quasi 0”
                if abs(val) < binary_tol:
                    val = 0.0

            if include_zeros:
                fam_vals[key] = val
            else:
                # fallback (se un giorno vuoi tornare a salvare solo attive)
                if vt == "B":
                    if val >= 0.5:
                        fam_vals[key] = 1.0
                else:
                    if abs(val) > binary_tol:
                        fam_vals[key] = val

        out[fam] = fam_vals

    return out







def deep_merge_fixmaps(
    base: Dict[str, Dict[Tuple, Any]],
    new: Dict[str, Dict[Tuple, Any]],
    *,
    check_conflict: bool = True,
    tol: float = 1e-9,
) -> Dict[str, Dict[Tuple, Any]]:
    """
    Merge profonda di fixmaps:
      base[fam][key] <- new[fam][key]

    - se una famiglia non esiste, viene creata
    - se una key esiste già:
        - se stesso valore: ok
        - se diverso:
            - check_conflict=True -> ValueError
            - check_conflict=False -> new sovrascrive
    """
    out = deepcopy(base)

    for fam, fixes in new.items():
        if fam not in out:
            out[fam] = dict(fixes)
            continue

        for key, val in fixes.items():
            if key in out[fam]:
                old = out[fam][key]
                if abs(old - val) > tol:
                    if check_conflict:
                        raise ValueError(
                            f"Conflict on {fam}{key}: old={old}, new={val}"
                        )
                    # else: sovrascrivi
            out[fam][key] = val

    return out