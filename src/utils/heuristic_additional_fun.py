import json
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Iterable, Optional
import math
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
from typing import Literal
from sklearn.cluster import KMeans
import numpy as np
from copy import deepcopy
import random

from utils.GP_def import *



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



### Per ora RANDOM, in futuro EURISTICA
def cluster_events_random_eventwise(
    events4d: List[dict],
    n_clusters: int,
    p_noise: float = 0.2,
    seed: int | None = None,
    enforce_pair_noise: bool = True,   # se uno è -1 => entrambi -1
) -> Dict[tuple[int, str], int]:
    rng = random.Random(seed)

    # assegna prima label indipendente per evento
    tmp: Dict[int, Dict[str, int]] = {}
    for e in events4d:
        k = int(e["k"])
        typ = str(e["type"])  # "P" or "D"

        if rng.random() < p_noise:
            c = -1
        else:
            c = rng.randrange(n_clusters)

        tmp.setdefault(k, {})[typ] = c

    # costruisci labels finali
    labels: Dict[tuple[int, str], int] = {}
    for k, d in tmp.items():
        lp = int(d.get("P", -1))
        ld = int(d.get("D", -1))

        if enforce_pair_noise and (lp == -1 or ld == -1):
            labels[(k, "P")] = -1
            labels[(k, "D")] = -1
        else:
            labels[(k, "P")] = lp
            labels[(k, "D")] = ld

    return labels



### Fissare a 0 il clsuter -1
def add_ignored_request_zero_constraints(mdl, I, r, a, b, w, s, ignored_ks, name="ign"):
    M = list(I.M)
    Nw = list(I.Nw)

    for k in ignored_ks:
        mdl.add_constraint(s[k] == 0, ctname=f"{name}_s0_k{k}")

        for t in I.DeltaT[k]:
            for m in M:
                if (k,t,m) in r: mdl.add_constraint(r[(k,t,m)] == 0, ctname=f"{name}_r0_k{k}_t{t}_m{m}")
                if (k,t,m) in a: mdl.add_constraint(a[(k,t,m)] == 0, ctname=f"{name}_a0_k{k}_t{t}_m{m}")
                if (k,t,m) in b: mdl.add_constraint(b[(k,t,m)] == 0, ctname=f"{name}_b0_k{k}_t{t}_m{m}")

            # w[k,i,t,m,mp]
            for i in Nw:
                for m in M:
                    for mp in M:
                        if m == mp: 
                            continue
                        key = (k, i, t, m, mp)
                        if key in w:
                            mdl.add_constraint(w[key] == 0, ctname=f"{name}_w0_k{k}_i{i}_t{t}_m{m}_mp{mp}")





### FISSARE A O B




### Pick the requests using the ID
def pick_by_ids(items, ids, key):
    sel = set(int(i) for i in ids)
    return [x for x in items if int(x[key]) in sel]




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



# From 7D requests build 4D
def build_events_4d_from_req7d(req7d: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Da richieste 7D (xo,yo,tP,xd,yd,tD,q) costruisce eventi 4D:
      - pickup: (xo,yo,tP,+q)
      - drop  : (xd,yd,tD,-q)
    """
    events = []
    for r in req7d:
        k = int(r["k"])
        q = float(r["q"])

        events.append({
            "k": k,
            "type": "P",
            "x": float(r["xo"]),
            "y": float(r["yo"]),
            "t": float(r["tP"]),
            "q": +q,
        })

        events.append({
            "k": k,
            "type": "D",
            "x": float(r["xd"]),
            "y": float(r["yd"]),
            "t": float(r["tD"]),
            "q": -q,
        })

    return events


















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