from __future__ import annotations
from typing import Any, Dict, Tuple, List, Iterable, Optional
import json
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import copy
import random

from utils.GP_fun import *
from models.model_MT_w import *
from utils.cplex_config import *



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
) -> Tuple[nx.DiGraph, List[Dict[str, Any]], List[Request], Dict[int, np.ndarray]]:
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





### Pick the requests using the ID
def pick_by_ids(items, ids, key):
    sel = set(int(i) for i in ids)
    return [x for x in items if int(x[key]) in sel]





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







### Clusterizzazione random (O e D clsuter diversi)
"""OLD VERSION"""
def cluster_PD_events_random(
    events4d: List[dict],
    n_clusters: int,
    p_noise: float = 0.2,
    seed: int | None = None,
) -> Dict[tuple[int, str], int]:
    """
    labels = {
        (1, "P"): 0,
        (1, "D"): 2,

        (3, "P"): -1,
        (3, "D"): -1,

        ...
    }
    """
    rng = random.Random(seed)
    labels: Dict[tuple[int, str], int] = {}

    # Si prendono gli ID (originali)
    requests = sorted({int(e["k"]) for e in events4d})

    for k in requests:
        # 1) decidi se la richiesta è rumore
        if rng.random() < p_noise:
            labels[(k, "P")] = -1
            labels[(k, "D")] = -1
            continue

        # 2) assegna cluster (qui P e D indipendenti)
        labels[(k, "P")] = rng.randrange(n_clusters)
        labels[(k, "D")] = rng.randrange(n_clusters)

    return labels



### Clusterizzazione random (O e D stesso cluster)
def cluster_PD_events_random(
    events4d: List[dict],
    n_clusters: int,
    seed: int | None = None,
) -> Dict[tuple[int, str], int]:
    """
    Random clustering per richiesta k:
      - pickup e delivery della stessa richiesta finiscono nello STESSO cluster

    Output:
      labels[(k,"P")] = c
      labels[(k,"D")] = c
    """
    rng = random.Random(seed)
    labels: Dict[tuple[int, str], int] = {}

    requests = sorted({int(e["k"]) for e in events4d})

    for k in requests:
        c = rng.randrange(n_clusters)
        labels[(k, "P")] = c
        labels[(k, "D")] = c

    return labels




### Variabili a e b già fissate in passato (soluzioni ottenute precedentemente)
FixMapAB = Dict[str, Dict[Tuple[int, int, int], float]]  # {"a": {(k,t,m):1}, "b": {...}}
def fix_constraints_ab(
    model,
    var_dicts: Dict[str, Dict[Tuple[int, int, int], Any]],
    fix_map: Optional[FixMapAB],
    tol: float = 1e-9,
    name_prefix: str = "fix",
    check_missing_var: bool = True,
) -> int:
    """
    Aggiunge SOLO vincoli di fixing per famiglie 'a' e 'b':
      a[k,t,m] == 0/1
      b[k,t,m] == 0/1

    Return: numero vincoli aggiunti.
    """
    if not fix_map:
        return 0

    n_added = 0
    for fam in ("a", "b"):
        fixes = fix_map.get(fam)
        if not fixes:
            continue

        if fam not in var_dicts:
            raise KeyError(f"var_dicts non contiene la famiglia '{fam}'")

        V = var_dicts[fam]  # (k,t,m) -> Var

        for key, val in fixes.items():
            if key not in V:
                if check_missing_var:
                    raise KeyError(f"Variabile {fam}{key} non trovata nel modello")
                else:
                    continue

            v = V[key]
            value = float(val)

            # skip se già fissata a value
            lb = getattr(v, "lb", None)
            ub = getattr(v, "ub", None)
            if lb is not None and ub is not None and abs(lb - value) <= tol and abs(ub - value) <= tol:
                continue

            k, t, m = key
            ct_name = f"{name_prefix}_{fam}_k{k}_t{t}_m{m}"
            model.add_constraint(v == value, ctname=ct_name)
            n_added += 1

    return n_added






### Si estraggono gli ID delle richieste da ignorare e da servire
def split_ignored_and_active_ks(labels_event: dict, cluster_ks):
    """
    ignored_ks: k se almeno uno tra (k,'P') o (k,'D') è -1
    active_ks : k clusterizzati con P e D entrambi != -1
    """
    cluster_set = set(int(k) for k in cluster_ks)
    status = {}  # k -> set dei label visti (es. {-1, 2})

    for (k, typ), c in labels_event.items():
        k = int(k)
        if k not in cluster_set:
            continue
        status.setdefault(k, set()).add(int(c))

    ignored = []
    active = []

    for k, labels in status.items():
        if -1 in labels:
            ignored.append(k)
        else:
            active.append(k)

    return sorted(ignored), sorted(active)





### Fissare a 0 il cluster -1
""" TEMPORANEAMENTE NON USATA """
def add_ignored_request_zero_constraints_ab(mdl, I, a, b, ignored_ks, name="ign"):
    M = list(I.M)
    ignored_ks = [int(k) for k in ignored_ks]
    for k in ignored_ks:
        k = int(k)
        for t in I.DeltaT[k]:
            for m in M:
                key = (k, t, m)
                if key in a:
                    mdl.add_constraint(a[key] == 0, ctname=f"{name}_a0_k{k}_t{t}_m{m}")
                if key in b:
                    mdl.add_constraint(b[key] == 0, ctname=f"{name}_b0_k{k}_t{t}_m{m}")
    return len(ignored_ks)





### Fissare lo stesso modulo per gli eventi OD appartenenti allo stesso cluster
def add_cluster_same_module_constraints_ab(
    mdl, I, a, b,
    labels_PD: dict,       # {(k,"P"/"D"): c or -1}
    active_ids: list[int], # richieste NON ignored
    name="cl_mod",
):
    M = list(I.M)
    active_set = set(int(k) for k in active_ids)

    # cluster -> eventi (k,typ)
    cluster_events: dict[int, list[tuple[int, str]]] = {}
    for (k, typ), c in labels_PD.items():
        k = int(k)
        c = int(c)
        typ = str(typ)

        if k not in active_set:
            continue
        if c == -1:
            continue
        if typ not in ("P", "D"):
            continue

        cluster_events.setdefault(c, []).append((k, typ))

    if not cluster_events:
        return None

    # 2) Nuova variabile u[c,m]: il modello sceglie il modulo del cluster
    # u[c,m]
    u = mdl.binary_var_dict(
        keys=[(c, m) for c in cluster_events.keys() for m in M],
        name=f"u_{name}"
    )


    # 3) un modulo per cluster
    #   ∑_{m ∈ M} u[c,m] = 1    ∀ cluster c
    for c in cluster_events.keys():
        mdl.add_constraint(
            mdl.sum(u[c, m] for m in M) == 1,
            ctname=f"{name}_one_module_c{c}"
        )

    # 4) linking: eventi del cluster possono attivare solo a/b sul modulo scelto
    # Pickup (typ == "P"):
    #   a[k,t,m] ≤ u[c,m]
    # Delivery (typ == "D"):
    #   b[k,t,m] ≤ u[c,m]
    for c, evs in cluster_events.items():
        for (k, typ) in evs:
            for t in I.DeltaT[int(k)]:
                for m in M:
                    key = (int(k), t, m)
                    if typ == "P":   # "P", si fissa a
                        if key in a:
                            mdl.add_constraint(
                                a[key] <= u[c, m],
                                ctname=f"{name}_A_c{c}_k{k}_t{t}_m{m}"
                            )
                    else:  # "D", si fissa b
                        if key in b:
                            mdl.add_constraint(
                                b[key] <= u[c, m],
                                ctname=f"{name}_B_c{c}_k{k}_t{t}_m{m}"
                            )

    return u






### Si estraggono a e b fissati dei GRIGI
def extract_solution_values_only_selected_k_ab(
    var_dicts: Dict[str, Dict[Any, Any]],
    sol: Any,
    selected_ids: Iterable[int],   # richieste da fissare (GRIGI)
) -> Dict[str, Dict[Any, float]]:
    """
    Estrae dalla soluzione SOLO le variabili a e b = 1 per le richieste in selected_ids.
    Output:
      {
        "a": {(k,t,m): 1.0, ...},
        "b": {(k,t,m): 1.0, ...}
      }
    """
    if sol is None:
        raise ValueError("sol is None")

    selected = set(int(k) for k in selected_ids)

    out: Dict[str, Dict[Any, float]] = {
        "a": {},
        "b": {},
    }

    for fam in ("a", "b"):
        if fam not in var_dicts:
            continue

        V = var_dicts[fam]   # (k,t,m) -> Var

        for (k, t, m), var in V.items():
            if int(k) not in selected:
                continue

            val = sol.get_value(var)
            if val is None:
                continue

            if float(val) >= 0.5:   # attiva
                out[fam][(int(k), t, m)] = 1.0

    return {fam: d for fam, d in out.items() if d}






### Si uniscono tutte le constraints fino a questo momento
def merge_constraints_ab(base, new, check_conflict=True, tol=1e-9):
    """
    Merge dei vincoli di fixing per le sole variabili a e b, combinando quelli già fissati con i nuovi e rilevando eventuali conflitti.
    """
    base = {} if base is None else copy.deepcopy(base)
    new  = {} if new  is None else new

    out = {"a": dict(base.get("a", {})), "b": dict(base.get("b", {}))}

    for fam in ("a", "b"):
        for key, val in new.get(fam, {}).items():
            v = float(val)
            if key in out[fam]:
                old = float(out[fam][key])
                if abs(old - v) > tol and check_conflict:   #Se lo stesso (famiglia, key) ha valori diversi (oltre tol), errore (ValueError).
                    raise ValueError(f"Conflict on {fam}{key}: old={old}, new={v}")
            out[fam][key] = v

    return {fam: d for fam, d in out.items() if d}









### Costruzione modello con fixed constraints
def build_base_model_with_fixed_constraints(
    instance,
    model_name: str,
    fixed_constr: Optional[Dict[str, Dict[tuple, float]]],
    cplex_cfg: dict | None = None,
):
    """
    Costruisce il modello BASE
    Applica SOLO i fixed constraints
    """

    ### Build model
    if model_name == "w":
        model, x, y, r, w, s, a, b, D, U, z, kappa, h = create_MT_model_w(instance)
        var_dicts = {
            "x": x, "y": y,
            "r": r, "w": w, "s": s,
            "a": a, "b": b,
            "D": D, "U": U,
            "z": z, "kappa": kappa,
            "h": h,
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    configure_cplex(model, cplex_cfg)

    ### Apply FIXED constraints (ROSA)
    if fixed_constr:
        fix_constraints_ab(model, var_dicts, fixed_constr)

    return model, var_dicts





### Funzione per mappare le variaibli quando si clona il modello
def map_vars_by_name(model, base_var_dicts):
    """
    Rimappa le variabili del clone usando i nomi.
    base_var_dicts: dict fam -> dict key -> docplex_var (del base model)
    return: dict fam -> dict key -> docplex_var (del clone)
    """
    mapped = {}
    for fam, V in base_var_dicts.items():
        mapped[fam] = {}
        for key, var in V.items():
            v2 = model.get_var_by_name(var.name)
            if v2 is None:
                raise KeyError(f"Var not found in clone: fam={fam}, key={key}, name={var.name}")
            mapped[fam][key] = v2
    return mapped




### Adatta la soluzione warmup al nuovo modello
def add_mip_start_by_name(model, start_by_name: dict | None):
    """
    Crea una SolveSolution (docplex) con i valori e la passa a add_mip_start().
    Ritorna quante variabili sono state impostate.
    """
    if not start_by_name:
        return 0

    sol = model.new_solution()   # SolveSolution legata a questo model
    n = 0

    for name, val in start_by_name.items():
        v = model.get_var_by_name(name)
        if v is None:
            continue
        sol.add_var_value(v, val)
        n += 1

    if n > 0:
        model.add_mip_start(sol)   # <-- qui ora è un oggetto Solution, non dict
    return n



### Funzione per estrarre soluzioni parziali 
def extract_mip_start_by_name(sol, var_dicts, thr=0.5):
    """
    Ritorna dict {var_name: value} valido per rimappare su qualunque clone
    che contenga variabili con gli stessi nomi.
    """
    if sol is None:
        return None

    fams_bin = ("x", "y")
    fams_int = ("D", "U", "z", "kappa") 

    start = {}

    for fam in fams_bin:
        V = var_dicts.get(fam)
        if not V:
            continue
        for var in V.values():
            val = sol.get_value(var)
            if val is None:
                continue
            start[var.name] = 1 if float(val) >= thr else 0

    for fam in fams_int:
        V = var_dicts.get(fam)
        if not V:
            continue
        for var in V.values():
            val = sol.get_value(var)
            if val is None:
                continue
            v = int(round(float(val)))
            lb = getattr(var, "lb", None)
            ub = getattr(var, "ub", None)
            if lb is not None:
                v = max(v, int(lb))
            if ub is not None:
                v = min(v, int(ub))
            start[var.name] = v

    return start






### Funzione di supporto per warm start
def fix_all_requests_except_k_ab(
    mdl,
    I,
    var_dicts,
    k_keep: list[int],
    *,
    name="onlyK",
):
    """
    Fissa a=b=0 per tutte le richieste k NON in k_keep.
    (opzionale) se esiste s[k], fissa s[k]=0 per le non keep.
    Return: numero vincoli aggiunti.
    """
    keep = set(int(k) for k in k_keep)
    M = list(I.M)

    a = var_dicts["a"]
    b = var_dicts["b"]
    s = var_dicts.get("s", None)

    n_added = 0
    for k in I.K:
        k = int(k)
        if k in keep:
            continue

        # spegni servizio per coerenza (se s esiste)
        if s is not None and k in s:
            mdl.add_constraint(s[k] == 0, ctname=f"{name}_s0_k{k}")
            n_added += 1

        # spegni a,b su tutte le finestre temporali possibili
        for t in I.DeltaT[k]:
            for m in M:
                key = (k, t, m)
                if key in a:
                    mdl.add_constraint(a[key] == 0, ctname=f"{name}_a0_k{k}_t{t}_m{m}")
                    n_added += 1
                if key in b:
                    mdl.add_constraint(b[key] == 0, ctname=f"{name}_b0_k{k}_t{t}_m{m}")
                    n_added += 1

    return n_added





### Soluzione greedy cluster-aware => n_cluster = n_richieste soddisfatte
def warmstart_one_request_per_cluster(
    I,
    labels_PD,          # {(k,"P"/"D"): c}
    active_ids,         # richieste attive
    name="cl_mod",
    seed=0,
):
    rng = random.Random(seed)

    M = list(I.M)
    active = set(int(k) for k in active_ids)

    # cluster -> set richieste
    cl2ks = {}
    for (k, typ), c in labels_PD.items():
        k = int(k); c = int(c)
        if k not in active: 
            continue
        cl2ks.setdefault(c, set()).add(k)

    clusters = sorted(cl2ks.keys())
    rng.shuffle(clusters)

    # assegna moduli diversi finché possibile
    mods = M[:]
    rng.shuffle(mods)

    start = {}

    for idx, c in enumerate(clusters):
        if not cl2ks[c]:
            continue

        k = rng.choice(list(cl2ks[c]))
        m = mods[idx % len(mods)]  # diverso finché #clusters <= #mods

        # u[c,m]=1 e gli altri a 0 (opzionale ma meglio)
        for mm in M:
            start[f"u_{name}_{c}_{mm}"] = 1 if mm == m else 0

        # scegli un tempo semplice
        t0 = min(I.DeltaT[k])

        # attiva pickup e delivery (1 sola scelta) sul modulo m
        start[f"a_{k}_{t0}_{m}"] = 1
        start[f"b_{k}_{t0}_{m}"] = 1

    return start

