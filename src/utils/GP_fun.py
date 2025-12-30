from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import mean, pvariance
from typing import Dict, Tuple, List, Optional, Iterable


LabelKey = Tuple[int, str]          # (k, "P"/"D")
LabelsPD = Dict[LabelKey, int]      # -> cluster_id


### Distanza euclidea
def _eucl(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return sqrt(dx * dx + dy * dy)



### Estrazione campi di un evento
def _get_event_fields_xy(e: dict):
    """
    Evento 4D come in build_events_4d_from_req7d:
      {"k":.., "type":"P/D", "x":.., "y":.., "t":.., "q":..}
    """
    k = int(e["k"])
    typ = str(e["type"])            # "P" / "D"
    x = float(e["x"])
    y = float(e["y"])
    t = float(e["t"])
    q = float(e.get("q", 0.0))
    return k, typ, x, y, t, q



### Dato un clustering, si va a calcolare le features
def compute_cluster_features(
    events4d: List[dict],                           # lista di eventi (pickup e delivery separati)
    labels_PD: LabelsPD,                            # mapping (k,"P"/"D") -> cluster_id
    node_xy: Dict[int, Tuple[float, float]],        # mapping node_id -> (x,y)
    swap_nodes: Iterable[int] = None,               # nodi Nw, da swap_nodes = I.Nw
) -> Dict[int, Dict[str, float]]:
    """
    Clustering features over EVENTS (P/D separate)
    """
    swap_nodes = list(swap_nodes) if swap_nodes is not None else []   # Se swap_nodes è dato, lo converte in lista, altrimenti lista vuota.
    swap_xy = [node_xy[n] for n in swap_nodes if n in node_xy]        # Calcola le coordinate (x,y) dei nodi di scambio.

    # Si ragruppa gli eventi in buckets (uno per ogni cluster)
    buckets: Dict[int, List[Tuple[int, str, float, float, float, float]]] = defaultdict(list)
    for e in events4d:
        k, typ, x, y, t, q = _get_event_fields_xy(e)
        c = int(labels_PD[(k, typ)])         
        buckets[c].append((k, typ, x, y, t, q))

    # Dizionario output: cluster_id  →  { nome_feature → valore }
    out: Dict[int, Dict[str, float]] = {}


    ### Costruzioni features
    for c, lst in buckets.items():
        # Quante richieste diverse si hanno in ogni cluster (non quanti eventi).
        ks = {k for (k, _, _, _, _, _) in lst}
        n_req = float(len(ks)) 

        # Estrazione coordinate x e y di ogni evento
        xs = [x for (_, _, x, _, _, _) in lst]
        ys = [y for (_, _, _, y, _, _) in lst]
        
        # Estrazione dei tempi degli eventi
        ts = [t for (_, _, _, _, t, _) in lst]

        # Estrazione delle quantità degli eventi
        qs = [q for (_, _, _, _, _, q) in lst]

        # Calcolo centroide spaziale del cluster: media delle x e media delle y
        cx = mean(xs)
        cy = mean(ys)
        centroid = (cx, cy)

        # Calcolo distanza dal centroide per ogni evento
        d_cent = [_eucl((x, y), centroid) for x, y in zip(xs, ys)]

        # Calcolo raggio medio (distanza media dal centroide)
        r_mean = mean(d_cent)
        # Calcolo raggio massimo (distanza massima dal centroide)
        r_max = max(d_cent)
        # Calcolo "varianza spaziale" definita come media delle distanze al quadrato
        var_sp = mean([d * d for d in d_cent])

        # Calcolo range temporale (differenza tra max tempo e min tempo)
        t_min = min(ts)
        t_max = max(ts)
        t_range = t_max - t_min

        # Calcolo varianza temporale
        var_t = pvariance(ts) if len(ts) > 1 else 0.0

        # Calcolo capacità totale come somma delle q degli eventi del cluster
        cap_tot = sum(qs)

        # Calcolo per ogni evento della distanza minima/massima/media dal nodo di scambio più vicino
        if swap_xy:     # Se si hanno dei nodi di scambio Nw
            d_swap = []
            for (_, _, x, y, _, _) in lst:
                p = (x, y)
                dmin = min(_eucl(p, sxy) for sxy in swap_xy)
                d_swap.append(dmin)
            min_sc = min(d_swap)
            avg_sc = mean(d_swap)
            max_sc = max(d_swap)
        else:   # Tutto 0.0 se non si hanno  Nw
            min_sc = avg_sc = max_sc = 0.0

        ### Dizionario delle features
        out[c] = {
            "posizione_media_x": float(cx),
            "posizione_media_y": float(cy),
            "raggio_mean": float(r_mean),
            "raggio_max": float(r_max),
            "varianza_spaziale": float(var_sp),
            "range_temporale": float(t_range),
            "varianza_temporale": float(var_t),
            "capacita_totale": float(cap_tot),
            "numero_richieste": float(n_req),
            "min_dist_scambio": float(min_sc),
            "avg_dist_scambio": float(avg_sc),
            "max_dist_scambio": float(max_sc),
        }

    return out

    


    
