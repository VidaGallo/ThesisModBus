"""
Network Generator
=================

Creates the base transportation network used in the experiments.
Supports two modes:

1. GRID network:
   - Builds a side × side grid

2. CITY network:
   - Loaded from OSM and normalized to the same JSON structure

Output JSON contains:
    - nodes
    - edges
    - metadata (e.g., type, grid size)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import random
import osmnx as ox
import networkx as nx
import geopandas as gpd
import os


# seed
import random
import numpy as np
seed = 23
random.seed(seed)
np.random.seed(seed)



def save_network_json(network_dict: dict, output_dir: str, filename: str = "network.json") -> None:
    """
    Save a network dictionary to a JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(network_dict, f, indent=2)

    #print(f"[INFO] Saved network to: {out_path}")





#################
# GRID GENERATOR:
#################

def generate_grid_network(
    side: int,
    edge_length_km: float = 1.0,
    speed_kmh: float = 40.0,
) -> dict:
    
    """
    Generate a side × side GRID network.

    Parameters:
        side : int
            Number of nodes per side (total nodes = side * side).
        edge_length_km : float
            Length of each edge in kilometers (default: 1 km).
        speed_kmh : float
            Travel speed in km/h used to compute travel time (default: 40 km/h).
    """

    ### Nodes ###
    nodes = []
    for row in range(side):
        for col in range(side):
            node_id = row * side + col
            nodes.append({
                "id": node_id,
                "row": row,    # x 
                "col": col     # y
            })

    ### Travel time per edge (in minutes) ###
    time_hours = edge_length_km / speed_kmh          # h
    time_minutes = time_hours * 60.0                 # min (= ready for the discretization)

    
    ### Edges (4-neighbors: up, down, left, right) ###
    edges = []

    def node_id_from_rc(r: int, c: int) -> int:    # To give an ID to each node 
        return r * side + c

    directions = [
        (1, 0),   # down
        (-1, 0),  # up
        (0, 1),   # right
        (0, -1),  # left
    ]

    for row in range(side):
        for col in range(side):
            u = node_id_from_rc(row, col)
            for dr, dc in directions:
                r2 = row + dr
                c2 = col + dc
                if 0 <= r2 < side and 0 <= c2 < side:   # check for the borders
                    v = node_id_from_rc(r2, c2)
                    edges.append({              # append the edges 
                        "u": u,
                        "v": v,
                        "length_km": float(edge_length_km),
                        "time_min": float(time_minutes),
                    })

    
    ### Creation of the network dict
    network_dict = {
        "type": "GRID",
        "side": side,
        "nodes": nodes,
        "edges": edges,
        "speed_kmh": float(speed_kmh),
    }
    return network_dict



def generate_grid_network_asym(
    side: int,
    edge_length_km: float = 1.0,
    speed_kmh: float = 40.0,
    rel_std: float = 0.2,
) -> dict:
    """
    Generate a side × side GRID network with asymmetric edge lengths.

    Each directed edge (u -> v) gets a length drawn from a normal
    distribution with mean = edge_length_km and std = rel_std * edge_length_km.

    Parameters:
        side : int
            Number of nodes per side (total nodes = side * side).
        edge_length_km : float
            Base mean length of each edge in kilometers (default: 1 km).
        speed_kmh : float
            Travel speed in km/h used to compute travel time.
        rel_std : float
            Relative standard deviation for the normal noise
            (e.g. 0.2 -> std = 0.2 * edge_length_km).
    """

    # --- Nodes ---
    nodes = []
    for row in range(side):
        for col in range(side):
            node_id = row * side + col
            nodes.append({
                "id": node_id,
                "row": row,    # x
                "col": col     # y
            })

    def node_id_from_rc(r: int, c: int) -> int:
        return r * side + c

    directions = [
        (1, 0),   # down
        (-1, 0),  # up
        (0, 1),   # right
        (0, -1),  # left
    ]

    edges = []

    # funzione di supporto per campionare una lunghezza > 0
    def sample_length():
        mu = edge_length_km
        sigma = rel_std * edge_length_km
        # campiona finché non ottieni una lunghezza positiva
        length = -1.0
        while length <= 0:
            length = random.gauss(mu, sigma)
        return float(length)

    for row in range(side):
        for col in range(side):
            u = node_id_from_rc(row, col)
            for dr, dc in directions:
                r2 = row + dr
                c2 = col + dc
                if 0 <= r2 < side and 0 <= c2 < side:
                    v = node_id_from_rc(r2, c2)

                    # lunghezza asimmetrica (per ogni direzione)
                    length_km = sample_length()

                    # travel time (in minuti) per questo arco specifico
                    time_hours = length_km / speed_kmh
                    time_minutes = time_hours * 60.0

                    edges.append({
                        "u": u,
                        "v": v,
                        "length_km": length_km,
                        "time_min": float(time_minutes),
                    })

    network_dict = {
        "type": "GRID_ASYM",
        "side": side,
        "nodes": nodes,
        "edges": edges,
        "speed_kmh": float(speed_kmh),
        "base_edge_length_km": float(edge_length_km),
        "rel_std": float(rel_std),
    }
    return network_dict





############################
# NETWORK GENERATOR - TURIN:
############################
def plot_city_suburb_network(
    osm_graph,
    network_dict: dict,
    place: str,
    output_dir: str,
    filename: str = "torino_suburbs.png",
    title_suffix: str = "",
) -> None:
    """
    Plot a macro city network (suburbs as nodes, macro edges on top of OSM graph).
    """

    plt.switch_backend("Agg")

    # Base OSM background
    fig, ax = ox.plot_graph(osm_graph, show=False, close=False, bgcolor="white")

    # Build mapping id -> (lon, lat)
    nodes = network_dict["nodes"]
    edges = network_dict["edges"]

    id_to_coord = {
        n["id"]: (n["lon"], n["lat"])
        for n in nodes
    }

    # Plot macro edges
    for e in edges:
        u = e["u"]
        v = e["v"]
        if u not in id_to_coord or v not in id_to_coord:
            continue

        lon_u, lat_u = id_to_coord[u]
        lon_v, lat_v = id_to_coord[v]

        ax.plot(
            [lon_u, lon_v],
            [lat_u, lat_v],
            linewidth=0.8,
            alpha=0.5,
        )

    # Plot centroids with labels
    for n in nodes:
        lon, lat = n["lon"], n["lat"]
        ax.scatter(lon, lat, c="red", s=40)
        ax.text(lon, lat, n["name"], fontsize=9)

    # Title
    if title_suffix:
        full_title = f"Suburbs + macro edges for {place} {title_suffix}"
    else:
        full_title = f"Suburbs + macro edges for {place}"

    plt.title(full_title)
    plt.tight_layout()

    # Save to specific folder
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=500)
    print(f"Saved plot as {output_path}")



### REMARK: OSM has the info of 32 Turin suburbs under the "suburb" tag
def generate_city_network_raw(
    place: str,
    default_speed_kmh: float = 40.0,
    plot: bool = True,
    k_lateral: int = 2,      
    k_center: int = 4,
    central_suburbs: list = None,
    output_dir: str = "instances/CITY/TORINO_SUB",
    filename: str = "torino_suburbs_raw.png",
) -> tuple[dict, nx.MultiDiGraph]:
    """
    Generate a macro CITY network for a given place based on OSM suburbs.

    - Each node is a suburb (place=suburb) in the given place.
    - Each suburb is mapped to the nearest OSM street node (centroid).
    - Undirected edges are built as:
        * central suburbs  -> connected to their k_center nearest suburbs
        * non-central ones -> connected to their k_lateral nearest suburbs

    Parameters
    ----------
    place : str
        Name of the place to query in OSM (e.g. "Torino, Italia").
    default_speed_kmh : float
        Speed used to compute travel time if needed.
    plot : bool
        If True, plot the city graph, suburbs and macro edges.
    k_lateral : int
        Number of nearest neighbors for central suburbs. 
    k_center : int
        Number of nearest neighbors for central suburbs.
    central_suburbs : list
        List of suburb names considered as "central hubs".

    Returns
    -------
    network_dict : dict
    """
    if central_suburbs is None:
        central_suburbs = []

    # ------------------------
    # 1) Download city graph
    # ------------------------
    osm_graph = ox.graph_from_place(
        place,
        network_type="drive",
        simplify=True,
    )

    # ------------------------
    # 2) Get suburbs polygons
    # ------------------------
    tags = {"place": ["suburb"]}
    gdf = ox.features_from_place(place, tags=tags)

    gdf_sub = gdf[["name", "geometry"]].dropna(subset=["name", "geometry"])
    gdf_grouped = gdf_sub.dissolve(by="name")  # index = name

    # ------------------------
    # 3) Centroids -> nearest OSM node
    # ------------------------
    suburb_nodes = {}   # name -> osm node id
    node_entries = []   # list of dicts for "nodes" in JSON

    for name, row in gdf_grouped.iterrows():
        centroid = row["geometry"].centroid
        lat, lon = centroid.y, centroid.x

        osm_node = ox.distance.nearest_nodes(osm_graph, X=lon, Y=lat)

        suburb_nodes[name] = osm_node
        node_entries.append({
            "id": int(osm_node),
            "name": str(name),
            "lat": float(lat),
            "lon": float(lon),
        })

    # ------------------------------------------------------------
    # Edge construction: central = k_center, lateral = k_lateral
    # ------------------------------------------------------------
    suburb_names = sorted(suburb_nodes.keys())

    # Mapping: suburb name -> (lat, lon)
    coords = {entry["name"]: (entry["lat"], entry["lon"]) for entry in node_entries}

    def geo_dist2(a, b):
        """
        Squared geographic distance between two centroids (lat, lon).
        We use squared distance because it preserves ordering and avoids sqrt().
        """
        (lat1, lon1), (lat2, lon2) = a, b
        return (lat1 - lat2) ** 2 + (lon1 - lon2) ** 2

    # Undirected adjacency list: suburb_name -> set(neighbor_names)
    adjacency = {name: set() for name in suburb_names}

    # ------------------------------------------------------------
    # (1) INITIAL STEP: each suburb picks its nearest neighbors
    #     - central suburbs use k_center
    #     - lateral suburbs use k_lateral
    # ------------------------------------------------------------
    for name_u in suburb_names:
        cu = coords[name_u]

        # Build candidate list: all *other* suburbs with their distance
        cand = []
        for name_v in suburb_names:
            if name_v == name_u:
                continue
            cv = coords[name_v]
            d2 = geo_dist2(cu, cv)
            cand.append((name_v, d2))

        # Sort candidates by proximity
        cand.sort(key=lambda x: x[1])

        # Decide how many neighbors to keep (central vs lateral)
        if name_u in central_suburbs:
            k = min(k_center, len(cand))
        else:
            k = min(k_lateral, len(cand))

        # List of the selected nearest neighbor names
        chosen = [name_v for (name_v, _) in cand[:k]]

        # Update adjacency (undirected)
        for name_v in chosen:
            adjacency[name_u].add(name_v)
            adjacency[name_v].add(name_u)

    # ------------------------------------------------------------
    # (2) ADJUSTMENT STEP:
    #     Try to enforce:
    #       - central suburbs: exactly k_center neighbors
    #       - non-central suburbs: exactly k_lateral neighbors
    # ------------------------------------------------------------
    for name_u in suburb_names:
        # target degree for this suburb
        if name_u in central_suburbs:
            target_deg = k_center
        else:
            target_deg = k_lateral

        cu = coords[name_u]

        # ----------------------------
        # (2a) Se ha meno del target:
        #      aggiungi vicini mancanti
        # ----------------------------
        neighbors = list(adjacency[name_u])
        if len(neighbors) < target_deg:
            # candidati = tutti gli altri suburb che non sono già vicini
            cand = []
            for name_v in suburb_names:
                if name_v == name_u:
                    continue
                if name_v in adjacency[name_u]:
                    continue
                cv = coords[name_v]
                d2 = geo_dist2(cu, cv)
                cand.append((name_v, d2))

            # ordina per distanza geometrica
            cand.sort(key=lambda x: x[1])

            # aggiungi i più vicini finché non raggiungi target_deg
            for (name_v, _) in cand:
                adjacency[name_u].add(name_v)
                adjacency[name_v].add(name_u)  # grafo non orientato
                neighbors.append(name_v)
                if len(neighbors) >= target_deg:
                    break

        # ----------------------------
        # (2b) Se ha più del target:
        #      tieni solo i target_deg più vicini
        # ----------------------------
        neighbors = list(adjacency[name_u])  # aggiorna dopo eventuali aggiunte
        if len(neighbors) > target_deg:
            # ordina i vicini per distanza
            neighbors.sort(key=lambda name_v: geo_dist2(cu, coords[name_v]))

            keep = set(neighbors[:target_deg])
            remove = set(neighbors[target_deg:])

            for name_v in remove:
                adjacency[name_u].discard(name_v)
                adjacency[name_v].discard(name_u)



    # ------------------------------------------------------------
    # (3) Convert adjacency structure into final edge list
    #     Each edge (u,v) is undirected → only one entry per pair.
    #     Distances come from OSM shortest-path length.
    # ------------------------------------------------------------
    edges = []
    seen_pairs = set()  # prevent duplicates of (u,v) and (v,u)

    for name_u in suburb_names:
        u_node = suburb_nodes[name_u]
        for name_v in adjacency[name_u]:
            v_node = suburb_nodes[name_v]

            # Normalize undirected pair
            pair_key = tuple(sorted((u_node, v_node)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Try shortest path u→v; if no path, try v→u
            try:
                length_m = nx.shortest_path_length(
                    osm_graph, source=u_node, target=v_node, weight="length"
                )
            except nx.NetworkXNoPath:
                try:
                    length_m = nx.shortest_path_length(
                        osm_graph, source=v_node, target=u_node, weight="length"
                    )
                except nx.NetworkXNoPath:
                    continue  # no path exists in either direction → skip

            length_km = float(length_m) / 1000.0
            time_min = (length_km / default_speed_kmh) * 60.0

            edges.append({
                "u": int(u_node),
                "v": int(v_node),  # undirected edge represented as (u,v)
                "length_km": length_km,
                "time_min": float(time_min),
            })



    network_dict = {
        "type": "CITY_SUBURB_UNDIRECTED",
        "place": place,
        "default_speed_kmh": float(default_speed_kmh),
        "nodes": node_entries,
        "edges": edges,
    }

    # --------------
    # Optional: plot
    # --------------
    if plot:
        plot_city_suburb_network(
            osm_graph=osm_graph,
            network_dict=network_dict,
            place=place,
            output_dir=output_dir,
            filename=filename,
            title_suffix="(raw)",
        )

    return network_dict, osm_graph


def generate_city_network_merged(
    osm_graph,
    base_network: dict,
    merge_groups: list[tuple[str, ...]],
    place: str,
    plot: bool = True,
    output_dir: str = "/home/vida/Desktop/TESI/thesis_code/instances/CITY/TORINO_SUB",
    filename: str = "torino_suburbs_merged.png",
) -> dict:
    """
    Merge groups of suburbs into single macro-nodes.

    Parameters
    ----------
    osm_graph : networkx.MultiDiGraph
        Grafo OSM originale (serve solo per il plot finale).
    base_network : dict
        Network dict con chiavi "nodes" e "edges" (output di generate_city_network_raw).
    merge_groups : list[tuple[str, ...]]
        Ogni tupla è un gruppo di quartieri da unire, es:
            [("Vanchiglietta", "Madonna del Pilone"),
             ("Vallette", "Lucento", "Borgo Vittoria")]
    place : str
        Nome del posto (per il titolo del plot).
    """

    nodes = base_network["nodes"]
    edges = base_network["edges"]

    # Map name -> node dict
    name_to_node = {n["name"]: n for n in nodes}

    # Track which original node IDs are merged into which new node ID
    old_id_to_new_id = {}

    new_nodes = []
    used_ids = {n["id"] for n in nodes}
    next_id = max(used_ids) + 1 if used_ids else 0

    # 1) copia tutti i nodi NON mergiati
    merged_names_flat = {name for group in merge_groups for name in group}
    for n in nodes:
        if n["name"] not in merged_names_flat:
            new_nodes.append(n)

    # 2) crea un nodo mergiato per ogni gruppo
    for group in merge_groups:
        group_nodes = []
        for name in group:
            if name not in name_to_node:
                print(f"[WARN] merge_suburbs: suburb '{name}' not found in nodes.")
                continue
            group_nodes.append(name_to_node[name])

        if not group_nodes:
            continue

        # Centroid (media lat/lon)
        avg_lat = sum(n["lat"] for n in group_nodes) / len(group_nodes)
        avg_lon = sum(n["lon"] for n in group_nodes) / len(group_nodes)

        # Nome del nodo mergiato
        base_name = " + ".join(n["name"] for n in group_nodes)
        merged_name = base_name

        merged_id = next_id
        next_id += 1

        # Mappa gli id vecchi -> nuovo id
        for n in group_nodes:
            old_id_to_new_id[n["id"]] = merged_id

        new_nodes.append({
            "id": merged_id,
            "name": merged_name,
            "lat": float(avg_lat),
            "lon": float(avg_lon),
        })

    # 3) Ricollega gli archi ai nuovi id, eliminando self-loop e duplicati
    new_edges = []
    seen_pairs = set()  # per evitare duplicati (u,v) e (v,u) per grafo non orientato

    for e in edges:
        u = e["u"]
        v = e["v"]

        u_new = old_id_to_new_id.get(u, u)
        v_new = old_id_to_new_id.get(v, v)

        # Self-loop creati dal merge → via
        if u_new == v_new:
            continue

        pair_key = tuple(sorted((u_new, v_new)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        new_edges.append({
            "u": int(u_new),
            "v": int(v_new),
            "length_km": e["length_km"],
            "time_min": e["time_min"],
        })

    new_network = dict(base_network)
    new_network["nodes"] = new_nodes
    new_network["edges"] = new_edges
    new_network["type"] = "CITY_SUBURB_MERGED"

    # Plot opzionale
    if plot:
        plot_city_suburb_network(
            osm_graph=osm_graph,
            network_dict=new_network,
            place=place,
            output_dir=output_dir,
            filename=filename,
            title_suffix="(merged)",
        )

    return new_network


def generate_city_network_reworked(
    osm_graph,
    base_network: dict,
    place: str,
    # possono essere nomi (str) o id (int)
    remove_edges: list[tuple[object, object]] | None = None,
    add_edges: list[tuple[object, object]] | None = None,
    default_speed_kmh: float | None = None,
    plot: bool = True,
    output_dir: str = "/home/vida/Desktop/TESI/thesis_code/instances/CITY/TORINO_SUB",
    filename: str = "torino_suburbs_final.png",
) -> dict:
    """
    Terzo step: pulizia manuale del grafo.
    - remove_edges / add_edges possono usare nomi o id dei nodi del base_network.
    - Per le distanze, ogni nodo (anche mergiato) viene mappato a un nodo OSM
      usando lat/lon e ox.distance.nearest_nodes.
    """
    if remove_edges is None:
        remove_edges = []
    if add_edges is None:
        add_edges = []

    nodes = list(base_network["nodes"])
    edges = list(base_network["edges"])

    # -----------------------------
    # 1) Mapping name <-> id (MERGED)
    # -----------------------------
    name_to_id = {n["name"]: n["id"] for n in nodes}

    def resolve_node(ref):
        """Accetta un nome (str) o un id (int) e restituisce sempre l'id MERGED."""
        if isinstance(ref, int):
            return ref
        if isinstance(ref, str):
            if ref not in name_to_id:
                raise ValueError(f"Node name '{ref}' not found in network nodes.")
            return name_to_id[ref]
        raise TypeError(f"Node reference must be int or str, got {type(ref)}")

    # -----------------------------
    # 2) Mapping id MERGED -> id OSM
    # -----------------------------
    # Alcuni id sono già nodi OSM, altri (mergiati) no.
    # Per tutti, associamo un osm_id valido (usando lat/lon se serve).
    nodeid_to_osmid = {}
    for n in nodes:
        nid = n["id"]
        lat = n["lat"]
        lon = n["lon"]

        if nid in osm_graph.nodes:
            osm_id = nid
        else:
            # nodo mergiato: trova il nodo OSM più vicino
            osm_id = ox.distance.nearest_nodes(osm_graph, X=lon, Y=lat)

        nodeid_to_osmid[nid] = osm_id

    # -----------------------------
    # 3) Normalizza remove/add in termini di id MERGED
    # -----------------------------
    remove_id_pairs = []
    for (u_ref, v_ref) in remove_edges:
        u_id = resolve_node(u_ref)
        v_id = resolve_node(v_ref)
        remove_id_pairs.append(tuple(sorted((u_id, v_id))))

    add_id_pairs = []
    for (u_ref, v_ref) in add_edges:
        u_id = resolve_node(u_ref)
        v_id = resolve_node(v_ref)
        add_id_pairs.append(tuple(sorted((u_id, v_id))))

    remove_set = set(remove_id_pairs)
    add_set = set(add_id_pairs)

    if default_speed_kmh is None:
        default_speed_kmh = float(base_network.get("default_speed_kmh", 40.0))

    # -----------------------------
    # 4) Rimuovi archi indesiderati
    # -----------------------------
    new_edges = []
    for e in edges:
        pair = tuple(sorted((e["u"], e["v"])))
        if pair in remove_set:
            continue
        new_edges.append(e)

    # -----------------------------
    # 5) Aggiungi archi nuovi (usando OSM per la distanza)
    # -----------------------------
    existing_pairs = {tuple(sorted((e["u"], e["v"]))) for e in new_edges}

    for (u_id, v_id) in add_set:
        pair = tuple(sorted((u_id, v_id)))
        if pair in existing_pairs:
            continue  # già esiste

        # recupera i nodi OSM corrispondenti (sempre validi)
        u_osm = nodeid_to_osmid[u_id]
        v_osm = nodeid_to_osmid[v_id]

        # prova shortest path u_osm → v_osm
        try:
            length_m = nx.shortest_path_length(
                osm_graph, source=u_osm, target=v_osm, weight="length"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                length_m = nx.shortest_path_length(
                    osm_graph, source=v_osm, target=u_osm, weight="length"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"[WARN] No path between merged nodes {u_id} and {v_id} (OSM {u_osm}-{v_osm}), skipping.")
                continue

        length_km = float(length_m) / 1000.0
        time_min = (length_km / default_speed_kmh) * 60.0

        new_edges.append({
            "u": int(u_id),      # id MERGED nel tuo network
            "v": int(v_id),
            "length_km": length_km,
            "time_min": float(time_min),
        })
        existing_pairs.add(pair)

    network_dict = {
        "type": "CITY_SUBURB_FINAL",
        "place": place,
        "default_speed_kmh": float(default_speed_kmh),
        "nodes": nodes,
        "edges": new_edges,
    }

    # Plot opzionale
    if plot:
        plot_city_suburb_network(
            osm_graph=osm_graph,
            network_dict=network_dict,
            place=place,
            output_dir=output_dir,
            filename=filename,
            title_suffix="(final)",
        )

    return network_dict


def make_network_directed(network_dict: dict) -> dict:
    """
    Prende un network con archi non orientati (u,v) unici
    e restituisce un nuovo network con archi orientati:
        (u,v) e (v,u) con stessi length_km e time_min.
    """
    nodes = list(network_dict["nodes"])
    undirected_edges = network_dict["edges"]

    directed_edges = []
    for e in undirected_edges:
        u = e["u"]
        v = e["v"]
        length_km = e["length_km"]
        time_min = e["time_min"]

        # arco u -> v
        directed_edges.append({
            "u": int(u),
            "v": int(v),
            "length_km": float(length_km),
            "time_min": float(time_min),
        })
        # arco v -> u (ritorno)
        directed_edges.append({
            "u": int(v),
            "v": int(u),
            "length_km": float(length_km),
            "time_min": float(time_min),
        })

    new_network = dict(network_dict)
    new_network["type"] = network_dict.get("type", "") + "_DIRECTED"
    new_network["edges"] = directed_edges
    new_network["is_directed"] = True

    return new_network


def generate_grid_network_city(
    place: str = "Torino, Italia",
    central_suburbs: list[str] = ["Centro", "Crocetta", "Santa Rita", "Aurora"],
    speed_kmh: float = 40.0
):

    # 1) RAW network
    city_network_raw, osm_graph = generate_city_network_raw(
        place=place,
        default_speed_kmh=40.0,
        plot=True,
        k_lateral=2,
        k_center=4,
        central_suburbs=central_suburbs
    )

    merge_groups = [
        ("Vanchiglia", "Vanchiglietta", "Madonna del Pilone"),
        ("Madonna di Campagna", "Borgo Vittoria"),
        ("San Donato", "Campidoglio", "Cit Turin"),
        ("Cenisia", "Borgo San Paolo"),
        ("Regio Parco", "Barca"),
        ("Lingotto", "Filadelfia"),
        ("Parella", "Pozzo Strada"),
        ("Vallette", "Lucento"),
        ("Villaretto", "Falchera"),
        ("Mirafiori Sud", "Mirafiori Nord")

    ]

    # 2) MERGED network (starting from RAW)
    city_network_merged = generate_city_network_merged(
        osm_graph=osm_graph,
        base_network=city_network_raw,
        merge_groups=merge_groups,
        place=place,
        plot=True,
    )

    remove_edges = [
        ("San Donato + Campidoglio + Cit Turin", "Crocetta"),
        ("Rebaudengo", "Regio Parco + Barca"),
        ("Borgo Po", "Vanchiglia + Vanchiglietta + Madonna del Pilone"),
        ("Madonna di Campagna + Borgo Vittoria", "Aurora"),
        ("Vallette + Lucento", "Barriera di Lanzo"),
        ("Vallette + Lucento", "Parella + Pozzo Strada"),
        ("Santa Rita",  "Cenisia + Borgo San Paolo")
    ]
    add_edges = [
        ("San Donato + Campidoglio + Cit Turin", "Centro"),
        ("Centro", "Crocetta"),
        #("Vallette + Lucento", "San Donato + Campidoglio + Cit Turin"),
        ("Regio Parco + Barca", "Barriera di Milano"),
        ("Santa Rita", "Crocetta"),
        #("San Donato + Campidoglio + Cit Turin", "Aurora"),
        #("Vanchiglia + Vanchiglietta + Madonna del Pilone", "Aurora"),
        ("Madonna di Campagna + Borgo Vittoria", "Barriera di Milano"),
        ("Vallette + Lucento", "Madonna di Campagna + Borgo Vittoria"),
        ("San Donato + Campidoglio + Cit Turin", "Madonna di Campagna + Borgo Vittoria") 
    ]

    # 3) REWORKED network (starting from MERGED)
    city_network_final = generate_city_network_reworked(
        osm_graph=osm_graph,
        base_network=city_network_merged,
        place=place,
        remove_edges=remove_edges,
        add_edges=add_edges,
        plot=True,
    )

    # 4) Convert to directed 
    city_network_directed = make_network_directed(city_network_final)

    # 5) Print
    num_nodes = len(city_network_directed["nodes"])
    num_edges = len(city_network_directed["edges"])

    print(f"# N°NODES:    {num_nodes}")
    print(f"# N°EDGES:   {num_edges}")

    # 6) Network dict
    nodes = city_network_directed["nodes"]
    edges = city_network_directed["edges"]

    network_dict = {
        "type": "CITY",
        "place": place,
        "nodes": nodes,          
        "edges": edges,          
        "speed_kmh": float(speed_kmh),
    }
    return network_dict





### TEST MAIN ###
if __name__ == "__main__":
    place = "Torino, Italia",
    name = "Torino",
    city_output_folder = "instances/CITY/TORINO_SUB",
    central_suburbs = ["Centro", "Crocetta", "Santa Rita", "Aurora"],

    # 1) RAW network
    city_network_raw, osm_graph = generate_city_network_raw(
        place=place,
        default_speed_kmh=40.0,
        plot=True,
        k_lateral=2,
        k_center=4,
        central_suburbs=central_suburbs
    )

    

    merge_groups = [
        ("Vanchiglia", "Vanchiglietta", "Madonna del Pilone"),
        ("Madonna di Campagna", "Borgo Vittoria"),
        ("San Donato", "Campidoglio", "Cit Turin"),
        ("Cenisia", "Borgo San Paolo"),
        ("Regio Parco", "Barca"),
        ("Lingotto", "Filadelfia"),
        ("Parella", "Pozzo Strada"),
        ("Vallette", "Lucento"),
        ("Villaretto", "Falchera")

    ]

    # 2) MERGED network (starting from RAW)
    city_network_merged = generate_city_network_merged(
        osm_graph=osm_graph,
        base_network=city_network_raw,
        merge_groups=merge_groups,
        place=place,
        plot=True,
    )

    remove_edges = [
        ("San Donato + Campidoglio + Cit Turin", "Crocetta"),
        ("Rebaudengo", "Regio Parco + Barca"),
        ("Borgo Po", "Vanchiglia + Vanchiglietta + Madonna del Pilone"),
        ("Madonna di Campagna + Borgo Vittoria", "Aurora"),
        ("Vallette + Lucento", "Barriera di Lanzo"),
        ("Vallette + Lucento", "Parella + Pozzo Strada"),
        ("Santa Rita",  "Cenisia + Borgo San Paolo")
    ]
    add_edges = [
        ("San Donato + Campidoglio + Cit Turin", "Centro"),
        ("Centro", "Crocetta"),
        #("Vallette + Lucento", "San Donato + Campidoglio + Cit Turin"),
        ("Regio Parco + Barca", "Barriera di Milano"),
        ("Santa Rita", "Crocetta"),
        #("San Donato + Campidoglio + Cit Turin", "Aurora"),
        #("Vanchiglia + Vanchiglietta + Madonna del Pilone", "Aurora"),
        ("Madonna di Campagna + Borgo Vittoria", "Barriera di Milano"),
        ("Vallette + Lucento", "Madonna di Campagna + Borgo Vittoria"),
        ("San Donato + Campidoglio + Cit Turin", "Madonna di Campagna + Borgo Vittoria") 
    ]

    # 3) REWORKED network (starting from MERGED)
    city_network_final = generate_city_network_reworked(
        osm_graph=osm_graph,
        base_network=city_network_merged,
        place=place,
        remove_edges=remove_edges,
        add_edges=add_edges,
        plot=True,
    )

    # 4) Convert to directed 
    city_network_directed = make_network_directed(city_network_final)

    # 5) Print
    num_nodes = len(city_network_directed["nodes"])
    num_edges = len(city_network_directed["edges"])
    print(f"# NODI FINALI:    {num_nodes}")
    print(f"# ARCHI FINALI:   {num_edges} (diretti, andata+ritorno)")

    # 6) Save
    save_network_json(
        network_dict=city_network_directed,
        output_dir=city_output_folder,
        filename=f"network_{name}_directed.json",
    )










#################################################
# NETWORK GENERATOR - COEARSENING/SPARSIFICATION:
#################################################


"""
PSEUDOCODE:
- if there are suburb in labels => use suburb , otherwise use some coarsening algorithm
- then use sparsification: keep k nearest neighbors for each node
   + consider threshold => above treshold do not connect if there are already some neighborhoods even if less thank k,
                        => below threshold connect also more then k if the supernode is central
- to define CENTRAL we use centrality (prima betweenness centrality + poi closeness centrality, non basta la degree centrality)
"""



def get_osm_zones_or_none(place: str):
    """
    Try multiple OSM tag strategies to extract city subdivisions (zones).
    Returns a GeoDataFrame dissolved by name (index=name) or None if nothing useful found.
    """
    tag_candidates = [
        {"place": ["suburb", "neighbourhood", "quarter", "borough"]},
        {"boundary": ["administrative"]},  # spesso richiede filtro extra (admin_level)
    ]

    # gdf = GeoDataFrame (ogni riga = un oggetto OSM (poligono, multipoligono, ecc.))
    for tags in tag_candidates:
        try:
            gdf = ox.features_from_place(place, tags=tags)     # funzione che controlla le features del grafo senza doverlo scaricare
        except Exception as e:
            print(f"[OSM] errore query tags={tags}: {e}")
            continue

        if gdf is None or len(gdf) == 0:
            continue

        # Teniamo solo oggetti con name e geometry validi
        keep_cols = [c for c in ["name", "geometry", "admin_level"] if c in gdf.columns]
        gdf_sub = gdf[keep_cols].dropna(subset=["geometry"])

        # Se "name" manca o è vuoto, non è utile come “zona”
        if "name" not in gdf_sub.columns:
            continue
        gdf_sub = gdf_sub.dropna(subset=["name"])
        gdf_sub = gdf_sub[gdf_sub["name"].astype(str).str.strip() != ""]

        # Se è boundary=administrative, spesso conviene filtrare admin_level (es: 8/9/10)
        # Osm ha confini a più livelli (2-nazione, 4-regione, 6-provincia, 8-comune, 10-frazione, ecc.)
        if "boundary" in tags:
            if "admin_level" in gdf_sub.columns:
                gdf_sub = gdf_sub[gdf_sub["admin_level"].astype(str).isin(["8", "9", "10"])]

        if len(gdf_sub) == 0:
            continue

        # Dissolve per ottenere 1 geometria per nome-zona
        # Si possono avere MuliPoligon (ovvero zone spezzate) => gli fondiamo insieme by "name"
        gdf_grouped = gdf_sub[["name", "geometry"]].dissolve(by="name")
        # Rimuovi geometrie vuote/non valide
        gdf_grouped = gdf_grouped[~gdf_grouped.geometry.is_empty & gdf_grouped.geometry.notna()]

        if len(gdf_grouped) > 0:
            found_type = list(tags.items())[0]
            print(f"[OSM] trovate zone via tags={tags} -> #zone={len(gdf_grouped)}")
            return gdf_grouped

    # Se siamo arrivati fino a qui...
    print("[OSM] nessuna suddivisione (suburb/neighbourhood/quarter/...) trovata.")
    return None




"""
IN PROGRESS...

def build_supernodes(G: nx.MultiDiGraph, place: str, n_coarsened: int, gdf_nodes: gpd.GeoDataFrame | None = None): 

    Build super-nodes for a city:
    - from scratch
    - starting from existing zones

    

    # Caso: zone trovate
    num_zones = len(gdf_zones)
    print(f"[CHECK] zone OSM trovate: {num_zones} (target={n_coarsened})")

    if num_zones > n_coarsened:
        print(f"[COARSEN] troppe zone OSM ({num_zones}) -> riduco con algoritmo fino a {n_coarsened}")
        # puoi passare le zone come input al tuo algoritmo, se lo supporta
        # altrimenti fai coarsening “da rete” e basta
        return coarsen_algo_fn(place=place, target_n=n_coarsened, zones_gdf=gdf_zones, **coarsen_kwargs)

    print("[OK] zone OSM già <= target: le uso direttamente")
    return gdf_zones





def generate_city_network_coarsened(
    place: str,
    n_super_nodes: int = 20,
    speed_kmh: float = 40.0,
    plot: bool = True,
    dir: str = "instances/CITY/",
    filename: str = "network_coarsened.png",
) -> tuple[dict, nx.MultiDiGraph]:
    
    output_dir = "instances/CITY/" / place.replace(",","").replace(" ","_")

    # -------------------
    # Download city graph 
    # -------------------
    osm_graph = ox.graph_from_place(
        place,
        network_type="drive",
        simplify=True,
    )

    # ----------------
    # Coarsening step:
    # ----------------
    # Check if there are "suburb" labels in OSM data
    gdf = get_osm_zones_or_none(place)

    # Se non si hanno già zone o si hanno troppi pochi nodi
    if gdf is None or len(gdf) < n_super_nodes:   #NOTA: len(gdf) non funzionerebbe se gdf è None, ma cmq viene valutata priam l'istanza a sx    
        gdf = build_supernodes(osm_graph, place, n_super_nodes, gdf)

    # ------------------------
    # 3) Centroids -> nearest OSM node
    # ------------------------
    suburb_nodes = {}   # name -> osm node id
    node_entries = []   # list of dicts for "nodes" in JSON

    for name, row in gdf_grouped.iterrows():
        centroid = row["geometry"].centroid
        lat, lon = centroid.y, centroid.x

        osm_node = ox.distance.nearest_nodes(osm_graph, X=lon, Y=lat)

        suburb_nodes[name] = osm_node
        node_entries.append({
            "id": int(osm_node),
            "name": str(name),
            "lat": float(lat),
            "lon": float(lon),
        })

    # ------------------------------------------------------------
    # Edge construction: central = k_center, lateral = k_lateral
    # ------------------------------------------------------------
    suburb_names = sorted(suburb_nodes.keys())

    # Mapping: suburb name -> (lat, lon)
    coords = {entry["name"]: (entry["lat"], entry["lon"]) for entry in node_entries}

    def geo_dist2(a, b):

        Squared geographic distance between two centroids (lat, lon).
        We use squared distance because it preserves ordering and avoids sqrt().

        (lat1, lon1), (lat2, lon2) = a, b
        return (lat1 - lat2) ** 2 + (lon1 - lon2) ** 2

    # Undirected adjacency list: suburb_name -> set(neighbor_names)
    adjacency = {name: set() for name in suburb_names}

    # ------------------------------------------------------------
    # (1) INITIAL STEP: each suburb picks its nearest neighbors
    #     - central suburbs use k_center
    #     - lateral suburbs use k_lateral
    # ------------------------------------------------------------
    for name_u in suburb_names:
        cu = coords[name_u]

        # Build candidate list: all *other* suburbs with their distance
        cand = []
        for name_v in suburb_names:
            if name_v == name_u:
                continue
            cv = coords[name_v]
            d2 = geo_dist2(cu, cv)
            cand.append((name_v, d2))

        # Sort candidates by proximity
        cand.sort(key=lambda x: x[1])

        # Decide how many neighbors to keep (central vs lateral)
        if name_u in central_suburbs:
            k = min(k_center, len(cand))
        else:
            k = min(k_lateral, len(cand))

        # List of the selected nearest neighbor names
        chosen = [name_v for (name_v, _) in cand[:k]]

        # Update adjacency (undirected)
        for name_v in chosen:
            adjacency[name_u].add(name_v)
            adjacency[name_v].add(name_u)

    # ------------------------------------------------------------
    # (2) ADJUSTMENT STEP:
    #     Try to enforce:
    #       - central suburbs: exactly k_center neighbors
    #       - non-central suburbs: exactly k_lateral neighbors
    # ------------------------------------------------------------
    for name_u in suburb_names:
        # target degree for this suburb
        if name_u in central_suburbs:
            target_deg = k_center
        else:
            target_deg = k_lateral

        cu = coords[name_u]

        # ----------------------------
        # (2a) Se ha meno del target:
        #      aggiungi vicini mancanti
        # ----------------------------
        neighbors = list(adjacency[name_u])
        if len(neighbors) < target_deg:
            # candidati = tutti gli altri suburb che non sono già vicini
            cand = []
            for name_v in suburb_names:
                if name_v == name_u:
                    continue
                if name_v in adjacency[name_u]:
                    continue
                cv = coords[name_v]
                d2 = geo_dist2(cu, cv)
                cand.append((name_v, d2))

            # ordina per distanza geometrica
            cand.sort(key=lambda x: x[1])

            # aggiungi i più vicini finché non raggiungi target_deg
            for (name_v, _) in cand:
                adjacency[name_u].add(name_v)
                adjacency[name_v].add(name_u)  # grafo non orientato
                neighbors.append(name_v)
                if len(neighbors) >= target_deg:
                    break

        # ----------------------------
        # (2b) Se ha più del target:
        #      tieni solo i target_deg più vicini
        # ----------------------------
        neighbors = list(adjacency[name_u])  # aggiorna dopo eventuali aggiunte
        if len(neighbors) > target_deg:
            # ordina i vicini per distanza
            neighbors.sort(key=lambda name_v: geo_dist2(cu, coords[name_v]))

            keep = set(neighbors[:target_deg])
            remove = set(neighbors[target_deg:])

            for name_v in remove:
                adjacency[name_u].discard(name_v)
                adjacency[name_v].discard(name_u)



    # ------------------------------------------------------------
    # (3) Convert adjacency structure into final edge list
    #     Each edge (u,v) is undirected → only one entry per pair.
    #     Distances come from OSM shortest-path length.
    # ------------------------------------------------------------
    edges = []
    seen_pairs = set()  # prevent duplicates of (u,v) and (v,u)

    for name_u in suburb_names:
        u_node = suburb_nodes[name_u]
        for name_v in adjacency[name_u]:
            v_node = suburb_nodes[name_v]

            # Normalize undirected pair
            pair_key = tuple(sorted((u_node, v_node)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Try shortest path u→v; if no path, try v→u
            try:
                length_m = nx.shortest_path_length(
                    osm_graph, source=u_node, target=v_node, weight="length"
                )
            except nx.NetworkXNoPath:
                try:
                    length_m = nx.shortest_path_length(
                        osm_graph, source=v_node, target=u_node, weight="length"
                    )
                except nx.NetworkXNoPath:
                    continue  # no path exists in either direction → skip

            length_km = float(length_m) / 1000.0
            time_min = (length_km / default_speed_kmh) * 60.0

            edges.append({
                "u": int(u_node),
                "v": int(v_node),  # undirected edge represented as (u,v)
                "length_km": length_km,
                "time_min": float(time_min),
            })



    network_dict = {
        "type": "CITY_SUBURB_UNDIRECTED",
        "place": place,
        "default_speed_kmh": float(default_speed_kmh),
        "nodes": node_entries,
        "edges": edges,
    }

    # --------------
    # Optional: plot
    # --------------
    if plot:
        plot_city_suburb_network(
            osm_graph=osm_graph,
            network_dict=network_dict,
            place=place,
            output_dir=output_dir,
            filename=filename,
            title_suffix="(raw)",
        )

    return network_dict, osm_graph

"""