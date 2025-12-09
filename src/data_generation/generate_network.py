"""
Network Generator
=================

Creates the base transportation network used in the experiments.
Supports two modes:

1. GRID network:
   - Builds a side × side grid
   - Nodes = intersections, edges = adjacent links
   - Each edge includes distance γ(i,j) and travel time τ(i,j)
   - Saved as `grid_network.json`

2. CITY network (optional):
   - Loaded from OSM and normalized to the same JSON structure

Output JSON contains:
    - nodes
    - edges
    - metadata (e.g., type, grid size)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import random
import osmnx as ox
import networkx as nx


# seed
import random
import numpy as np
seed = 23
random.seed(seed)
np.random.seed(seed)



#################
# GRID GENERATOR:
#################
"""
Creates a GRID network (side × side) and saves it as `grid_network.json`
under the given output directory.

Nodes:
    id      : integer
    row,col : grid coordinates

Edges:
    u, v    : tail and head node id
    length  : distance γ(i,j)
    time    : travel time τ(i,j)
"""


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



####################
# NETWORK GENERATOR:
####################
"""
Downloads or loads a real city road network (via OSMnx) and converts it into
a unified JSON format compatible with the optimization model.

Optionally applies graph reduction:
    - intersection consolidation (merge nearby nodes)
    - geometric simplification (remove intermediary shape points)
    - centrality-based filtering (keep only most important nodes)

Nodes:
    id        : unique identifier from OSM
    x, y      : geographic coordinates (longitude/latitude or projected coords)

Edges:
    u, v      : tail and head node id
    length_km : physical distance of the road segment (γ(i,j))
    time_min  : travel time (τ(i,j)) computed from distance and speed
    speed_kmh : speed assigned to the segment (OSM-based or default)

Notes:
    - Length is extracted from the road data (meters converted to km).
    - Speed is obtained from OSM attributes; if missing, a default speed is used.
    - Travel time is computed as:
            time_min = (length_km / speed_kmh) * 60
    - If reduction is enabled, intersections within a tolerance radius
      are merged, unnecessary nodes are removed, and (optionally) only
      the most central nodes are kept.
"""



### REMARK: OSM has the info of 32 Turin suburbs under the "suburb" tag
def generate_city_network_raw(
    place: str,
    default_speed_kmh: float = 40.0,
    plot: bool = True,
    k_neighbors: int = 2,
    k_center: int = 4,
    central_suburbs: list = None,
) -> dict:
    """
    Generate a macro CITY network for a given place based on OSM suburbs.

    - Each node is a suburb (place=suburb) in the given place.
    - Node id is the nearest OSM street node to the suburb centroid.
    - Edges are built with a hub–centric neighborhood strategy:
        * local k-nearest neighbor edges for each suburb
        * extra radial edges from non-central suburbs to central ones.

    Parameters
    ----------
    place : str
        Name of the place to query in OSM (e.g. "Torino, Italia").
    default_speed_kmh : float
        Speed used to compute travel time if needed.
    plot : bool
        If True, plot the city graph, suburbs and macro edges.
    k_neighbors : int
        Number of nearest neighbors for non-central suburbs.
    k_center : int
        Number of nearest neighbors for central suburbs.
    central_suburbs : list
        List of suburb names considered as "central hubs".

    Returns
    -------
    network_dict : dict
        Unified network dictionary with:
            - type
            - place
            - nodes: [ {id, name, lat, lon}, ... ]
            - edges: [ {u, v, length_km, time_min}, ... ]
            - default_speed_kmh
    """
    if central_suburbs is None:
        central_suburbs = []

    ### CITY GRAPH
    osm_graph = ox.graph_from_place(
        place,
        network_type="drive",
        simplify=True,
    )

    ### SUBURBS
    tags = {"place": ["suburb"]}
    gdf = ox.features_from_place(place, tags=tags)

    # Keep only name + geometry
    gdf_sub = gdf[["name", "geometry"]].dropna(subset=["name", "geometry"])

    # Merge geometries with the same name (in case of multiple polygons)
    gdf_grouped = gdf_sub.dissolve(by="name")  # index = name

    ### Compute centroids and map each suburb to nearest OSM street node
    suburb_nodes = {}   # name -> osm node id
    node_entries = []   # list of dicts for "nodes" in JSON

    for name, row in gdf_grouped.iterrows():
        centroid = row["geometry"].centroid
        lat, lon = centroid.y, centroid.x

        # nearest street node in the downloaded osm_graph
        osm_node = ox.distance.nearest_nodes(osm_graph, X=lon, Y=lat)

        suburb_nodes[name] = osm_node

        node_entries.append({
            "id": int(osm_node),
            "name": str(name),
            "lat": float(lat),
            "lon": float(lon),
        })

    # ------------------------------------------------------------
    # Edge construction: hub–centric neighborhood graph
    # ------------------------------------------------------------
    edges = []
    suburb_names = sorted(suburb_nodes.keys())

    # name -> (lat, lon)
    coords = {entry["name"]: (entry["lat"], entry["lon"]) for entry in node_entries}

    def geo_dist2(a, b):
        """Squared distance between two (lat, lon) points (no sqrt needed for ordering)."""
        (lat1, lon1), (lat2, lon2) = a, b
        return (lat1 - lat2) ** 2 + (lon1 - lon2) ** 2

    # (1) Local edges: k-nearest neighbors per suburb
    for name_u in suburb_names:
        u = suburb_nodes[name_u]
        cu = coords[name_u]

        # central suburbs are more connected
        if name_u in central_suburbs:
            k = k_center
        else:
            k = k_neighbors

        # candidate neighbors (all others)
        cand = []
        for name_v in suburb_names:
            if name_v == name_u:
                continue
            cv = coords[name_v]
            d2 = geo_dist2(cu, cv)
            cand.append((name_v, d2))

        cand.sort(key=lambda x: x[1])
        nearest = [name_v for (name_v, _) in cand[:k]]

        for name_v in nearest:
            v = suburb_nodes[name_v]

            try:
                length_m = nx.shortest_path_length(
                    osm_graph, source=u, target=v, weight="length"
                )
            except nx.NetworkXNoPath:
                continue

            length_km = float(length_m) / 1000.0
            time_min = (length_km / default_speed_kmh) * 60.0

            edges.append({
                "u": int(u),
                "v": int(v),
                "length_km": length_km,
                "time_min": float(time_min),
            })

    # (2) Radial edge: each non-central suburb connects to its closest central hub
    central_coords = {name: coords[name] for name in suburb_names if name in central_suburbs}
    if central_coords:
        for name_u in suburb_names:
            if name_u in central_suburbs:
                continue  # hubs already dense

            u = suburb_nodes[name_u]
            cu = coords[name_u]

            best_name = None
            best_d2 = None
            for name_c, cc in central_coords.items():
                d2 = geo_dist2(cu, cc)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_name = name_c

            if best_name is None:
                continue

            v = suburb_nodes[best_name]

            # avoid duplicates if already created in k-nearest step
            if any(e["u"] == int(u) and e["v"] == int(v) for e in edges):
                continue

            try:
                length_m = nx.shortest_path_length(
                    osm_graph, source=u, target=v, weight="length"
                )
            except nx.NetworkXNoPath:
                continue

            length_km = float(length_m) / 1000.0
            time_min = (length_km / default_speed_kmh) * 60.0

            edges.append({
                "u": int(u),
                "v": int(v),
                "length_km": length_km,
                "time_min": float(time_min),
            })

    network_dict = {
        "type": "CITY_SUBURB",
        "place": place,
        "default_speed_kmh": float(default_speed_kmh),
        "nodes": node_entries,
        "edges": edges,
    }

    # ----------------------------------------
    # Print current edges as MANUAL_EDGES list (DIRECTED)
    # ----------------------------------------
    id_to_name = {n["id"]: n["name"] for n in node_entries}

    for e in edges:
        u = e["u"]
        v = e["v"]
        name_u = id_to_name.get(u)
        name_v = id_to_name.get(v)
        if name_u is None or name_v is None:
            continue
        print(f'("{name_u}", "{name_v}"),')





    # -----------------------
    # Optional plot (to file)
    # -----------------------
    plt.switch_backend("Agg")
    if plot:
        # background: city street graph
        fig, ax = ox.plot_graph(osm_graph, show=False, close=False, bgcolor="white")

        # node_id -> (lon, lat)
        id_to_coord = {
            entry["id"]: (entry["lon"], entry["lat"])
            for entry in node_entries
        }

        # draw macro edges
        print(len(edges))
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

        # centroids + labels
        for entry in node_entries:
            ax.scatter(entry["lon"], entry["lat"], c="red", s=40)
            ax.text(entry["lon"], entry["lat"], entry["name"], fontsize=8)

        plt.title(f"Suburbs + macro edges for {place}")
        plt.tight_layout()
        plt.savefig("torino_suburbs.png", dpi=500)
        print("Saved plot as torino_suburbs.png")

    return network_dict




"""
Edge construction: hub–centric neighborhood graph
-------------------------------------------------

Instead of a fully-connected graph, we build a sparse, road-like structure:

(1) Local edges (k-nearest neighbors):
    - Each suburb u connects only to its geographically closest neighbors.
    - Central suburbs use k_center neighbors (denser core), while peripheral
      suburbs use k_neighbors (sparser periphery).
    - For each selected neighbor v, edge cost is the shortest-path distance
      on the OSM road graph (in km and minutes).

(2) Radial edges to the center:
    - Each NON-central suburb gets at least one edge to the nearest central
      suburb, unless already connected.
    - This creates a hub–and–spoke structure and ensures global connectivity.
"""

#def rielaborate_city_network_Turin()






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






### TEST MAIN ###
if __name__ == "__main__":
    ### GRID TEST ###
    side = 3
    edge_length_km = 1.0
    speed_kmh = 40.0

    folder_name = f"{side}x{side}"
    output_folder = f"instances/GRID/{folder_name}"

    network = generate_grid_network(
        side=side,
        edge_length_km=edge_length_km,
        speed_kmh=speed_kmh,
    )

    save_network_json(
        network_dict=network,
        output_dir=output_folder,
        filename="network.json"
    )

    ### CITY TEST (TORINO SUBURB) ###
    place = "Torino, Italia"

    # Download detailed city graph from OSM
    G_city = ox.graph_from_place(
        place,
        network_type="drive",
        simplify=True,
    )

    central_suburbs = [   # EXISTING IN OSM
        "Centro",    
        "San Salvario",
        "Crocetta",
        "Aurora",
        "Vanchiglia",
    ]

    city_network = generate_city_network_raw(
        place=place,
        default_speed_kmh=40.0,
        plot=True,
        k_neighbors=1,
        k_center=2,
        central_suburbs=central_suburbs,
    )


    #city_network = rielaborate_city_network_Turin()




    city_output_folder = "instances/CITY/TORINO_SUBURB"
    save_network_json(
        network_dict=city_network,
        output_dir=city_output_folder,
        filename="network.json"
    )
