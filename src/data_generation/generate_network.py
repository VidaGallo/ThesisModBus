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




#def generate_city_network(osm_graph, default_speed_kmh: float = 40.0) -> dict:
#    ...   
#    return network_dict







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
    side = 3                # grid side dimension
    edge_length_km = 1.0    
    speed_kmh = 40.0        

    # Folder name automatically generated from side
    folder_name = f"{side}x{side}"
    output_folder = f"instances/GRID/{folder_name}"

    # 1) Generate network
    network = generate_grid_network(
        side=side,
        edge_length_km=edge_length_km,
        speed_kmh=speed_kmh,
    )

    # 2) Save JSON
    save_network_json(
        network_dict=network,
        output_dir=output_folder,
        filename="network.json"
    )

    #print(f"[INFO] Network generated: GRID {folder_name}")
    #print(f"[INFO] Saved in: {output_folder}/network.json")

