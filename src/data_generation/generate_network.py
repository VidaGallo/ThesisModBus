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

