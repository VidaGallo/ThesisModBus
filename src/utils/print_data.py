"""
print_data.py
=============
 - print of the Iinstance Class
 - print of the Results
"""

from .instance import Instance
import json
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt





def print_instance_summary(inst: Instance):
    """Pretty-print full content of an Instance."""

    print("\n\n\n" + "="*100)
    print("="*100)
    print("INSTANCE SUMMARY:1n")

    ### Global info
    print("\n--->Global Info")
    print(f"  dt (minutes):     {inst.dt}")
    print(f"  t_max (slots):    {inst.t_max}")
    print(f"  Time periods T:   {inst.T}")

    ### Sets
    print("\n--->Sets")
    print(f"  |N| Nodes:        {len(inst.N)}")
    print(f"  |A| Arcs:         {len(inst.A)}")
    print(f"  |M| Modules:      {len(inst.M)}")
    print(f"  |K| Requests:     {len(inst.K)}")

    ### Parameters
    print("\n--->Parameters")
    print(f"  Module capacity Q:    {inst.Q}")
    print(f"  Cost per km:          {inst.c_km}")
    print(f"  Cost uns. demand:     {inst.c_uns_taxi}")

    ### Network
    print("\n--->Network: first 10 arcs")
    for idx, arc in enumerate(sorted(inst.A)):
        if idx >= 10:
            print("  ...")
            break
        gamma = inst.gamma.get(arc, None)
        tau = inst.tau_arc.get(arc, None)
        print(f"  Arc {arc}: length={gamma}, tau={tau}")

    ### Requests
    print("\n\n--->Requests Summary")
    for k in sorted(inst.K):
        print(f"\n  Request {k}:")
        print(f"    origin:        {inst.origin[k]}")
        print(f"    destination:   {inst.dest[k]}")
        print(f"    q_k:           {inst.q[k]}")
        print(f"    ΔT_k:          {inst.DeltaT[k]}")
        print(f"    ΔT_in:         {inst.DeltaT_in[k]}")
        print(f"    ΔT_out:        {inst.DeltaT_out[k]}")

    ### Sparse matrices
    print("\n--->Sparse d_in (first 10)")
    print(inst.d_in)

    print("\n--->Sparse d_out (first 10)")
    print(inst.d_out)

    print("\n" + "="*100 + "\n\n")









def load_node_positions_from_network(network_path: str) -> Dict[int, Tuple[float, float]]:
    """
    Load node positions (row, col) from a GRID network JSON.

    Returns:
        pos[node_id] = (x, y)
    """
    net_path = Path(network_path)
    with net_path.open("r", encoding="utf-8") as f:
        net = json.load(f)

    pos = {}
    for node in net["nodes"]:
        node_id = int(node["id"])
        row = node.get("row", 0)
        col = node.get("col", 0)
        # use (col, -row) so that row increases upwards visually
        pos[node_id] = (float(col), float(-row))

    return pos


def precompute_travel_positions(
    I: Instance,
    solution,
    y,
) -> Dict[Tuple[int, int], Tuple[int, int, float]]:
    """
    Precompute positions of traveling modules.

    For each y[m,i,j,t0] = 1 and each offset = 1,...,tau_ij-1 we create:
        travel_pos[(m, t0+offset)] = (i, j, lambda)
    where lambda = offset / tau_arc[(i,j)].

    Returns:
        travel_pos[(m,t)] = (i, j, lam)
    """
    travel_pos: Dict[Tuple[int, int], Tuple[int, int, float]] = {}

    T_min = min(I.T)
    T_max = max(I.T)

    for m in I.M:
        for (i, j) in I.A:
            tau_ij = I.tau_arc[(i, j)]
            # se tau_ij <= 1 non abbiamo istanti intermedi
            if tau_ij <= 1:
                continue

            for t0 in I.T:
                if solution.get_value(y[m, i, j, t0]) > 0.5:
                    # module departs from i at time t0, travels for tau_ij steps
                    for offset in range(1, tau_ij):
                        t = t0 + offset
                        if t < T_min or t > T_max:
                            continue
                        lam = offset / float(tau_ij)
                        # in una soluzione fattibile non dovrebbero esserci conflitti
                        travel_pos[(m, t)] = (i, j, lam)

    return travel_pos



def print_solution_movements_and_positions(instance, solution, x, y):
    """
    Prints:
        1. Module movements (y)
        2. Module positions (x)
       + If module is traveling at time t → prints "i→j"
    """

    M = instance.M
    T = instance.T

    # ------------------------------------------------
    # Precompute travel positions
    # ------------------------------------------------
    travel_pos = precompute_travel_positions(instance, solution, y)

    # ------------------------------------------------
    # 1. MOVEMENTS
    # ------------------------------------------------
    print("\n" + "=" * 77)
    print("MODULE MOVEMENTS (y[m,i,j,t] = 1)")
    movements_by_module = {m: [] for m in M}

    for (m, i, j, t), var in y.items():
        if solution.get_value(var) > 0.99:
            movements_by_module[m].append((t, i, j))

    for m in M:
        print(f"\nModule {m}:")
        if movements_by_module[m]:
            movements_by_module[m].sort(key=lambda x: x[0])
            for t, i, j in movements_by_module[m]:
                print(f"  t={t}: from {i} → {j}")
        else:
            print("  No movements recorded.")

    # ------------------------------------------------
    # 2. POSITIONS
    # ------------------------------------------------
    print("\n" + "=" * 77)
    print("MODULE POSITIONS (x[m,i,t] = 1)")
    print("If module is traveling at time t → prints 'i→j' instead of a node.")

    positions_by_module = {m: [] for m in M}

    for (m, i, t), var in x.items():
        if solution.get_value(var) > 0.99:
            positions_by_module[m].append((t, i))

    for m in M:
        print(f"\nModule {m}:")
        if not positions_by_module[m]:
            print("  No positions recorded.")
            continue

        positions_by_module[m].sort(key=lambda x: x[0])

        for t, node in positions_by_module[m]:

            # -------------------------
            # CHECK IF MODULE IS TRAVELING
            # -------------------------
            if (m, t) in travel_pos:
                i, j, lam = travel_pos[(m, t)]
                print(f"  t={t}: {i}→{j}  (travel)")
            else:
                print(f"  t={t}: node {node}")





def plot_initial_grid_with_modules(
    I: Instance,
    solution,
    x,
    network_path: str,
    t0: int,
    output_path: Path,
):
    """
    Plot a single image of the GRID network at initial time t0:
    - nodes with labels (node id)
    - edges in blue with travel time labels (tau_arc or duration on each arc)
    - modules as big red dots at their initial node (x[m,i,t0] = 1),
      with module id as text next to the dot.
    """
    output_path = Path(output_path)

    # 1) Load node positions (you already have this helper)
    node_pos: Dict[int, Tuple[float, float]] = load_node_positions_from_network(network_path)

    # 2) Evaluate x at time t0
    x_val = {idx: solution.get_value(var) for idx, var in x.items()}

    fig, ax = plt.subplots(figsize=(6, 6))

    # ------------------------------------------------------------------
    # Draw edges in BLUE with travel time labels
    # ------------------------------------------------------------------
    for (i, j) in I.A:
        xi, yi = node_pos[i]
        xj, yj = node_pos[j]

        # edge in blue
        ax.plot([xi, xj], [yi, yj], color="blue", alpha=0.5, linewidth=1)

        # label with travel time (tau_arc or minutes)
        tau_ij = I.tau_arc[(i, j)]
        xm = 0.5 * (xi + xj)
        ym = 0.5 * (yi + yj)

        # If tau_ij is in time steps and you want minutes, you can multiply by I.dt, ecc.
        ax.text(
            xm,
            ym,
            f"{tau_ij}",
            fontsize=7,
            color="blue",
            ha="center",
            va="center",
            alpha=0.8,
        )

    # ------------------------------------------------------------------
    # Draw nodes with labels
    # ------------------------------------------------------------------
    for i in I.N:
        x_i, y_i = node_pos[i]
        # node point (can be small)
        ax.scatter([x_i], [y_i], s=30, color="black", zorder=3)
        # node label next to node
        ax.text(
            x_i + 0.05,
            y_i + 0.05,
            str(i),
            fontsize=8,
            color="black",
            zorder=4,
        )

    # ------------------------------------------------------------------
    # Draw modules as big RED dots at t0, with labels
    # ------------------------------------------------------------------
    for m in I.M:
        # find node i where x[m,i,t0] = 1
        node_found = None
        for i in I.N:
            val = x_val.get((m, i, t0), 0.0)
            if val > 0.5:
                node_found = i
                break

        if node_found is None:
            # module not present / not assigned at t0
            continue

        xm, ym = node_pos[node_found]

        # big red dot for module
        ax.scatter([xm], [ym], s=120, color="red", zorder=5)

        # label with module id
        ax.text(
            xm + 0.07,
            ym + 0.07,
            f"m{m}",
            fontsize=9,
            color="red",
            zorder=6,
        )

    ax.set_title(f"Initial configuration at t = {t0}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    plt.tight_layout()
    print(f"[INFO] Saving initial grid snapshot at t={t0} -> {output_path}")
    fig.savefig(output_path)
    plt.close(fig)



