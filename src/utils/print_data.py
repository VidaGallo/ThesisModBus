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

    print("\n" + "="*123)
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

    print("\n" + "="*123 + "\n")









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








def debug_full_system_timeline(
    K,
    M,
    N,
    T,
    DeltaT,
    t0,
    solution,
    x,
    y,
    r,
    w,
    s=None,
):
    """
    Print a detailed, time-step-by-time-step view of the solution.

    For each time t, it prints:
        - Module positions (x)
        - Module movements (y)
        - Swaps between modules (w)
        - Request status (r, s)
        - Events for each request: board / alight / swap-in / swap-out

    """

    print("\n\n\n" + "#" * 80)
    print("DEBUG: FULL SYSTEM TIMELINE")


    # Pre-extract r values in a dict for faster access
    r_val = {}
    for (k, t, m), var in r.items():
        r_val[(k, t, m)] = solution.get_value(var)

    # Optional: s values
    s_val = {}
    if s is not None:
        for k, var in s.items():
            s_val[k] = solution.get_value(var)

    # Pre-extract y, w, x values
    x_val = {idx: solution.get_value(var) for idx, var in x.items()}
    y_val = {idx: solution.get_value(var) for idx, var in y.items()}
    w_val = {idx: solution.get_value(var) for idx, var in w.items()}

    # ------------------------------------------------------------------
    # MAIN LOOP ON TIME
    # ------------------------------------------------------------------
    for t in T:
        print("\n" + "=" * 50)
        print(f"TIME t = {t}")
        print("=" * 50)

        # ==============================================================
        # 1) MODULE POSITIONS AND MOVEMENTS
        # ==============================================================

        print("\n[1] MODULE STATES (x and y)")

        # Positions on nodes
        for m in M:
            # find node with x[m,i,t] = 1
            node_here = []
            for i in N:
                val = x_val.get((m, i, t), 0.0)
                if val > 0.5:
                    node_here.append(i)

            # movements starting at time t
            moves_here = []
            for (mm, i, j, tt), val in y_val.items():
                if mm == m and tt == t and val > 0.5:
                    moves_here.append((i, j))

            if not node_here and not moves_here:
                # module not present / idle out of horizon
                continue

            print(f"\n  Module m={m}:")
            if node_here:
                print(f"    - at nodes: {node_here}")
            else:
                print(f"    - not on any node (according to x)")

            if moves_here:
                for (i, j) in moves_here:
                    print(f"    - movement: departs from {i} → {j}")
            else:
                print(f"    - no departures (y) at time t={t}")

        # ==============================================================
        # 2) SWAPS (w) AT TIME t
        # ==============================================================

        print("\n[2] SWAPS (w) at time t")

        swaps_here = []
        for (k, i, tt, m, mp), val in w_val.items():
            if tt == t and val > 0.5:
                swaps_here.append((k, i, m, mp))

        if swaps_here:
            for (k, i, m, mp) in swaps_here:
                print(f"  Request k={k} swaps at node i={i}: m={m} -> m'={mp}")
        else:
            print("  No swaps at this time.")

        # ==============================================================
        # 3) REQUEST STATUS AT TIME t
        # ==============================================================

        print("\n[3] REQUEST STATUS (r and s)")

        for k in K:
            if t not in DeltaT[k]:
                # request not active at this time
                continue

            # which modules are carrying k at time t?
            mods = []
            for m in M:
                val = r_val.get((k, t, m), 0.0)
                if val > 0.5:
                    mods.append(m)

            if not mods and s is None:
                # nothing special, request is simply not on any module now
                continue

            print(f"\n  Request k={k}:")
            if mods:
                print(f"    - carried by modules: {mods}")
            else:
                print("    - not carried by any module.")

            if s is not None:
                sval = s_val.get(k, 0.0)
                if sval > 0.5:
                    print(f"    - s[k]={sval}  (request marked as served)")
                else:
                    print(f"    - s[k]={sval}  (request not served)")

                # else: s not defined for this (k,t)

        # ==============================================================
        # 4) EVENTS (BOARD / ALIGHT / SWAP) between t-1 and t
        # ==============================================================

        if t == t0:
            print("\n[4] EVENTS (board/alight/swap) between t-1 and t")
            print("  Skipped at t0 (no previous time).")
            continue

        print("\n[4] EVENTS (board/alight/swap) between t-1 and t")

        t_prev = t - 1

        for k in K:
            if t not in DeltaT[k] or t_prev not in DeltaT[k]:
                continue

            for m in M:
                prev_val = r_val.get((k, t_prev, m), 0.0)
                curr_val = r_val.get((k, t, m), 0.0)

                delta = curr_val - prev_val

                if delta > 0.5:
                    # r goes 0 -> 1 (or increases): boarding or swap-in
                    # check if there's a swap w[*,t,*,*,m] involving this k,m
                    swap_in = []
                    for (kk, i, tt, mm, mp), val in w_val.items():
                        if (
                            kk == k and
                            tt == t and
                            mp == m and
                            val > 0.5
                        ):
                            swap_in.append((i, mm))

                    if swap_in:
                        for (i, mm) in swap_in:
                            print(
                                f"  k={k}, m={m}: SWAP-IN at node i={i} "
                                f"(coming from module m={mm}) between t={t_prev} and t={t}"
                            )
                    else:
                        print(
                            f"  k={k}, m={m}: BOARDING between t={t_prev} and t={t} "
                            f"(r went {prev_val} -> {curr_val})"
                        )

                elif delta < -0.5:
                    # r goes 1 -> 0 (or decreases): alighting or swap-out
                    swap_out = []
                    for (kk, i, tt, mm, mp), val in w_val.items():
                        if (
                            kk == k and
                            tt == t and
                            mm == m and
                            val > 0.5
                        ):
                            swap_out.append((i, mp))

                    if swap_out:
                        for (i, mp) in swap_out:
                            print(
                                f"  k={k}, m={m}: SWAP-OUT at node i={i} "
                                f"(going to module m'={mp}) between t={t_prev} and t={t}"
                            )
                    else:
                        print(
                            f"  k={k}, m={m}: ALIGHTING between t={t_prev} and t={t} "
                            f"(r went {prev_val} -> {curr_val})"
                        )



def debug_requests_details(
    K,
    M,
    N,
    T,
    DeltaT,
    t0,
    solution,
    r,
    d_in,
    d_out,
    q=None,
    s=None,
):
    """
    Riepilogo compatto per ogni richiesta k:
      - finestra DeltaT[k]
      - domanda q[k]
      - se è servita (s[k])
      - in quali tempi e moduli viene trasportata
      - primo boarding e ultima alighting (se esistono)
    """

    print("\n" + "#" * 80)
    print("DEBUG: REQUEST-LEVEL SUMMARY")

    # valori r
    r_val = {(k, t, m): solution.get_value(var) for (k, t, m), var in r.items()}

    # valori s[k]
    s_val = {}
    if s is not None:
        for k, var in s.items():
            s_val[k] = solution.get_value(var)

    for k in K:
        print("\n" + "=" * 50)
        print(f"REQUEST k = {k}")
        print("=" * 50)

        times_k = sorted(DeltaT[k])
        print(f"- DeltaT[k]: {times_k}")
        if q is not None and k in q:
            print(f"- q[k]: {q[k]}")

        # status globale
        if s is not None:
            sval = s_val.get(k, 0.0)
            print(f"- s[k]: {sval}  ({'SERVITA' if sval > 0.5 else 'NON servita'})")

        # dove e quando è a bordo
        carried_times = []
        for t in times_k:
            mods = [m for m in M if r_val.get((k, t, m), 0.0) > 0.5]
            if mods:
                carried_times.append((t, mods))

        if not carried_times:
            print("\n[carico] Mai a bordo (tutti r[k,t,m] = 0).")
        else:
            print("\n[carico] Tempi in cui è a bordo:")
            for t, mods in carried_times:
                print(f"  t={t}: moduli m={mods}")

        # inferire primo boarding e ultima alighting
        first_board = None
        last_alight = None

        for t in times_k:
            if t == t0:
                continue
            t_prev = t - 1
            if t_prev not in times_k:
                continue

            for m in M:
                prev_val = r_val.get((k, t_prev, m), 0.0)
                curr_val = r_val.get((k, t, m), 0.0)
                delta = curr_val - prev_val

                if delta > 0.5 and first_board is None:
                    first_board = (t, m)
                if delta < -0.5:
                    last_alight = (t, m)

        print("\n[eventi]")
        if first_board is not None:
            t_b, m_b = first_board
            print(f"  Primo BOARD: t={t_b}, modulo m={m_b}")
        else:
            print("  Nessun BOARD individuato.")

        if last_alight is not None:
            t_a, m_a = last_alight
            print(f"  Ultimo ALIGHT: t={t_a}, modulo m={m_a}")
        else:
            print("  Nessun ALIGHT individuato.")








def full_report(
    instance,
    solution,
    x,
    y,
    r,
    w,
    s,
    network_path,
    output_folder,
    display_instance_summary=True,
    display_movements=True,
    display_initial_grid=True,
    display_timeline=True,
    display_request_details=True,
):
    """
    Master verbose/debug report.

    Each component can be enabled/disabled individually using flags.

    Parameters:
    -----------
    display_instance_summary : bool
        If True → prints full instance summary.
    display_movements : bool
        If True → prints module movements and positions.
    display_initial_grid : bool
        If True → plots the initial grid with module locations.
    display_timeline : bool
        If True → prints full step-by-step system timeline.
    display_request_details : bool
        If True → prints detailed request-level breakdown.
    """

    t0 = min(instance.T)


    ### (1) Instance Summary
    if display_instance_summary:
        print_instance_summary(instance)



    ### (2) Initial Grid Snapshot
    if display_initial_grid:
        initial_snapshot_path = output_folder / f"initial_grid_t{t0}.png"
        plot_initial_grid_with_modules(
            I=instance,
            solution=solution,
            x=x,
            network_path=network_path,
            t0=t0,
            output_path=initial_snapshot_path,
        )
    
    
    ### (3) Movements + Positions
    if display_movements:
        print_solution_movements_and_positions(instance, solution, x, y)



    ### (4) Full Timeline Debug
    if display_timeline:
        debug_full_system_timeline(
            K=instance.K,
            M=instance.M,
            N=instance.N,
            T=instance.T,
            DeltaT=instance.DeltaT,
            t0=t0,
            solution=solution,
            x=x,
            y=y,
            r=r,
            w=w,
            s=s,
        )


    ### (5) Request-Level Details
    if display_request_details:
        debug_requests_details(
            K=instance.K,
            M=instance.M,
            N=instance.N,
            T=instance.T,
            DeltaT=instance.DeltaT,
            t0=t0,
            solution=solution,
            r=r,
            d_in=instance.d_in,
            d_out=instance.d_out,
            q=getattr(instance, "q", None),
            s=s,
        )
