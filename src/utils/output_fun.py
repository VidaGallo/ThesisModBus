"""
CPLEX Logging and Output Utilities
==================================

Utilities to manage and store the outputs of CPLEX optimization runs.
The module supports structured result folders, export of LP models and solver
logs, and saving of solution summaries and decision variables for
post-processing and reproducibility.
"""


from pathlib import Path
from docplex.mp.model import Model



def build_output_folder(base_dir: str, network_path: str, t_max: int, dt: int):
    """
    Build a structured folder for logs and results:
    base_dir/network_name/tmaxXXX_dtYY/

    Example:
        base_dir = "results"
        network_path = "instances/GRID/MT/5x5/network_disc5min.json"

    Result folder:
        results/GRID/MT/5x5/tmax100_dt5/
    """

    net = Path(network_path)

    # ex.extract GRID/5x5 from path
    network_dir = net.parent.name      # ex."5x5"
    network_group = net.parent.parent.name   # ex."GRID"

    folder = Path(base_dir) / network_group / "MT" / network_dir / f"tmax{t_max*dt}_dt{dt}"
    if not folder.exists():
        folder.mkdir(parents=True)

    return folder





def save_cplex_log(mdl: Model, output_folder: Path):
    """
    Save CPLEX log and LP model file in the given folder.
    """
    # path del file .lp come stringa
    lp_path = output_folder / "model.lp"
    mdl.export_as_lp(str(lp_path))

    log_path = output_folder / "cplex_log.txt"
    with log_path.open("w") as f:
        f.write("==== CPLEX SOLVE LOG ====\n")
        f.write(str(mdl.solve_details))

    #print(f"[INFO] CPLEX model exported to: {lp_path}")
    #print(f"[INFO] CPLEX log saved to: {log_path}")





def save_solution_summary(solution, output_folder: Path):
    """
    Save basic summary info about the solution.
    """
    summary_path = output_folder / "solution_summary.txt"
    with summary_path.open("w") as f:
        f.write("==== SOLUTION SUMMARY ====\n")
        f.write(f"Objective Value: {solution.objective_value}\n")
        f.write(f"Status: {solution.solve_status}\n")
        f.write(f"Gap: {solution.solve_details.mip_relative_gap}\n")
        f.write(f"Time: {solution.solve_details.time}\n")
    #print(f"[INFO] Summary saved to: {summary_path}")


def save_model_stats(mdl: Model, output_folder: Path):
    """
    Write number of variables, constraints, and other model stats.
    """
    stats_path = output_folder / "model_stats.txt"
    with stats_path.open("w") as f:
        f.write("==== MODEL STATISTICS ====\n")
        f.write(f"Variables: {mdl.number_of_variables}\n")
        f.write(f"Binary Vars: {mdl.number_of_binary_variables}\n")
        f.write(f"Integer Vars: {mdl.number_of_integer_variables}\n")
        f.write(f"Constraints: {mdl.number_of_constraints}\n")
    #print(f"[INFO] Model stats saved to: {stats_path}")





def save_solution_variables_flex(
    solution,
    output_folder: Path,
    *,
    x=None,
    y=None,
    r=None,
    w=None,
    z=None,          # vecchia z (se la usi ancora in altri modelli)
    s=None,
    L=None,
    R=None,
    a=None,
    b=None,
    h=None,         # h può essere (i,j,t) oppure (m,i,j,t)
    D=None,         # NEW: D[m, i, t]
    U=None,         # NEW: U[m, i, t]
    z_main=None,    # NEW: z[m, t]
    kappa=None,     # NEW: kappa[i, t]
    thr: float = 0.5,
):
    """
    Save decision variables (only those > thr) into CSV files.

    Tutti i dizionari di variabili sono opzionali: se sono None vengono ignorati.

    Parametri tipici per i vari modelli:
        x[(m, i, t)]
        y[(m, i, j, t)]
        r[(k, t, m)]
        w[(k, i, t, m, mp)]
        z[(k, t, m, i)]           # vecchio modello (linearizzazione)
        s[k]
        L[(k, i, t, m)]
        R[(k, i, t, m)]
        a[(k, t, m)]
        b[(k, t, m)]
        h[(i, j, t)] oppure h[(m, i, j, t)]

        D[(m, i, t)]              # AB + TRAIL
        U[(m, i, t)]
        z_main[(m, t)]
        kappa[(i, t)]
    """

    # -------------------------
    # Create subfolder: variables/
    # -------------------------
    var_folder = output_folder / "variables"
    var_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # X: module positions
    # -------------------------
    if x is not None:
        with (var_folder / "x_positions.csv").open("w") as f:
            f.write("m,i,t,val\n")
            for (m, i, t), var in x.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{m},{i},{t},{val}\n")

    # -------------------------
    # Y: module movements
    # -------------------------
    if y is not None:
        with (var_folder / "y_movements.csv").open("w") as f:
            f.write("m,i,j,t,val\n")
            for (m, i, j, t), var in y.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{m},{i},{j},{t},{val}\n")

    # -------------------------
    # R: request assignments
    # -------------------------
    if r is not None:
        with (var_folder / "r_assignments.csv").open("w") as f:
            f.write("k,t,m,val\n")
            for (k, t, m), var in r.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{k},{t},{m},{val}\n")

    # -------------------------
    # W: swaps (taxi-like vecchia versione)
    # -------------------------
    if w is not None:
        with (var_folder / "w_swaps.csv").open("w") as f:
            f.write("k,i,t,m,mp,val\n")
            for (k, i, t, m, mp), var in w.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{k},{i},{t},{m},{mp},{val}\n")

    # -------------------------
    # Z (vecchia): exchange nodes (linearizzazione r*x sui nodi di scambio)
    # -------------------------
    if z is not None:
        with (var_folder / "z_exchange_nodes.csv").open("w") as f:
            f.write("k,t,m,i,val\n")
            for (k, t, m, i), var in z.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{k},{t},{m},{i},{val}\n")

    # -------------------------
    # S: served
    # -------------------------
    if s is not None:
        with (var_folder / "s_served.csv").open("w") as f:
            f.write("k,val\n")
            for k, var in s.items():
                val = solution.get_value(var)
                f.write(f"{k},{val}\n")

    # -------------------------
    # L: leave (modello Tris)
    # -------------------------
    if L is not None:
        with (var_folder / "L_leave.csv").open("w") as f:
            f.write("k,i,t,m,val\n")
            for (k, i, t, m), var in L.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{k},{i},{t},{m},{val}\n")

    # -------------------------
    # R: receive (modello Tris)
    # -------------------------
    if R is not None:
        with (var_folder / "R_receive.csv").open("w") as f:
            f.write("k,i,t,m,val\n")
            for (k, i, t, m), var in R.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{k},{i},{t},{m},{val}\n")

    # -------------------------
    # A: boarding events
    # -------------------------
    if a is not None:
        with (var_folder / "a_boarding.csv").open("w") as f:
            f.write("k,t,m,val\n")
            for (k, t, m), var in a.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{k},{t},{m},{val}\n")

    # -------------------------
    # B: alighting events
    # -------------------------
    if b is not None:
        with (var_folder / "b_alighting.csv").open("w") as f:
            f.write("k,t,m,val\n")
            for (k, t, m), var in b.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{k},{t},{m},{val}\n")

    # -------------------------
    # H: arc flows (supporta sia (i,j,t) sia (m,i,j,t))
    # -------------------------
    if h is not None and len(h) > 0:
        # guarda la struttura della chiave
        sample_key = next(iter(h.keys()))
        if len(sample_key) == 3:
            # h[(i,j,t)]
            with (var_folder / "h_arc_flows.csv").open("w") as f:
                f.write("i,j,t,val\n")
                for (i, j, t), var in h.items():
                    val = solution.get_value(var)
                    if val > thr:
                        f.write(f"{i},{j},{t},{val}\n")
        elif len(sample_key) == 4:
            # h[(m,i,j,t)] nuovo modello
            with (var_folder / "h_arc_flows_m.csv").open("w") as f:
                f.write("m,i,j,t,val\n")
                for (m, i, j, t), var in h.items():
                    val = solution.get_value(var)
                    if val > thr:
                        f.write(f"{m},{i},{j},{t},{val}\n")

    # -------------------------
    # D: TRAIL down
    # -------------------------
    if D is not None:
        with (var_folder / "D_trail_down.csv").open("w") as f:
            f.write("m,i,t,val\n")
            for (m, i, t), var in D.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{m},{i},{t},{val}\n")

    # -------------------------
    # U: TRAIL up
    # -------------------------
    if U is not None:
        with (var_folder / "U_trail_up.csv").open("w") as f:
            f.write("m,i,t,val\n")
            for (m, i, t), var in U.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{m},{i},{t},{val}\n")

    # -------------------------
    # z_main: n° TRAIL attaccati al MAIN m,t
    # -------------------------
    if z_main is not None:
        with (var_folder / "z_main_trail.csv").open("w") as f:
            f.write("m,t,val\n")
            for (m, t), var in z_main.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{m},{t},{val}\n")

    # -------------------------
    # kappa: TRAIL parcheggiati nei nodi di scambio
    # -------------------------
    if kappa is not None:
        with (var_folder / "kappa_trail_nodes.csv").open("w") as f:
            f.write("i,t,val\n")
            for (i, t), var in kappa.items():
                val = solution.get_value(var)
                if val > thr:
                    f.write(f"{i},{t},{val}\n")
