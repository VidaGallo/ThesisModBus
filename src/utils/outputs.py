"""
CPLEX Logging and Output Utilities

Provides functions to:
- create structured output folders for experiments,
- save CPLEX logs and LP model files,
- store solution summaries (objective, status, gap, time),
- save basic model statistics (variables, constraints).

These utilities help keep optimization experiments organized and reproducible.
"""


from pathlib import Path
from docplex.mp.model import Model


def build_output_folder(base_dir: str, network_path: str, t_max: int, dt: int):
    """
    Build a structured folder for logs and results:
    base_dir/network_name/tmaxXXX_dtYY/

    Example:
        base_dir = "results"
        network_path = "instances/GRID/5x5/network_disc5min.json"

    Result folder:
        results/GRID/5x5/tmax100_dt5/
    """

    net = Path(network_path)

    # ex.extract GRID/5x5 from path
    network_dir = net.parent.name      # ex."5x5"
    network_group = net.parent.parent.name   # ex."GRID"

    folder = Path(base_dir) / network_group / network_dir / f"tmax{t_max*dt}_dt{dt}"
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










def save_solution_variables(solution, x, y, r, w, s, output_folder):
    """
    Save all decision variables that are equal to 1 (or >0.5 threshold)
    into CSV files inside:  output_folder / "variables" / ...

    Creates:
        variables/x_positions.csv
        variables/y_movements.csv
        variables/r_assignments.csv
        variables/w_swaps.csv
        variables/s_served.csv
    """

    thr = 0.5  # threshold

    # -------------------------
    # Create subfolder: variables/
    # -------------------------
    var_folder = output_folder / "variables"
    var_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # X: module positions
    # -------------------------
    with (var_folder / "x_positions.csv").open("w") as f:
        f.write("m,i,t,val\n")
        for (m, i, t), var in x.items():
            val = solution.get_value(var)
            if val > thr:
                f.write(f"{m},{i},{t},{val}\n")

    # -------------------------
    # Y: module movements
    # -------------------------
    with (var_folder / "y_movements.csv").open("w") as f:
        f.write("m,i,j,t,val\n")
        for (m, i, j, t), var in y.items():
            val = solution.get_value(var)
            if val > thr:
                f.write(f"{m},{i},{j},{t},{val}\n")

    # -------------------------
    # R: request assignments
    # -------------------------
    with (var_folder / "r_assignments.csv").open("w") as f:
        f.write("k,t,m,val\n")
        for (k, t, m), var in r.items():
            val = solution.get_value(var)
            if val > thr:
                f.write(f"{k},{t},{m},{val}\n")

    # -------------------------
    # W: swaps
    # -------------------------
    with (var_folder / "w_swaps.csv").open("w") as f:
        f.write("k,i,t,m,mp,val\n")
        for (k, i, t, m, mp), var in w.items():
            val = solution.get_value(var)
            if val > thr:
                f.write(f"{k},{i},{t},{m},{mp},{val}\n")

    # -------------------------
    # S: served
    # -------------------------
    with (var_folder / "s_served.csv").open("w") as f:
        f.write("k,val\n")
        for k, var in s.items():
            val = solution.get_value(var)
            f.write(f"{k},{val}\n")

    #print(f"[INFO] Decision variables saved in: {var_folder}")







def save_solution_variables_new(solution, x, y, r, s, z, output_folder: Path):
    """
    Save all decision variables that are equal to 1 (or >0.5 threshold)
    into CSV files inside:  output_folder / "variables" / ...

    Creates:
        variables/x_positions.csv
        variables/y_movements.csv
        variables/r_assignments.csv
        variables/z_exchange_nodes.csv
        variables/s_served.csv
    """

    thr = 0.5  # threshold

    # -------------------------
    # Create subfolder: variables/
    # -------------------------
    var_folder = output_folder / "variables"
    var_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # X: module positions
    # -------------------------
    with (var_folder / "x_positions.csv").open("w") as f:
        f.write("m,i,t,val\n")
        for (m, i, t), var in x.items():
            val = solution.get_value(var)
            if val > thr:
                f.write(f"{m},{i},{t},{val}\n")

    # -------------------------
    # Y: module movements
    # -------------------------
    with (var_folder / "y_movements.csv").open("w") as f:
        f.write("m,i,j,t,val\n")
        for (m, i, j, t), var in y.items():
            val = solution.get_value(var)
            if val > thr:
                f.write(f"{m},{i},{j},{t},{val}\n")

    # -------------------------
    # R: request assignments (k on m at time t)
    # -------------------------
    with (var_folder / "r_assignments.csv").open("w") as f:
        f.write("k,t,m,val\n")
        for (k, t, m), var in r.items():
            val = solution.get_value(var)
            if val > thr:
                f.write(f"{k},{t},{m},{val}\n")

    # -------------------------
    # Z: request k on module m at exchange node i, time t
    #     (linearizzazione di r*x sui nodi di scambio Nw)
    # -------------------------
    with (var_folder / "z_exchange_nodes.csv").open("w") as f:
        f.write("k,t,m,i,val\n")
        for (k, t, m, i), var in z.items():
            val = solution.get_value(var)
            if val > thr:
                f.write(f"{k},{t},{m},{i},{val}\n")

    # -------------------------
    # S: served (s_k)
    # -------------------------
    with (var_folder / "s_served.csv").open("w") as f:
        f.write("k,val\n")
        for k, var in s.items():
            val = solution.get_value(var)
            f.write(f"{k},{val}\n")
