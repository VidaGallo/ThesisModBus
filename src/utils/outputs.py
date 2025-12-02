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

    folder = Path(base_dir) / network_group / network_dir / f"tmax{t_max}_dt{dt}"
    folder.mkdir(parents=True, exist_ok=True)

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

    print(f"[INFO] CPLEX model exported to: {lp_path}")
    print(f"[INFO] CPLEX log saved to: {log_path}")





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
    print(f"[INFO] Summary saved to: {summary_path}")


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
    print(f"[INFO] Model stats saved to: {stats_path}")
