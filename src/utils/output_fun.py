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
from typing import Any, Dict, Tuple, Iterable



def build_output_folder(base_dir: str, network_path: str):
    """
    Build a structured folder for logs and results:

    Example:
        base_dir = "results"
        network_path = "instances/GRID/5x5/network_disc5min.json"

    Result folder:
        results/GRID/5x5/
    """

    net = Path(network_path)

    # ex.extract GRID/5x5 from network_path
    network_dir = net.parent.name      # ex."5x5"
    network_group = net.parent.parent.name   # ex."GRID"

    folder = Path(base_dir) / network_group / network_dir
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




### Save solution summary from solution
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


### Save solution summary from a dict
def save_solution_dict_summary(
    sol_dict: Dict[str, Dict[Any, float]],
    objective: float | None,
    output_folder: Path,
    served_keys: Iterable[Any] | None = None,
    extra: Dict[str, Any] | None = None,     # Related to heuristic info!
):
    """
    Salva solution_summary.txt partendo da una soluzione come dizionario.
    """
    output_folder = Path(output_folder)
    summary_path = output_folder / "solution_summary.txt"

    # served
    if served_keys is not None:
        served = len(list(served_keys))
        total = served
    else:
        s = sol_dict.get("s", {})
        served_keys = [k for k, v in s.items() if v is not None and v > 0.5]
        served = len(served_keys)
        total = len(s)

    served_ratio = served / total if total > 0 else 0.0

    with summary_path.open("w") as f:
        f.write("==== SOLUTION SUMMARY (DICT) ====\n")
        f.write(f"Objective Value: {objective}\n")
        f.write("Status: Heuristic / Dict-based\n")
        f.write(f"Served: {served}/{total}\n")
        f.write(f"Served ratio: {served_ratio:.4f}\n")

        if extra:
            f.write("\n---- EXTRA ----\n")
            for k, v in extra.items():
                f.write(f"{k}: {v}\n")





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






### Scrittura variaibli nel file
def _write_var_block(solution, f, name, var_dict, header, thr=0.5):
    """
    Write a block like:
    === x ===
    m,i,t,val
    ...
    """
    if var_dict is None or len(var_dict) == 0:
        return

    f.write(f"\n=== {name} ===\n")
    f.write(header + "\n")

    for key, var in var_dict.items():
        val = solution.get_value(var)
        if val is None:
            continue
        if val > thr:
            # key can be tuple or scalar
            if isinstance(key, tuple):
                f.write(",".join(map(str, key)) + f",{val}\n")
            else:
                f.write(f"{key},{val}\n")



### Salvataggio singole variabili
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

    ### Scrittura variabili in singolo file
    bundle_path = var_folder / "all_variables.txt"
    with bundle_path.open("w") as f:
        # same thresholds as single files
        _write_var_block(solution, f, "x", x, "m,i,t,val", thr=thr)
        _write_var_block(solution, f, "y", y, "m,i,j,t,val", thr=thr)
        _write_var_block(solution, f, "r", r, "k,t,m,val", thr=thr)
        _write_var_block(solution, f, "w", w, "k,i,t,m,mp,val", thr=thr)
        _write_var_block(solution, f, "z_old", z, "k,t,m,i,val", thr=thr)
        _write_var_block(solution, f, "s", s, "k,val", thr=0.0)        # s voglio sempre
        _write_var_block(solution, f, "L", L, "k,i,t,m,val", thr=thr)
        _write_var_block(solution, f, "R", R, "k,i,t,m,val", thr=thr)
        _write_var_block(solution, f, "a", a, "k,t,m,val", thr=thr)
        _write_var_block(solution, f, "b", b, "k,t,m,val", thr=thr)

        # h: detect key size like you already do
        if h is not None and len(h) > 0:
            sample_key = next(iter(h.keys()))
            if isinstance(sample_key, tuple) and len(sample_key) == 3:
                _write_var_block(solution, f, "h", h, "i,j,t,val", thr=thr)
            elif isinstance(sample_key, tuple) and len(sample_key) == 4:
                _write_var_block(solution, f, "h", h, "m,i,j,t,val", thr=thr)

        _write_var_block(solution, f, "D", D, "m,i,t,val", thr=thr)
        _write_var_block(solution, f, "U", U, "m,i,t,val", thr=thr)
        _write_var_block(solution, f, "z_main", z_main, "m,t,val", thr=thr)
        _write_var_block(solution, f, "kappa", kappa, "i,t,val", thr=thr)







### Save a solution directly from a dictionary!
def save_solution_dict_variables_flex(
    sol_dict: Dict[str, Dict[Any, float]],
    output_folder: Path,
    thr: float = 0.5,
) -> Path:
    """
    Salva una soluzione già estratta come dizionario (current_solution) in CSV,
    creando la sottocartella: output_folder/variables/.

    Atteso:
      sol_dict["x"][(m,i,t)] = val
      sol_dict["y"][(m,i,j,t)] = val
      sol_dict["r"][(k,t,m)] = val
      ...
      sol_dict["s"][k] = val   (qui salva sempre, anche sotto soglia)

    Scrive:
      variables/x_positions.csv, y_movements.csv, ..., + all_variables.txt
    """
    output_folder = Path(output_folder)
    var_folder = output_folder / "variables"
    var_folder.mkdir(parents=True, exist_ok=True)

    # nome -> (filename, header)
    spec: Dict[str, Tuple[str, str]] = {
        "x": ("x_positions.csv", "m,i,t,val"),
        "y": ("y_movements.csv", "m,i,j,t,val"),
        "r": ("r_assignments.csv", "k,t,m,val"),
        "w": ("w_swaps.csv", "k,i,t,m,mp,val"),
        "z_old": ("z_exchange_nodes.csv", "k,t,m,i,val"),
        "s": ("s_served.csv", "k,val"),
        "L": ("L_leave.csv", "k,i,t,m,val"),
        "R": ("R_receive.csv", "k,i,t,m,val"),
        "a": ("a_boarding.csv", "k,t,m,val"),
        "b": ("b_alighting.csv", "k,t,m,val"),
        "h": ("h_arc_flows.csv", "i,j,t,val"),
        "h_m": ("h_arc_flows_m.csv", "m,i,j,t,val"),
        "D": ("D_trail_down.csv", "m,i,t,val"),
        "U": ("U_trail_up.csv", "m,i,t,val"),
        "z_main": ("z_main_trail.csv", "m,t,val"),
        "kappa": ("kappa_trail_nodes.csv", "i,t,val"),
    }

    def iter_rows(d: Dict[Any, float], keep_all: bool = False):
        for key, val in d.items():
            if val is None:
                continue
            if (not keep_all) and val <= thr:
                continue
            if isinstance(key, tuple):
                yield (*key, val)
            else:
                yield (key, val)

    def write_csv(path: Path, header: str, rows):
        with path.open("w") as f:
            f.write(header + "\n")
            for row in rows:
                f.write(",".join(map(str, row)) + "\n")

    # singoli csv
    for name, (fname, header) in spec.items():
        d = sol_dict.get(name)
        if not d:
            continue
        keep_all = (name == "s")
        write_csv(var_folder / fname, header, iter_rows(d, keep_all=keep_all))

    # bundle unico identico stile "all_variables.txt"
    bundle_path = var_folder / "all_variables.txt"
    with bundle_path.open("w") as f:
        for name in [
            "x", "y", "r", "w", "z_old", "s", "L", "R", "a", "b", "h", "h_m",
            "D", "U", "z_main", "kappa"
        ]:
            d = sol_dict.get(name)
            if not d:
                continue
            f.write(f"\n=== {name} ===\n")
            f.write(spec[name][1] + "\n")  # header
            keep_all = (name == "s")
            for row in iter_rows(d, keep_all=keep_all):
                f.write(",".join(map(str, row)) + "\n")

