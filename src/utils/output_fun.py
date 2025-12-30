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
import json, hashlib
from datetime import datetime
import pandas as pd



RUNNER_VERSION = "v1"     # If there was soemthing modified the hash should change


### Funzioni per controllare se la run è già stata fatta e fare hash
def canonical_dumps(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def make_hash(params: dict, n: int = 12) -> str:
    s = canonical_dumps(params)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def write_meta(base: Path, meta: dict) -> None:
    meta = dict(meta)
    meta["created_at"] = datetime.now().isoformat(timespec="seconds")
    (base / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


### Lettura summary 
SUMMARY_FILENAMES = {
    "exact": "summary_exact.csv",
    "heuristic": "summary_heuristic.csv",
    "exact_flow": "summary_exact_flow.csv",          # futuro
    #...
}
def summary_csv_path(run_folder: Path, run_kind: str) -> Path:
    try:
        filename = SUMMARY_FILENAMES[run_kind]
    except KeyError:
        raise ValueError(
            f"Unknown run_kind '{run_kind}'. "
        )
    return run_folder / filename


### Controlalre se si può saltare il run (se il summary ed il file meta esistono già + non sono vuoti)
def can_skip_run(run_folder: Path, run_kind: str) -> bool:
    meta_ok = (run_folder / "meta.json").exists()
    summary_path = summary_csv_path(run_folder, run_kind)
    if not meta_ok or not summary_path.exists():
        return False
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return False

    # non vuoto: almeno 1 riga e almeno 1 colonna
    return (not df.empty) and (df.shape[1] > 0)


### Lettura della hash dell'istanza
def read_instance_hash(instance_folder: Path) -> str:
    meta_path = instance_folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return str(meta["hash"])



### Costruzione cartella dove salvare i results
def build_run_folder(
    results_root: Path,
    dataset: str,                 # "GRID_ASYM", "GRID", "CITY"
    instance_hash: str,
    run_kind: str,                # "exact" o "heuristic"
    run_params: dict,             # cose che cambiano il run
) -> tuple[Path, str]:
    params = dict(
        runner_version=RUNNER_VERSION,
        dataset=dataset,
        instance_hash=instance_hash,
        kind=run_kind,
        run_params=run_params,
    )
    run_hash = make_hash(params)

    folder = results_root / dataset / instance_hash / run_kind / run_hash
    folder.mkdir(parents=True, exist_ok=True)
    return folder, run_hash


### Salvataggio sel summary
def save_summary_csv(res: dict, run_folder: Path, run_kind: str) -> Path:
    p = summary_csv_path(run_folder, run_kind)
    pd.DataFrame([res]).to_csv(p, index=False)
    return p

### Lettura del summary
def load_summary_csv(run_folder: Path, run_kind: str) -> dict:
    p = summary_csv_path(run_folder, run_kind)
    df = pd.read_csv(p)
    return df.iloc[0].to_dict()



### Salvataggio log cplex
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




### Salvataggio stato modello
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




### Salvataggio delle variabili
def save_solution_variables_flex(
    solution,
    output_folder: Path,
    x=None,
    y=None,
    r=None,
    w=None,        
    s=None,
    L=None,
    R=None,
    a=None,
    b=None,
    h=None,         
    D=None,         
    U=None,         
    z_main=None,  
    kappa=None, 
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
    if solution is None:  ### Non c'è niente da salvare
        return

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
