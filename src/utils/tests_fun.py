# You provide:
#   - grid_n (2,3,4,...)
#   - n_instances (how many random parameter-regimes to generate for that grid)
#   - repeat_seeds (list of seeds: each generated instance is re-run for all these seeds)
#
# The generator creates n_instances DIFFERENT inst_params (everything random except grid_n),
# and for each inst_params it runs ALL repeat_seeds.

import random
from pathlib import Path
import pandas as pd
import time
from utils.runs_fun import *
from utils.runs_heu_random_fun import *


# -----------------------------
# 1) Random inst_params generator (ONLY grid_n is fixed)
# -----------------------------
def sample_inst_params_for_grid(grid_n: int, rng: random.Random) -> dict:
    N = grid_n * grid_n

    # dt & horizon (ensure at least 4 time steps)
    dt = rng.choice([2, 3, 4, 5, 6, 10, 15])
    horizon = rng.choice([30, 40, 60, 90, 120, 180])
    while horizon // dt < 4:
        horizon = rng.choice([30, 40, 60, 90, 120, 180])

    # network
    mean_edge_length_km = rng.choice([1.0, 2.0, 3.33, 5.0, 7.5])
    mean_speed_kmh = rng.choice([20, 30, 40, 50, 60])
    rel_std = rng.choice([0.0, 0.15, 0.33, 0.50, 0.66, 0.85, 1.10])

    # depot
    depot = rng.choice([0, N - 1, grid_n - 1, N - grid_n, (grid_n // 2) * grid_n + (grid_n // 2)])

    # demand
    num_requests = rng.choice([2, 4, 6, 8, 10, 12, 15, 20, 25, 30])
    q_min = 1
    q_max = rng.choice([10, 15, 20, 30, 50])
    slack_min = rng.choice([10.0, 15.0, 20.0, 30.0, 40.0, 60.0])
    alpha = rng.choice([0.01, 0.03, 0.05, 0.10, 0.12, 0.20, 0.40, 0.60])

    # fleet
    num_modules = rng.choice([1, 2, 3, 4, 5, 8, 10, 15, 20])
    Q = rng.choice([4, 6, 8, 10, 12, 15])
    
    # trails: allow parking, so can exceed M*z_max, but keep it somewhat bounded
    # (still random and "free")
    base_trails = rng.choice([0, num_modules, 2 * num_modules, 2 * num_modules])
    num_trails = max(0, int(base_trails + rng.choice([-2, -1, 0, 1, 2])))
    
    # z_max must be <= |P| = num_trails
    z_candidates = [z for z in [0, 2, 4, 6, 8, 10] if z <= num_trails]
    z_max = rng.choice(z_candidates) if z_candidates else num_trails

    # Nw
    num_Nw = rng.choice([min(N, 2), min(N, 4), min(N, 6), max(2, N // 2), N])

    # costs
    c_km = rng.choice([0.2, 0.5, 1.0, 2.0, 5.0])
    c_uns = rng.choice([10.0, 50.0, 100.0, 200.0, 500.0])

    return {
        "number": grid_n,
        "horizon": horizon,
        "dt": dt,
        "mean_edge_length_km": float(mean_edge_length_km),
        "mean_speed_kmh": float(mean_speed_kmh),
        "rel_std": float(rel_std),

        "num_requests": int(num_requests),
        "q_min": int(q_min),
        "q_max": int(q_max),
        "slack_min": float(slack_min),
        "alpha": float(alpha),

        "num_modules": int(num_modules),
        "num_trails": int(num_trails),
        "Q": int(Q),
        "z_max": int(z_max),

        "c_km": float(c_km),
        "c_uns": float(c_uns),

        "num_Nw": int(num_Nw),
        "depot": int(depot),
    }


# -----------------------------
# 2) Create a batch of UNIQUE instances (parameter regimes)
# -----------------------------
def generate_instance_regimes(
    *,
    grid_n: int,
    n_instances: int,
    generator_seed: int = 999,   # seed for the generator itself (NOT the instance seeds)
):
    rng = random.Random(generator_seed)
    regimes = []
    seen = set()

    def key(p: dict):
        # stable key to avoid duplicates
        items = tuple(sorted(p.items()))
        return items

    while len(regimes) < n_instances:
        p = sample_inst_params_for_grid(grid_n, rng)
        k = key(p)
        if k in seen:
            continue
        seen.add(k)
        regimes.append(p)

    return regimes


# -----------------------------
# 3) Runner: for each regime, run ALL repeat_seeds
# -----------------------------
def regime_id(p: dict) -> str:
    return (
        f"GRID_n{p['number']}_H{p['horizon']}_dt{p['dt']}"
        f"_K{p['num_requests']}_q{p['q_min']}-{p['q_max']}_sl{p['slack_min']}_a{p['alpha']}"
        f"_M{p['num_modules']}_P{p['num_trails']}_Q{p['Q']}_Z{p.get('z_max')}"
        f"_Nw{p['num_Nw']}_dep{p['depot']}"
        f"_spd{p['mean_speed_kmh']}_L{p['mean_edge_length_km']}_std{p['rel_std']}"
        f"_ckm{p['c_km']}_cuns{p['c_uns']}"
    )


def run_generated_regimes(
    *,
    regimes: list[dict],
    repeat_seeds: list[int],
    model_name: str,
    base_results_dir: Path,
    run_exact: bool,
    run_heur: bool,
    cplex_cfg_exact: dict,
    cplex_cfg_heur: dict,
    heu_params: dict,
):
    all_rows = []

    for idx, inst_params in enumerate(regimes, start=1):
        rid = regime_id(inst_params)
        regime_root = base_results_dir / rid
        (regime_root / "summary").mkdir(parents=True, exist_ok=True)

        for seed in repeat_seeds:
            set_seed(seed)

            exp_id = f"{rid}_seed{seed}"

            seed_root = regime_root / f"seed_{seed}"
            paths = {
                "exact": seed_root / "exact",
                "heur": seed_root / "heuristic_prob",
                "summary": seed_root / "summary",
            }
            for p in paths.values():
                p.mkdir(parents=True, exist_ok=True)

            instance, network_cont_path, requests_cont_path, network_disc_path, requests_disc_path, t_max = build_instance_and_paths(
                inst_params=inst_params,
                seed=seed
            )

            res_exact = None
            res_heu = None

            if run_exact:
                t0 = time.time()
                res_exact = run_single_model(
                    instance=instance,
                    model_name=model_name,
                    inst_params=inst_params,
                    seed=seed,
                    exp_id=exp_id,
                    base_output_folder=paths["exact"],
                    cplex_cfg=cplex_cfg_exact,
                )
                res_exact.update({
                    "regime_index": idx,
                    "regime_id": rid,
                    "seed": seed,
                    "method": "EXACT",
                    "wall_time": time.time() - t0,
                })
                pd.DataFrame([res_exact]).to_csv(paths["summary"] / "summary_exact.csv", index=False)
                all_rows.append(res_exact)

            if run_heur:
                t0 = time.time()
                res_heu = run_heu_model(
                    instance=instance,
                    model_name=model_name,
                    network_cont_path=network_cont_path,
                    requests_cont_path=requests_cont_path,
                    network_disc_path=network_disc_path,
                    requests_disc_path=requests_disc_path,
                    inst_params=inst_params,
                    heu_params=heu_params,
                    seed=seed,
                    exp_id=exp_id,
                    base_output_folder=paths["heur"],
                    cplex_cfg=cplex_cfg_heur,
                )
                res_heu.update({
                    "regime_index": idx,
                    "regime_id": rid,
                    "seed": seed,
                    "method": "HEUR",
                    "wall_time": time.time() - t0,
                })
                pd.DataFrame([res_heu]).to_csv(paths["summary"] / "summary_heur.csv", index=False)
                all_rows.append(res_heu)

            pd.DataFrame([r for r in [res_exact, res_heu] if r is not None]).to_csv(
                paths["summary"] / "summary_all.csv", index=False
            )

        # regime-level summary across all seeds
        df_reg = pd.DataFrame([r for r in all_rows if r["regime_id"] == rid])
        df_reg.to_csv(regime_root / "summary" / "summary_all_seeds.csv", index=False)

    # global summary across everything
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(base_results_dir / "summary_ALL_regimes.csv", index=False)
    return df_all


