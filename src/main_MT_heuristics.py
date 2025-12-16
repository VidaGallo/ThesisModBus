from utils.MT.heuristic_fun import *
from utils.MT.runs_fun import *





if __name__ == "__main__":

    # -----------------------------
    # CHOOSE MODE
    # -----------------------------
    MODE = "GRID"   # "GRID" or "CITY"

    # -----------------------------
    # COMMON PARAMS
    # -----------------------------
    seed = 23
    horizon = 120          # minutes
    dt = 3
    num_requests = 30
    q_min, q_max = 1, 6
    slack_min = 20.0
    depot = 0

    number = 3
    num_modules = 3
    num_trails  = 6
    Q = 10
    c_km = 1.0
    c_uns = 100.0
    g_plat = None
    num_Nw = 3

    # output base
    base_output_folder = Path("results") / f"{MODE}_h{horizon}_dt{dt}_K{num_requests}_seed{seed}"
    base_output_folder.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # DATA GENERATION + INSTANCE
    # -----------------------------
    if MODE == "GRID":

        # 1) generate network + requests JSON (your existing function)
        network_path, requests_path = generate_all_data_asym(
            number=number,
            horizon=horizon,
            dt=dt,
            num_requests=num_requests,
            q_min=q_min,
            q_max=q_max,
            slack_min=slack_min,
            depot=depot
        )
        network_cont_path  = f"instances/GRID/{number}x{number}/network.json"
        requests_cont_path = f"instances/GRID/{number}x{number}/taxi_like_requests_{horizon}maxmin.json"

        G = load_network_continuous_as_graph(network_cont_path)

        req4d = build_requests_4d_from_file(requests_cont_path)

        print("\nK =", len(req4d))
        print("\n--- REQUEST NODES (4D representation) ---")
        for r in req4d:
            print(
                f"k={r['k']:2d} | "
                f"O={r['o']}, tP={r['tP']:.2f} -> "
                f"D={r['d']}, tD={r['tD']:.2f} | "
                f"q={r['q']}"
            )


        # 1) embedding nodi (x,y) che “imitano” shortest path time_min
        node_xy = mds_embed_nodes_from_sp(G, weight="time_min", symmetrize="avg", dim=2)

        # 2) richieste 6D
        req6d = build_requests_6d_from_4d(req4d, node_xy)


        print("\n\nK =", len(req6d))
        print("\n--- REQUEST NODES (6D representation) ---")
        for i in range(len(req4d)):
            r4 = req4d[i]
            r6 = req6d[i]
            print(
                f"k={r4['k']:2d} | "
                f"O={r4['o']} -> ({r6['xo']:.3f},{r6['yo']:.3f}), "
                f"D={r4['d']} -> ({r6['xd']:.3f},{r6['yd']:.3f}), "
                f"tP={r4['tP']:.2f}, tD={r4['tD']:.2f}, q={r4['q']}"
            )

        W6 = build_fully_connected_request_6d_matrix(
            req6d,
            alpha_O=1.0, beta_P=1.0,
            alpha_D=1.0, beta_D=1.0,
            standardize=True
        )

        print("\nW6 shape =", W6.shape)





        # ---- 6D ----
        X, Xn, c, dist, mu, sd = compute_centroid_distances(req6d, use_capacity=False, standardize=True)
        print_req_centroid_debug(req6d, Xn, dist, use_capacity=False)

        # richiesta “media fittizia” (centroide) in spazio standardizzato
        print("\nCentroid (6D, standardized):", np.round(c, 4))


        # ---- 7D (aggiungo capacità q) ----
        X7, Xn7, c7, dist7, mu7, sd7 = compute_centroid_distances(req6d, use_capacity=True, capacity_key="q", standardize=True)
        print_req_centroid_debug(req6d, Xn7, dist7, use_capacity=True)
        print("\nCentroid (7D, standardized):", np.round(c7, 4))



        KEEP = 4  # quante “più rappresentative” vuoi, con minor varianza, che se rimosse cambiano meno la distribuzione

        ### NON ITERATIVE VERSION ###
        # ---- 6D ----
        idx6, top6, dist6, c6 = topk_closest_to_centroid(req6d, k=KEEP, use_capacity=False, standardize=True)
        print_topk(top6, dist6, use_capacity=False)
        print("\nIndices 6D:", idx6.tolist())

        # ---- 7D ----
        idx7, top7, dist7, c7 = topk_closest_to_centroid(req6d, k=KEEP, use_capacity=True, capacity_key="q", standardize=True)
        print_topk(top7, dist7, use_capacity=True)      
        print("Indices 7D:", idx7.tolist())



        ### ITERATIVE VERSION ###
        # -------- 6D --------
        steps6, remaining6 = iterative_remove_by_centroid(req6d, n_remove=KEEP, use_capacity=False, standardize=True, mode="closest")
        print_removed_steps(steps6, use_capacity=False)
        print("Selected (iterative 6D):", [s["removed_k"] for s in steps6])

        # -------- 7D --------
        steps7, remaining7 = iterative_remove_by_centroid(req6d, n_remove=KEEP, use_capacity=True, capacity_key="q", standardize=True, mode="closest")
        print_removed_steps(steps7, use_capacity=True)
        print("Selected (iterative 7D):", [s["removed_k"] for s in steps7])





        #### Crea richieste fittizie da remaining (usando 7D clustering) ####

        # --- 1) richieste fisse ---
        fixed_k = [s["removed_k"] for s in steps7]
        print("Fixed k:", fixed_k)

        FICT = 5
        # --- 2) crea 4 fittizie dalle altre 26 ---
        fict6d, remaining, labels = make_fictitious_requests_from_remaining(
            req6d_all=req6d,
            fixed_k=fixed_k,
            n_fict=FICT,
            use_capacity=True,      # clustering in 7D
            standardize=True,
            random_state=23,
        )

        print("\n--- FICTITIOUS (in embedding coords) ---")
        for f in fict6d:
            print(f"k={f['k']} | n_agg={f['n_agg']} | "
                f"O=({f['xo']:.3f},{f['yo']:.3f}) -> D=({f['xd']:.3f},{f['yd']:.3f}) | "
                f"tP={f['tP']:.2f} tD={f['tD']:.2f} q={f['q']}")

        # --- 3) snap su nodi reali ---
        fict_graph = [snap_fict_request_to_graph_nodes(f, node_xy) for f in fict6d]

        print("\n--- FICTITIOUS (snapped to graph node IDs) ---")
        for f in fict_graph:
            print(f"k={f['k']} | n_agg={f['n_agg']} | "
                f"o={f['o']} d={f['d']} | tP={f['tP']:.2f} tD={f['tD']:.2f} q={f['q']}")
            


        fixed_reqs_real = [r for r in req4d if r["k"] in set(fixed_k)]

        # converti fict_graph nel formato req4d
        fict_reqs_4d = [{
            "k": f["k"],
            "o": f["o"],
            "tP": f["tP"],
            "d": f["d"],
            "tD": f["tD"],
            "q": f["q"],
        } for f in fict_graph]

        final_requests = fixed_reqs_real + fict_reqs_4d
        print("\nFINAL K =", len(final_requests))


        # salva json nel formato identico a quello originale
        out_path = base_output_folder / "requests_REDUCED.json"
        out_json = []
        for r in final_requests:
            out_json.append({
                "id": int(r["k"]),  # ok anche negativo se il loader lo accetta
                "origin": int(r["o"]),
                "destination": int(r["d"]),
                "q_k": int(r["q"]),
                "desired_departure_min": float(r["tP"]),
                "desired_arrival_min": float(r["tD"]),
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2)

        print("Saved reduced requests to:", out_path)

        










    elif MODE == "CITY":
        city = "Torino, Italia"
        subdir = "TORINO_SUB"
        central_suburbs = ["Centro", "Crocetta", "Santa Rita", "Aurora"]
        depot = 1198867366

        # 1) generate network + requests JSON (your existing function)
        network_path, requests_path = generate_all_data_city(
            city=city,
            subdir=subdir,
            central_suburbs=central_suburbs,
            horizon=horizon,
            dt=dt,
            num_requests=num_requests,
            q_min=q_min,
            q_max=q_max,
            slack_min=slack_min,
            depot=depot
        )

        # 2) build instance
        t_max = horizon // dt
        instance = load_instance_discrete(
            network_path=network_path,
            requests_path=requests_path,
            dt=dt,
            t_max=t_max,
            num_modules=num_modules,
            num_trail=num_trails,
            Q=Q,
            c_km=c_km,
            c_uns=c_uns,
            g_plat=g_plat,
            depot=depot,
            num_Nw=num_Nw
        )



    else:
        raise ValueError(f"Unknown MODE: {MODE}")
