from docplex.mp.model import Model
from utils.MT.instance_def import Instance




def create_decision_variables_ab(mdl: Model, I: Instance):
    """
    Create all MILP decision variables for the taxi-like model.
    """
    M = I.M
    N = I.N
    Nw = I.Nw
    print("Nw:", Nw)
    A = I.A
    T = I.T

    # x[m,i,t]
    x = mdl.binary_var_dict(
        keys=[(m, i, t) for m in M for i in N for t in T],
        name="x"
    )

    # y[m,i,j,t]
    y = mdl.binary_var_dict(
        keys=[(m, i, j, t) for m in M for (i, j) in A for t in T],
        name="y"
    )

    # r[k,t,m], only for t ∈ ΔT_k
    r = mdl.binary_var_dict(
        keys=[(k, t, m)
              for k in I.K
              for t in I.DeltaT[k]
              for m in M],
        name="r"
    )

    # w[k,i,t,m,mp] only for:
    #   - t ∈ ΔT_k (request is "alive")
    #   - i ∈ Nw (internal nodes where swaps are allowed)
    w = mdl.binary_var_dict(
        keys=[
            (k, i, t, m, mp)
            for k in I.K
            for t in I.DeltaT[k]
            if (t > T[0]) and (t < T[-1])    # no exchanges at the beginning and at the end
            for i in Nw
            for m in M
            for mp in M
            if m != mp
        ],
        name="w"
    )

    # s[k]
    s = mdl.binary_var_dict(
        keys=[k for k in I.K],
        name="s"
    )

    
    # a[k,t,m]
    a = mdl.binary_var_dict(
        keys=[
            (k, t, m)
            for k in I.K
            for t in I.DeltaT[k]
            for m in M
        ],
        name="a"
    )

    # b[k,t,m]
    b = mdl.binary_var_dict(
        keys=[
            (k, t, m)
            for k in I.K
            for t in I.DeltaT[k]
            for m in M
        ],
        name="b"
    )


    Z_max = I.Z_max
    P = I.P

    # D[m,i,t] = n° TRAIL rilasciati dal MAIN m nel nodo i al tempo t
    D = mdl.integer_var_dict(
        keys=[(m, i, t)
              for m in M
              for i in Nw
              for t in T
              if t != T[0]],
        lb=0,
        ub=Z_max,
        name="D"
    )

        # U[m,i,t] = n° TRAIL presi dal MAIN m nel nodo i al tempo t
    U = mdl.integer_var_dict(
        keys=[(m, i, t)
              for m in M
              for i in Nw
              for t in T
              if t != T[0]],
        lb=0,
        ub=Z_max,
        name="U"
    )

    # z[m,t] = n° TRAIL attaccati al MAIN m al tempo t
    z = mdl.integer_var_dict(
        keys=[(m, t) for m in M for t in T],
        lb=0,
        ub=Z_max,
        name="z"
    )

    # kappa[i,t] = n° TRAIL “parcheggiati” nel nodo di scambio i al tempo t
    kappa = mdl.integer_var_dict(
        keys=[(i, t) for i in Nw for t in T],
        lb=0,
        ub=len(P),
        name="kappa"
    )

    # h[m,i,j,t] or h[i,j,t]
    h = mdl.integer_var_dict(
        keys=[(m, i, j, t) for m in M for (i, j) in A for t in T],
        lb=0,
        ub=Z_max+1,
        name="h"
    )

    return x, y, r, w, s, a, b, D, U, z, kappa, h







def add_taxi_like_constraints_ab(mdl, I, x, y, r, w, s, a, b, D, U, z, kappa, h):
    """
    Add all constraints of the taxi-like MILP model to the docplex model.
    """
    N = I.N
    Nw = I.Nw
    A = I.A
    M = I.M
    Z_max = I.Z_max
    P = I.P
    K = I.K
    T = I.T
    tau = I.tau_arc
    q = I.q
    Q = I.Q
    DeltaT = I.DeltaT
    d_in = I.d_in
    d_out = I.d_out

    # T = [1, 2, ..., t_max]
    t0 = T[0]           # first time period (1)
    T_pos = T[1:]       # all t > t0

    depot = I.depot

    # ------------------------------------------------------------------
    # 0) Initial conditions:
    #    ...
    # ------------------------------------------------------------------ 
    for m in M:
        mdl.add_constraint(
            x[m, depot, t0] == 1,
            ctname=f"initial_position_at_depot_m{m}"
        )
    
    # Tutti i moduli TRAIL sono connessi a qualche MAIN al tempo t0:
    mdl.add_constraint(
        mdl.sum(z[m, t0] for m in M) + mdl.sum(kappa[i, t0] for i in Nw) == len(P),
        ctname="initial_trail_attached_t0"
    )
    mdl.add_constraint(
        mdl.sum(kappa[i, t0] for i in Nw) == 0,
        ctname="initial_trail_stock_zero"
    )


    # ------------------------------------------------------------------
    # 0.bis) Z_max:
    #    ...
    # ------------------------------------------------------------------ 
    # Nessun MAIN può avere più di Z_max moduli TRAIL:
    Z_max = I.Z_max
    for m in M:
        for t in T:
            mdl.add_constraint(
                z[m, t] <= Z_max,
                ctname=f"z_upper_bound_m{m}_t{t}"
            )




    # ------------------------------------------------------------------
    # 1) Modules position: 
    #    sum_i x[m,i,t] <= 1  ∀m,t
    # ------------------------------------------------------------------
    for m in M:
        for t in T:
            mdl.add_constraint(
                mdl.sum(x[m, i, t] for i in N) <= 1,
                ctname=f"module_location_m{m}_t{t}"
            )


    # ------------------------------------------------------------------
    # 2) Module movement: 
    #    sum_{j:(i,j)∈A} y[m,i,j,t] <= x[m,i,t]   ∀m,i,t
    # ------------------------------------------------------------------
    for m in M:
        for i in N:
            outgoing = [ (u,v) for (u,v) in A if u == i ]
            if not outgoing:
                continue
            for t in T:
                mdl.add_constraint(
                    mdl.sum(y[m, i, j, t] for (i2,j) in outgoing) <= x[m, i, t],
                    ctname=f"one_departure_m{m}_i{i}_t{t}"
                )


    # ------------------------------------------------------------------
    # 3) Movement consistency:
    #
    # x[m,i,t] =
    #   x[m,i,t-1]
    #   - sum_{j:(i,j)∈A} y[m,i,j,t-1]
    #   + sum_{h:(h,i)∈A, t - tau(h,i) >= t0} y[m,h,i,t - tau(h,i)]
    #
    # ∀m,i, t > t0
    # ------------------------------------------------------------------
    for m in M:
        for i in N:
            outgoing = [ (u,v) for (u,v) in A if u == i ]
            incoming = [ (hh,v) for (hh,v) in A if v == i ]

            for t in T_pos:   # t > t0
                lhs = x[m, i, t]

                # x[m,i,t-1]
                rhs = x[m, i, t-1]

                # - sum_{j:(i,j)∈A} y[m,i,j,t-1]
                if outgoing:
                    rhs -= mdl.sum(y[m, i, j, t-1] for (i2,j) in outgoing)

                # + sum_{h:(h,i)∈A, t - tau(h,i) >= t0} y[m,h,i,t - tau(h,i)]
                incoming_terms = []
                for (hh, i2) in incoming:
                    travel = tau[(hh, i)]
                    t_depart = t - travel
                    if t_depart >= t0:
                        incoming_terms.append(y[m, hh, i, t_depart])
                if incoming_terms:
                    rhs += mdl.sum(incoming_terms)

                mdl.add_constraint(
                    lhs == rhs,
                    ctname=f"move_consistency_m{m}_i{i}_t{t}"
                )





    # ------------------------------------------------------------------
    # 4) Module capacity with TRAIL:
    #    sum_{k∈K} q_k * r[k,t,m] <= Q * (1 + z[m,t])   ∀m,t
    #    (only k s.t. (k,t,m) exists in r)
    # ------------------------------------------------------------------
    for m in M:
        for t in T:
            mdl.add_constraint(
                mdl.sum(
                    q[k] * r[k, t, m]
                    for k in K
                    if (k, t, m) in r
                ) <= Q * (1 + z[m, t]),
                ctname=f"capacity_with_trail_m{m}_t{t}"
            )

    # ------------------------------------------------------------------
    # 5) SCAMBIO MODULI TRAIL (D, U, z, kappa)
    # ------------------------------------------------------------------
    
    Z_max = I.Z_max
    t0 = T[0]

    # 5.1) Lo scambio può avvenire solo se si ha qualche TRAIL da scambiare
    #    D^m_{i,t} <= z^m_t              ∀ m∈M, i∈Nw, t∈T \ {t0}
    for m in M:
        for t in T:
            if t == t0:
                continue
            for i in Nw:
                mdl.add_constraint(
                    D[m, i, t] <= z[m, t],
                    ctname=f"D_le_z_m{m}_i{i}_t{t}"
                )

    # 5.2) Il MAIN deve essere presente in quel nodo per poter rilasciare TRAIL
    #    D^m_{i,t} <= Z_max * x^m_{i,t}  ∀ m∈M, i∈Nw, t∈T \ {t0}
    for m in M:
        for t in T:
            if t == t0:
                continue
            for i in Nw:
                mdl.add_constraint(
                    D[m, i, t] <= Z_max * x[m, i, t],
                    ctname=f"D_le_Zx_m{m}_i{i}_t{t}"
                )

    # 5.3) Un MAIN può ricevere TRAIL solo se si trova in quel nodo
    #    U^m_{i,t} <= Z_max * x^m_{i,t}  ∀ m∈M, i∈Nw, t∈T \ {t0}
    for m in M:
        for t in T:
            if t == t0:
                continue
            for i in Nw:
                mdl.add_constraint(
                    U[m, i, t] <= Z_max * x[m, i, t],
                    ctname=f"U_le_Zx_m{m}_i{i}_t{t}"
                )

    # 5.4) Conservazione scambio TRAIL nel nodo:
    #    ∑_m U^m_{i,t} <= ∑_m D^m_{i,t} + κ_{i,t-1}
    #    ∀ i∈Nw, t∈T \ {t0}
    for i in Nw:
        for t in T:
            if t == t0:
                continue
            mdl.add_constraint(
                mdl.sum(U[m, i, t] for m in M)
                <= mdl.sum(D[m, i, t] for m in M) + kappa[i, t-1],
                ctname=f"trail_conservation_node_i{i}_t{t}"
            )

    # 5.5) Dinamica di z^m_t:
    #    z^m_t = z^m_{t-1} - ∑_{i∈Nw} D^m_{i,t} + ∑_{i∈Nw} U^m_{i,t}
    #    ∀ m∈M, t∈T \ {t0}
    for m in M:
        for t in T:
            if t == t0:
                continue
            mdl.add_constraint(
                z[m, t] ==
                z[m, t-1]
                - mdl.sum(D[m, i, t] for i in Nw)
                + mdl.sum(U[m, i, t] for i in Nw),
                ctname=f"z_balance_m{m}_t{t}"
            )

    # 5.6) Disponibilità di TRAIL nel nodo i:
    #    κ_{i,t} = κ_{i,t-1} + ∑_m D^m_{i,t} - ∑_m U^m_{i,t}
    #    ∀ i∈Nw, t∈T \ {t0}
    for i in Nw:
        for t in T:
            if t == t0:
                continue
            mdl.add_constraint(
                kappa[i, t] ==
                kappa[i, t-1]
                + mdl.sum(D[m, i, t] for m in M)
                - mdl.sum(U[m, i, t] for m in M),
                ctname=f"kappa_balance_i{i}_t{t}"
            )


    # ------------------------------------------------------------------
    # 6) Request completion and definition of s_k (via b)
    #
    #   s_k = 1  se e solo se la richiesta k effettua una discesa finale
    #             in uno degli istanti ammessi ΔT_k_out.
    #
    #   sum_{m∈M} sum_{t ∈ ΔT_k_out} b[k,t,m] = s_k
    #
    # Se non esistono tempi di discesa ammissibili per k → s_k = 0.
    # ------------------------------------------------------------------
    for k in K:
        # Tutti gli istanti in cui la richiesta k può scendere in qualche nodo i
        out_times_k = sorted({t for (kk, i, t) in d_out.keys() if kk == k})

        if not out_times_k:
            # Nessun tempo di discesa → la richiesta non può mai essere completata
            mdl.add_constraint(s[k] == 0, ctname=f"served_empty_k{k}")
            continue

        mdl.add_constraint(
            mdl.sum(
                b[k, t, m]
                for m in M
                for t in out_times_k
                if (k, t, m) in b
            ) == s[k],
            ctname=f"served_via_drop_k{k}"
        )






    # ------------------------------------------------------------------
    # 7) At most one module serves k:
    #    sum_m r[k,t,m] <= 1  ∀k, t ∈ ΔT_k
    # ------------------------------------------------------------------
    for k in K:
        for t in DeltaT[k]:
            mdl.add_constraint(
                mdl.sum(r[k, t, m] for m in M) <= 1,
                ctname=f"one_module_per_req_k{k}_t{t}"
            )




    # ------------------------------------------------------------------
    # 8) Excange constraints w:
    #    (a) w <= r[k,t-1,m]
    #    (b) w <= x[m,i,t]
    #    (c) w <= x[mp,i,t]
    #    (d) sum_{i,m,mp≠m} w <= 1 
    #    (e) r[k,t,m] <= 1 - sum_{i,mp≠m} w[k,i,t,m,mp]
    #    (f) r[k,t,mp] >= sum_{i,m≠mp} w[k,i,t,m,mp]
    # ------------------------------------------------------------------

    # (a), (b), (c): local constraints on each w
    for k in K:
        for t in DeltaT[k]:
            for i in Nw:
                for m in M:
                    for mp in M:
                        if m == mp:
                            continue
                        # skip if the key does not exist 
                        if (k, i, t, m, mp) not in w:
                            continue

                        # (a) The swap can occur only if the request was on m at time t-1
                        prev_t = t - 1
                        if (k, prev_t, m) in r:
                            mdl.add_constraint(
                                w[k, i, t, m, mp] <= r[k, prev_t, m],
                                ctname=f"swap_only_if_onboard_k{k}_i{i}_t{t}_m{m}_mp{mp}"
                            )
                        else:
                            # se r[k,t-1,m] non esiste, vuol dire "non può essere a bordo" → w<=0
                            mdl.add_constraint(
                                w[k, i, t, m, mp] <= 0,
                                ctname=f"swap_only_if_onboard_k{k}_i{i}_t{t}_m{m}_mp{mp}"
                            )

                        # (b)(c) The modules must be at the same node i at time t
                        mdl.add_constraint(
                            w[k, i, t, m, mp] <= x[m, i, t],
                            ctname=f"swap_same_node1_k{k}_i{i}_t{t}_m{m}_mp{mp}"
                        )
                        mdl.add_constraint(
                            w[k, i, t, m, mp] <= x[mp, i, t],
                            ctname=f"swap_same_node2_k{k}_i{i}_t{t}_m{m}_mp{mp}"
                        )


    # (d) At most 1 swap per k,t
    for k in K:
        for t in DeltaT[k]:  
            mdl.add_constraint(
                mdl.sum(
                    w[k, i, t, m, mp]
                    for i in Nw
                    for m in M
                    for mp in M
                    if m != mp and (k, i, t, m, mp) in w
                ) <= 1,
                ctname=f"one_swap_k{k}_t{t}"
            )


    # (e) Module m "loses" the request
    for k in K:
        for t in DeltaT[k]:
            for m in M:
                mdl.add_constraint(
                    r[k, t, m] <=
                    1 - mdl.sum(
                        w[k, i, t, m, mp]
                        for i in Nw
                        for mp in M
                        if mp != m and (k, i, t, m, mp) in w
                    ),
                    ctname=f"lose_req_k{k}_t{t}_m{m}"
                )



    # (f) Module mp "receives" the request
    for k in K:
        for t in DeltaT[k]:
            for mp in M:
                mdl.add_constraint(
                    r[k, t, mp] >=
                    mdl.sum(
                        w[k, i, t, m, mp]
                        for i in Nw
                        for m in M
                        if m != mp and (k, i, t, m, mp) in w
                    ),
                    ctname=f"receive_req_k{k}_t{t}_mp{mp}"
                )

    


     # ------------------------------------------------------------------
    # 10) Boarding / Alighting con a,b + coerenza di r 
    #
    # (10.1) Al più un evento (salita o discesa) per (k,t,m):
    #         a[k,t,m] + b[k,t,m] ≤ 1
    #
    # (10.2) Dinamica di r:
    #    Per i tempi di k ordinati t^1_k < t^2_k < ... < t^{|ΔT_k|}_k:
    #
    #    r[k, t^1_k, m] = a[k, t^1_k, m] - b[k, t^1_k, m]
    #    r[k, t^i_k, m] = r[k, t^{i-1}_k, m] + a[k, t^i_k, m] - b[k, t^i_k, m]
    #                     - sum_{i∈Nw} sum_{m'≠m} w[k,i,t^i_k,m,m']
    #                     + sum_{i∈Nw} sum_{m'≠m} w[k,i,t^i_k,m',m]
    #
    # (10.3) Attivazione di a e b (solo da terra):
    #
    #   a[k,t,m] ≤ sum_{i∈N} x[m,i,t] d_in[k,i,t]
    #   b[k,t,m] ≤ sum_{i∈N} x[m,i,t] d_out[k,i,t]
    #
    # (10.4) Coerenza con lo stato precedente:
    #
    #   a[k,t,m] ≤ 1 - r[k,t_prev,m]
    #   b[k,t,m] ≤     r[k,t_prev,m]
    # ------------------------------------------------------------------

    # (10.1) Al più un evento per (k,t,m)
    #for k in K:
    #    for t in DeltaT[k]:
    #        for m in M:
    #            mdl.add_constraint(
    #                a[k, t, m] + b[k, t, m] <= 1,
    #                ctname=f"one_event_k{k}_t{t}_m{m}"
    #            )

    # (10.1bis) Al più UNA salita e UNA discesa per richiesta k
    for k in K:
        # max 1 salita in tutta la finestra ΔT_k su tutti i moduli
        mdl.add_constraint(
            mdl.sum(
                a[k, t, m]
                for t in DeltaT[k]
                for m in M
            ) <= 1,
            ctname=f"max_one_boarding_k{k}"
        )

        # max 1 discesa in tutta la finestra ΔT_k su tutti i moduli
        mdl.add_constraint(
            mdl.sum(
                b[k, t, m]
                for t in DeltaT[k]
                for m in M
            ) <= 1,
            ctname=f"max_one_alighting_k{k}"
        )

    # (10.2) Dinamica di r tramite a,b e w
    for k in K:
        times_k = sorted(DeltaT[k])
        if not times_k:
            continue

        for m in M:
            # primo tempo della finestra di k
            t0_k = times_k[0]

            # Al primo istante della finestra non consideriamo scambi e non si ha r per t-1:
            mdl.add_constraint(
                r[k, t0_k, m] == a[k, t0_k, m] - b[k, t0_k, m],
                ctname=f"r_chain_first_k{k}_t{t0_k}_m{m}"
            )

            # tempi successivi
            for t_prev, t in zip(times_k[:-1], times_k[1:]):

                # scambio "in uscita" dal modulo m al tempo t
                swap_out = mdl.sum(
                    w[k, i, t, m, mp]
                    for i in Nw
                    for mp in M
                    if mp != m and (k, i, t, m, mp) in w
                )

                # scambio "in entrata" sul modulo m al tempo t
                swap_in = mdl.sum(
                    w[k, i, t, mp, m]
                    for i in Nw
                    for mp in M
                    if mp != m and (k, i, t, mp, m) in w
                )

                mdl.add_constraint(
                    r[k, t, m] ==
                    r[k, t_prev, m] + a[k, t, m] - b[k, t, m]
                    - swap_out + swap_in,
                    ctname=f"r_chain_k{k}_t{t}_m{m}"
                )

    # (10.3) Attivazione di a (solo salite da terra)
    for k in K:
        for t in DeltaT[k]:
            for m in M:
                boarding_terms = [
                    x[m, i, t]
                    for i in N
                    if (k, i, t) in d_in
                ]

                rhs_up = mdl.sum(boarding_terms)
                mdl.add_constraint(
                    a[k, t, m] <= rhs_up,
                    ctname=f"a_activation_k{k}_t{t}_m{m}"
                )

    # (10.3) Attivazione di b (solo discese a terra)
    for k in K:
        for t in DeltaT[k]:
            for m in M:
                alight_terms = [
                    x[m, i, t]
                    for i in N
                    if (k, i, t) in d_out
                ]

                rhs_down = mdl.sum(alight_terms)
                mdl.add_constraint(
                    b[k, t, m] <= rhs_down,
                    ctname=f"b_activation_k{k}_t{t}_m{m}"
                )

    # (10.4) Coerenza con lo stato precedente: a/b compatibili con r_{t_prev}
    for k in K:
        times_k = sorted(DeltaT[k])
        if not times_k:
            continue

        for m in M:
            # imponiamo i vincoli solo a partire dal secondo istante della finestra
            for t_prev, t in zip(times_k[:-1], times_k[1:]):
                mdl.add_constraint(
                    a[k, t, m] <= 1 - r[k, t_prev, m],
                    ctname=f"a_prev_state_k{k}_t{t}_m{m}"
                )
                mdl.add_constraint(
                    b[k, t, m] <= r[k, t_prev, m],
                    ctname=f"b_prev_state_k{k}_t{t}_m{m}"
                )
    # ------------------------------------------------------------------
    # 11) Linearization h[m,i,j,t]
    # ------------------------------------------------------------------ 
    # h[m,i,j,t] <= (Z_max + 1) * y[m,i,j,t]
    for m in M:
        for (i, j) in A:
            for t in T:
                mdl.add_constraint(
                    h[m, i, j, t] <= (Z_max + 1) * y[m, i, j, t],
                    ctname=f"h_le_bigM_y_m{m}_i{i}_j{j}_t{t}"
                )
    # h[m,i,j,t] <= z[m,t] + 1
    for m in M:
        for (i, j) in A:
            for t in T:
                mdl.add_constraint(
                    h[m, i, j, t] <= z[m, t] + 1,
                    ctname=f"h_le_zplus1_m{m}_i{i}_j{j}_t{t}"
                )
    # h[m,i,j,t] >= (z[m,t] + 1) - (Z_max + 1) * (1 - y[m,i,j,t])
    for m in M:
        for (i, j) in A:
            for t in T:
                mdl.add_constraint(
                    h[m, i, j, t] >= (z[m, t] + 1) - (Z_max + 1) * (1 - y[m, i, j, t]),
                    ctname=f"h_ge_zplus1_bigM_m{m}_i{i}_j{j}_t{t}"
                )




def add_taxi_like_objective_ab(mdl, I, h, s):
    """
    Add the full taxi-like MILP objective function:
        min ( C_oper + C_uns )

    Parameters
    ----------
    mdl : docplex.mp.model.Model
    I   : Instance
        Data container with sets and parameters.

    h : docplex integer var dict
        h[m,i,j,t] = number if modules main + trail m departs from i to j at time t

    s : docplex binary var dict
        s[k] = 1 if request k is served
    """

    # -------------------------
    # 1) Operational cost
    #     C_oper = c_km * sum_m,i,j,t gamma(i,j) * h(m,i,j,t)
    # -------------------------
    C_oper = I.c_km * mdl.sum(
        I.gamma[(i, j)] * h[m, i, j, t]
        for m in I.M
        for (i, j) in I.A
        for t in I.T
    )

    # -------------------------
    # 2) Unserved demand cost
    #     C_uns = c_uns * sum_k q_k * (1 - s_k)
    # -------------------------
    C_uns = mdl.sum(
        I.c_uns * I.q[k] * (1 - s[k])
        for k in I.K
    )

    # -------------------------
    # Minimize total cost
    # -------------------------
    mdl.minimize(C_oper + C_uns)

    # Return objective expression if needed
    return C_oper + C_uns







def create_MT_model_ab(I: Instance):
    """
    Create:
        - Model()
        - decision variables
        - constraints
        - objective
    """

    mdl = Model(name="TaxiLike")

    # 1) variables
    x, y, r, w, s, a, b, D, U, z, kappa,h = create_decision_variables_ab(mdl, I)

    # 2) constraints
    add_taxi_like_constraints_ab(mdl, I, x, y, r, w, s, a, b, D, U, z, kappa,h)

    # 3) objective
    add_taxi_like_objective_ab(mdl, I, h, s)

    return mdl, x, y, r, w, s, a, b, D, U, z, kappa,h



