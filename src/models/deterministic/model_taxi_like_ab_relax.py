"""
Taxi-like deterministic MILP model
=================================
- creation of a CPLEX/DOcplex model
- creation of all the decision variables
- addition of all the MILP constraints
- addition of the objective function
- return of the model and variables
"""


from docplex.mp.model import Model
from utils.instance_def import Instance



# seed
import random
import numpy as np
seed = 23
random.seed(seed)
np.random.seed(seed)



def create_decision_variables_ab_relax(mdl: Model, I: Instance):
    """
    Create all MILP decision variables for the taxi-like model.
    """


    # x[m,i,t]
    x = mdl.binary_var_dict(
        keys=[(m, i, t) for m in I.M for i in I.N for t in I.T],
        name="x"
    )

    # y[m,i,j,t]
    y = mdl.binary_var_dict(
        keys=[(m, i, j, t) for m in I.M for (i, j) in I.A for t in I.T],
        name="y"
    )

    # r[k,t,m], only for t ∈ ΔT_k
    r = mdl.binary_var_dict(
        keys=[(k, t, m)
              for k in I.K
              for t in I.DeltaT[k]
              for m in I.M],
        name="r"
    )

    print("Nw:", I.Nw)
    # w[k,i,t,m,mp] only for:
    #   - t ∈ ΔT_k (request is "alive")
    #   - i ∈ Nw (internal nodes where swaps are allowed)
    w = mdl.binary_var_dict(
        keys=[
            (k, i, t, m, mp)
            for k in I.K
            for t in I.DeltaT[k]
            if (t > I.T[0]) and (t < I.T[-1])    # no excanges at the beginning and at the end
            for i in I.Nw
            for m in I.M
            for mp in I.M
            if m != mp
        ],
        name="w"
    )

    # s[k]
    s = mdl.continuous_var_dict(
    keys=[k for k in I.K],
    lb=0,
    ub=1,
    name="s"
    )


    
    # a[k,t,m]
    a = mdl.binary_var_dict(
        keys=[
            (k, t, m)
            for k in I.K
            for t in I.DeltaT[k]
            for m in I.M
        ],
        name="a"
    )

    # b[k,t,m]
    b = mdl.binary_var_dict(
        keys=[
            (k, t, m)
            for k in I.K
            for t in I.DeltaT[k]
            for m in I.M
        ],
        name="b"
    )


    return x, y, r, w, s, a, b







def add_taxi_like_constraints_ab_relax(mdl, I, x, y, r, w, s, a, b):
    """
    Add all constraints of the taxi-like MILP model to the docplex model.
    """
    N = I.N
    Nw = I.Nw
    A = I.A
    M = I.M
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
            incoming = [ (h,v) for (h,v) in A if v == i ]

            for t in T_pos:   # t > t0
                lhs = x[m, i, t]

                # x[m,i,t-1]
                rhs = x[m, i, t-1]

                # - sum_{j:(i,j)∈A} y[m,i,j,t-1]
                if outgoing:
                    rhs -= mdl.sum(y[m, i, j, t-1] for (i2,j) in outgoing)

                # + sum_{h:(h,i)∈A, t - tau(h,i) >= t0} y[m,h,i,t - tau(h,i)]
                incoming_terms = []
                for (h, i2) in incoming:
                    travel = tau[(h, i)]
                    t_depart = t - travel
                    if t_depart >= t0:
                        incoming_terms.append(y[m, h, i, t_depart])
                if incoming_terms:
                    rhs += mdl.sum(incoming_terms)

                mdl.add_constraint(
                    lhs == rhs,
                    ctname=f"move_consistency_m{m}_i{i}_t{t}"
                )

    # ------------------------------------------------------------------
    # 4) Initial condition:
    #    All modules start at the depot node (node 0) at time t0.
    # ------------------------------------------------------------------
    depot = I.depot

    for m in M:
        mdl.add_constraint(
            x[m, depot, t0] == 1,
            ctname=f"initial_position_at_depot_m{m}"
        )



    # ------------------------------------------------------------------
    # 5) Module capacity:
    #    sum_{k∈K} q_k * r[k,t,m] <= Q  ∀m,t
    #    (only k s.t. (k,t,m) exists in r)
    # ------------------------------------------------------------------
    for m in M:
        for t in T:
            mdl.add_constraint(
                mdl.sum(
                    q[k] * r[k, t, m]
                    for k in K
                    if (k, t, m) in r
                ) <= Q,
                ctname=f"capacity_m{m}_t{t}"
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




def add_taxi_like_objective_ab_relax(mdl, I, y, s):
    """
    Add the full taxi-like MILP objective function:
        min ( C_oper + C_uns )

    Parameters
    ----------
    mdl : docplex.mp.model.Model
    I   : Instance
        Data container with sets and parameters.

    y : docplex binary var dict
        y[m,i,j,t] = 1 if module m departs from i to j at time t

    s : docplex binary var dict
        s[k] = 1 if request k is served
    """

    # -------------------------
    # 1) Operational cost
    #     C_oper = c_km * sum_m,i,j,t gamma(i,j) * y(m,i,j,t)
    # -------------------------
    C_oper = I.c_km * mdl.sum(
        I.gamma[(i, j)] * y[m, i, j, t]
        for m in I.M
        for (i, j) in I.A
        for t in I.T
    )

    # -------------------------
    # 2) Unserved demand cost
    #     C_uns = c_uns * sum_k q_k * (1 - s_k)
    # -------------------------
    C_uns = mdl.sum(
        I.c_uns_taxi * I.q[k] * (1 - s[k])
        for k in I.K
    )

    # -------------------------
    # Minimize total cost
    # -------------------------
    mdl.minimize(C_oper + C_uns)

    # Return objective expression if needed
    return C_oper + C_uns







def create_taxi_like_model_ab_relax(I: Instance):
    """
    Create:
        - Model()
        - decision variables
        - constraints
        - objective
    """

    mdl = Model(name="TaxiLike")

    # 1) variables
    x, y, r, w, s, a, b = create_decision_variables_ab_relax(mdl, I)

    # 2) constraints
    add_taxi_like_constraints_ab_relax(mdl, I, x, y, r, w, s, a, b)

    # 3) objective
    add_taxi_like_objective_ab_relax(mdl, I, y, s)

    return mdl, x, y, r, w, s, a, b



