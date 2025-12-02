"""
Taxi-like deterministic MILP model
=================================
- creation of a CPLEX/DOcplex model
- creation of all the decision variables
- addition of all the MILP constraints
- addition of the objective function
- return of the model and variables
"""


from typing import Dict, Tuple
from docplex.mp.model import Model
from utils.instance import Instance






def create_decision_variables(mdl: Model, I: Instance):
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

    # r[k,t,m]
    r = mdl.binary_var_dict(
        keys=[(k, t, m) for k in I.K for t in I.T for m in I.M],
        name="r"
    )

    # w[k,i,t,m,mp]
    w = mdl.binary_var_dict(
        keys=[
            (k, i, t, m, mp)
            for k in I.K
            for i in I.N
            for t in I.T
            for m in I.M
            for mp in I.M
            if m != mp
        ],
        name="w"
    )

    # s[k]
    s = mdl.binary_var_dict(
        keys=[k for k in I.K],
        name="s"
    )

    return x, y, r, w, s



def add_taxi_like_constraints(mdl, I, x, y, r, w, s):
    """
    Add all constraints of the taxi-like MILP model to the docplex model.
    """
    # Convenience
    N = I.N
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
    # 1) Moduli: presenza in un nodo o in viaggio
    #    sum_i x[m,i,t] <= 1  ∀m,t
    # ------------------------------------------------------------------
    for m in M:
        for t in T:
            mdl.add_constraint(
                mdl.sum(x[m, i, t] for i in N) <= 1,
                ctname=f"module_location_m{m}_t{t}"
            )


    # ------------------------------------------------------------------
    # 2) Movimento moduli
    #    sum_{j:(i,j)∈A} y[m,i,j,t] <= x[m,i,t]   ∀m,i,t
    # ------------------------------------------------------------------
    for m in M:
        for (i, j) in A:
            for t in T:
                # Per ogni arco (i,j), vincolo locale: y[m,i,j,t] <= x[m,i,t]
                mdl.add_constraint(
                    y[m, i, j, t] <= x[m, i, t],
                    ctname=f"depart_possible_m{m}_i{i}_j{j}_t{t}"
                )

    # Se vuoi la versione "una sola partenza per nodo", come in LaTeX:
    # sum_{j:(i,j)∈A} y[m,i,j,t] <= x[m,i,t]
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
    # 3) Coerenza del movimento (no teletrasporto)
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
    # 4) Condizione iniziale: ogni modulo in un nodo all'istante t0
    #
    # sum_{i∈N} x[m,i,t0] = 1  ∀m
    # ------------------------------------------------------------------
    for m in M:
        mdl.add_constraint(
            mdl.sum(x[m, i, t0] for i in N) == 1,
            ctname=f"initial_position_m{m}"
        )


    # ------------------------------------------------------------------
    # 5) Capacità moduli
    #
    # sum_{k∈K} q_k * r[k,t,m] <= Q  ∀m,t
    # (nota: r è indicizzato (k,t,m) nel tuo codice)
    # ------------------------------------------------------------------
    for m in M:
        for t in T:
            mdl.add_constraint(
                mdl.sum(q[k] * r[k, t, m] for k in K) <= Q,
                ctname=f"capacity_m{m}_t{t}"
            )


    # ------------------------------------------------------------------
    # 6) Definizione s_k
    #
    # s_k <= sum_{t∈ΔT_k} sum_m r[k,t,m]
    # sum_{t∈ΔT_k} sum_m r[k,t,m] <= |ΔT_k| * s_k
    # ------------------------------------------------------------------
    for k in K:
        times_k = DeltaT[k]
        if not times_k:
            # in pratica, se ΔT_k è vuoto, non può mai essere servita
            mdl.add_constraint(s[k] == 0, ctname=f"served_empty_k{k}")
            continue

        mdl.add_constraint(
            s[k] <= mdl.sum(r[k, t, m] for t in times_k for m in M),
            ctname=f"served_lo_k{k}"
        )

        mdl.add_constraint(
            mdl.sum(r[k, t, m] for t in times_k for m in M)
            <= len(times_k) * s[k],
            ctname=f"served_hi_k{k}"
        )


    # ------------------------------------------------------------------
    # 7) Fuori da ΔT_k: la richiesta non può essere a bordo
    #
    # sum_m r[k,t,m] = 0  ∀k, t ∉ ΔT_k
    # ------------------------------------------------------------------
    all_T = set(T)
    for k in K:
        allowed_T = set(DeltaT[k])
        forbidden_T = all_T - allowed_T
        for t in forbidden_T:
            mdl.add_constraint(
                mdl.sum(r[k, t, m] for m in M) == 0,
                ctname=f"outside_window_k{k}_t{t}"
            )


    # ------------------------------------------------------------------
    # 8) Al più un modulo serve k in un certo istante
    #
    # sum_m r[k,t,m] <= 1  ∀k, t ∈ ΔT_k
    # ------------------------------------------------------------------
    for k in K:
        for t in DeltaT[k]:
            mdl.add_constraint(
                mdl.sum(r[k, t, m] for m in M) <= 1,
                ctname=f"one_module_per_req_k{k}_t{t}"
            )


    # ------------------------------------------------------------------
    # 9) Vincoli di SCAMBIO w
    #    (a) w <= r[k,t-1,m]
    #    (b) w <= x[m,i,t]
    #    (c) w <= x[mp,i,t]
    #    (d) sum_{i,m,mp≠m} w <= 1  (per k,t)
    #    (e) r[k,t,m] <= 1 - sum_{i,mp≠m} w[k,i,t,m,mp]
    #    (f) r[k,t,mp] >= sum_{i,m≠mp} w[k,i,t,m,mp]
    # ------------------------------------------------------------------

    # (a), (b), (c): vincoli locali su ogni w
    for k in K:
        for t in T_pos:  # occorre t-1
            for i in N:
                for m in M:
                    for mp in M:
                        if m == mp:
                            continue
                        # skip se chiave non esiste (ma tu le hai create tutte)
                        if (k, i, t, m, mp) not in w:
                            continue

                        # Lo scambio può avvenire solo se la richiesta era su m al tempo t-1
                        mdl.add_constraint(
                            w[k, i, t, m, mp] <= r[k, t-1, m],
                            ctname=f"swap_only_if_onboard_k{k}_i{i}_t{t}_m{m}_mp{mp}"
                        )
                        # I moduli devono essere nello stesso nodo i al tempo t
                        mdl.add_constraint(
                            w[k, i, t, m, mp] <= x[m, i, t],
                            ctname=f"swap_same_node1_k{k}_i{i}_t{t}_m{m}_mp{mp}"
                        )
                        mdl.add_constraint(
                            w[k, i, t, m, mp] <= x[mp, i, t],
                            ctname=f"swap_same_node2_k{k}_i{i}_t{t}_m{m}_mp{mp}"
                        )

    # (d) Al più 1 scambio per k,t
    for k in K:
        for t in DeltaT[k]:  # solo quando la richiesta esiste
            mdl.add_constraint(
                mdl.sum(
                    w[k, i, t, m, mp]
                    for i in N
                    for m in M
                    for mp in M
                    if m != mp and (k, i, t, m, mp) in w
                ) <= 1,
                ctname=f"one_swap_k{k}_t{t}"
            )

    # (e) Il modulo m "perde" la richiesta
    for k in K:
        for t in DeltaT[k]:
            for m in M:
                mdl.add_constraint(
                    r[k, t, m] <=
                    1 - mdl.sum(
                        w[k, i, t, m, mp]
                        for i in N
                        for mp in M
                        if mp != m and (k, i, t, m, mp) in w
                    ),
                    ctname=f"lose_req_k{k}_t{t}_m{m}"
                )

    # (f) Il modulo mp "riceve" la richiesta
    for k in K:
        for t in DeltaT[k]:
            for mp in M:
                mdl.add_constraint(
                    r[k, t, mp] >=
                    mdl.sum(
                        w[k, i, t, m, mp]
                        for i in N
                        for m in M
                        if m != mp and (k, i, t, m, mp) in w
                    ),
                    ctname=f"receive_req_k{k}_t{t}_mp{mp}"
                )


    # ------------------------------------------------------------------
    # 10) Vincoli SALITA / DISCESA con d_in, d_out
    #
    # Salita o ingresso per scambio:
    #   r[k,t,m] - r[k,t-1,m] <=
    #       sum_i x[m,i,t] d_in[k,i,t] +
    #       sum_i sum_{m'≠m} w[k,i,t,m',m]
    #
    # Discesa o uscita per scambio:
    #   r[k,t,m] - r[k,t-1,m] >=
    #      - sum_i x[m,i,t] d_out[k,i,t]
    #      - sum_i sum_{m'≠m} w[k,i,t,m,m']
    # ------------------------------------------------------------------
    for k in K:
        times_k = DeltaT[k]
        for t in times_k:
            if t == t0:
                # se vuoi, puoi saltare t0 o gestire r[k,t0-1,m] come 0
                continue
            for m in M:
                # LHS
                lhs = r[k, t, m] - r[k, t-1, m]

                # RHS salita: sum_i x[m,i,t] * d_in[k,i,t]
                boarding_terms = []
                for i in N:
                    if (k, i, t) in d_in:
                        boarding_terms.append(x[m, i, t])
                # + scambi in ingresso w[k,i,t,m',m]
                swap_in_terms = []
                for i in N:
                    for mp in M:
                        if mp == m:
                            continue
                        if (k, i, t, mp, m) in w:
                            swap_in_terms.append(w[k, i, t, mp, m])

                rhs_up = mdl.sum(boarding_terms) + mdl.sum(swap_in_terms)

                mdl.add_constraint(
                    lhs <= rhs_up,
                    ctname=f"boarding_or_swap_in_k{k}_t{t}_m{m}"
                )

                # RHS discesa: - sum_i x[m,i,t] * d_out[k,i,t]
                alight_terms = []
                for i in N:
                    if (k, i, t) in d_out:
                        alight_terms.append(x[m, i, t])
                # + scambi in uscita w[k,i,t,m,m']
                swap_out_terms = []
                for i in N:
                    for mp in M:
                        if mp == m:
                            continue
                        if (k, i, t, m, mp) in w:
                            swap_out_terms.append(w[k, i, t, m, mp])

                rhs_down = - mdl.sum(alight_terms) - mdl.sum(swap_out_terms)

                mdl.add_constraint(
                    lhs >= rhs_down,
                    ctname=f"alighting_or_swap_out_k{k}_t{t}_m{m}"
                )




def add_taxi_like_objective(mdl, I, y, s):
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







def create_taxi_like_model(I: Instance):
    """
    Create:
        - Model()
        - decision variables
        - constraints
        - objective
    """

    mdl = Model(name="TaxiLike")

    # 1) variables
    x, y, r, w, s = create_decision_variables(mdl, I)

    # 2) constraints
    add_taxi_like_constraints(mdl, I, x, y, r, w, s)

    # 3) objective
    add_taxi_like_objective(mdl, I, y, s)

    return mdl, x, y, r, w, s