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



def create_decision_variables_LR_relax(mdl: Model, I: Instance):
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
    # L[k,i,t,m], R[k,i,t,m] only for:
    #   - t ∈ ΔT_k
    #   - t > t0 (no swap at the very first instant)
    #   - i ∈ Nw (internal nodes where swaps are allowed)
    L = mdl.binary_var_dict(
        keys=[
            (k, i, t, m)
            for k in I.K
            for t in I.DeltaT[k]
            if (t > I.T[0]) and (t < I.T[-1])   # no excanges at the beginning and at the end
            for i in I.Nw
            for m in I.M
        ],
        name="L"
    )

    R = mdl.binary_var_dict(
        keys=[
            (k, i, t, m)
            for k in I.K
            for t in I.DeltaT[k]
            if (t > I.T[0]) and (t < I.T[-1])   # no excanges at the beginning and at the end
            for i in I.Nw
            for m in I.M
        ],
        name="R"
    )

    # s[k]
    s = mdl.continuous_var_dict(
    keys=[k for k in I.K],
    lb=0,
    ub=1,
    name="s"
    )


    return x, y, r, L, R, s







def add_taxi_like_constraints_LR_relax(mdl, I, x, y, r, L, R, s):
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
    # 6) Request completion and definition of s_k
    #     sum_{t ∈ ΔT_k_out} sum_{m ∈ M} (r[k,t-1,m] - r[k,t,m]) = s_k
    #
    # If no admissible drop-off times exist for request k, then s_k = 0.
    # ------------------------------------------------------------------
    for k in K:
        # All times t where request k is allowed to alight at some node i
        out_times_k = sorted({t for (kk, i, t) in d_out.keys() if kk == k})     ### <--- It is enough to check ΔT_out_k

        if not out_times_k:
            # No feasible drop-off time → the request can never be completed
            mdl.add_constraint(s[k] == 0, ctname=f"served_empty_k{k}")
            continue

        terms = []
        for m in M:
            for t in out_times_k:
                # r[k, t, m] esiste solo se (k, t, m) in r
                if (k, t, m) not in r:
                    continue

                prev_t = t - 1
                if (k, prev_t, m) in r:
                    terms.append(r[k, prev_t, m] - r[k, t, m])
                else:
                    # nessuna variabile r[k, t-1, m] definita → r[k,t-1,m]=0
                    terms.append(- r[k, t, m])

        expr = mdl.sum(terms)

        mdl.add_constraint(
            expr == s[k],
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
    # 8) Swap constraints via L and R
    #
    # (a) L <= r[k,t-1,m]
    # (b) L <= x[m,i,t]
    # (c) R <= x[m,i,t]
    # (d) sum_m L[k,i,t,m] = sum_m R[k,i,t,m]  (bilancio per nodo/tempo)
    # (e) sum_{i,m} L[k,i,t,m] <= 1 (al più uno scambio per k,t)
    # ------------------------------------------------------------------

    # (a) and (b) for L, (c) for R
    for k in K:
        for t in DeltaT[k]:
            if t == t0:
                continue  # no swaps at the first time
            for i in Nw:
                for m in M:
                    # some (k,i,t,m) might not exist if we restricted keys
                    if (k, i, t, m) in L:

                        # (a) L <= r[k,t-1,m] (if defined)
                        prev_t = t - 1
                        if (k, prev_t, m) in r:
                            mdl.add_constraint(
                                L[k, i, t, m] <= r[k, prev_t, m],
                                ctname=f"L_only_if_onboard_k{k}_i{i}_t{t}_m{m}"
                            )
                        else:
                            # cannot be onboard => L <= 0
                            mdl.add_constraint(
                                L[k, i, t, m] <= 0,
                                ctname=f"L_only_if_onboard_k{k}_i{i}_t{t}_m{m}"
                            )

                        # (b) L <= x[m,i,t]
                        mdl.add_constraint(
                            L[k, i, t, m] <= x[m, i, t],
                            ctname=f"L_same_node_k{k}_i{i}_t{t}_m{m}"
                        )

                    if (k, i, t, m) in R:
                        # (c) R <= x[m,i,t]
                        mdl.add_constraint(
                            R[k, i, t, m] <= x[m, i, t],
                            ctname=f"R_same_node_k{k}_i{i}_t{t}_m{m}"
                        )

    # (d) balance L and R at each (k,i,t)
    for k in K:
        for t in DeltaT[k]:
            if t == t0:
                continue
            for i in Nw:
                lhs = mdl.sum(
                    L[k, i, t, m]
                    for m in M
                    if (k, i, t, m) in L
                )
                rhs = mdl.sum(
                    R[k, i, t, m]
                    for m in M
                    if (k, i, t, m) in R
                )
                mdl.add_constraint(
                    lhs == rhs,
                    ctname=f"LR_balance_k{k}_i{i}_t{t}"
                )

    # (e) at most 1 swap per request/time (anywhere)
    for k in K:
        for t in DeltaT[k]:
            if t == t0:
                continue
            mdl.add_constraint(
                mdl.sum(
                    L[k, i, t, m]
                    for i in Nw
                    for m in M
                    if (k, i, t, m) in L
                ) <= 1,
                ctname=f"one_swap_LR_k{k}_t{t}"
            )



    # ------------------------------------------------------------------
    # 8bis) Discese e Salite
    #
    # Per ogni (k,t,m):
    #   r[k,t,m] <= 1 - sum_i L[k,i,t,m]   (se L=1, il modulo perde la richiesta)
    #   r[k,t,m] >= sum_i R[k,i,t,m]       (se R=1, il modulo riceve la richiesta)
    #
    # Questo rende gli L/R davvero "operativi" sulla variabile r.
    # ------------------------------------------------------------------
    for k in K:
        for t in DeltaT[k]:
            for m in M:
                # r[k,t,m] esiste solo se (k,t,m) in r
                if (k, t, m) not in r:
                    continue

                # Somma delle L che "lasciano" da m a tempo t
                sum_L = mdl.sum(
                    L[k, i, t, m]
                    for i in Nw
                    if (k, i, t, m) in L
                )

                # Somma delle R che "ricevono" su m a tempo t
                sum_R = mdl.sum(
                    R[k, i, t, m]
                    for i in Nw
                    if (k, i, t, m) in R
                )

                # (lose) se qualcuno lascia da m, r[k,t,m] deve andare a 0
                mdl.add_constraint(
                    r[k, t, m] <= 1 - sum_L,
                    ctname=f"r_lose_LR_k{k}_t{t}_m{m}"
                )

                # (receive) se qualcuno riceve su m, r[k,t,m] deve andare a 1
                mdl.add_constraint(
                    r[k, t, m] >= sum_R,
                    ctname=f"r_receive_LR_k{k}_t{t}_m{m}"
                )


    # ------------------------------------------------------------------
    # 9) Boarding / Alighting with L,R and d_in, d_out
    #
    # Boarding:
    #   r[k,t,m] - r[k,t-1,m] <=
    #       sum_i x[m,i,t] d_in[k,i,t] +
    #       sum_{i∈Nw} R[k,i,t,m]
    #
    # Alighting:
    #   r[k,t,m] - r[k,t-1,m] >=
    #      - sum_i x[m,i,t] d_out[k,i,t]
    #      - sum_{i∈Nw} L[k,i,t,m]
    #
    # If r[k,t-1,m] not defined (t-1 ∉ ΔT_k), use r[k,t,m] alone.
    # ------------------------------------------------------------------
    for k in K:
        times_k = DeltaT[k]
        for t in times_k:
            for m in M:
                if (k, t, m) not in r:
                    continue

                # LHS
                if (k, t-1, m) in r:
                    lhs = r[k, t, m] - r[k, t-1, m]
                else:
                    lhs = r[k, t, m]

                # Boarding side: sum_i x[m,i,t] d_in + sum_i R
                boarding_terms = []
                for i in N:
                    if (k, i, t) in d_in:
                        boarding_terms.append(x[m, i, t])

                swap_in_terms = []
                for i in Nw:
                    if (k, i, t, m) in R:
                        swap_in_terms.append(R[k, i, t, m])

                rhs_up = mdl.sum(boarding_terms) + mdl.sum(swap_in_terms)

                mdl.add_constraint(
                    lhs <= rhs_up,
                    ctname=f"boarding_or_swap_in_LR_k{k}_t{t}_m{m}"
                )

                # Alighting side: - sum_i x[m,i,t] d_out - sum_i L
                alight_terms = []
                for i in N:
                    if (k, i, t) in d_out:
                        alight_terms.append(x[m, i, t])

                swap_out_terms = []
                for i in Nw:
                    if (k, i, t, m) in L:
                        swap_out_terms.append(L[k, i, t, m])

                rhs_down = - mdl.sum(alight_terms) - mdl.sum(swap_out_terms)

                mdl.add_constraint(
                    lhs >= rhs_down,
                    ctname=f"alighting_or_swap_out_LR_k{k}_t{t}_m{m}"
                )









def add_taxi_like_objective_LR_relax(mdl, I, y, s):
    """
    Add the full taxi-like MILP objective function:
        min ( C_oper + C_uns )
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







def create_taxi_like_model_LR_relax(I: Instance):
    """
    Create:
        - Model()
        - decision variables
        - constraints
        - objective
    """

    mdl = Model(name="TaxiLike_LR")

    # 1) variables
    x, y, r, L, R, s = create_decision_variables_LR_relax(mdl, I)

    # 2) constraints
    add_taxi_like_constraints_LR_relax(mdl, I, x, y, r, L, R, s)

    # 3) objective
    add_taxi_like_objective_LR_relax(mdl, I, y, s)

    return mdl, x, y, r, L, R, s



