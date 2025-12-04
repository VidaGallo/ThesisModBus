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






def create_decision_variables_asym_new(mdl: Model, I: Instance):
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
    # z[k,t,m,i], solo nodi di scambio i ∈ Nw e t ∈ ΔT_k
    z = mdl.binary_var_dict(
        keys=[(k, t, m, i)
              for k in I.K
              for t in I.DeltaT[k]
              for m in I.M
              for i in I.Nw],
        name="z"
    )

    # s[k]
    s = mdl.binary_var_dict(
        keys=[k for k in I.K],
        name="s"
    )

    return x, y, r, s, z







def add_taxi_like_constraints_asym_new(mdl, I, x, y, r, s, z):
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
                    # se non esiste, contribuisce 0 al bilancio (sia r_t che r_{t-1})
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
    # 8) Linearization of z_{k,t,m,i} = r[k,t,m] * x[m,i,t]
    #
    # z <= r
    # z <= x
    # z >= r + x - 1
    # for k, t ∈ ΔT_k, m ∈ M, i ∈ Nw
    # ------------------------------------------------------------------
    for k in K:
        for t in DeltaT[k]:
            for m in M:
                for i in Nw:
                    if (k, t, m) not in r:
                        continue  # per sicurezza
                    mdl.add_constraint(
                        z[k, t, m, i] <= r[k, t, m],
                        ctname=f"z_le_r_k{k}_t{t}_m{m}_i{i}"
                    )
                    mdl.add_constraint(
                        z[k, t, m, i] <= x[m, i, t],
                        ctname=f"z_le_x_k{k}_t{t}_m{m}_i{i}"
                    )
                    mdl.add_constraint(
                        z[k, t, m, i] >= r[k, t, m] + x[m, i, t] - 1,
                        ctname=f"z_ge_r_plus_x_minus1_k{k}_t{t}_m{m}_i{i}"
                    )


    # ------------------------------------------------------------------
    # 9) "Exchange" constraint: 
    #
    #    sum_m z[k,t,m,i] <= sum_m z[k,t-1,m,i]
    #
    #    ∀ i ∈ Nw, ∀ k ∈ K, ∀ t ∈ ΔT_k : t > t0
    #    (if z[k,t-1,m,i] does not exist → treat it as 0)
    # ------------------------------------------------------------------
    for k in K:
        for t in DeltaT[k]:
            if t <= t0:
                continue
            for i in Nw:
                lhs = mdl.sum(
                    z[k, t, m, i]
                    for m in M
                    if (k, t, m, i) in z
                )
                rhs = mdl.sum(
                    z[k, t - 1, m, i]
                    for m in M
                    if (k, t - 1, m, i) in z
                )
                mdl.add_constraint(
                    lhs <= rhs,
                    ctname=f"swap_flow_k{k}_i{i}_t{t}"
                )

    # ------------------------------------------------------------------
    # 10) Handling boarding/alighting with d_in, d_out, and z:
    #
    # Boarding or exchange entry:
    #   r^m_{k,t} - r^m_{k,t-1} <=
    #       sum_i x^m_{i,t} d^k_{i,t}
    #       + sum_{i ∈ Nw} z^m_{k,i,t-1}
    #
    # Alighting or exchange exit:
    #   r^m_{k,t} - r^m_{k,t-1} >=
    #       - sum_i x^m_{i,t} \tilde d^k_{i,t}
    #       - sum_{i ∈ Nw} z^m_{k,i,t}
    #
    # with the convention r[k,t-1,m] = 0 if not defined.
    # ------------------------------------------------------------------
    for k in K:
        times_k = DeltaT[k]
        for t in times_k:
            for m in M:
                if (k, t, m) not in r:
                    continue

                # LHS
                if (k, t - 1, m) in r:
                    lhs = r[k, t, m] - r[k, t - 1, m]
                    t_prev_exists = True
                else:
                    lhs = r[k, t, m]
                    t_prev_exists = False

                # RHS up (boarding)
                boarding_terms = []
                for i in N:
                    if (k, i, t) in d_in:
                        boarding_terms.append(x[m, i, t])

                swap_in_terms = []
                if t_prev_exists:
                    for i in Nw:
                        if (k, t - 1, m, i) in z:
                            swap_in_terms.append(z[k, t - 1, m, i])

                rhs_up = mdl.sum(boarding_terms) + mdl.sum(swap_in_terms)
                mdl.add_constraint(
                    lhs <= rhs_up,
                    ctname=f"boarding_or_swap_in_k{k}_t{t}_m{m}"
                )

                # RHS down (alighting)
                alight_terms = []
                for i in N:
                    if (k, i, t) in d_out:
                        alight_terms.append(x[m, i, t])

                swap_out_terms = []
                for i in Nw:
                    if (k, t, m, i) in z:
                        swap_out_terms.append(z[k, t, m, i])

                rhs_down = -mdl.sum(alight_terms) - mdl.sum(swap_out_terms)
                mdl.add_constraint(
                    lhs >= rhs_down,
                    ctname=f"alighting_or_swap_out_k{k}_t{t}_m{m}"
                )


    
    # ------------------------------------------------------------------
    # 11) Symmetry-breaking:
    #
    # ????????
    # ------------------------------------------------------------------







def add_taxi_like_objective_asym_new(mdl, I, y, s):
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







def create_taxi_like_model_asym_new(I: Instance):
    """
    Create:
        - Model()
        - decision variables
        - constraints
        - objective
    """

    mdl = Model(name="TaxiLike")

    # 1) variables
    x, y, r, s, z = create_decision_variables_asym_new(mdl, I)

    # 2) constraints
    add_taxi_like_constraints_asym_new(mdl, I, x, y, r, s, z)

    # 3) objective
    add_taxi_like_objective_asym_new(mdl, I, y, s)

    return mdl, x, y, r, s, z



