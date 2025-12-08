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
from utils.instance_def import Instance


# seed
import random
import numpy as np
seed = 23
random.seed(seed)
np.random.seed(seed)



def create_decision_variables_ab_LR_plat(mdl: Model, I: Instance):
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

    t0 = I.T[0]
    print("Nw:", I.Nw)

    L = mdl.binary_var_dict(
        keys=[
            (k, i, t, m)
            for k in I.K
            for t in I.DeltaT[k]
            if t > t0                 # no excanges at t0
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
            if t > t0                # no excanges at t0
            for i in I.Nw
            for m in I.M
        ],
        name="R"
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

    # h[i,j,t]: numero di moduli che percorrono l'arco (i,j) al tempo t
    h = mdl.integer_var_dict(
        keys=[(i, j, t) for (i, j) in I.A for t in I.T],
        lb=0,
        ub=len(I.M),
        name="h"
    )

    return x, y, r, L, R, s, a, b, h







def add_taxi_like_constraints_ab_LR_plat(mdl, I, x, y, r, L, R, s, a, b, h):
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




     # -------------------------------------
    # 8) Exchange constraints with L,R  
    # ----------------------------------------

    # (8.1) L <= r_prev and L <= x
    for k in K:
        for t in DeltaT[k]:
            if t == t0:
                continue  # niente scambi al primo istante globale
            prev_t = t - 1
            for i in Nw:
                for m in M:
                    if (k, i, t, m) not in L:
                        continue

                    # può lasciare solo se era a bordo al tempo precedente
                    if (k, prev_t, m) in r:
                        mdl.add_constraint(
                            L[k, i, t, m] <= r[k, prev_t, m],
                            ctname=f"L_only_if_onboard_k{k}_i{i}_t{t}_m{m}"
                        )
                    else:
                        # se non esiste r[k,prev_t,m], forziamo L=0
                        mdl.add_constraint(
                            L[k, i, t, m] <= 0,
                            ctname=f"L_forbidden_k{k}_i{i}_t{t}_m{m}"
                        )

                    # modulo m deve essere nel nodo i
                    mdl.add_constraint(
                        L[k, i, t, m] <= x[m, i, t],
                        ctname=f"L_only_if_module_at_node_k{k}_i{i}_t{t}_m{m}"
                    )

    # (8.2) R <= x   (può ricevere solo se il modulo è nel nodo di scambio)
    for k in K:
        for t in DeltaT[k]:
            if t == t0:
                continue
            for i in Nw:
                for m in M:
                    if (k, i, t, m) not in R:
                        continue
                    mdl.add_constraint(
                        R[k, i, t, m] <= x[m, i, t],
                        ctname=f"R_only_if_module_at_node_k{k}_i{i}_t{t}_m{m}"
                    )

    # (8.3) Bilancio scambi in ogni (k,i,t): somma L = somma R
    for k in K:
        for t in DeltaT[k]:
            if t == t0:
                continue
            for i in Nw:
                mdl.add_constraint(
                    mdl.sum(
                        L[k, i, t, m]
                        for m in M
                        if (k, i, t, m) in L
                    )
                    ==
                    mdl.sum(
                        R[k, i, t, m]
                        for m in M
                        if (k, i, t, m) in R
                    ),
                    ctname=f"LR_balance_k{k}_i{i}_t{t}"
                )

    # (8.4) (opzionale) Al più uno scambio per richiesta e istante
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
                ctname=f"max_one_exchange_k{k}_t{t}"
            )

    


    # ------------------------------------------------------------------
    # 10) Boarding / Alighting with a,b + L,R + coerenza di r
    # ------------------------------------------------------------------

    # (10.1) Al più un evento (salita o discesa) per (k,t,m)
    #for k in K:
    #    for t in DeltaT[k]:
    #        for m in M:
    #            mdl.add_constraint(
    #                a[k, t, m] + b[k, t, m] <= 1,
    #                ctname=f"one_event_k{k}_t{t}_m{m}"
    #            )

    # (10.1bis) Max 1 salita e max 1 discesa per richiesta k
    for k in K:
        mdl.add_constraint(
            mdl.sum(a[k, t, m] for t in DeltaT[k] for m in M) <= 1,
            ctname=f"max_one_boarding_k{k}"
        )
        mdl.add_constraint(
            mdl.sum(b[k, t, m] for t in DeltaT[k] for m in M) <= 1,
            ctname=f"max_one_alighting_k{k}"
        )

    # (10.2) Dinamica di r con L,R (catena su ΔT_k ordinato)
    for k in K:
        times_k = sorted(DeltaT[k])
        if not times_k:
            continue

        for m in M:
            # primo tempo della finestra di k
            t_first = times_k[0]

            loss_first = mdl.sum(
                L[k, i, t_first, m]
                for i in Nw
                if (k, i, t_first, m) in L
            )
            gain_first = mdl.sum(
                R[k, i, t_first, m]
                for i in Nw
                if (k, i, t_first, m) in R
            )

            mdl.add_constraint(
                r[k, t_first, m]
                ==
                a[k, t_first, m]
                - b[k, t_first, m]
                - loss_first
                + gain_first,
                ctname=f"r_chain_first_k{k}_t{t_first}_m{m}"
            )

            # tempi successivi
            for t_prev, t in zip(times_k[:-1], times_k[1:]):
                loss_t = mdl.sum(
                    L[k, i, t, m]
                    for i in Nw
                    if (k, i, t, m) in L
                )
                gain_t = mdl.sum(
                    R[k, i, t, m]
                    for i in Nw
                    if (k, i, t, m) in R
                )

                mdl.add_constraint(
                    r[k, t, m]
                    ==
                    r[k, t_prev, m]
                    + a[k, t, m]
                    - b[k, t, m]
                    - loss_t
                    + gain_t,
                    ctname=f"r_chain_k{k}_t{t}_m{m}"
                )

    # (10.3) Attivazione di a (salite da terra)
    for k in K:
        for t in DeltaT[k]:
            for m in M:
                boarding_terms = [
                    x[m, i, t]
                    for i in N
                    if (k, i, t) in d_in
                ]
                if boarding_terms:
                    mdl.add_constraint(
                        a[k, t, m] <= mdl.sum(boarding_terms),
                        ctname=f"a_activation_k{k}_t{t}_m{m}"
                    )
                else:
                    # se non c'è alcun nodo di salita ammesso, a deve essere 0
                    mdl.add_constraint(
                        a[k, t, m] <= 0,
                        ctname=f"a_forbidden_k{k}_t{t}_m{m}"
                    )

    # (10.4) Attivazione di b (discese a terra)
    for k in K:
        for t in DeltaT[k]:
            for m in M:
                alight_terms = [
                    x[m, i, t]
                    for i in N
                    if (k, i, t) in d_out
                ]
                if alight_terms:
                    mdl.add_constraint(
                        b[k, t, m] <= mdl.sum(alight_terms),
                        ctname=f"b_activation_k{k}_t{t}_m{m}"
                    )
                else:
                    mdl.add_constraint(
                        b[k, t, m] <= 0,
                        ctname=f"b_forbidden_k{k}_t{t}_m{m}"
                    )


    # (10.5) Coerenza con lo stato precedente: a/b compatibili con r_{t_prev}
    for k in K:
        times_k = sorted(DeltaT[k])
        if not times_k:
            continue

        for m in M:
            # imponiamo i vincoli solo a partire dal secondo istante della finestra
            for t_prev, t in zip(times_k[:-1], times_k[1:]):
                if (k, t_prev, m) in r:
                    mdl.add_constraint(
                        a[k, t, m] <= 1 - r[k, t_prev, m],
                        ctname=f"a_prev_state_k{k}_t{t}_m{m}"
                    )
                    mdl.add_constraint(
                        b[k, t, m] <= r[k, t_prev, m],
                        ctname=f"b_prev_state_k{k}_t{t}_m{m}"
                    )

    # ------------------------------------------------------------------
    # 11) Definizione di h_{i,j,t}:
    #     h[i,j,t] = sum_{m in M} y[m,i,j,t]
    # ------------------------------------------------------------------
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                h[i, j, t] == mdl.sum(y[m, i, j, t] for m in M),
                ctname=f"h_def_i{i}_j{j}_t{t}"
            )

    # ------------------------------------------------------------------
    # 11) Definizione di h_{i,j,t}:
    #     h[i,j,t] = sum_{m in M} y[m,i,j,t]
    # ------------------------------------------------------------------
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                h[i, j, t] == mdl.sum(y[m, i, j, t] for m in M),
                ctname=f"h_def_i{i}_j{j}_t{t}"
            )




def add_taxi_like_objective_ab_LR_plat(mdl, I, y, s, h):
    """
    Add the full taxi-like MILP objective function:
        min ( C_oper + C_uns - G_plat)
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
    # 3) Platooning "reward"
    #     C_plat = - g_plat * sum_{i,j,t} gamma(i,j) * h[i,j,t]
    # -------------------------
    # Small check:
    if I.g_plat > 1.0 / max(1, len(I.M)):
        print(f"[WARN] g_plat={I.g_plat} > 1/|M|={1.0/len(I.M):.4f}")

    C_plat = - I.g_plat * mdl.sum(
        I.gamma[(i, j)] * h[i, j, t]
        for (i, j) in I.A
        for t in I.T
    )

    # -------------------------
    # Minimize total cost
    # -------------------------
    mdl.minimize(C_oper + C_uns - C_plat)

    # Return objective expression if needed
    return C_oper + C_uns - C_plat








def create_taxi_like_model_ab_LR_plat(I: Instance):
    """
    Create:
        - Model()
        - decision variables
        - constraints
        - objective
    """

    mdl = Model(name="TaxiLike")

    # 1) variables
    x, y, r, L, R, s, a, b, h = create_decision_variables_ab_LR_plat(mdl, I)

    # 2) constraints
    add_taxi_like_constraints_ab_LR_plat(mdl, I, x, y, r, L, R, s, a, b, h)

    # 3) objective
    add_taxi_like_objective_ab_LR_plat(mdl, I, y, s, h)

    return mdl, x, y, r, L, R, s, a, b, h



