from cplex import Cplex

def main():
    # Crea un problema di ottimizzazione lineare
    problem = Cplex()

    # Impostiamo il tipo di problema (minimizzazione)
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Variabili: x e y, con costi nell'obiettivo
    problem.variables.add(
        names=["x", "y"],
        obj=[1.0, 2.0],  # coefficienti dell'obiettivo
        lb=[0.0, 0.0]    # limiti inferiori
    )

    # Vincoli: x + y >= 10
    problem.linear_constraints.add(
        lin_expr=[[["x", "y"], [1.0, 1.0]]],
        senses=["G"],  # "G" = maggiore o uguale
        rhs=[10.0]
    )

    # Risolvi
    problem.solve()

    # Stampa il risultato
    print("Status:", problem.solution.get_status())
    print("Valore ottimo:", problem.solution.get_objective_value())
    print("Valori delle variabili:", problem.solution.get_values())

if __name__ == "__main__":
    main()
