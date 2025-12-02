"""
print_data.py
=============
"""

from .instance import Instance

def print_instance_summary(inst: Instance):
    """Pretty-print full content of an Instance."""

    print("\n" + "="*123)
    print("INSTANCE SUMMARY:1n")

    ### Global info
    print("\n--->Global Info")
    print(f"  dt (minutes):     {inst.dt}")
    print(f"  t_max (slots):    {inst.t_max}")
    print(f"  Time periods T:   {inst.T}")

    ### Sets
    print("\n--->Sets")
    print(f"  |N| Nodes:        {len(inst.N)}")
    print(f"  |A| Arcs:         {len(inst.A)}")
    print(f"  |M| Modules:      {len(inst.M)}")
    print(f"  |K| Requests:     {len(inst.K)}")

    ### Parameters
    print("\n--->Parameters")
    print(f"  Module capacity Q:    {inst.Q}")
    print(f"  Cost per km:          {inst.c_km}")
    print(f"  Cost uns. demand:     {inst.c_uns_taxi}")

    ### Network
    print("\n--->Network: first 10 arcs")
    for idx, arc in enumerate(sorted(inst.A)):
        if idx >= 10:
            print("  ...")
            break
        gamma = inst.gamma.get(arc, None)
        tau = inst.tau_arc.get(arc, None)
        print(f"  Arc {arc}: length={gamma}, tau={tau}")

    ### Requests
    print("\n\n--->Requests Summary")
    for k in sorted(inst.K):
        print(f"\n  Request {k}:")
        print(f"    origin:        {inst.origin[k]}")
        print(f"    destination:   {inst.dest[k]}")
        print(f"    q_k:           {inst.q[k]}")
        print(f"    ΔT_k:          {inst.DeltaT[k]}")
        print(f"    ΔT_in:         {inst.DeltaT_in[k]}")
        print(f"    ΔT_out:        {inst.DeltaT_out[k]}")

    ### Sparse matrices
    print("\n--->Sparse d_in (first 10)")
    print(inst.d_in)

    print("\n--->Sparse d_out (first 10)")
    print(inst.d_out)

    print("\n" + "="*123 + "\n")
