"""
print_data.py
=============
 - print of the Iinstance Class
 - print of the Results
"""

from .instance_def import Instance
import json
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt


# seed
import random
import numpy as np
seed = 23
random.seed(seed)
np.random.seed(seed)

def print_instance_summary(inst: Instance):
    """Pretty-print full content of an Instance."""

    print("\n\n" + "=" * 100)
    print("INSTANCE SUMMARY")
    print("=" * 100)

    # --------------------------------------------------------------
    # GLOBAL INFO
    # --------------------------------------------------------------
    print("\n---> Global Info")
    print(f"  dt (minutes):         {inst.dt}")
    print(f"  t_max (slots):        {inst.t_max}")
    print(f"  Time periods T:       {inst.T}")

    # --------------------------------------------------------------
    # SETS
    # --------------------------------------------------------------
    print("\n---> Sets")
    print(f"  |N| Nodes:            {len(inst.N)}")
    print(f"  |A| Arcs:             {len(inst.A)}")
    print(f"  |M| MAIN modules:     {len(inst.M)}")
    print(f"  |P| TRAIL modules:    {len(inst.P)}")
    print(f"  |K| Requests:         {len(inst.K)}")
    print(f"  |N_w| Swap nodes:     {len(inst.Nw)}")
    print(f"  N_w:                  {sorted(inst.Nw)}")

    # --------------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------------
    print("\n---> Parameters")
    print(f"  Capacity Q:           {inst.Q}")
    print(f"  c_km (cost/km):       {inst.c_km}")
    print(f"  c_uns (uns. penalty): {inst.c_uns}")
    print(f"  g_plat (reward):      {inst.g_plat}")
    print(f"  Depot node:           {inst.depot}")

    # Z_max = |P| / |M|
    try:
        print(f"  Z_max = |P|/|M|:      {inst.Z_max}")
    except Exception as e:
        print(f"  Z_max: ERROR -> {e}")

    # --------------------------------------------------------------
    # NETWORK (first 10 arcs)
    # --------------------------------------------------------------
    print("\n---> Network: first 10 arcs")
    for idx, arc in enumerate(sorted(inst.A)):
        if idx >= 10:
            print("  ...")
            break
        gamma = inst.gamma.get(arc, None)
        tau   = inst.tau_arc.get(arc, None)
        print(f"  Arc {arc}:  length_km={gamma},  tau_steps={tau}")

    # --------------------------------------------------------------
    # REQUESTS
    # --------------------------------------------------------------
    print("\n---> Requests Summary")
    for k in sorted(inst.K):
        print(f"\n  Request {k}:")
        print(f"    origin:        {inst.origin[k]}")
        print(f"    destination:   {inst.dest[k]}")
        print(f"    q_k:           {inst.q[k]}")
        print(f"    ΔT_k:          {inst.DeltaT[k]}")
        print(f"    ΔT_in:         {inst.DeltaT_in[k]}")
        print(f"    ΔT_out:        {inst.DeltaT_out[k]}")

    # --------------------------------------------------------------
    # SPARSE MATRICES d_in / d_out (first 10)
    # --------------------------------------------------------------
    print("\n---> Sparse d_in (first 10)")
    for idx, key in enumerate(inst.d_in):
        if idx >= 10:
            print("  ...")
            break
        print(f"  {key}: {inst.d_in[key]}")

    print("\n---> Sparse d_out (first 10)")
    for idx, key in enumerate(inst.d_out):
        if idx >= 10:
            print("  ...")
            break
        print(f"  {key}: {inst.d_out[key]}")

    print("\n" + "=" * 100 + "\n")
