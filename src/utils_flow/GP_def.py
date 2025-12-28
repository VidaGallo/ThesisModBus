from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import numpy as np


@dataclass(frozen = True, slots=True)   # Every time a new vector will be created, no possibility of adding new attributes
class Theta:
    """
    Class for parameters
    """

    p_noise: float = 0.2       # probability of -1 (ignored cluster)
    


    
