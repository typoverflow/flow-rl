import random

import numpy as np

from flowrl.types import *


def set_seed_everywhere(seed: Optional[Union[str, int]] = None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed
