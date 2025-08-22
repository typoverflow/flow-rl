import os
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

def wandb_sync_run(root_path: str, *args):
    import wandb

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"wandb_sync_offline: path {root_path} not found.")

    def recursive_upload(path):
        if os.path.isfile(path):
            return
        for sub_path in os.listdir(path):
            full_path = os.path.join(path, sub_path)
            if sub_path == "wandb":
                final_path = None
                for sub_sub_path in os.listdir(full_path):
                    if sub_sub_path.startswith("run-") or sub_sub_path.startswith("offline-run-"):
                        final_path = os.path.join(full_path, sub_sub_path)
                if final_path:
                    print(f"sync {final_path} to wandb ...")
                    os.system(f"wandb sync {final_path} {' '.join(args)}")
            else:
                recursive_upload(full_path)
    recursive_upload(root_path)
