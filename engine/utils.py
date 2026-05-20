__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import os
import random
import numpy as np
import torch
from typing import Iterable, Iterator


def infinite_batch_iterator(data_loader: Iterable) -> Iterator:
    """Yields batches indefinitely from the provided data loader."""
    while True:
        for batch in data_loader:
            yield batch


def seed_everything(default_seed: int = 32) -> None:
    env_seed = os.environ.get('PYTHONHASHSEED')
    if env_seed is not None:
        seed = int(env_seed)
        print(f"[Info] Found PYTHONHASHSEED in environment. Locking seed to: {seed}")
    else:
        seed = default_seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"[Info] No PYTHONHASHSEED found. Defaulting seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
