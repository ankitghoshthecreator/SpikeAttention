"""Random seed helpers."""

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy random generators."""
    random.seed(seed)
    np.random.seed(seed)
