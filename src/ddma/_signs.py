"""Monotonicity sign patterns.

A sign pattern ``signs`` has shape (m, n): ``signs[i, j]`` says how output i of the
unknown function depends on input j:

* ``+1``: non-decreasing (the standard monotone case),
* ``-1``: non-increasing (handled by reversing the order of coordinate j),
* `` 0``: output i does not depend on input j.

The all-ones pattern is the monotone setting of the paper. Patterns with -1 entries
are the order transformation used for unknown functions whose signs are known
(e.g. linear functions with known sign structure); 0 entries are an extension.
"""

from __future__ import annotations

import numpy as np


def normalize_signs(signs, m: int, n: int) -> np.ndarray:
    if signs is None:
        return np.ones((m, n), dtype=np.int8)
    s = np.asarray(signs)
    if s.ndim == 1:
        s = np.broadcast_to(s, (m, n))
    if s.shape != (m, n):
        raise ValueError(f"signs must have shape ({m}, {n}) or ({n},), got {s.shape}")
    if not np.all(np.isin(s, (-1, 0, 1))):
        raise ValueError("signs entries must be -1, 0 or +1")
    return np.array(s, dtype=np.int8)
