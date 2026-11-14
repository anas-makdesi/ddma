"""Shared helpers for the paper reproductions."""

import json
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def tac_partition(lo, hi, K, c):
    """TAC paper, Sec. V: K+1 cells per axis; K-1 uniform cells between
    alpha_1 = lo + c (hi - lo) and alpha_K = hi - c (hi - lo), plus two margin cells."""
    from ddma import RectangularPartition

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    K = np.broadcast_to(np.atleast_1d(K), lo.shape)
    return RectangularPartition.with_margins(lo, hi, lo + c * (hi - lo), hi - c * (hi - lo), K - 1)


def cell_volumes(partition):
    """Volume of every cell, array of shape partition.shape."""
    widths = [np.diff(e) for e in partition.edges]
    vol = widths[0]
    for w in widths[1:]:
        vol = np.multiply.outer(vol, w)
    return vol


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t


def save(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=conv))
