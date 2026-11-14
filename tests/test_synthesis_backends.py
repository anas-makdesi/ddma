"""The fixed points agree with the textbook iterations, in both backends, on random
abstractions (dimensions 1-4, all index types, arbitrary non-monotone transitions)."""

import numpy as np
import pytest

from ddma import (
    RectangularPartition,
    SymbolicAbstraction,
    compiled_backend_available,
    reach_avoid,
    safety,
)
from ddma.synthesis import _box_all_in

BACKENDS = ["numpy"] + (["cpp"] if compiled_backend_available() else [])


def random_abstraction(rng, shape, n_inputs, dtype, p_enabled=0.9, max_extent=2):
    """Abstraction object with random successor boxes (no map behind it): each input
    shifts the state by a random step plus noise; boxes that leave the grid are disabled,
    as are random pairs, so removals propagate through the fixed points."""
    a = object.__new__(SymbolicAbstraction)
    a.X = RectangularPartition.uniform(np.zeros(len(shape)), np.ones(len(shape)), shape)
    a.U = RectangularPartition.uniform([0.0], [1.0], n_inputs)
    nq, n, sh = a.X.n_cells, len(shape), np.array(shape)
    sub = a.X.all_indices()[:, None, :]
    drift = rng.integers(-2, 3, size=(1, n_inputs, n))
    first = sub + drift + rng.integers(-1, 2, size=(nq, n_inputs, n))
    last = first + rng.integers(0, max_extent + 1, size=first.shape)
    inside = np.all((first >= 0) & (last < sh), axis=2)
    a.enabled = inside & (rng.random((nq, n_inputs)) < p_enabled)
    a.first = np.where(a.enabled[..., None], first, 0).astype(dtype)
    a.last = np.where(a.enabled[..., None], last, 0).astype(dtype)
    return a


def textbook_safety(a, safe):
    W = safe.copy()
    while True:
        good = _box_all_in(a, W, "numpy")
        Wn = W & good.any(axis=1)
        if np.array_equal(Wn, W):
            return W, good & W[:, None]
        W = Wn


def textbook_reach(a, target, avoid):
    R = target & ~avoid
    value = np.where(R, 0.0, np.inf)
    allowed = np.zeros(a.enabled.shape, dtype=bool)
    k = 0
    while True:
        k += 1
        good = _box_all_in(a, R, "numpy")
        new = ~R & ~avoid & good.any(axis=1)
        if not new.any():
            return R, value, allowed
        allowed[new] = good[new]
        value[new] = k
        R = R | new


CASES = [
    ((40,), 7, np.uint8),
    ((14, 11), 6, np.uint8),
    ((9, 8, 7), 6, np.uint16),
    ((6, 5, 5, 4), 6, np.int32),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape,n_inputs,dtype", CASES)
def test_safety_equals_textbook(shape, n_inputs, dtype, backend):
    rng = np.random.default_rng(len(shape))
    fractions = []
    for trial in range(8):
        a = random_abstraction(rng, shape, n_inputs, dtype, max_extent=trial % 3)
        safe = rng.random(a.n_states) < 0.9
        W, allowed = textbook_safety(a, safe)
        c = safety(a, safe, backend=backend)
        np.testing.assert_array_equal(c.domain, W)
        np.testing.assert_array_equal(c.allowed, allowed)
        fractions.append(W.mean())
    assert any(0.05 < f < 0.95 for f in fractions)  # some cases are not all-or-nothing


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape,n_inputs,dtype", CASES)
def test_reach_avoid_equals_textbook(shape, n_inputs, dtype, backend):
    rng = np.random.default_rng(10 + len(shape))
    for trial in range(6):
        a = random_abstraction(rng, shape, n_inputs, dtype, max_extent=trial % 3)
        target = rng.random(a.n_states) < 0.15
        avoid = rng.random(a.n_states) < 0.1
        R, value, allowed = textbook_reach(a, target, avoid)
        c = reach_avoid(a, target, avoid, backend=backend)
        np.testing.assert_array_equal(c.domain, R)
        np.testing.assert_array_equal(c.value, value)
        np.testing.assert_array_equal(c.allowed, allowed)


@pytest.mark.skipif(not compiled_backend_available(), reason="compiled kernel not built")
@pytest.mark.parametrize("shape,n_inputs,dtype", CASES)
def test_box_all_in_backends_agree(shape, n_inputs, dtype):
    rng = np.random.default_rng(20)
    a = random_abstraction(rng, shape, n_inputs, dtype)
    W = rng.random(a.n_states) < 0.7
    np.testing.assert_array_equal(_box_all_in(a, W, "cpp"), _box_all_in(a, W, "numpy"))


def test_backend_argument_is_checked():
    a = random_abstraction(np.random.default_rng(0), (4,), 2, np.uint8)
    with pytest.raises(ValueError):
        safety(a, np.ones(a.n_states, bool), backend="fortran")
    with pytest.raises(ValueError):
        safety(a, np.ones(a.n_states + 1, bool))


# ---------------------------------------------------------------- lazy scan

def as_lazy(a):
    """The same abstraction, seen through pair_boxes only (as a lazy abstraction)."""
    lz = object.__new__(SymbolicAbstraction)
    lz.X, lz.U, lz.lazy, lz.n_evaluated = a.X, a.U, True, 0
    lz.first = lz.last = lz.enabled = None

    def pair_boxes(Q, P):
        lz.n_evaluated += np.size(Q)
        return a.enabled[Q, P], a.first[Q, P], a.last[Q, P]

    lz.pair_boxes = pair_boxes
    return lz


@pytest.mark.parametrize("shape,n_inputs,dtype", CASES)
def test_lazy_safety_equals_textbook(shape, n_inputs, dtype):
    rng = np.random.default_rng(30 + len(shape))
    for trial in range(8):
        a = random_abstraction(rng, shape, n_inputs, dtype, max_extent=trial % 3)
        safe = rng.random(a.n_states) < 0.9
        W, allowed = textbook_safety(a, safe)
        lz = as_lazy(a)
        c = safety(lz, safe)
        np.testing.assert_array_equal(c.domain, W)
        assert c.allowed is None
        for q in range(0, a.n_states, 7):
            np.testing.assert_array_equal(c.allowed_row(q), allowed[q])
        assert lz.n_evaluated <= a.n_states * n_inputs
