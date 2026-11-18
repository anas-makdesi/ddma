"""Assumption monitor, round 2: check_data, the three levels, quarantine, widening limits,
and identical results for every backend and number of threads."""

import numpy as np
import pytest

import ddma
from ddma import AssumptionMonitor, PartitionedSimulatingMap, RectangularPartition, check_data
from ddma.monitor import _A, _P
from ddma.partitioned import _core

needs_cpp = pytest.mark.skipif(_core is None, reason="compiled kernel not installed")


def monotone_data(rng, K, w=0.05):
    """Two outputs: increasing in (z0, z1), and constant; noise uses its full width."""
    Z = rng.uniform(0, 1, (K, 2))
    Y = np.c_[0.8 * Z[:, 0] + 0.3 * Z[:, 1] + 0.2 * Z[:, 0] ** 2 + rng.uniform(-w, w, K),
              3 + rng.uniform(-1, 1, K)]
    lo = np.zeros((2, 2))
    hi = np.array([[np.inf, np.inf], [0.0, 0.0]])
    return Z, Y, lo, hi, np.array([-w, -1.0]), np.array([w, 1.0])


def brute(Z, Y, lo, hi, theta, tol=1e-9):
    """All pairs in plain NumPy: count of violating pairs per output, needed width."""
    m = Y.shape[1]
    dz = Z[:, None, :] - Z[None, :, :]
    count, width = np.zeros(m, int), theta.astype(float).copy()
    for i in range(m):
        with np.errstate(invalid="ignore"):
            c = np.where(dz > 0, hi[i] * dz, np.where(dz < 0, lo[i] * dz, 0.0)).sum(-1)
        gap = Y[:, None, i] - Y[None, :, i] - theta[i] - c  # gap[a, b]: a above b
        v = np.maximum(gap, gap.T)[np.triu_indices(len(Z), 1)]
        count[i] = int(np.sum(v > tol))
        width[i] = max(width[i], theta[i] + gap.max())
    return count, width


def bits(x):
    x = np.asarray(x)
    return x.view(np.uint8).tobytes() if x.dtype == float else x.tobytes()


# ---------------------------------------------------------------------- check_data
def test_check_data_equals_brute_force_and_finds_the_bad_points():
    rng = np.random.default_rng(0)
    Z, Y, lo, hi, d_lo, d_hi = monotone_data(rng, 700)
    bad = rng.choice(700, 4, replace=False)
    Y[bad, 0] -= 0.4  # corrupted sensor readings
    res = check_data(Z, Y, lo, hi, d_lo, d_hi, input_names=["v", "u"])
    count, width = brute(Z, Y, lo, hi, d_hi - d_lo)
    assert not res.ok and np.array_equal(res.violating, count)
    assert np.allclose(res.needed["width"], width)
    assert set(res.suspects(4)) == set(bad)  # the corrupted points have the most conflicts
    assert np.isnan(res.needed["slope_hi"][0]).all()  # upper bounds cannot explain it
    assert "Most conflicts" in res.report() and "w.r.t. v" in res.report()
    # the sign form gives the same, and clean data pass
    res2 = check_data(Z, Y, d_lo=d_lo, d_hi=d_hi, signs=[[1, 1], [0, 0]])
    assert np.array_equal(res2.conflicts, res.conflicts)
    Y[bad, 0] += 0.4
    assert check_data(Z, Y, lo, hi, d_lo, d_hi).ok


@needs_cpp
def test_check_data_identical_for_backends_pruning_and_threads():
    rng = np.random.default_rng(1)
    Z, Y, lo, hi, d_lo, d_hi = monotone_data(rng, 900)
    Y[rng.choice(900, 6, replace=False), 0] += rng.uniform(-0.5, 0.5, 6)
    lo[0, 1], hi[0, 1] = 0.1, 0.6  # a finite slope interval too
    ref = check_data(Z, Y, lo, hi, d_lo, d_hi, backend="numpy")
    old = ddma.get_num_threads()
    try:
        for t in (1, 2, 3):
            ddma.set_num_threads(t)
            for prune in (False, True):
                r = check_data(Z, Y, lo, hi, d_lo, d_hi, prune=prune, backend="cpp")
                for f in ("violating", "worst", "worst_pair", "conflicts"):
                    assert bits(getattr(r, f)) == bits(getattr(ref, f)), f
                for k in ref.needed:
                    assert bits(r.needed[k]) == bits(ref.needed[k]), k
    finally:
        ddma.set_num_threads(old)
    assert r.pairs_evaluated < ref.pairs_evaluated  # pruning skipped work


@needs_cpp
def test_pruning_skips_most_pairs_on_consistent_data():
    rng = np.random.default_rng(2)
    Z, Y, lo, hi, d_lo, d_hi = monotone_data(rng, 20000)
    r = check_data(Z, Y, lo, hi, d_lo, d_hi)
    assert r.ok and r.pairs_evaluated < 0.05 * 20000 * 19999 / 2


# ---------------------------------------------------------------------- the three levels
def seeded(rng, K=3000, **kw):
    Z, Y, lo, hi, d_lo, d_hi = monotone_data(rng, K)
    mon = AssumptionMonitor(lo, hi, d_lo, d_hi, **kw)
    mon.add(Z, Y)
    return mon, Z, Y


def test_level_cells_catches_large_jumps_without_data():
    P = RectangularPartition.uniform([0, 0], [1, 1], [10, 10])
    rng = np.random.default_rng(3)
    Z, Y, lo, hi, d_lo, d_hi = monotone_data(rng, 2000)
    psm = PartitionedSimulatingMap.from_data(Z, Y, P, d_lo, d_hi, signs=[[1, 1], [0, 0]])
    mon = AssumptionMonitor.for_model(psm)  # no stored data: only level 1 can see it
    v = mon.check([[0.5, 0.5]], [[3.0, 3.0]])
    assert [x.level for x in v] == ["cells"] and v[0].output == 0


def test_level_sample_catches_a_broad_violation_at_once():
    rng = np.random.default_rng(4)
    mon, Z, Y = seeded(rng, full="later")
    # far below every stored point that is larger in (z0, z1): many witnesses
    v = mon.check([[0.9, 0.9]], [[0.0, 3.0]])
    assert len(v) == 1 and v[0].level == "sample" and v[0].found_at == v[0].index == 0
    assert mon.status()["quarantined"] == 1 and mon.pending == 1


def test_level_full_later_finds_what_the_sample_misses_and_equals_step_mode():
    rng = np.random.default_rng(5)
    Z, Y, lo, hi, d_lo, d_hi = monotone_data(rng, 3000)
    Zn, Yn, *_ = monotone_data(rng, 400)
    # subtle errors: each conflicts with only a few stored points
    sub = rng.choice(400, 8, replace=False)
    Yn[sub, 0] -= 0.12
    res = {}
    for mode in ("step", "later"):
        mon = AssumptionMonitor(lo, hi, d_lo, d_hi, full=mode, sample=0)
        mon.add(Z, Y)
        during = mon.check(Zn, Yn)
        assert mon.pending == (400 if mode == "later" else 0)
        after = mon.flush()
        assert mon.pending == 0
        if mode == "later":
            assert during == [] and after and all(v.found_at == 399 for v in after)
        res[mode] = mon
    key = lambda v: (v.index, v.output, v.gap, v.witness)  # noqa: E731
    assert sorted(map(key, res["step"].violations)) == sorted(map(key, res["later"].violations))
    for k in ("width", "slope_lo", "slope_hi"):
        assert bits(res["step"].needed()[k]) == bits(res["later"].needed()[k])
    assert set(v.index for v in res["step"].violations) <= set(sub.tolist())
    # needed() after the full check is exact: brute force over all stored data
    Zs, Ys = res["later"].data(quarantined=True)
    _, width = brute(Zs, Ys, lo, hi, d_hi - d_lo)
    assert np.allclose(res["later"].needed()["width"], width)


@needs_cpp
@pytest.mark.parametrize("full", ["step", "later"])
def test_monitor_identical_for_backends_pruning_and_threads(full):
    rng = np.random.default_rng(6)
    Z, Y, lo, hi, d_lo, d_hi = monotone_data(rng, 2500)
    Zn, Yn, *_ = monotone_data(rng, 300)
    Yn[rng.choice(300, 10, replace=False), 0] += rng.uniform(-0.3, 0.3, 10)
    Yn[rng.choice(300, 3, replace=False), 1] += 1.5

    def run(backend, prune):
        mon = AssumptionMonitor(lo, hi, d_lo, d_hi, full=full, sample=300, prune=prune,
                                backend=backend, seed=7)
        mon.add(Z, Y)
        mon.check(Zn, Yn)
        mon.flush()
        return mon

    ref = run("numpy", False)
    key = lambda v: (v.index, v.output, v.level, v.gap, v.witness, v.found_at)  # noqa: E731
    old = ddma.get_num_threads()
    try:
        for t in (1, 2, 3):
            ddma.set_num_threads(t)
            for prune in (False, True):
                mon = run("cpp", prune)
                assert [key(v) for v in mon.violations] == [key(v) for v in ref.violations]
                for k in ("width", "slope_lo", "slope_hi"):
                    assert bits(mon.needed()[k]) == bits(ref.needed()[k])
                assert np.array_equal(mon._st, ref._st)
    finally:
        ddma.set_num_threads(old)
    assert ref.violations


def test_sample_indices_are_minstd():
    """Level-2 samples follow x <- 48271 x mod (2^31 - 1), x0 = seed mod (P - 1) + 1, index
    = x mod V (the MATLAB code draws exactly the same)."""
    mon = AssumptionMonitor([[0.0]], [[1.0]], sample=5, seed=3)
    x, ref = 3 % (_P - 1) + 1, []
    for _ in range(10):
        x = (_A * x) % _P
        ref.append(x % 17)
    assert list(mon._sample(17)) + list(mon._sample(17)) == ref


# ---------------------------------------------------------------------- quarantine
def test_quarantine_prevents_follow_up_alarms():
    rng = np.random.default_rng(8)
    mon, Z, Y = seeded(rng, 500)
    bad_z, good = np.array([0.5, 0.5]), 0.8 * 0.5 + 0.3 * 0.5 + 0.2 * 0.25
    # one reading far too high, then good readings just below/right of it
    v = mon.check([bad_z], [[good + 0.5, 3.0]])
    assert v and mon.quarantine()["ids"].tolist() == [0]
    Zg = np.array([[0.5, 0.5], [0.52, 0.5], [0.5, 0.53]])
    Yg = np.c_[0.8 * Zg[:, 0] + 0.3 * Zg[:, 1] + 0.2 * Zg[:, 0] ** 2, [3.0] * 3]
    # these good points do contradict the bad one ...
    assert not check_data(np.r_[[bad_z], Zg], np.r_[[[good + 0.5, 3.0]], Yg],
                          mon.slope_lo, mon.slope_hi, mon.d_lo, mon.d_hi).ok
    # ... but the bad one is quarantined, so they raise no alarm
    assert mon.check(Zg, Yg) == []
    assert mon.data()[0].shape[0] == 503 and mon.data(quarantined=True)[0].shape[0] == 504
    assert len(mon.quarantine()["violations"]) == 1


def test_earlier_point_is_trusted_and_quarantined_partners_blame_no_one():
    mon = AssumptionMonitor.monotone(None, -0.01, 0.01, m=1, n=1)
    mon.check([[0.0]], [0.0])      # 0: first, trusted
    mon.check([[1.0]], [-1.0])     # 1: decreases: blamed (0 was first)
    mon.check([[2.0]], [-0.5])     # 2: conflicts only with 0 (and agrees with 1): blamed
    mon.check([[3.0]], [5.0])      # 3: fine with 0; conflicts with nobody clean
    assert [v.index for v in mon.violations] == [1, 2]
    assert mon.quarantine()["ids"].tolist() == [1, 2]


# ---------------------------------------------------------------------- responses
def _widen_case(on_violation, extra, **kw):
    rng = np.random.default_rng(9)
    Z = rng.uniform(0, 1, (800, 1))
    y = Z[:, 0] + rng.uniform(-0.05, 0.05, 800)
    mon = AssumptionMonitor([[0.0]], [[2.0]], -0.05, 0.05, on_violation=on_violation, **kw)
    mon.add(Z[:400], y[:400])
    y2 = Z[400:, 0] + rng.uniform(-0.05 - extra, 0.05 + extra, 400)
    mon.check(Z[400:], y2)
    return mon


def test_widen_then_fallback_widens_under_the_limit():
    mon = _widen_case("widen_then_fallback", 0.02)  # width 0.1 -> about 0.14 < 0.2
    assert mon.changes and not mon.fallback_active and mon.consistent()
    assert mon.status()["quarantined"] == 0  # quarantine lifted after widening
    assert mon.theta[0] <= 0.2


def test_widen_then_fallback_falls_back_beyond_the_limit():
    mon = _widen_case("widen_then_fallback", 0.2)  # would need about 0.5 > 2 x 0.1
    assert mon.fallback_active and "beyond the limit 0.2" in mon.fallback_reason
    assert mon.theta[0] <= 0.2 + 1e-12  # never widened past the limit
    assert "beyond the limit" in mon.report()
    n = len(mon.changes)
    mon.check([[0.5]], [3.0])  # after falling back: no more widening
    assert len(mon.changes) == n and mon.fallback_active


def test_plain_widen_has_no_limit():
    mon = _widen_case("widen", 0.2)
    assert not mon.fallback_active and mon.theta[0] > 0.4 and mon.consistent()


def test_slope_widening_only_with_a_limit():
    rng = np.random.default_rng(10)
    Z = rng.uniform(0, 1, (600, 1))
    y = 3.0 * Z[:, 0] + rng.uniform(-0.01, 0.01, 600)
    for limit, widened in ((4.0, True), (2.5, False)):
        mon = AssumptionMonitor([[0.0]], [[2.0]], -0.01, 0.01, on_violation="widen",
                                widen=("slope_hi", 0, 0), slope_limit=limit)
        mon.add(Z[:100], y[:100])
        mon.check(Z[100:], y[100:])
        if widened:
            assert 2.9 < mon.slope_hi[0, 0] <= 3.0 + 1e-9 and mon.consistent()
        else:
            assert mon.slope_hi[0, 0] == 2.0 and mon.notes and not mon.changes
        mon2 = AssumptionMonitor([[0.0]], [[2.0]], -0.01, 0.01,
                                 on_violation="widen_then_fallback",
                                 widen=("slope_hi", 0, 0), slope_limit=limit)
        mon2.add(Z[:100], y[:100])
        mon2.check(Z[100:], y[100:])
        assert mon2.fallback_active != widened


def test_add_runs_pending_checks_first():
    rng = np.random.default_rng(11)
    mon, Z, Y = seeded(rng, 300, full="later", sample=0)
    mon.check([[0.9, 0.9]], [[0.0, 3.0]])
    assert mon.pending == 1
    mon.add(Z[:5], Y[:5])
    assert mon.pending == 0 and mon.violations[0].level == "full"
