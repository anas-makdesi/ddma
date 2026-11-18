"""NumPy versions of the monitor's pair checks (the C++ kernels are in cpp/ddma_monitor.hpp).

Same arithmetic in the same order as the C++ code, so the results are bit-identical:

    d_j = zq_j - z_k,j;  tn = sum_j s(d_j);  to = sum_j s(-d_j)   (left to right, from 0)
    dy = yq_i - y_k,i;   gap_new = (dy - theta_i) - tn;  gap_old = (-dy - theta_i) - to

These versions check every pair (no cell grid): pruning never changes a result, only the
work. Return formats are those of ddma._core.monitor_list / monitor_full / monitor_all.
"""

from __future__ import annotations

import numpy as np


def cone(t, lo, hi):
    """s(t) = hi t (t > 0), lo t (t < 0), 0 (t = 0), entrywise; 0 * inf counts as 0."""
    with np.errstate(invalid="ignore"):
        return np.where(t > 0, hi * t, np.where(t < 0, lo * t, 0.0))


def new_need(m, n):
    """Empty need tuple (any, width_r, hi, hi_out, lo, lo_out)."""
    return (np.zeros(m, bool), np.full(m, -np.inf), np.full((m, n), -np.inf),
            np.zeros((m, n), bool), np.full((m, n), np.inf), np.zeros((m, n), bool))


def pair_gaps(lo, hi, theta, i, zq, yq, Z, Y):
    """gap_new, gap_old (L,) of output i between point (zq, yq) and the rows of Z, Y."""
    d = zq[None, :] - Z
    tn = np.zeros(Z.shape[0])
    to = np.zeros(Z.shape[0])
    for j in range(Z.shape[1]):
        tn = tn + cone(d[:, j], lo[i, j], hi[i, j])
        to = to + cone(-d[:, j], lo[i, j], hi[i, j])
    dy = yq[i] - Y[:, i]
    return (dy - theta[i]) - tn, (-dy - theta[i]) - to


def accumulate(need, lo, hi, theta, i, du, dyu, gap):
    """Violating pairs "a above b": du = z_a - z_b (P, n), dyu, gap (P,)."""
    any_, w, nh, nho, nl, nlo = need
    any_[i] = True
    w[i] = max(w[i], gap.max())
    n = du.shape[1]
    s = np.stack([cone(du[:, j], lo[i, j], hi[i, j]) for j in range(n)], axis=1)
    tot = np.zeros(du.shape[0])
    for j in range(n):
        tot = tot + s[:, j]
    for j in range(n):
        rest = (dyu - theta[i]) - (tot - s[:, j])
        pos, neg = du[:, j] > 0, du[:, j] < 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = rest / du[:, j]
        if pos.any():
            nh[i, j] = max(nh[i, j], ratio[pos].max())
        if not pos.all():
            nho[i, j] = True
        if neg.any():
            nl[i, j] = min(nl[i, j], ratio[neg].min())
        if not neg.all():
            nlo[i, j] = True


def _eval(need, lo, hi, theta, tol, zq, yq, Z, Y):
    """All outputs of one point against the rows of Z, Y: updates need, returns g (L, m)
    = max(gap_new, gap_old) (only entries > tol matter)."""
    m = Y.shape[1]
    g = np.empty((Z.shape[0], m))
    for i in range(m):
        gn, go = pair_gaps(lo, hi, theta, i, zq, yq, Z, Y)
        vn, vo = gn > tol, go > tol
        if vn.any():
            accumulate(need, lo, hi, theta, i, zq[None, :] - Z[vn], yq[i] - Y[vn, i], gn[vn])
        if vo.any():
            accumulate(need, lo, hi, theta, i, -(zq[None, :] - Z[vo]), -(yq[i] - Y[vo, i]),
                       go[vo])
        g[:, i] = np.maximum(gn, go)
    return g


def monitor_list(lo, hi, theta, tol, Z, Y, z, y, idx, status):
    m, n = lo.shape
    need = new_need(m, n)
    idx = np.asarray(idx, np.int64)
    idx = idx[status[idx] != 2]
    worst = np.full(m, -np.inf)
    wit = np.full(m, -1, np.int64)
    if idx.size:
        g = _eval(need, lo, hi, theta, tol, z, y, Z[idx], Y[idx])
        for i in range(m):
            v = g[:, i] > tol
            if v.any():
                gi = np.where(v, g[:, i], -np.inf)
                best = gi.max()
                worst[i] = best
                wit[i] = idx[gi == best].min()  # ties: smaller stored index
    return need, worst, wit


def monitor_full(lo, hi, theta, tol, Z, Y, status, first, last, grid=None, prune=True):
    m, n = lo.shape
    need = new_need(m, n)
    P = last - first
    worst = np.full((P, m), -np.inf)
    wit = np.full((P, m), -1, np.int64)
    rows, gaps = [], []
    pairs = 0
    for q in range(first, last):
        if q == 0:
            continue
        g = _eval(need, lo, hi, theta, tol, Z[q], Y[q], Z[:q], Y[:q])
        pairs += q
        k = np.arange(q)
        clean = (k < first) & (status[:q] == 0)
        for i in range(m):
            v = g[:, i] > tol
            vc = v & clean
            if vc.any():
                gi = np.where(vc, g[:, i], -np.inf)
                worst[q - first, i] = gi.max()
                wit[q - first, i] = k[gi == gi.max()].min()
        late = np.flatnonzero((k >= first)[:, None] & (g > tol))  # row-major: (k, i)
        for t in late:
            kk, i = divmod(int(t), m)
            rows.append((q, kk, i))
            gaps.append(g[kk, i])
    ir = np.array(rows, np.int64).reshape(-1, 3)
    return need, worst, wit, ir, np.array(gaps, float), pairs


def monitor_all(lo, hi, theta, tol, Z, Y, prune=True):
    m, n = lo.shape
    K = Z.shape[0]
    need = new_need(m, n)
    count = np.zeros(m, np.int64)
    worst = np.full(m, -np.inf)
    pair = np.full((m, 2), -1, np.int64)
    conflicts = np.zeros(K, np.int64)
    for q in range(1, K):
        g = _eval(need, lo, hi, theta, tol, Z[q], Y[q], Z[:q], Y[:q])
        v = g > tol
        count += v.sum(0)
        for i in range(m):
            if v[:, i].any():
                gi = np.where(v[:, i], g[:, i], -np.inf)
                b = gi.max()
                if b > worst[i]:  # earlier q wins ties, then the smaller k
                    worst[i] = b
                    pair[i] = (q, int(np.flatnonzero(gi == b)[0]))
        anyv = v.any(1)
        conflicts[q] += anyv.sum()
        conflicts[:q] += anyv
    return need, count, worst, pair, conflicts, K * (K - 1) // 2
