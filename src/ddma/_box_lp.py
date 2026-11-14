"""Certified bounds of max over a box of a minimum of affine functions.

Used by the "LP" final steps of the bounded-derivative models (thesis, eqs. 3.6-3.7
and 3.24-3.25). For every cell k we need

    U_k = max_{z in [lo_k, hi_k]}  min_l  (c_kl + a_l . z).

This is a small linear program. All cells of a batch are solved at once as one
block-diagonal LP (HiGHS). The solver's value is not used directly: from its dual
multipliers lam (lam >= 0, sum lam = 1) we evaluate the bound

    max_z sum_l lam_l (c_kl + a_l . z)  =  sum_l lam_l c_kl + sum_j max(mu_j lo_kj, mu_j hi_kj),
    mu = sum_l lam_l a_l,

which is >= U_k for *any* such lam (weak duality). So the returned bound is sound
whatever the solver accuracy, and equals U_k up to that accuracy. It is also capped
by ``fallback`` (a bound known to be sound, e.g. the min-max bound).
"""

from __future__ import annotations

import numpy as np

__all__ = ["max_min_affine"]


def max_min_affine(offsets, slopes, lo, hi, fallback=None, batch: int = 4000):
    """Certified upper bounds of max_{z in [lo_k, hi_k]} min_l (offsets[k, l] + slopes[l] . z).

    offsets (N, L), entries may be +inf (constraint absent); slopes (L, n);
    lo, hi (N, n) finite. ``fallback`` (N,) sound upper bounds (default +inf).
    Returns (N,) bounds; +inf where every offset of a cell is +inf.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    offsets = np.asarray(offsets, float)
    slopes = np.asarray(slopes, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    N, L = offsets.shape
    n = slopes.shape[1]
    out = np.full(N, np.inf) if fallback is None else np.array(fallback, float, copy=True)
    finite = np.isfinite(offsets)
    todo = np.flatnonzero(finite.any(axis=1))
    nv = n + 1  # per cell: z (n), c
    for s in range(0, todo.size, batch):
        cells = todo[s : s + batch]
        B = cells.size
        fin = finite[cells]  # (B, L)
        kk, ll = np.nonzero(fin)  # one row per finite constraint
        R = kk.size
        rows = np.arange(R)
        # row r:  c_k - a_l . z_k <= off_kl
        r_idx = np.concatenate([np.repeat(rows, n), rows])
        c_idx = np.concatenate([(kk[:, None] * nv + np.arange(n)).ravel(), kk * nv + n])
        vals = np.concatenate([(-slopes[ll]).ravel(), np.ones(R)])
        A = coo_matrix((vals, (r_idx, c_idx)), shape=(R, B * nv)).tocsr()
        b = offsets[cells][fin]
        cost = np.zeros(B * nv)
        cost[n::nv] = -1.0
        bnds = np.empty((B * nv, 2))
        bnds[:, 0], bnds[:, 1] = -np.inf, np.inf
        zi = (np.arange(B)[:, None] * nv + np.arange(n)).ravel()
        bnds[zi, 0] = lo[cells].ravel()
        bnds[zi, 1] = hi[cells].ravel()
        res = linprog(cost, A_ub=A, b_ub=b, bounds=bnds, method="highs")
        if res.status != 0 or res.ineqlin is None:
            continue  # keep the fallback for this batch (sound)
        lam = np.maximum(-np.asarray(res.ineqlin.marginals, float), 0.0)
        tot = np.bincount(kk, weights=lam, minlength=B)
        ok = tot > 1e-12
        lam = lam / np.where(ok, tot, 1.0)[kk]
        mu = np.zeros((B, n))
        np.add.at(mu, kk, lam[:, None] * slopes[ll])
        val = np.bincount(kk, weights=lam * b, minlength=B)
        val += np.maximum(mu * lo[cells], mu * hi[cells]).sum(axis=1)
        val = np.where(ok, val, np.inf)
        out[cells] = np.minimum(out[cells], val)
    return out
