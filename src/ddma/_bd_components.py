"""One-output building blocks of :class:`ddma.BoundedDerivativeModel`.

Each component learns, on a rectangular partition of its input coordinates z, interval
bounds of f_i(z) + [d_lo, d_hi] per cell, for a scalar function f_i whose partial
derivatives lie in [g_lo_j, g_hi_j]. Data are pairs (z_k, y_k) with
y_k in f_i(z_k) + [d_lo, d_hi]. Thesis, Chapter 3 (Section 3.4 numbers the approaches).

* :class:`SignMatrixComponent` (approaches 2 and 3, "All matrix" and "Lin + Mono").
  For every slope vector a with a_j in {min(g_lo_j, 0), max(g_hi_j, 0)}, the function
  h = f_i - a . z is monotone: non-decreasing in z_j when a_j is the lower choice,
  non-increasing when it is the upper one (Remark 5: a coordinate flip). Its
  simulating map sigma_a is learned from y_k - a . z_k, and f_i + D lies in
  a . z + sigma_a for every a (Prop. 9, all matrices). Final step: "box" takes
  the intersection of the interval hulls per cell (eqs. 3.8-3.9), "lp" the interval
  hull of the intersection (eqs. 3.6-3.7).

* :class:`GrowthComponent` (approaches 4 and 5, "Growth" and "Lin + Growth").
  Every data point gives the growth cone (Def. 8)

      y_k - theta + phi_lo(z - z_k)  <=  f_i(z) + d  <=  y_k + theta + phi_hi(z - z_k),
      phi_hi(t) = sum_j max(g_hi_j t_j, g_lo_j t_j),  phi_lo(t) = sum_j min(...),

  theta = d_hi - d_lo. The tightest bounds at the grid vertices (eqs. 3.20-3.21) are
  computed exactly: each point is written to the 2^n vertices of its cell, then a
  running min/max along every axis spreads it over the grid. This is the fixed point
  of Prop. 11, reached in one pass per axis because phi is a sum of one-dimensional
  terms, each linear on either side of 0 (cost O(|K| 2^n + |V| n), cf. Prop. 12).
  Cell bounds then follow from the vertex values: "box" = eqs. 3.26-3.27,
  "lp" = eqs. 3.24-3.25 (the slopes are those of the cone from each vertex, which are
  fixed inside the cell). :meth:`GrowthComponent.update_local` is the windowed update
  of Section 3.2 (Prop. 13).
"""

from __future__ import annotations

import itertools

import numpy as np

from ._box_lp import max_min_affine
from .partitioned import PartitionedSimulatingMap

__all__ = ["SignMatrixComponent", "GrowthComponent", "sign_patterns"]


def sign_patterns(g_lo, g_hi, which: str = "all"):
    """Slope vectors a (L, n) and sign patterns s (L, n) of the monotone transforms.

    ``which="all"``: every combination (2^n, less for coordinates with g_lo = g_hi = 0,
    which get one choice a_j = 0, s_j = 0). ``which="two"``: only the all-lower and
    all-upper choices (approach 1).
    """
    g_lo, g_hi = np.asarray(g_lo, float), np.asarray(g_hi, float)
    choices = []
    for lo, hi in zip(g_lo, g_hi):
        if lo == 0.0 and hi == 0.0:
            choices.append([(0.0, 0)])
        else:
            choices.append([(min(lo, 0.0), 1), (max(hi, 0.0), -1)])
    if which == "two":
        combos = [tuple(c[0] for c in choices), tuple(c[-1] for c in choices)]
        combos = list(dict.fromkeys(combos))
    elif which == "all":
        combos = list(itertools.product(*choices))
    else:
        raise ValueError("which must be 'all' or 'two'")
    a = np.array([[c[0] for c in combo] for combo in combos], float)
    s = np.array([[c[1] for c in combo] for combo in combos], np.int8)
    return a, s


def _grid_corners(partition):
    lo, hi = partition.grid_bounds()
    return lo, hi


def _check_bounds(g_lo, g_hi):
    g_lo = np.asarray(g_lo, float).ravel()
    g_hi = np.asarray(g_hi, float).ravel()
    if g_lo.shape != g_hi.shape or np.any(g_lo > g_hi):
        raise ValueError("derivative bounds must satisfy g_lo <= g_hi")
    if not (np.all(np.isfinite(g_lo)) and np.all(np.isfinite(g_hi))):
        raise ValueError("derivative bounds must be finite")
    return g_lo, g_hi


class SignMatrixComponent:
    """Approaches 2 ("box") and 3 ("lp"); approach 1 with ``which="two"``."""

    def __init__(self, partition, g_lo, g_hi, d_lo, d_hi, final="box", which="all",
                 backend="auto", compact=False):
        self.partition = partition
        self.g_lo, self.g_hi = _check_bounds(g_lo, g_hi)
        self.d_lo, self.d_hi = float(d_lo), float(d_hi)
        self.final = final
        self.slopes, self.signs = sign_patterns(self.g_lo, self.g_hi, which)
        self.backend, self.compact = backend, compact
        self.map = None
        self.lower = self.upper = None

    @property
    def n_patterns(self) -> int:
        return self.slopes.shape[0]

    def update(self, Z, y):
        Z = np.atleast_2d(np.asarray(Z, float))
        y = np.asarray(y, float).ravel()
        Y = y[:, None] - Z @ self.slopes.T
        L = self.n_patterns
        new = PartitionedSimulatingMap.from_data(
            Z, Y, self.partition, np.full(L, self.d_lo), np.full(L, self.d_hi),
            signs=self.signs, backend=self.backend, compact=self.compact)
        self.map = new if self.map is None else self.map.intersect_(new)
        self._finalize()

    def _finalize(self):
        clo, chi = _grid_corners(self.partition)  # (*shape, n)
        a = self.slopes  # (L, n)
        # interval of a . z over each cell, per pattern: (*shape, L)
        az_lo = np.einsum("...j,lj->...l", clo, np.maximum(a, 0)) + np.einsum(
            "...j,lj->...l", chi, np.minimum(a, 0))
        az_hi = np.einsum("...j,lj->...l", chi, np.maximum(a, 0)) + np.einsum(
            "...j,lj->...l", clo, np.minimum(a, 0))
        s_lo = self.map.lower.astype(float)
        s_hi = self.map.upper.astype(float)
        with np.errstate(invalid="ignore"):
            upper = np.min(az_hi + s_hi, axis=-1)
            lower = np.max(az_lo + s_lo, axis=-1)
        if self.final == "lp":
            shape, n = self.partition.shape, self.partition.ndim
            lo2, hi2 = clo.reshape(-1, n), chi.reshape(-1, n)
            upper = max_min_affine(s_hi.reshape(-1, self.n_patterns), a, lo2, hi2,
                                   upper.ravel()).reshape(shape)
            lower = -max_min_affine(-s_lo.reshape(-1, self.n_patterns), -a, lo2, hi2,
                                    -lower.ravel()).reshape(shape)
        self.lower, self.upper = lower, upper


class GrowthComponent:
    """Approaches 4 ("box") and 5 ("lp"): growth cones evaluated at the grid vertices.

    ``local`` (optional) intersects the result with a map built directly from the cones
    of some data points, evaluated on each cell (max of the cone over the cell's
    vertices): ``"cell"`` uses the points inside the cell, ``"slab"`` the points whose
    coordinate lies in the cell's range on at least one axis. With ``"slab"`` the
    bounds are at least as tight as min over all data of the max of the cone over the
    cell's vertices (eqs. 3.10-3.11): a point outside the slabs is dominated by the
    vertex values, because a vertex of the cell lies between it and the cell on every
    axis. Cost per point: O(n) for "cell", O(sum_j |Q| / K_j) for "slab".
    """

    def __init__(self, partition, g_lo, g_hi, d_lo, d_hi, final="box", local=None):
        self.partition = partition
        self.g_lo, self.g_hi = _check_bounds(g_lo, g_hi)
        self.d_lo, self.d_hi = float(d_lo), float(d_hi)
        self.theta = self.d_hi - self.d_lo
        self.final = final
        vshape = tuple(e.size for e in partition.edges)
        self.v_upper = np.full(vshape, np.inf)  # tightest bounds of f_i + D at vertices
        self.v_lower = np.full(vshape, -np.inf)
        self.lower = np.full(partition.shape, -np.inf)
        self.upper = np.full(partition.shape, np.inf)
        if local not in (None, "cell", "slab"):
            raise ValueError("local must be None, 'cell' or 'slab'")
        self.local = local
        if local is not None:
            self.loc_upper = np.full(partition.shape, np.inf)
            self.loc_lower = np.full(partition.shape, -np.inf)
        n = partition.ndim
        self._corners = np.array(list(itertools.product((0, 1), repeat=n)), np.intp)

    # -------------------------------------------------------------- cone helpers
    def _phi(self, dz):
        hi = np.maximum(self.g_hi * dz, self.g_lo * dz).sum(axis=-1)
        lo = np.minimum(self.g_hi * dz, self.g_lo * dz).sum(axis=-1)
        return lo, hi

    def _cells(self, Z):
        idx = np.empty(Z.shape, np.intp)
        for j, e in enumerate(self.partition.edges):
            idx[:, j] = np.clip(self.partition.search(j, Z[:, j], "right") - 1, 0, e.size - 2)
        return idx

    def _axis_terms(self, z, j, first, last):
        """Max / min over the two edges of each cell first..last on axis j of the cone
        terms phi_hi_j(e - z), phi_lo_j(e - z), for points z (N,). Returns (N, cells)."""
        e = self.partition.edges[j][first : last + 2]
        t = e[None, :] - z[:, None]
        gl, gh = self.g_lo[j], self.g_hi[j]
        ph = np.maximum(gh * t, gl * t)
        pl = np.minimum(gh * t, gl * t)
        return np.maximum(ph[:, :-1], ph[:, 1:]), np.minimum(pl[:, :-1], pl[:, 1:])

    def _local_update(self, Z, y, block=None):
        """Min/max-scatter the per-cell cone bounds of the points into loc_upper/lower."""
        P = self.partition
        n = P.ndim
        K = np.array(P.shape)
        first, last = (np.zeros(n, np.intp), K - 1) if block is None else block
        cshape = tuple(int(b - a + 1) for a, b in zip(first, last))
        sl = tuple(slice(first[j], last[j] + 1) for j in range(n))
        lu, ll = self.loc_upper[sl], self.loc_lower[sl]  # views
        cells = self._cells(Z) - first  # cell index relative to the block
        inside = np.all((cells >= 0) & (cells < np.array(cshape)), axis=1)
        Z, y, cells = Z[inside], y[inside], cells[inside]
        if Z.shape[0] == 0:
            return
        terms = [self._axis_terms(Z[:, j], j, first[j], last[j]) for j in range(n)]
        base_u = y + self.theta
        base_l = y - self.theta
        if self.local == "cell":
            tu = base_u + sum(terms[j][0][np.arange(len(y)), cells[:, j]] for j in range(n))
            tl = base_l + sum(terms[j][1][np.arange(len(y)), cells[:, j]] for j in range(n))
            flat = np.ravel_multi_index(tuple(cells.T), cshape)
            fu, fl = lu.reshape(-1).copy(), ll.reshape(-1).copy()
            np.minimum.at(fu, flat, tu)
            np.maximum.at(fl, flat, tl)
            lu[...] = fu.reshape(cshape)
            ll[...] = fl.reshape(cshape)
            return
        # slabs: for axis j, all cells of the block with index_j = cell_j of the point
        other = int(np.prod(cshape))
        chunk = max(1, 5_000_000 // max(1, other))
        for j in range(n):
            for s0 in range(0, len(y), chunk):
                r = slice(s0, s0 + chunk)
                N = len(y[r])
                ix = np.arange(N)
                vu = (base_u[r] + terms[j][0][r][ix, cells[r, j]])
                vl = (base_l[r] + terms[j][1][r][ix, cells[r, j]])
                shp_u, shp_l = vu.reshape(N, *([1] * n)), vl.reshape(N, *([1] * n))
                for i in range(n):
                    if i == j:
                        continue
                    sh = [N] + [1] * n
                    sh[1 + i] = cshape[i]
                    shp_u = shp_u + terms[i][0][r].reshape(sh)
                    shp_l = shp_l + terms[i][1][r].reshape(sh)
                # shp_*: (N, ..., 1 on axis j, ...); reduce per slab index along axis j
                for c in np.unique(cells[r, j]):
                    m = cells[r, j] == c
                    idx = [slice(None)] * n
                    idx[j] = slice(int(c), int(c) + 1)
                    idx = tuple(idx)
                    np.minimum(lu[idx], shp_u[m].min(axis=0), out=lu[idx])
                    np.maximum(ll[idx], shp_l[m].max(axis=0), out=ll[idx])

    # -------------------------------------------------------------- full fit
    def vertex_values(self, Z, y):
        """Exact bounds at all vertices from the data (Z, y) alone (eqs. 3.20-3.21)."""
        Z = np.atleast_2d(np.asarray(Z, float))
        y = np.asarray(y, float).ravel()
        P = self.partition
        vshape = self.v_upper.shape
        up = np.full(int(np.prod(vshape)), np.inf)
        lo = np.full(up.size, -np.inf)
        cell = self._cells(Z)
        for b in self._corners:
            vi = cell + b
            v = np.stack([e[vi[:, j]] for j, e in enumerate(P.edges)], axis=1)
            plo, phi = self._phi(v - Z)
            flat = np.ravel_multi_index(tuple(vi.T), vshape)
            np.minimum.at(up, flat, y + self.theta + phi)
            np.maximum.at(lo, flat, y - self.theta + plo)
        up, lo = up.reshape(vshape), lo.reshape(vshape)
        for j, e in enumerate(P.edges):
            sh = [1] * len(vshape)
            sh[j] = e.size
            e = e.reshape(sh)
            gl, gh = self.g_lo[j], self.g_hi[j]
            # upper: forward (moving up costs g_hi per unit), backward (-g_lo per unit)
            w = up - gh * e
            np.minimum.accumulate(w, axis=j, out=w)
            np.minimum(up, w + gh * e, out=up)
            w = np.flip(up - gl * e, j)
            np.minimum.accumulate(w, axis=j, out=w)
            np.minimum(up, np.flip(w, j) + gl * e, out=up)
            # lower: forward (+g_lo per unit), backward (-g_hi per unit)
            w = lo - gl * e
            np.maximum.accumulate(w, axis=j, out=w)
            np.maximum(lo, w + gl * e, out=lo)
            w = np.flip(lo - gh * e, j)
            np.maximum.accumulate(w, axis=j, out=w)
            np.maximum(lo, np.flip(w, j) + gh * e, out=lo)
        return lo, up

    def update(self, Z, y):
        """Add data. Exact: equals a refit on all data seen so far."""
        lo, up = self.vertex_values(Z, y)
        np.minimum(self.v_upper, up, out=self.v_upper)
        np.maximum(self.v_lower, lo, out=self.v_lower)
        if self.local is not None:
            self._local_update(np.atleast_2d(np.asarray(Z, float)),
                               np.asarray(y, float).ravel())
        self._finalize()

    # -------------------------------------------------------------- final step
    def _finalize(self, block=None):
        """Cell bounds from vertex values, on all cells or on a block of cells.

        ``block`` = (first, last) cell index arrays (inclusive)."""
        P = self.partition
        n = P.ndim
        if block is None:
            first, last = np.zeros(n, np.intp), np.array(P.shape) - 1
        else:
            first, last = block
        cshape = tuple(int(b - a + 1) for a, b in zip(first, last))
        h = []
        clo = []
        for j, e in enumerate(P.edges):
            sh = [1] * n
            sh[j] = cshape[j]
            seg = e[first[j] : last[j] + 2]
            h.append(np.diff(seg).reshape(sh))
            clo.append(seg[:-1].reshape(sh))
        gl, gh = self.g_lo, self.g_hi
        upper = np.full(cshape, np.inf)
        lower = np.full(cshape, -np.inf)
        offs_u, offs_l = [], []
        for b in self._corners:
            sl = tuple(slice(first[j] + b[j], last[j] + b[j] + 1) for j in range(n))
            vu, vl = self.v_upper[sl], self.v_lower[sl]
            ru = sum(np.maximum(0.0, (gh[j] if b[j] == 0 else -gl[j]) * h[j]) for j in range(n))
            rl = sum(np.minimum(0.0, (gl[j] if b[j] == 0 else -gh[j]) * h[j]) for j in range(n))
            with np.errstate(invalid="ignore"):
                np.minimum(upper, vu + ru, out=upper)
                np.maximum(lower, vl + rl, out=lower)
            if self.final == "lp":
                # cone from vertex v, inside the cell: slope g_hi_j (upper) / g_lo_j (lower)
                # where v_j is the lower edge, the other bound where it is the upper edge
                su = np.where(b == 0, gh, gl)
                sl_ = np.where(b == 0, gl, gh)
                vcoord = [clo[j] + b[j] * h[j] for j in range(n)]
                offs_u.append(vu - sum(su[j] * vcoord[j] for j in range(n)))
                offs_l.append(vl - sum(sl_[j] * vcoord[j] for j in range(n)))
        if self.final == "lp":
            lo_c = np.stack(np.broadcast_arrays(*clo), axis=-1).reshape(-1, n)
            hi_c = lo_c + np.stack(np.broadcast_arrays(*h), axis=-1).reshape(-1, n)
            su = np.where(self._corners == 0, gh, gl)
            sl_ = np.where(self._corners == 0, gl, gh)
            ou = np.stack([o.ravel() for o in offs_u], axis=1)
            ol = np.stack([o.ravel() for o in offs_l], axis=1)
            upper = max_min_affine(ou, su, lo_c, hi_c, upper.ravel()).reshape(cshape)
            lower = -max_min_affine(-ol, -sl_, lo_c, hi_c, -lower.ravel()).reshape(cshape)
        sl = tuple(slice(first[j], last[j] + 1) for j in range(n))
        if self.local is not None:
            upper = np.minimum(upper, self.loc_upper[sl])
            lower = np.maximum(lower, self.loc_lower[sl])
        self.upper[sl] = upper
        self.lower[sl] = lower

    # -------------------------------------------------------------- local update
    def update_local(self, Z, y, window):
        """Windowed update (Section 3.2, Prop. 13): each new point tightens the vertex
        values of the cells within ``window`` cells of its own cell (per axis), and only
        those cells are recomputed. Sound, cheaper than :meth:`update`, and never
        tighter than it."""
        Z = np.atleast_2d(np.asarray(Z, float))
        y = np.asarray(y, float).ravel()
        P = self.partition
        n = P.ndim
        w = np.broadcast_to(np.asarray(window, np.intp), (n,))
        K = np.array(P.shape)
        cells = self._cells(Z)
        changed = []
        for z, yk, c in zip(Z, y, cells):
            first = np.maximum(c - w, 0)
            last = np.minimum(c + w, K - 1)
            changed.append((first, last))
            axes = [P.edges[j][first[j] : last[j] + 2] for j in range(n)]
            V = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            plo, phi = self._phi(V - z)
            sl = tuple(slice(first[j], last[j] + 2) for j in range(n))
            np.minimum(self.v_upper[sl], yk + self.theta + phi, out=self.v_upper[sl])
            np.maximum(self.v_lower[sl], yk - self.theta + plo, out=self.v_lower[sl])
            if self.local is not None:
                self._local_update(z[None], np.array([yk]), (first, last))
            self._finalize((first, last))
        return changed
