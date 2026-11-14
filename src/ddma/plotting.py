"""Figures in the style of the papers (requires matplotlib; 3-D sets use scikit-image if
installed, voxels otherwise).

* :func:`plot_staircase_bounds` - transparent piecewise-constant upper / lower bounds of a
  simulating map over a 2-D input space, optionally with the true function (TAC Fig. 8).
* :func:`plot_set_3d` - a set of cells of a 3-D partition (controllable set, invariant set)
  as a shaded surface (TAC Figs. 11, 12; ADHS Fig. 6).
* :func:`plot_set_2d` - the same for 2-D partitions (L4DC Fig. 5).
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_staircase_bounds", "plot_set_3d", "plot_set_2d"]


def _staircase_grid(edges, values):
    """Coordinates and values of a piecewise-constant surface (each cell a flat top).

    Every inner edge is duplicated (left / right limit), like the kron(., ones(2)) trick
    of the original MATLAB scripts.
    """
    ex, ey = edges
    eps_x = 1e-9 * (ex[-1] - ex[0])
    eps_y = 1e-9 * (ey[-1] - ey[0])
    xs = np.sort(np.r_[ex[:-1], ex[1:] - eps_x])
    ys = np.sort(np.r_[ey[:-1], ey[1:] - eps_y])
    Z = np.kron(values, np.ones((2, 2)))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return X, Y, Z


def plot_staircase_bounds(sigma, output=0, ax=None, true_fn=None, trim=1, clip=None,
                          alpha=0.3, cmap="viridis", bound_color="0.25",
                          bound_cmap="plasma", region=None):
    """Transparent staircase upper/lower bounds of a 2-D-input simulating map.

    sigma : PartitionedSimulatingMap with two inputs. ``trim`` outer cells are dropped on
    every side (their bounds are typically infinite). ``true_fn(Z) -> values`` draws the
    true function as a solid surface. ``clip=(lo, hi)`` clips infinite values for display.
    The bounds are transparent, coloured by height with ``bound_cmap`` (or plain
    ``bound_color`` if ``bound_cmap=None``). ``region=(i0, i1, j0, j1)`` zooms on the
    cells i0 <= i < i1, j0 <= j < j1 (after trimming), where the steps are visible.
    """
    import matplotlib.pyplot as plt

    if sigma.partition.ndim != 2:
        raise ValueError("plot_staircase_bounds needs a map with two inputs")
    if ax is None:
        ax = plt.figure(figsize=(6.5, 4.8)).add_subplot(projection="3d")
    sl = slice(trim, -trim if trim else None)
    edges = [e[trim: len(e) - trim] for e in sigma.partition.edges]
    lo = sigma.lower[sl, sl, output].astype(float)
    hi = sigma.upper[sl, sl, output].astype(float)
    if region is not None:
        i0, i1, j0, j1 = region
        lo, hi = lo[i0:i1, j0:j1], hi[i0:i1, j0:j1]
        edges = [edges[0][i0:i1 + 1], edges[1][j0:j1 + 1]]
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    if true_fn is not None:
        gx = np.linspace(edges[0][0], edges[0][-1], 80)
        gy = np.linspace(edges[1][0], edges[1][-1], 80)
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        GZ = np.asarray(true_fn(np.stack([GX.ravel(), GY.ravel()], 1))).reshape(GX.shape)
        ax.plot_surface(GX, GY, GZ, cmap=cmap, linewidth=0, antialiased=True, alpha=1.0,
                        rstride=1, cstride=1)
    cm = plt.get_cmap(bound_cmap)
    zmin = np.nanmin(lo[np.isfinite(lo)]) if np.isfinite(lo).any() else 0.0
    zmax = np.nanmax(hi[np.isfinite(hi)]) if np.isfinite(hi).any() else 1.0
    for vals in (lo, hi):
        X, Y, Z = _staircase_grid(edges, vals)
        if bound_cmap is None:
            ax.plot_surface(X, Y, Z, color=bound_color, alpha=alpha, linewidth=0.2,
                            edgecolor=(0, 0, 0, 0.3), shade=False)
        else:
            fc = cm((Z - zmin) / max(zmax - zmin, 1e-12))
            fc[..., 3] = alpha
            ax.plot_surface(X, Y, Z, facecolors=fc, linewidth=0.2,
                            edgecolor=(0, 0, 0, 0.25), shade=False)
    return ax


def plot_set_3d(partition, mask, ax=None, color="#2a9d8f", alpha=0.95, step=1,
                axis_labels=None):
    """A set of cells of a 3-D partition as a shaded surface.

    ``mask`` is a boolean array over the cells (flat or of shape partition.shape). Uses the
    marching-cubes surface of the cell indicator (scikit-image) mapped to the cell edges; if
    scikit-image is missing, falls back to voxels.
    """
    import matplotlib.pyplot as plt

    if partition.ndim != 3:
        raise ValueError("plot_set_3d needs a 3-D partition")
    M = np.asarray(mask, bool).reshape(partition.shape)[::step, ::step, ::step]
    edges = [e[::step] if step == 1 else np.r_[e[:-1:step], e[-1]] for e in partition.edges]
    if ax is None:
        ax = plt.figure(figsize=(6.5, 5)).add_subplot(projection="3d")
    centers = [0.5 * (e[1:] + e[:-1]) for e in edges]
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes

        vol = np.pad(M.astype(float), 1)
        if vol.max() == 0:
            _mark_empty(ax, edges, three_d=True)
            return ax
        verts, faces, _, _ = marching_cubes(vol, 0.5)
        # map padded index coordinates to physical coordinates (cell centers)
        phys = np.empty_like(verts)
        for j in range(3):
            c = centers[j]
            idx = np.arange(-1, len(c) + 1, dtype=float)
            cc = np.r_[2 * c[0] - c[1], c, 2 * c[-1] - c[-2]] if len(c) > 1 else np.r_[c, c, c]
            phys[:, j] = np.interp(verts[:, j] - 1, idx, cc)
        light = np.array([0.3, -0.5, 0.8])
        tri = phys[faces]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        shade = 0.45 + 0.55 * np.abs(nrm @ (light / np.linalg.norm(light)))
        base = np.array(plt.matplotlib.colors.to_rgb(color))
        fc = np.clip(base[None] * shade[:, None], 0, 1)
        coll = Poly3DCollection(tri, facecolors=np.c_[fc, np.full(len(fc), alpha)],
                                edgecolor="none")
        ax.add_collection3d(coll)
    except ImportError:
        X, Y, Z = np.meshgrid(*edges, indexing="ij")
        ax.voxels(X, Y, Z, M, facecolors=color, edgecolor="none", alpha=alpha)
    for j, set_lim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        set_lim(edges[j][0], edges[j][-1])
    if axis_labels:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
    return ax


def plot_set_2d(partition, mask, ax=None, color="#d62828", points=None, alpha=0.85):
    """A set of cells of a 2-D partition, optionally with data points."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        ax = plt.figure(figsize=(6.4, 4)).add_subplot()
    M = np.asarray(mask, bool).reshape(partition.shape).astype(float)
    if not M.any():
        _mark_empty(ax, partition.edges)
    M[M == 0] = np.nan
    ax.pcolormesh(*partition.edges, M.T, cmap=ListedColormap([color]), alpha=alpha,
                  shading="flat")
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], ".", ms=3, color="#1d3557")
    ax.set_xlim(partition.edges[0][0], partition.edges[0][-1])
    ax.set_ylim(partition.edges[1][0], partition.edges[1][-1])
    return ax


def _mark_empty(ax, edges, three_d=False):
    """Say on the axes that the set is empty (instead of leaving a blank panel)."""
    lims = [(e[0], e[-1]) for e in edges]
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    kw = dict(ha="center", va="center", color="0.35", fontsize=10)
    if three_d:
        ax.set_zlim(*lims[2])
        ax.text2D(0.5, 0.5, "empty set", transform=ax.transAxes, **kw)
    else:
        ax.text(0.5, 0.5, "empty set", transform=ax.transAxes, **kw)
