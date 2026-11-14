"""Write reference_invariance.mat: gridless-envelope problems solved by Python, for MATLAB.

Run from the repository root:  python matlab/tests/make_reference_invariance.py
(kept separate from reference.mat so the two can be regenerated independently)
"""
from pathlib import Path

import numpy as np
from scipy.io import savemat

from ddma.invariance import embedding_envelope, max_min_cones, maximal_points, monotone_envelope


def rooms(T, u):
    a, b, c, Ts = 0.05, 0.3, 0.6, 0.25
    cpl = np.stack([T[:, 1] - T[:, 0], T[:, 0] + T[:, 2] - 2 * T[:, 1], T[:, 1] - T[:, 2]], 1)
    return T + Ts * (a * (34.0 - T) + b * cpl - c * u * (T - 14.0))


def competition(x, u):
    return np.stack([x[:, 0] * (2 - x[:, 0] - 0.2 * x[:, 1]) + u[:, 0],
                     x[:, 1] * (2 - x[:, 1] - 0.2 * x[:, 0]) + u[:, 1]], 1)


def main():
    rng = np.random.default_rng(2026)
    out = {}
    # maximal points
    P = rng.integers(0, 8, (400, 4)).astype(float)
    out["mp_P"] = P
    out["mp_M"] = maximal_points(P)
    # max-min of two cones
    p, q = rng.normal(size=(2, 500))
    al = rng.exponential(size=(500, 6)) * (rng.random((500, 6)) > 0.2)
    be = rng.exponential(size=(500, 6)) * (rng.random((500, 6)) > 0.2)
    out.update(mm_p=p[:, None], mm_q=q[:, None], mm_alpha=al, mm_beta=be,
               mm_val=max_min_cones(p, q, al, be)[:, None])
    # monotone envelope: rooms (input flipped) and a 2-D autonomous system with flipped state
    N = 400
    X = rng.uniform(16, 32, (N, 3))
    U = rng.uniform(0, 1, (N, 1))
    Xn = rooms(X, U) + rng.uniform(0, 0.15, (N, 3))
    E = monotone_envelope(X, Xn, U, w_lo=0, w_hi=0.15, input_signs=[-1], x_hi=[26] * 3,
                          u_lo=[0], u_hi=[1])
    Q = rng.uniform(16, 26, (2000, 3))
    out.update(r_X=X, r_U=U, r_Xn=Xn, r_corners=E.corners, r_pT=E.pairs_T, r_pU=E.pairs_U,
               r_Q=Q, r_in=E.contains(Q)[:, None].astype(float),
               r_u=np.array([E.safe_input(x, [0.0])[0] if E.contains(x)[0] else np.nan
                             for x in Q[:200]])[:, None])
    X2 = rng.uniform(-1, 1, (300, 2))
    Xn2 = np.stack([0.5 * X2[:, 0] - 0.1 * X2[:, 1], -0.2 * X2[:, 0] + 0.6 * X2[:, 1]], 1)
    E2 = monotone_envelope(X2, Xn2, None, w_lo=[-0.02, -0.02], w_hi=[0.02, 0.02],
                           state_signs=[1, -1], x_hi=[0.8, np.inf], x_lo=[-np.inf, -0.7])
    out.update(a_X=X2, a_Xn=Xn2, a_corners=E2.corners)
    # embedding envelope: competition with inputs and disturbance
    LO, HI = 0.6, 1.1
    N = 150
    Xc = rng.uniform(LO, HI, (N, 2))
    Uc = rng.uniform(-0.02, 0.02, (N, 2))
    Xcn = competition(Xc, Uc) + rng.uniform(-0.005, 0.005, (N, 2))
    jl = np.array([[2 - 2.2 * HI, -0.2 * HI], [-0.2 * HI, 2 - 2.2 * HI]])
    jh = np.array([[2 - 2.2 * LO, -0.2 * LO], [-0.2 * LO, 2 - 2.2 * LO]])
    B = embedding_envelope(Xc, Xcn, Uc, jl, jh, np.eye(2), np.eye(2), w_lo=-0.005, w_hi=0.005,
                           x_lo=[LO, LO], x_hi=[HI, HI])
    Qc = rng.uniform(LO, HI, (2000, 2))
    out.update(c_X=Xc, c_U=Uc, c_Xn=Xcn, c_jl=jl, c_jh=jh, c_lo=B.lo, c_hi=B.hi, c_ulo=B.ulo,
               c_uhi=B.uhi, c_pairs=(B.pairs + 1).astype(float), c_Q=Qc,
               c_in=B.contains(Qc)[:, None].astype(float),
               c_u=np.array([B.safe_input(x, [-0.02, -0.02]) if B.contains(x)[0]
                             else [np.nan, np.nan] for x in Qc[:200]]))
    path = Path(__file__).with_name("reference_invariance.mat")
    savemat(path, out, do_compression=True)
    print(f"wrote {path}: {len(E.corners)} room corners, {len(E2.corners)} 2-D corners, "
          f"{B.n_boxes} competition boxes")


if __name__ == "__main__":
    main()
