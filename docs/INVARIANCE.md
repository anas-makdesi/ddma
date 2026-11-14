# Gridless safe envelopes (`ddma.invariance`)

Controlled invariant sets computed directly from transitions, with no grid and no model
identification. Both methods give deterministic guarantees: every state in the envelope can
be kept in it forever, for every disturbance within the stated bounds, with the inputs the
envelope certifies.

| | `monotone_envelope` | `embedding_envelope` |
|---|---|---|
| Systems | monotone (after optional coordinate flips), additive bounded disturbance | any, with interval bounds on the Jacobian, additive bounded disturbance |
| Specification | lower-closed (e.g. "every temperature ≤ 26 °C") | a box |
| Result | union of lower sets, stored as maximal corners with the largest allowed input of each | union of boxes spanned by pairs of data points, each with a box of allowed inputs |
| Method | Makdesi & Zamani, *Data-driven synthesis of controlled invariant sets for monotone systems*, IFAC WC 2026 | Makdesi, Zamani & Jafarpour, *Data-driven controlled invariance via monotone embeddings*, IFAC WC 2026 |

Disturbances are given in state units: the change they can cause in `x+` in one step.

## Implementation notes

- **Monotone search.** One data point is chosen per coordinate (a point whose worst-case
  successor did not increase in that coordinate); the corner is their componentwise
  minimum. The search goes coordinate by coordinate and keeps only partial choices that
  are not dominated (larger corner, larger input limit, smaller thresholds); a final
  sort-filter skyline (`maximal_points`) keeps the maximal (corner, input) pairs. It
  returns exactly the valid corners of the paper's enumeration (tested against brute force).
- **Pair bound in closed form.** The LP "max over the box of the minimum of two cones" is
  solved exactly by duality: a convex piecewise-linear function of one variable, minimised
  after one sort of its breakpoints (`max_min_cones`; tested against HiGHS). All pairs are
  processed in vectorized blocks, after a cheap necessary test on the two end points.

## Examples

- `examples/envelope_room_cooling.py`: three rooms, one cooling input, unknown heat gains in
  [0, 0.15] °C per step, limit 26 °C. From 1,000 transitions the envelope certifies about
  98 % of the safe temperature box (0.1–1 s); the grid abstraction of
  `examples/lazy_synthesis.py`, from 200,000 transitions, certifies about 74 %. Closed loop
  with the least cooling the envelope allows: 0 violations.
- `examples/envelope_nonlinear.py`: (1) two competing species,
  `x_i+ = x_i (2 − x_i − 0.2 x_j) + u_i + w_i`, on [0.6, 1.1]², inputs in [−0.02, 0.02]²,
  disturbance in [−0.005, 0.005]², Jacobian bounds computed on that domain: 400 transitions
  certify about 84 % of the domain in under 0.1 s; (2) the ring of ten coupled units of the
  IFAC paper, `x_i+ = 0.5 x_i + 0.1 sin(x_{i−1}) + 0.1 cos(x_{i+1})` on [−2, 2]¹⁰, from
  1,000 transitions (0.2 s).
- MATLAB: `matlab/examples/envelope_room_cooling.m`, `matlab/examples/envelope_nonlinear.m`.

## Python / MATLAB

| Python (`ddma.invariance`) | MATLAB |
|---|---|
| `monotone_envelope(X, Xn, U, w_lo=, w_hi=, x_lo=, x_hi=, state_signs=, input_signs=, u_lo=, u_hi=)` | `ddma.monotone_envelope(X, Xn, U, opts)` (same option names in a struct) |
| `embedding_envelope(X, Xn, U, jx_lo, jx_hi, ju_lo, ju_hi, w_lo=, w_hi=, x_lo=, x_hi=, pairs=)` | `ddma.embedding_envelope(X, Xn, U, jx_lo, jx_hi, ju_lo, ju_hi, opts)` (pairs 1-based) |
| `env.contains(x)` | `ddma.envelope_contains(E, x)` |
| `env.safe_input(x, u_desired)` | `ddma.envelope_input(E, x, u_desired)` |
| `maximal_points(P)`, `max_min_cones(p, q, alpha, beta)` | `ddma.maximal_points`, `ddma.max_min_cones` |

Tested against the Python results in `matlab/tests/run_invariance_tests.m`
(reference: `matlab/tests/make_reference_invariance.py`).
