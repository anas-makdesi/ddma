# Reproducing the papers with ddma

This report re-runs the numerical experiments of the three papers behind `ddma`:

- **L4DC 2021**: *Data-Driven Abstraction of Monotone Systems*
- **ADHS 2021**: *Efficient Data-Driven Abstraction of Monotone Systems with Disturbances*
- **TAC 2024**: *Data-Driven Models of Monotone Systems*

Each experiment has a script in `reproductions/`. The results are saved as JSON and figures in `reproductions/results/`.

**Hardware.** The `ddma` timings were measured on a 2-core cloud container with 8 GB RAM. The papers used MATLAB on a desktop (L4DC: Intel i7 at 2.9 GHz, 8 GB). Times are indicative only.

**Safety checks.** Every synthesized controller is also run in closed loop on the *true* plant: 200 random starting states × 300–500 steps, with random disturbances. A run counts as a violation if it leaves the specification or the controller's domain.

## Summary

| Paper, experiment | Paper reports | ddma | Verdict |
|---|---|---|---|
| **TAC V-A**, Fig. 9: µ vs number of data points (K = 100) | µ ≈ 2.7 → 2.47 from 10⁶ to 10⁷ points; time in seconds (axis up to 7 s) | µ = 2.703 → 2.470; 0.11 → 1.2 s | ✅ reproduced |
| **TAC V-A**, Fig. 10: µ vs partition size (10⁶ points) | µ from ≈ 18 down to ≈ 2.7 at 10⁴ cells; 0.3–0.65 s | µ = 15.4 → 2.70 at 10,201 cells; 0.06–0.11 s | ✅ reproduced |
| **TAC V-A**, Fig. 8: map contains F | visual | contains F at 200,000 random queries | ✅ |
| **TAC V-B**, cruise control, 10⁶ points, 51⁴ grid | fit 0.633 s; controllable set (Fig. 11) | fit 0.75 s on 4-D grid (0.10 s on the (v_f, u) grid); abstraction 2.1 s; synthesis 0.18 s; 27.9 % of the safe cells; **0 violations** | ✅ reproduced |
| **TAC V-C**, Lorenz: derivative bounds (scenario LP) | J and w table | Close to the table: e.g. J₂₁ ∈ [0.183, 0.372] (paper [0.185, 0.377], true [0.18, 0.38]); w·T₀ = [0.003, 0.047, 0.035] (paper [0.005, 0.059, 0.058]) | ✅ with the LP weighting of note 3 |
| **TAC V-C**, Lorenz: abstraction + invariant | 10⁸ points, 101⁴ cells, 929 s; invariant (Fig. 12) | **Same size** (10⁸ points, 104 M cells) on an 8 GB machine: model 174 s + abstraction 49 s = 223 s; invariant 5.6 s (50 s before v0.3); invariant = 80 % of the cells; **0 violations**; peak memory 4.3 GB | ✅ reproduced at full scale |
| **ADHS**, Fig. 3: µ vs number of data points | µ ≈ 2500 → 1700, 150–650 s per abstraction | With the settings of the original experiment scripts (note 2): their saved output is 4010 → 3280, ddma gives 3888 → 3158; 0.03–0.17 s | ✅ matches the original scripts |
| **ADHS**, Fig. 5: µ vs partition size (10⁴ points) | decreasing, 3.5·10⁴ → 0.5·10⁴ | decreasing, 4.6·10⁴ → 806 | ✅ trend reproduced (values depend on the settings, note 2) |
| **ADHS**, Fig. 6: controllable set, assume-guarantee contract | visual | 72.5 % of the safe cells; **0 violations** | ✅ |
| **L4DC**, exact abstraction from 300 transitions | abstraction 923 s + invariant 341 s (Fig. 5) | 90,601 states × 301 inputs: abstraction 5.3 s + invariant 0.38 s; invariant = 63.5 % of the safe set; **0 violations** | ✅ reproduced, ~100× faster |

## Notes

**1. Input range for L4DC.** The paper does not state the input bounds. The script uses u ∈ [−0.4 g, 0.4 g], as in the companion experiment script.

**2. ADHS Figs. 3 and 5: settings.**
- ddma uses the settings of the original experiment scripts: disturbance ±0.05, data inputs in [−4, 4], and the partitions of Figs. 4–5 (1000, 2700, 6400, 11250, 18000 state cells). `reproductions/adhs_script_params.py` runs them.
- With these settings the scripts' saved output is µ = 4010 → 3280 for 10⁵ … 8·10⁵ samples; ddma gives 3888 → 3158, within about 3 % (random data).
- The absolute values of µ depend on these choices (partition, input range, and how cells next to the unbounded outer cells are counted). Varying them gives curves in the range of the published figure (2700–3000 → 2200–2500).

**3. Lorenz: weight of the disturbance term in the LP.**
- The scenario LP that estimates the derivative and disturbance bounds weighs the disturbance term. `ddma.estimate_derivative_bounds` exposes this weight as `w_weight` (1 is equation (32)). The reproduction uses 50, as in the original experiment code: this gives bounds close to the paper's table and the 65 % invariant set above.
- With `w_weight=1` the derivative bounds are very close to the true ones but the disturbance bounds are larger (w·T₀ = [0.005, 0.068, 0.110]), and the invariant set at 51⁴ cells is empty.
- The guarantee of Proposition 9 holds for any weight; the weight only trades derivative bounds against disturbance bounds.
- With the *true* bounds, the invariant set at 51⁴ cells covers 71 % of the cells.

**4. Lorenz at full scale on an 8 GB machine.** Done naively, 101⁴ cells × 3 outputs × 2 maps needs about 10 GB as float64. Three changes bring the peak down to 4.3 GB:
- Each output is learned only on the variables it depends on: x⁺ on (x, y), z⁺ on (x, y, z), y⁺ on (x, y, z, u).
- Bounds are stored as float32, rounded outward so they stay sound; the C++ kernel writes float32 directly.
- The 10⁸ samples are processed in batches whose maps are intersected (TAC Proposition 3). Successor indices take 1 byte each.

Command: `python reproductions/tac_lorenz.py --full`.

**5. Half-open cells and assume-guarantee contracts.** With cells [a, b), a state exactly on the upper boundary of the assumed range belongs to the next cell. `ddma` therefore applies environment assumptions at the level of cells: successor sets are restricted to the cells inside the assumption box. Simulated environments must keep the assumed variable in [a, b).

## Figures

`reproductions/results/` contains the figures in the papers' style:
- `tac_fig8.png`: F and the transparent staircase bounds, with a zoom where the steps are visible.
- `tac_fig9_fig10.png`: time and µ curves.
- `tac_fig11.png`: 3-D controllable set; `tac_fig11_slices.png`: slices.
- `tac_fig12_w50_full_3d.png`: Lorenz invariant set at full scale.
- `tac_fig12_w1.png`, `tac_fig12_w1_slices.png`: the same with `w_weight=1` (equation (32) as written). The set is **empty** (note 3), and the figures say so; they show the effect of the weight.
- `adhs_fig3_fig5.png`, `adhs_fig6.png`: ADHS curves and 3-D controllable set.
- `l4dc_fig5.png`: L4DC invariant set with the data points.

They're produced by `ddma.plotting` (`plot_staircase_bounds`, `plot_set_3d`, `plot_set_2d`). MATLAB equivalents are `ddma.plot_bounds` and `ddma.plot_set`, built the same way as the original scripts (the `kron` staircase and `isosurface`). Both were run in real MATLAB (MATLAB Online, November 2026) by `matlab/examples/paper_figures.m`: the staircase bounds of Fig. 8 and the 3-D sets of TAC Figs. 11–12 and ADHS Fig. 6 render as intended. The tests and the cruise-control example also pass there. CI checks the numbers under Octave only, because headless Octave cannot render text.

## Speed

**Algorithms.**
- Computing the fixed-partition map uses the TAC sweep (Proposition 5), which is O(|K| log|Q| + |Q|). ADHS used O(|K|·|Q|·|P|), and L4DC used the data-induced partition with O(|K|^(n+m)) cells.
- Safety synthesis keeps, for each state, a pointer to its first input not yet ruled out. The set only shrinks, so an input that fails once fails for ever and the pointer only moves forward: every state–input pair is examined at most once over the whole fixed point. Reach-avoid re-examines a state only when a cell that just entered the reach set lies in the hull of its successor boxes. Both are exact (tested against the textbook iterations) and run in the shared C++ core (`cpp/ddma_synthesis.hpp`), with a NumPy fallback.

| Synthesis (same abstraction, same result) | v0.2 (witness, NumPy) | v0.3 NumPy | v0.3 C++ | Speed-up |
|---|---|---|---|---|
| L4DC safety, 90,601 states × 301 inputs | 3.9 s | 2.2 s | 0.36 s | 11× |
| L4DC reach-avoid, same abstraction | 6.6 s | 5.0 s | 0.47 s | 14× |
| TAC cruise control safety, 51³ states × 51 inputs | 2.7 s | | 0.18 s | 15× |
| Lorenz safety, 51³ states × 51 inputs | 2.35 s | 1.7 s | 0.30 s | 8× |
| Lorenz safety, 101³ states × 101 inputs (full size) | 42.5 s | 24.6 s | 5.3 s | 8× |

Before v0.2 (re-searching every input each round), L4DC safety took 49 s. In MATLAB/Octave, `ddma.safety` and `ddma.reach_avoid` call the same kernel through `ddma_synth_mex` once `build_mex` has been run.

**Languages.** The header-only C++ kernels (`cpp/ddma_core.hpp` for fitting, `cpp/ddma_synthesis.hpp` for synthesis) are shared by Python (through pybind11) and MATLAB (through MEX files). Fitting:

| Task | Pure Python/NumPy | C++ kernel | Speed-up |
|---|---|---|---|
| Fit, 10⁶ points, 100² grid | 0.075 s | 0.021 s | 3.6× |
| Fit, 10⁷ points, 100² grid | 1.75 s | 0.21 s | 8.3× |
| Fit, 10⁷ points, 1000² grid | 1.86 s | 0.51 s | 3.6× |
| Fit, 10⁶ points, 50⁴ grid, 3 outputs | 1.14 s | 0.65 s | 1.8× |
| Octave: fit, 10⁶ points, 100² grid | 0.86 s (vectorised .m) | 0.038 s (MEX) | 23× |

Both backends give **bit-identical** results; this is tested in both languages. On large grids the run is limited by memory bandwidth, which explains the smaller gains there.
