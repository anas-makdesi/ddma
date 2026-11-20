"""Check that the examples and paper reproductions still print the same numbers.

Every script below is run, its output is normalized (timings and file paths are masked)
and compared line by line with the reference in ``reproductions/expected/``. A change in
any reported number (safe-set percentage, violation count, mu value, ...) is a failure.

Run:
    python reproductions/check_numbers.py            # quick set, about 2 min
    python reproductions/check_numbers.py --full     # everything, about 10 min
    python reproductions/check_numbers.py --matlab   # MATLAB examples in Octave (MEX built)
    python reproductions/check_numbers.py --only tac_acc l4dc_acc
    python reproductions/check_numbers.py --full --update   # after an intended change

After ``--update``, review ``git diff reproductions/expected`` before committing: the diff
shows exactly which reported numbers changed.

The Python scripts run from a temporary copy of ``examples/`` and ``reproductions/``, so
figures and result files in the repository are not touched. The RL demo
(``examples/rl_safety_filter.py``) is not included: its numbers depend on the PyTorch
version and it takes over an hour.
"""

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ROOT / "reproductions" / "expected"

# name: (command relative to the run folder, tier). Tier "quick" runs by default
# (CI, every push), "full" only with --full, "matlab" only with --matlab.
CHECKS = {
    "scalar_map": (["examples/scalar_map.py"], "quick"),
    "bounded_derivative_map": (["examples/bounded_derivative_map.py"], "quick"),
    "adaptive_cruise_control": (["examples/adaptive_cruise_control.py"], "quick"),
    "lazy_synthesis": (["examples/lazy_synthesis.py"], "quick"),
    "envelope_nonlinear": (["examples/envelope_nonlinear.py"], "quick"),
    "envelope_room_cooling": (["examples/envelope_room_cooling.py"], "quick"),
    "assumption_monitor": (["examples/assumption_monitor.py", "--no-plot"], "quick"),
    "l4dc_acc": (["reproductions/l4dc_acc.py"], "quick"),
    "tac_acc": (["reproductions/tac_acc.py"], "quick"),
    "adhs_acc": (["reproductions/adhs_acc.py"], "quick"),
    "adhs_script_params": (["reproductions/adhs_script_params.py"], "quick"),
    "pendulum_bounded_derivatives": (["examples/pendulum_bounded_derivatives.py"], "full"),
    "tac_overapproximation": (["reproductions/tac_overapproximation.py"], "full"),
    "thesis_ch3": (["reproductions/thesis_ch3.py"], "full"),
    "tac_lorenz_w50": (["reproductions/tac_lorenz.py"], "full"),
    "tac_lorenz_w1": (["reproductions/tac_lorenz.py", "--w-weight", "1"], "full"),
}
# online_refinement.m is left out: its very first safe set already differs between Octave 8.4
# (Ubuntu 24.04) and Octave 11 (Ubuntu 26.04, CI), 49.0 % vs 48.8 %, before any random
# integer is drawn; a platform difference in the growth fit, not a change in ddma.
MATLAB = ["scalar_map", "bounded_derivative_map", "acc_safety_filter", "lazy_synthesis",
          "pendulum_bounded_derivatives", "rl_safety_filter", "assumption_monitor",
          "envelope_room_cooling", "envelope_nonlinear"]
for _m in MATLAB:
    CHECKS[f"matlab_{_m}"] = (_m, "matlab")


def normalize(text):
    """Mask what legitimately changes between runs: timings, file paths, Octave banners."""
    out = []
    for line in text.splitlines():
        if "X11 DISPLAY" in line or "disabling GUI features" in line:
            continue
        line = re.sub(r"(/[\w.\-]+)+/[\w.\-]*", "<path>", line)
        # simulated time ("t = 75.0 s") is a result, not a timing: protect it
        line = re.sub(r"t =\s+(-?[\d.]+) s\b", r"t=\1sec", line)
        line = re.sub(r"\d+(\.\d+)? / \d+(\.\d+)? / \d+(\.\d+)?", "<t> / <t> / <t>", line)
        # timing columns of the pendulum table (model, abstraction, safety, then the %)
        line = re.sub(r"^(\w+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(?=[\d.]+\s?%)",
                      r"\1  <t> <t> <t>  ", line)
        line = re.sub(r"\b\d+(\.\d+)?(e[-+]?\d+)?\s?(s|ms|µs|us)\b", "<t>", line)
        out.append(line.rstrip())
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


def run_python(cmd, workdir, env):
    return subprocess.run([sys.executable, *cmd], cwd=workdir, env=env,
                          capture_output=True, text=True)


def run_matlab(example, env):
    pre = "K = 20; " if example == "lazy_synthesis" else ""  # as in CI
    expr = f"set(0,'defaultfigurevisible','off'); {pre}run examples/{example}"
    return subprocess.run(["octave", "--no-gui", "--eval", expr], cwd=ROOT / "matlab", env=env,
                          capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also the slow reproductions (~10 min)")
    ap.add_argument("--matlab", action="store_true", help="the MATLAB examples in Octave instead")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="run these checks only")
    ap.add_argument("--update", action="store_true", help="write the outputs as the new references")
    a = ap.parse_args()

    if a.only:
        unknown = set(a.only) - set(CHECKS)
        if unknown:
            ap.error(f"unknown check(s): {', '.join(sorted(unknown))}; "
                     f"choose from {', '.join(CHECKS)}")
        names = a.only
    elif a.matlab:
        names = [n for n, (_, t) in CHECKS.items() if t == "matlab"]
    else:
        names = [n for n, (_, t) in CHECKS.items() if t == "quick" or (a.full and t == "full")]

    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    EXPECTED.mkdir(exist_ok=True)
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for d in ("examples", "reproductions"):
            skip = shutil.ignore_patterns("results", "expected", "__pycache__")
            shutil.copytree(ROOT / d, tmp / d, ignore=skip)
        (tmp / "docs" / "figures").mkdir(parents=True)
        for name in names:
            cmd, tier = CHECKS[name]
            t0 = time.perf_counter()
            res = run_matlab(cmd, env) if tier == "matlab" else run_python(cmd, tmp, env)
            dt = time.perf_counter() - t0
            got = normalize(res.stdout)
            ref_file = EXPECTED / f"{name}.txt"
            rel = ref_file.relative_to(ROOT)
            if res.returncode != 0:
                failed.append(name)
                print(f"FAIL  {name}: exit code {res.returncode} ({dt:.0f} s)")
                tail = (res.stdout + res.stderr).strip().splitlines()[-15:]
                print("      " + "\n      ".join(tail))
                continue
            if a.update:
                ref_file.write_text(got)
                print(f"wrote {rel} ({dt:.0f} s)")
                continue
            if not ref_file.exists():
                failed.append(name)
                print(f"FAIL  {name}: no reference {rel} (run with --update)")
                continue
            ref = ref_file.read_text()
            if got == ref:
                print(f"ok    {name} ({dt:.0f} s)")
            else:
                failed.append(name)
                print(f"FAIL  {name}: output differs from {rel} ({dt:.0f} s)")
                diff = difflib.unified_diff(ref.splitlines(), got.splitlines(), "expected", "now",
                                            lineterm="", n=0)
                print("      " + "\n      ".join(diff))
    if failed:
        print(f"\n{len(failed)} of {len(names)} checks failed: {', '.join(failed)}")
        sys.exit(1)
    if not a.update:
        print(f"\nall {len(names)} checks passed")


if __name__ == "__main__":
    main()
