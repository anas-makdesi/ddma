"""Generate the MATLAB API page (api/matlab.md) from the help text of matlab/+ddma/*.m.

MATLAB help text is plain text (``help ddma.fit`` shows it as written), so it is reproduced
verbatim under each function's signature rather than parsed as reStructuredText. The page is
written before every build; the grouping below must list every function of the package
(the build fails otherwise, so a new function can't be left out of the reference).
"""

from __future__ import annotations

import re
from pathlib import Path

GROUPS = [
    ("Partitions",
     "Grids are cell arrays of edge vectors, one per coordinate. Cell subscripts are 0-based "
     "(as in Python); linear indices are 1-based, column-major.",
     "uniform_partition margin_partition data_partition grid_shape cell_bounds cells_inside "
     "cells_meeting locate searchsorted all_subs all_subs_of flat_index"),
    ("Models of monotone systems", "",
     "transitions_to_data fit fit_update intersect compact bounds f_bounds contains widths "
     "bounded_fraction is_empty msm_bounds msm_f_bounds msm_estimate msm_is_consistent"),
    ("Models of systems with bounded derivatives", "",
     "bd_fit bd_update bd_update_local growth_map estimate_derivative_bounds "
     "scenario_confidence max_min_affine"),
    ("Abstraction", "", "abstraction abstraction_refresh affine_bounds box_all_in"),
    ("Synthesis and closed loop", "", "safety reach_avoid controller_input simulate"),
    ("Safe envelopes without a grid", "",
     "monotone_envelope embedding_envelope envelope_contains envelope_input maximal_points "
     "max_min_cones"),
    ("Reinforcement learning",
     "Wrappers for environments in the form used by the Reinforcement Learning Toolbox "
     "(`rlFunctionEnv`).",
     "safe_step safe_reset"),
    ("Threads, plotting and helpers", "",
     "set_num_threads get_num_threads plot_bounds plot_set seed"),
]


def parse(path: Path):
    """(signature line, help text) of a MATLAB function file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    sig = next(ln for ln in lines if ln.startswith("function"))
    start = lines.index(sig) + 1
    help_lines = []
    for ln in lines[start:]:
        if not ln.lstrip().startswith("%"):
            break
        help_lines.append(re.sub(r"^\s*%", "", ln))
    # first line "%FIT Summary": drop the upper-case name, keep the summary
    if help_lines:
        help_lines[0] = re.sub(r"^[A-Z_0-9]+\s+", "", help_lines[0].strip())
    # common indentation of the remaining lines
    body = help_lines[1:]
    ind = min((len(b) - len(b.lstrip()) for b in body if b.strip()), default=0)
    body = [b[ind:] for b in body]
    first = help_lines[0] if help_lines else ""
    if first and not first.endswith("."):
        # the H1 line continues on the next lines: keep the text in one piece
        return sig, "", "\n".join([first] + body).rstrip()
    return sig, first, "\n".join(body).rstrip()


def write_page(src_dir: Path, out: Path) -> None:
    pkg = src_dir / "+ddma"
    have = {p.stem: p for p in pkg.glob("*.m")}
    listed = [n for _, _, names in GROUPS for n in names.split()]
    missing = sorted(set(have) - set(listed))
    unknown = sorted(set(listed) - set(have))
    if missing or unknown:
        raise RuntimeError(f"docs/_ext/matlab_api.py: functions not listed {missing}, "
                           f"listed but absent {unknown}")
    md = ["# MATLAB", "",
          "The `+ddma` package in `matlab/`. Functions work on structs (`S = ddma.fit(...)`,",
          "`A = ddma.abstraction(...)`). Below each signature is the function's help text, as",
          "`help ddma.<name>` shows it in MATLAB. Run `build_mex` once to use the compiled",
          "kernels; without it the `.m` code gives the same results.", "",
          "Each function's Python counterpart is listed in {doc}`../guides/parity`.", ""]
    for title, intro, names in GROUPS:
        md += [f"## {title}", ""]
        if intro:
            md += [intro, ""]
        for name in names.split():
            sig, summary, body = parse(have[name])
            md += [f"(matlab-{name})=", f"### `ddma.{name}`", ""]
            if summary:
                md += [summary, ""]
            md += ["```matlab", sig, "```", ""]
            if body:
                md += ["```text", body, "```", ""]
    text = "\n".join(md)
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.write_text(text, encoding="utf-8")


def _on_init(app):
    write_page(Path(app.config.matlab_src_dir), Path(app.srcdir) / "api" / "matlab.md")


def setup(app):
    app.add_config_value("matlab_src_dir", "", "env")
    app.connect("builder-inited", _on_init)
    return {"parallel_read_safe": True}
