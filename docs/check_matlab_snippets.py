"""Run the MATLAB snippets of the tutorials in Octave (or MATLAB), so they can't go stale.

Every ```matlab block inside a "MATLAB" tab of docs/tutorials/*.md is run on its own, in a
fresh session, from the matlab/ folder (after build_mex if MEX files are present, they are
used; otherwise the .m code runs). Usage:

    python docs/check_matlab_snippets.py [--matlab]     # default: octave
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r":::\{tab-item\} MATLAB\n```matlab\n(.*?)```", re.S)


def main() -> int:
    use_matlab = "--matlab" in sys.argv
    failures = 0
    for md in sorted((ROOT / "docs" / "tutorials").glob("[0-9]*.md")):
        for i, code in enumerate(BLOCK.findall(md.read_text(encoding="utf-8"))):
            with tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / f"snippet_{i}.m"
                mdir = (ROOT / "matlab").as_posix()
                # run() changes to the script's folder: go back to matlab/ first
                script.write_text(f"cd('{mdir}'); addpath('{mdir}');\n" + code,
                                  encoding="utf-8")
                if use_matlab:
                    cmd = ["matlab", "-batch", f"run('{script}')"]
                else:
                    cmd = ["octave", "--no-gui", "--quiet", "--eval", f"run('{script}')"]
                r = subprocess.run(cmd, cwd=ROOT / "matlab", capture_output=True, text=True)
            ok = r.returncode == 0 and "error:" not in r.stderr
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {md.name} (block {i + 1})")
            if r.stdout.strip():
                print("     " + r.stdout.strip().replace("\n", "\n     "))
            if not ok:
                print(r.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
