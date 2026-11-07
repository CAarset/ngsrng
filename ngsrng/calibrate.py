"""Calibrate this machine for ngsrng: ``python -m ngsrng.calibrate`` (or the ``ngsrng-calibrate`` command).

Measures the white-noise samplers and solvers that can win here (on the CPU, and on the GPU if one is usable),
checks every measurement for accuracy, rejects the inaccurate, failing and far slower candidates, and installs the
recommendations as this machine's calibration. ngsrng then uses them for its automatic choices, ahead of the
calibration it ships with (``python -m ngsrng.tuning show``). It works wherever ngsrng is installed, also as a
dependency of another project: it needs no repository, and no packages beyond ngsrng's own (the full report uses
pandas, tabulate and matplotlib if they are installed).

    python -m ngsrng.calibrate               # 2D and 3D, 1e4 to 2.5e5 dofs: 13 min on an A100 box, longer on laptops
    python -m ngsrng.calibrate --quick       # a few minutes: checks that calibrating works here (installs nothing)
    python -m ngsrng.calibrate --large       # up to 1e6 dofs (an hour or more)
    python -m ngsrng.calibrate --resume DIR  # continue an interrupted calibration
    python -m ngsrng.tuning show             # what is installed, and what applies on this machine

Install the accelerator extra that fits the machine first (``python -m ngsrng.tuning show`` names it): ``cuda`` for
cuDSS on NVIDIA GPUs, ``rocm`` for AMG-preconditioned CG (pyamg) on AMD GPUs, the GPU candidate where cuDSS is
missing.
"""

import argparse
import os

from .calibration import runner


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ngsrng.calibrate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="a few minutes at small sizes; installs nothing")
    ap.add_argument("--large", action="store_true", help="also 1e6 dofs")
    ap.add_argument("--devices", nargs="+", default=None, help="subset of: cpu cuda (default: all usable)")
    ap.add_argument("--out", default=None, help="run directory (default: ~/.cache/ngsrng/calibration/...)")
    ap.add_argument("--resume", default=None, help="run directory of an interrupted calibration")
    ap.add_argument("--no-install", action="store_true", help="measure only; install later with "
                                                                "python -m ngsrng.tuning install <run dir>")
    args = ap.parse_args(argv)
    cmd = ["--profile", "smoke" if args.quick else "large" if args.large else "calibrate"]
    if args.devices:
        cmd += ["--devices", *args.devices]
    cmd += ["--resume", args.resume] if args.resume else (["--out", args.out] if args.out else [])
    install = not (args.quick or args.no_install)
    if install:
        cmd.append("--install")
    run_dir = runner.main(cmd)
    print(f"\ncalibration: {os.path.join(run_dir, 'report.md')}")
    print("installed as this machine's calibration (python -m ngsrng.tuning show)" if install else
          f"not installed (python -m ngsrng.tuning install {run_dir})")


if __name__ == "__main__":
    main()
