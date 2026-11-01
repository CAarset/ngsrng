"""Benchmark-fitted algorithm selection.

The benchmark (bench/run_all.py) writes ``recommendations.json``: the fastest valid white-noise method and solver
configuration for every measured case and platform. Installing it makes ``Gaussian`` use these measured choices
for the nearest benchmarked case (same dimension and space type; nearest order, then nearest size on a log scale).
Without tuning data, the built-in rules apply.

    python -m ngsrng.tuning install results/runs/<run>    # make a run's measurements the active tuning
    python -m ngsrng.tuning show                          # print the active tuning summary

The active file is ``$NGSRNG_TUNING`` if set, else ``~/.cache/ngsrng/recommendations.json``.
"""

import json
import math
import os
import shutil
import sys
import warnings

ENV_VAR = "NGSRNG_TUNING"


def default_path() -> str:
    cache = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(cache, "ngsrng", "recommendations.json")


def platform_of(engine: str, device: str, dtype: str) -> str:
    if engine == "ngsolve":
        return "cpu-ngsolve"
    if device == "cpu":
        return "cpu-torch"
    return "cuda-f32" if "32" in str(dtype) else "cuda-f64"


# Key fields of the tables bench/report.py writes (power may be fractional, e.g. "2.6" or "2.0")
KEY_FIELDS = {"noise": "dim|kind|order|ndof", "prior": "dim|kind|order|power|ndof"}


class Tuning:
    def __init__(self, data: dict, source: str):
        _check(data, source)
        self.data, self.source = data, source

    @classmethod
    def load(cls, path=None):
        path = path or os.environ.get(ENV_VAR) or default_path()
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return cls(json.load(fh), path)

    @property
    def run(self) -> str:
        return self.data.get("run", "?")

    def noise(self, platform, dim, kind, order, ndof):
        hit = _nearest(self.data.get("noise", {}).get(platform, {}), dim, kind, order, ndof)
        return hit["method"] if hit else None

    def prior(self, platform, dim, kind, order, power, ndof):
        return _nearest(self.data.get("prior", {}).get(platform, {}), dim, kind, order, ndof, power)


def _nearest(table, dim, kind, order, ndof, power=None):
    best, best_d = None, None
    for key, val in table.items():
        parts = key.split("|")
        d, k, o = int(float(parts[0])), parts[1], int(float(parts[2]))
        if d != dim or k != kind:
            continue
        if power is not None:
            pw, target = float(parts[3]), float(parts[4])
            pdist = abs(pw - power)
        else:
            target, pdist = float(parts[3]), 0
        dist = (pdist, abs(o - order), abs(math.log10(max(target, 1)) - math.log10(max(ndof, 1))))
        if best_d is None or dist < best_d:
            best, best_d = {**val, "_target": target}, dist
    return best


def _check(data, source):
    """Reject tables this version cannot read (e.g. written by another version of bench/report.py)."""
    if not isinstance(data, dict):
        raise ValueError(f"{source}: not a tuning table")
    for suite, fields in KEY_FIELDS.items():
        for platform, table in data.get(suite, {}).items():
            for key, val in table.items():
                parts = key.split("|")
                try:
                    ok = len(parts) == len(fields.split("|")) and isinstance(val, dict) and "method" in val
                    [float(p) for i, p in enumerate(parts) if i != 1]
                except ValueError:
                    ok = False
                if not ok:
                    raise ValueError(f"{source}: unreadable {suite} entry {platform}/{key!r} (expected {fields} "
                                     "with a 'method')")


_ACTIVE = {}


def active():
    """The active tuning (cached per path), or None."""
    path = os.environ.get(ENV_VAR) or default_path()
    if path not in _ACTIVE:
        try:
            _ACTIVE[path] = Tuning.load(path)
        except (OSError, ValueError) as err:
            warnings.warn(f"ignoring the installed tuning ({err}); the built-in rules apply")
            _ACTIVE[path] = None
    return _ACTIVE[path]


def install(source: str, dest=None) -> str:
    src = os.path.join(source, "recommendations.json") if os.path.isdir(source) else source
    dest = dest or os.environ.get(ENV_VAR) or default_path()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)
    _ACTIVE.clear()
    return dest


def _main(argv):
    if len(argv) >= 2 and argv[0] == "install":
        print("installed", install(argv[1]))
    elif argv and argv[0] == "show":
        t = active()
        if t is None:
            print("no tuning installed; built-in rules are used")
            return
        print(f"tuning from run {t.run} ({t.source})")
        for suite in ("noise", "prior"):
            for plat, table in t.data.get(suite, {}).items():
                print(f"  {suite:5s} {plat:12s} {len(table)} cases")
    else:
        print(__doc__)


if __name__ == "__main__":
    _main(sys.argv[1:])
