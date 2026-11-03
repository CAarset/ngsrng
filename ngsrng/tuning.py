"""Calibrated algorithm selection: sensible defaults on every machine, per-machine calibration on request.

Modelled on FFTW's planner and wisdom. The built-in rules (``gaussian.py``) play the part of FFTW's estimate mode:
always valid, never measured. A calibration measures the candidates on a machine and keeps the winners as wisdom:

1. this machine's calibration (``python -m ngsrng.calibrate`` measures and installs it in
   ``~/.cache/ngsrng/recommendations.json``; calibrations of several GPUs sharing a home directory coexist there);
2. the calibration shipped with ngsrng (``ngsrng/data/default_tuning.json``: an NVIDIA A100 with a 32-thread AMD
   EPYC), so that a fresh clone uses measured choices;
3. the built-in rules, wherever no calibration applies.

A table applies only to its platform. CPU tables (``cpu-ngsolve``) apply on any CPU: they rank algorithms (fill-in
against iteration counts), not chips. GPU tables apply only on the device they were measured on
(``cuda-f64@NVIDIA A100-PCIE-40GB``, ``rocm-f64@<name>``): an A100 table says nothing about an AMD card. An entry
applies only near its calibrated case: same dimension and space type, order within one, size within a factor of ten.
And a tuned choice is only a suggestion: ``Gaussian`` uses it if it is available on this machine and builds without
error, else the built-in rule (``plan["tuning"]`` says which).

    python -m ngsrng.calibrate              # measure this machine and install the result
    python -m ngsrng.tuning show            # this machine, the calibrations and which of their tables apply
    python -m ngsrng.tuning install <run>   # install (merge) a calibration run by hand
    python -m ngsrng.tuning reset           # forget this machine's calibration

Environment: ``NGSRNG_TUNING=<file>`` uses only that file, ``NGSRNG_TUNING=none`` only the built-in rules;
``NGSRNG_QUIET=1`` silences the once-per-session notice that a machine has no calibration of its own.
"""

import json
import math
import os
import sys
import tempfile
import warnings

ENV_VAR = "NGSRNG_TUNING"
QUIET_VAR = "NGSRNG_QUIET"
SHIPPED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "default_tuning.json")
FORMAT = 2
# How far an entry reaches beyond its calibrated case
MAX_ORDER_GAP = 1
MAX_DECADES = 1.0
# Key fields of the tables the calibration writes (power may be fractional, e.g. "2.6" or "2.0")
KEY_FIELDS = {"noise": "dim|kind|order|ndof", "prior": "dim|kind|order|power|ndof"}
_DISABLED = ("", "none", "off", "0")


class CalibrationNotice(UserWarning):
    """This machine has no calibration of its own (filterable like any warning; or set NGSRNG_QUIET=1)."""


def default_path() -> str:
    """This machine's calibration (where ``install`` merges calibration runs)."""
    cache = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(cache, "ngsrng", "recommendations.json")


def gpu_backend() -> str:
    """``rocm`` for PyTorch's ROCm build (AMD GPUs appear as device "cuda" there), else ``cuda``."""
    import torch

    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def device_name(device) -> str:
    import torch

    return torch.cuda.get_device_name(device)


def platform_of(engine: str, device: str, dtype, name=None) -> str:
    """Tuning table of a Gaussian: ``cpu-ngsolve``, ``cpu-torch``, or ``<backend>-f64[@<device name>]``; ``dtype`` is
    its arithmetic (work) dtype, float64 unless opted out on a GPU."""
    if engine == "ngsolve":
        return "cpu-ngsolve"
    if device == "cpu":
        return "cpu-torch"
    base = f"{gpu_backend()}-{'f32' if '32' in str(dtype) else 'f64'}"
    return f"{base}@{name}" if name else base


def _is_gpu(platform) -> bool:
    return not platform.startswith("cpu-")


def _normalise(data: dict) -> dict:
    """Format 2: GPU platforms carry the device they were measured on. A run's report (or an older table) names them
    ``cuda-f64``; the device comes from its environment record, and GPU tables of unknown device are dropped."""
    if data.get("format") == FORMAT:
        return data
    env = data.get("environment") or {}
    gpu = (env.get("cuda") or {}).get("device")
    out = {k: v for k, v in data.items() if k not in KEY_FIELDS}
    out["format"] = FORMAT
    for suite in KEY_FIELDS:
        tables = {}
        for plat, table in data.get(suite, {}).items():
            if _is_gpu(plat) and "@" not in plat:
                if not gpu:
                    continue
                plat = f"{plat}@{gpu}"
            tables[plat] = table
        out[suite] = tables
    return out


class Tuning:
    """One calibration (a table per suite and platform), optionally layered over a less specific one."""

    def __init__(self, data: dict, source: str, fallback=None, label=None):
        _check(data, source)
        self.data, self.source, self.fallback = _normalise(data), source, fallback
        self.label = label or ("shipped" if os.path.abspath(source) == SHIPPED else "file")

    @classmethod
    def load(cls, path, fallback=None, label=None):
        if path is None or not os.path.exists(path):
            return fallback
        with open(path) as fh:
            return cls(json.load(fh), path, fallback, label)

    @property
    def run(self) -> str:
        return self.data.get("run", "?")

    def layers(self):
        t = self
        while t is not None:
            yield t
            t = t.fallback

    def covers(self, platform):
        """The label of the most specific layer with a table for ``platform``, or None."""
        for t in self.layers():
            if t.data.get("prior", {}).get(platform) or t.data.get("noise", {}).get(platform):
                return t.label
        return None

    def noise(self, platform, dim, kind, order, ndof):
        hit = self._lookup("noise", platform, dim, kind, order, ndof)
        return hit["method"] if hit else None

    def prior(self, platform, dim, kind, order, power, ndof):
        return self._lookup("prior", platform, dim, kind, order, ndof, power)

    def _lookup(self, suite, platform, *case):
        for t in self.layers():
            hit = _nearest(t.data.get(suite, {}).get(platform, {}), *case)
            if hit is not None:
                return {**hit, "_layer": t.label, "_run": t.run}
        return None


def _nearest(table, dim, kind, order, ndof, power=None):
    """The closest calibrated case: same dimension and space type, nearest power, order and size -- or None when even
    the closest one is more than MAX_ORDER_GAP orders or MAX_DECADES decades of size away."""
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
        decades = abs(math.log10(max(target, 1)) - math.log10(max(ndof, 1)))
        if abs(o - order) > MAX_ORDER_GAP or decades > MAX_DECADES:
            continue
        dist = (pdist, abs(o - order), decades)
        if best_d is None or dist < best_d:
            best, best_d = {**val, "_target": target}, dist
    return best


def _check(data, source):
    """Reject tables this version cannot read (e.g. written by another version of the calibration)."""
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


def explicit() -> bool:
    """Whether NGSRNG_TUNING chooses the tuning (a file, or none) instead of the calibration layers."""
    return os.environ.get(ENV_VAR) is not None


def active():
    """The tuning in effect: this machine's calibration over the shipped one (or what NGSRNG_TUNING says); cached
    until the installed file changes."""
    env = os.environ.get(ENV_VAR)
    if env is not None:
        key = ("env", env)
    else:
        user = default_path()
        key = ("layers", user, os.path.getmtime(user) if os.path.exists(user) else None)
    if key not in _ACTIVE:
        try:
            if env is not None:
                _ACTIVE[key] = None if env.strip().lower() in _DISABLED else Tuning.load(env, label="file")
            else:
                shipped = Tuning.load(SHIPPED)
                _ACTIVE[key] = Tuning.load(default_path(), fallback=shipped, label="installed")
        except (OSError, ValueError) as err:
            warnings.warn(f"ignoring the installed tuning ({err}); the built-in rules apply")
            _ACTIVE[key] = None
    return _ACTIVE[key]


def install(source: str, dest=None) -> str:
    """Merge a calibration run (its directory or recommendations.json) into this machine's calibration: its platform
    tables replace the same platforms' earlier tables; other platforms (another GPU, say) are kept."""
    src = os.path.join(source, "recommendations.json") if os.path.isdir(source) else source
    env = os.environ.get(ENV_VAR)
    dest = dest or (env if env and env.strip().lower() not in _DISABLED else default_path())
    with open(src) as fh:
        new = json.load(fh)
    _check(new, src)
    new = _normalise(new)
    old = {"format": FORMAT, "noise": {}, "prior": {}, "runs": {}}
    if os.path.exists(dest):
        with open(dest) as fh:
            old = _normalise(json.load(fh))
    merged = {"format": FORMAT, "run": new.get("run", "?"), "environment": new.get("environment"),
              "runs": dict(old.get("runs", {}))}
    for suite in KEY_FIELDS:
        merged[suite] = {**old.get(suite, {}), **new.get(suite, {})}
        for plat in new.get(suite, {}):
            merged["runs"][plat] = new.get("run", "?")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(merged, fh, indent=1, default=str)
    os.replace(tmp, dest)
    _ACTIVE.clear()
    return dest


def reset() -> bool:
    """Forget this machine's calibration (the shipped one applies again)."""
    path = default_path()
    if os.path.exists(path):
        os.remove(path)
        _ACTIVE.clear()
        return True
    return False


# ------------------------------------------------------------------ the once-per-session notice

_NOTICED = False


def _session_marker():
    """A file that marks this terminal session (os.getsid: every process started from one shell shares it), or None
    where sessions are unknown (then the notice appears once per process)."""
    try:
        sid = os.getsid(0)
    except (AttributeError, OSError):
        return None
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return os.path.join(tempfile.gettempdir(), f"ngsrng-notice-{uid}-{sid}")


def notice(platform, tuning, plan):
    """Once per terminal session: tell a user whose machine has no calibration of its own how to get one. Silent
    when NGSRNG_TUNING or NGSRNG_QUIET is set, or when this machine's calibration covers the platform."""
    global _NOTICED
    if _NOTICED or explicit() or os.environ.get(QUIET_VAR):
        return
    layer = tuning.covers(platform) if tuning is not None else None
    if layer == "installed":
        return
    _NOTICED = True
    marker = _session_marker()
    if marker is not None:
        if os.path.exists(marker):
            return
        try:
            open(marker, "w").close()
        except OSError:
            pass
    where = platform.split("@", 1)[1] if "@" in platform else ("this CPU" if platform.startswith("cpu") else platform)
    if layer == "shipped":
        what = f"ngsrng uses the calibration shipped with it (measured on another machine) for {where}."
    else:
        what = (f"ngsrng has no calibration for {where} and uses its built-in rules "
                f"(solver: {plan.get('solver', '?')}).")
    warnings.warn(f"{what} Calibrating this machine takes 15-35 minutes and makes the automatic choices of solver "
                  "and white-noise sampler measured ones: python -m ngsrng.calibrate (python -m ngsrng.tuning show "
                  "explains; NGSRNG_QUIET=1 silences this notice).", CalibrationNotice, stacklevel=3)


# ------------------------------------------------------------------ command line


def machine() -> dict:
    out = {"cpu": _cpu_model(), "threads": os.cpu_count()}
    try:
        import torch

        if torch.cuda.is_available():
            out["gpus"] = [f"{gpu_backend()}: {torch.cuda.get_device_name(i)}"
                           for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 - informational only
        pass
    return out


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or "unknown"


def torch_build() -> str:
    """The installed torch and the GPU platform it was built for."""
    import torch

    if getattr(torch.version, "hip", None):
        return f"torch {torch.__version__} (ROCm {torch.version.hip} build)"
    if torch.version.cuda:
        return f"torch {torch.__version__} (CUDA {torch.version.cuda} build)"
    return f"torch {torch.__version__} (CPU-only build)"


_PROBE = """
import json, sys
for module in sys.argv[1:]:
    __import__(module)
import torch
out = {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(), "error": None}
try:
    torch.zeros(1, device="cuda")
except Exception as err:
    out["error"] = (type(err).__name__ + ": " + str(err)).splitlines()[0][:200]
print(json.dumps(out))
"""


def _probe(*first) -> dict:
    """Whether torch sees a usable GPU in a fresh process of this Python, after importing ``first``."""
    import json
    import subprocess

    try:
        run = subprocess.run([sys.executable, "-c", _PROBE, *first], capture_output=True, text=True, timeout=180)
        return json.loads(run.stdout.strip().splitlines()[-1])
    except Exception as err:  # noqa: BLE001 - a diagnosis must not fail
        return {"available": False, "count": 0, "error": f"probe failed: {type(err).__name__}: {err}"[:200]}


def gpu_diagnosis() -> list[str]:
    """Why no GPU is usable in this process, as lines of text; empty if one is.

    Checks the torch build against the GPU devices of the machine (/dev/kfd: AMD, /dev/nvidiactl: NVIDIA), and, for
    a GPU build of torch, whether a fresh process of this Python sees the GPU with torch alone and after importing
    NGSolve first (NGSolve loads its own bundled libraries)."""
    import torch

    from .backend import cuda_usable

    if cuda_usable():
        return []
    lines = [f"{torch_build()}, from {os.path.dirname(torch.__file__)} (Python {sys.executable})"]
    lines += [f"{v}={os.environ[v]}" for v in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                                "HSA_OVERRIDE_GFX_VERSION") if v in os.environ]
    amd = os.path.exists("/dev/kfd")
    nvidia = os.path.exists("/dev/nvidiactl") or os.path.exists("/dev/nvidia0")
    rocm, cuda = bool(getattr(torch.version, "hip", None)), bool(torch.version.cuda)
    if not (rocm or cuda):
        if amd or nvidia:
            kind, build = ("AMD", "ROCm") if amd else ("NVIDIA", "CUDA")
            lines.append(f"this torch is a CPU-only build, but the machine has an {kind} GPU: install the {build} "
                         "build of torch in this environment (pytorch.org lists the index URLs)")
        else:
            lines.append("no GPU device found")
        return lines
    if amd and not nvidia and not rocm:
        lines.append("this torch is an NVIDIA (CUDA) build, which cannot drive AMD GPUs, and the machine's GPU is "
                     "AMD: install the ROCm build of torch in this environment (installing a package that depends "
                     "on torch may have replaced it with the default wheel)")
        return lines
    alone, after = _probe(), _probe("ngsolve")

    def seen(p):
        return p["available"] and p["error"] is None

    if seen(alone) and not seen(after):
        lines.append("torch finds the GPU on its own, but not when NGSolve is imported first: import torch before "
                     f"ngsolve (ngsrng does; with NGSolve first: {after['error'] or 'no device'})")
    elif seen(alone):
        lines.append("a fresh process of this Python finds the GPU, also after NGSolve, but this process does not: "
                     "something imported or done earlier here interferes")
    else:
        lines.append(f"torch finds no usable GPU in this Python even on its own ({alone['error'] or 'no device'}): "
                     "check that this is the environment where the GPU works"
                     + ("; ROCm also needs the user in the groups render and video, and a ROCm release that supports "
                        "the card" if rocm else ""))
    return lines


def _show():
    m = machine()
    print(f"machine: {m['cpu']} ({m['threads']} threads); GPUs: {', '.join(m.get('gpus', [])) or 'none'}")
    try:
        if not m.get("gpus"):
            print("  torch reports no usable GPU here (ngsrng uses any GPU torch reports); why:")
            for line in gpu_diagnosis():
                print(f"  no GPU: {line}")
    except Exception as err:  # noqa: BLE001 - informational only
        print(f"  no GPU: diagnosis failed ({type(err).__name__}: {err})")
    env = os.environ.get(ENV_VAR)
    if env is not None:
        print(f"{ENV_VAR}={env}: " + ("built-in rules only" if env.strip().lower() in _DISABLED else "only this file"))
    t = active()
    if t is None:
        print("no calibration applies: built-in rules")
        return
    names = ["cpu-ngsolve"]
    try:
        import torch

        if torch.cuda.is_available():
            names += [platform_of("torch", "cuda", "float64", torch.cuda.get_device_name(i))
                      for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        pass
    for layer in t.layers():
        print(f"{layer.label} calibration: {layer.source}")
        for suite in KEY_FIELDS:
            for plat, table in layer.data.get(suite, {}).items():
                here = " (applies here)" if plat in names else ""
                print(f"  {suite:5s} {plat}: {len(table)} cases{here}")
    for plat in names:
        if t.covers(plat) != "installed":
            print(f"{plat}: no calibration of this machine -- python -m ngsrng.calibrate")


def _main(argv):
    if len(argv) >= 2 and argv[0] == "install":
        print("installed", install(argv[1]))
    elif argv and argv[0] == "show":
        _show()
    elif argv and argv[0] == "reset":
        print("removed this machine's calibration" if reset() else "this machine had no calibration")
    else:
        print(__doc__)


if __name__ == "__main__":
    _main(sys.argv[1:])
