import torch  # noqa: F401, I001  (before NGSolve, which the test modules import)
import os

# Tests check the built-in selection rules; an installed benchmark tuning must not change them.
os.environ["NGSRNG_TUNING"] = os.path.join(os.path.dirname(__file__), "no-tuning.json")

# The tests use coarse meshes on purpose (fast, exact checks of laws that do not need fine meshes): the check that the
# mesh resolves a Matérn field's length is off here, and on in tests/test_resolution.py.
import ngsrng.resolution as _resolution  # noqa: E402

_resolution.RESOLUTION_TOL = None


def cuda_usable() -> bool:
    """CUDA present *and* a context can be created (the GPU may be fully occupied by other processes)."""
    from ngsrng.backend import cuda_usable as usable

    return usable()


def gpu_solver():
    """The GPU's sparse solver: cuDSS (NVIDIA, a direct solver), else AMG-preconditioned CG (AMD and others, with
    pyamg); None without a usable GPU or either solver. Tests of GPU routes pass it as ``method``."""
    from ngsrng.solvers import cudss_available, pyamg_available

    if not cuda_usable():
        return None
    if cudss_available():
        return {"solver": "cudss"}
    return {"solver": "amg-pcg"} if pyamg_available() else None
