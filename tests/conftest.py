import torch  # noqa: F401, I001  (before NGSolve, which the test modules import)
import os

# Tests check the built-in selection rules; an installed benchmark tuning must not change them.
os.environ["NGSRNG_TUNING"] = os.path.join(os.path.dirname(__file__), "no-tuning.json")


def cuda_usable() -> bool:
    """CUDA present *and* a context can be created (the GPU may be fully occupied by other processes)."""
    from ngsrng.backend import cuda_usable as usable

    return usable()
