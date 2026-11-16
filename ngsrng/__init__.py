"""ngsrng: function-space random variables in NGSolve finite element spaces."""

import os as _os

# NGSRNG_NUM_THREADS=n: the BLAS and OpenMP libraries create their thread pools when they load, so their own variables
# are set too (where unset) before torch loads them; ngsrng.set_num_threads(n) below lowers them again if they loaded
# earlier (see ngsrng.backend.set_num_threads)
if _os.environ.get("NGSRNG_NUM_THREADS", "").strip().isdigit() and int(_os.environ["NGSRNG_NUM_THREADS"]) > 0:
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ.setdefault(_name, _os.environ["NGSRNG_NUM_THREADS"].strip())

# torch before NGSolve: NGSolve loads its bundled libraries (OpenBLAS, the C++ runtime) on import, and torch loaded
# after them has crashed; a GPU runtime may also fail to start (ngsrng.tuning.gpu_diagnosis
# checks)
import torch  # noqa: F401, E402, I001

from .boundary import BoundarySampler, boundary_covariance, boundary_dofs, boundary_mass  # [bnd] boundary sampling
from .correlation import MaternCorrelation
from .correlation import matern as matern_correlation
from .covariance import (
    LocalCovariance,
    TraceEstimate,
    covariance_function,
    covariance_trace,
    functional_covariance,
    local_covariance,
    pointwise_variance,
    sample_covariance,
    xnystrace,
)
from .correlated import CorrelatedFields, correlated
from .dense import DenseSampler, ObservedField
from .gaussian import Gaussian
from .generator import FieldDistribution, Generator, default_rng
from .lowrank import LowRankGaussian, karhunen_loeve, operator_kl
from .observed import ObservedGaussian
from .observations import observation_operator
from .posterior import LinearObservations, MatheronPosterior, downdate
from .robin import RobinTuning, tune_robin
from .space import SpaceInfo
from .specs import AssembledPrecision, OperatorCovariance, OperatorPrecision, PDEPrecision, matern, matern_dg
from .stationary import StationaryMatern

__all__ = [
    "default_rng",
    "Generator",
    "FieldDistribution",
    "DenseSampler",
    "ObservedField",
    "correlated",
    "CorrelatedFields",
    "Gaussian",
    "LowRankGaussian",
    "karhunen_loeve",
    "operator_kl",
    "local_covariance",
    "pointwise_variance",
    "tune_robin",
    "RobinTuning",
    "covariance_trace",
    "covariance_function",
    "functional_covariance",
    "xnystrace",
    "sample_covariance",
    "LocalCovariance",
    "TraceEstimate",
    "LinearObservations",
    "MatheronPosterior",
    "ObservedGaussian",
    "downdate",
    "observation_operator",
    "SpaceInfo",
    "PDEPrecision",
    "AssembledPrecision",
    "OperatorPrecision",
    "OperatorCovariance",
    "MaternCorrelation",
    "StationaryMatern",
    "matern_correlation",
    "matern",
    "matern_dg",
]
__all__ += ["BoundarySampler", "boundary_covariance", "boundary_dofs", "boundary_mass"]  # [bnd] boundary sampling

from .backend import check_import_order, get_num_threads, set_num_threads  # noqa: E402

__all__ += ["set_num_threads", "get_num_threads"]

check_import_order()  # warns if ngsolve was imported before torch (sys.modules keeps the order)


def _apply_thread_limit():
    """NGSRNG_NUM_THREADS=n, set before the import: as ``set_num_threads(n)``."""
    import os
    import warnings

    from .backend import NUM_THREADS_ENV

    value = os.environ.get(NUM_THREADS_ENV, "").strip()
    if not value:
        return
    try:
        n = int(value)
        if n < 1:
            raise ValueError
    except ValueError:
        warnings.warn(f"{NUM_THREADS_ENV}={value!r} is not a positive integer; ignored", stacklevel=3)
        return
    set_num_threads(n)


_apply_thread_limit()
