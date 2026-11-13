"""ngsrng: function-space random variables in NGSolve finite element spaces."""

# torch before NGSolve: NGSolve loads its bundled libraries (OpenBLAS, the C++ runtime) on import, and torch loaded
# after them has crashed; a GPU runtime may also fail to start (ngsrng.tuning.gpu_diagnosis
# checks)
import torch  # noqa: F401, I001

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

from .backend import check_import_order  # noqa: E402

check_import_order()  # warns if ngsolve was imported before torch (sys.modules keeps the order)
