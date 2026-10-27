"""Gaussian scale mixtures that stay FE functions exactly: Student-t processes and type-G Levy-noise fields.

Both are linear in a Gaussian sample, so they run on the Gaussian's engine and device. The mixing variables are
iid, so here the fastest iid samplers apply: NumPy's (Marsaglia-Tsang gamma, its inverse-Gaussian sampler) on the CPU
engine, and on the torch engine device-native, seeded versions of Marsaglia & Tsang (2000) for gamma and Michael,
Schucany & Haas (1976) for the inverse Gaussian, so white noise and mixing never leave the GPU.
"""

import warnings

import ngsolve as ngs
import numpy as np
import torch

from ..noise import DiagonalNoise
from .pointwise import is_conditioned


class _FEField:
    """Output: n=None gives a GridFunction, otherwise (ndof x n) coefficients (a tensor on the Gaussian's device)."""

    torch_capable = True

    def _finish(self, c_free: torch.Tensor, n):
        g = self.gauss
        out = torch.from_numpy(g.mean).to(c_free.device, c_free.dtype)[:, None].repeat(1, c_free.shape[1])
        out[torch.from_numpy(g.space.free).to(c_free.device)] += c_free
        if n is not None:
            return out
        gf = ngs.GridFunction(g.space.fes)
        gf.vec.FV().NumPy()[:] = out[:, 0].double().cpu().numpy()
        return gf


def torch_gamma(shape: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Gamma(shape, 1) variates on shape's device: Marsaglia & Tsang (2000), with the U^(1/a) boost for a < 1."""
    boost = shape < 1
    a = torch.where(boost, shape + 1, shape)
    d = a - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    out = torch.empty_like(a)
    todo = torch.ones_like(a, dtype=torch.bool)
    while todo.any():
        idx = todo.nonzero(as_tuple=True)
        x = torch.randn(idx[0].shape, generator=gen, device=a.device, dtype=a.dtype)
        u = torch.rand(idx[0].shape, generator=gen, device=a.device, dtype=a.dtype)
        v = (1 + c[idx] * x) ** 3
        ok = (v > 0) & (torch.log(u) < 0.5 * x * x + d[idx] - d[idx] * v + d[idx] * torch.log(v.clamp_min(1e-300)))
        acc = tuple(i[ok] for i in idx)
        out[acc] = d[idx][ok] * v[ok]
        todo[acc] = False
    if boost.any():
        u = torch.rand(shape.shape, generator=gen, device=a.device, dtype=a.dtype)
        out = torch.where(boost, out * u ** (1.0 / shape), out)
    return out


def torch_inverse_gaussian(mean: torch.Tensor, shape: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Inverse-Gaussian(mean, shape) variates on the device: Michael, Schucany & Haas (1976), no rejection."""
    nu = torch.randn(mean.shape, generator=gen, device=mean.device, dtype=mean.dtype)
    y = nu * nu
    x = mean + mean * mean * y / (2 * shape) - mean / (2 * shape) * torch.sqrt(4 * mean * shape * y + (mean * y) ** 2)
    u = torch.rand(mean.shape, generator=gen, device=mean.device, dtype=mean.dtype)
    return torch.where(u <= mean / (mean + x), x, mean * mean / x)


class StudentTField(_FEField):
    """Student-t process: f = m + (g - m) sqrt(nu / V), V ~ chi^2_nu, one V per sample (an elliptical process).

    Every finite-dimensional marginal is multivariate t with nu degrees of freedom and scale matrix Sigma, and
    Cov(f) = nu / (nu - 2) Sigma for nu > 2. The sample is an FE function exactly.
    """

    def __init__(self, gauss, nu, seed=None):
        if is_conditioned(gauss):
            warnings.warn(
                "Student-t process on a conditioned Gaussian: this is the elliptical process with the posterior's mean "
                "and covariance, not the posterior of a Student-t prior (that would need inference over the scale)",
                stacklevel=3,
            )
        self.gauss, self.nu = gauss, float(nu)
        self.rng = np.random.default_rng(seed)
        self.space = gauss.space
        self.info = {"law": f"Student-t process (elliptical, nu={nu})", "gaussian": gauss.plan}

    def sample(self, n=None):
        ns = 1 if n is None else int(n)
        g = self.gauss
        c = g.sample(ns)
        c = c[torch.from_numpy(g.space.free).to(c.device)] - torch.from_numpy(g.mean[g.space.free]).to(c)[:, None]
        scale = torch.from_numpy(np.sqrt(self.nu / self.rng.chisquare(self.nu, ns))).to(c)
        return self._finish(c * scale[None, :], n)


class TypeGField(_FEField):
    """Type-G Levy-noise field: the Gaussian pipeline driven by variance-mixed white noise (Bolin 2014;
    Wallin & Bolin 2015).

    The white noise on element e is sqrt(V_e) times Gaussian white noise there, with independent V_e, E[V_e] = 1 and
    Var(V_e) = nu / |e|:
    - ``mixing="nig"``: V_e ~ InverseGaussian(mean 1, shape |e|/nu), giving normal-inverse-Gaussian noise;
    - ``mixing="gamma"``: V_e ~ Gamma(shape |e|/nu, scale nu/|e|), giving variance-gamma (Laplace-type) noise.

    Both families are closed under summation over elements, so the noise measure of any set has the right law on every
    mesh: the construction is consistent in function space. The covariance is exactly the Gaussian one (E[V] = 1),
    the marginals are heavier-tailed, and the sample is an FE function exactly. It needs a white-noise pipeline with
    element-wise noise (local-gram, local-gram-explicit, quadrature or diagonal; even or fractional powers).
    """

    def __init__(self, gauss, mixing="nig", nu=1.0, seed=None, torch_seed=None):
        if is_conditioned(gauss) or not hasattr(gauss, "noise"):
            raise ValueError("type-G fields are defined for Gaussian priors with a white-noise pipeline, not for "
                             "conditioned or low-rank Gaussians")
        if "stiffness" in gauss.plan.get("structure", "") or "assembled" in gauss.plan.get("structure", "").lower():
            raise NotImplementedError("type-G fields need white-noise pipelines (even or fractional powers)")
        if mixing not in ("nig", "gamma"):
            raise ValueError("mixing must be 'nig' or 'gamma'")
        self.gauss, self.mixing, self.nu = gauss, mixing, float(nu)
        self.space = gauss.space
        self.rng = np.random.default_rng(seed)
        self.element_of_column = _element_map(gauss)
        self.volumes = ngs.Integrate(ngs.CoefficientFunction(1.0), self.space.mesh, element_wise=True).NumPy().copy()
        self.on_device = gauss.engine == "torch"
        if self.on_device:
            from ..backend import make_generator

            dev = gauss.device
            self.tgen = make_generator(self.rng.integers(2**62) if torch_seed is None else torch_seed, dev)
            self.t_shape = torch.as_tensor(self.volumes / self.nu, device=dev, dtype=torch.float64)
            self.t_emap = torch.as_tensor(self.element_of_column, device=dev)
        self.info = {
            "law": f"type-G ({'normal-inverse-Gaussian' if mixing == 'nig' else 'variance-gamma'}) noise field",
            "mixing": ("Michael-Schucany-Haas" if mixing == "nig" else "Marsaglia-Tsang")
            + (" (torch, on device)" if self.on_device else " (NumPy)"),
            "gaussian": gauss.plan,
        }

    def mixing_variables(self, n) -> np.ndarray:
        """V_e for every element and sample (ne x n): mean 1, variance nu / |e|."""
        shape = self.volumes / self.nu
        if self.mixing == "nig":
            return self.rng.wald(1.0, np.repeat(shape[:, None], n, axis=1))
        return self.rng.gamma(np.repeat(shape[:, None], n, axis=1), 1.0 / shape[:, None])

    def _device_mixing(self, n):
        shape = self.t_shape[:, None].expand(-1, n).contiguous()
        if self.mixing == "nig":
            return torch_inverse_gaussian(torch.ones_like(shape), shape, self.tgen)
        return torch_gamma(shape, self.tgen) / shape

    def sample(self, n=None):
        ns = 1 if n is None else int(n)
        g = self.gauss
        if self.on_device:
            v = self._device_mixing(ns).to(g._work_dtype)
            eta = torch.randn(g.nrand, ns, generator=self.tgen, device=g.device, dtype=g._work_dtype)
            c = g.apply(eta * torch.sqrt(v[self.t_emap]))
        else:
            v = self.mixing_variables(ns)
            eta = self.rng.standard_normal((g.nrand, ns)) * np.sqrt(v[self.element_of_column])
            c = g.apply(torch.from_numpy(eta))
        return self._finish(c.to(g.dtype), n)


def _element_map(gauss) -> np.ndarray:
    noise = gauss.noise
    emap = getattr(noise, "element_of_column", None)
    if emap is not None:
        return emap
    if isinstance(noise, DiagonalNoise) and gauss.space.kind == "L2":
        owner = np.empty(gauss.space.ndof, dtype=np.int64)
        for el in gauss.space.fes.Elements(ngs.VOL):
            owner[list(el.dofs)] = el.nr
        return owner[gauss.space.free]
    raise NotImplementedError(f"type-G fields need element-wise white noise; {noise.name} has no element map")
