"""Dense ground truth and accuracy diagnostics (small problems)."""

import numpy as np
import scipy.linalg as sla
import torch

from .specs import AssembledPrecision, OperatorPrecision, PDEPrecision


def exact_covariance(g) -> np.ndarray:
    """Exact discrete coefficient covariance Sigma_h on the free dofs of a ``Gaussian``."""
    s, spec = g.space, g.spec
    if isinstance(spec, PDEPrecision):
        k = s.assemble_free(spec.form).toarray()
        m = s.mass_free.toarray()
        if spec.is_integer:
            kim = np.linalg.solve(k, m)
            sigma = np.linalg.matrix_power(kim, spec.power) @ np.linalg.inv(m)
        else:  # K V = M V diag(lam), V^T M V = I  =>  Sigma = (K^-1 M)^beta M^-1 = V diag(lam^-beta) V^T
            lam, v = sla.eigh(k, m)
            sigma = (v * lam ** (-spec.power)) @ v.T
    elif isinstance(spec, AssembledPrecision):
        sigma = np.linalg.inv(s.assemble_free(spec.form).toarray())
    elif isinstance(spec, OperatorPrecision):
        # dense Sigma^-1 by applying the operator to the identity (dual->dual: Sigma^-1 = Q_dd M)
        arg = (
            torch.from_numpy(s.mass_free.toarray())
            if spec.convention == "dual->dual"
            else torch.eye(s.nfree, dtype=torch.float64)
        )
        q = spec.apply(arg).double().cpu().numpy()
        sigma = np.linalg.inv(0.5 * (q + q.T))
    else:
        raise TypeError(type(spec).__name__)
    sigma = 0.5 * (sigma + sigma.T)
    return spec.scale**2 * sigma


def sampler_covariance(g, chunk=2048) -> np.ndarray:
    """Exact covariance L L^T of the sampler's linear map eta -> c (valid for linear pipelines)."""
    n = g.nrand
    cov = np.zeros((g.space.nfree, g.space.nfree))
    for start in range(0, n, chunk):
        eye = torch.zeros(n, min(chunk, n - start), dtype=torch.float64, device=g.device)
        eye[start + torch.arange(eye.shape[1]), torch.arange(eye.shape[1])] = 1.0
        cols = g.apply(eye).double().cpu().numpy()
        cov += cols @ cols.T
    return cov


def covariance_errors(approx: np.ndarray, exact: np.ndarray) -> dict:
    """Relative Frobenius error, max relative variance error, and KL(N(0,approx) || N(0,exact))."""
    fro = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    var = np.max(np.abs(np.diag(approx) - np.diag(exact)) / np.diag(exact))
    try:
        le = np.linalg.cholesky(exact)
        x = np.linalg.solve(le, np.linalg.solve(le, approx).T)
        sign, logdet_a = np.linalg.slogdet(approx)
        kl = 0.5 * (np.trace(x) - len(exact) + 2 * np.log(np.diag(le)).sum() - logdet_a) if sign > 0 else np.inf
    except np.linalg.LinAlgError:
        kl = np.nan
    return {"rel_fro": float(fro), "max_rel_var": float(var), "kl": float(kl)}
