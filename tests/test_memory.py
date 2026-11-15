"""Samplers are freed when their last reference goes, without the cyclic garbage collector.

ngsrng's samplers hold factorisations, noise samplers and per-thread work vectors: memory the garbage collector does
not see, so reference cycles through them (an object keeping a bound method of itself, e.g. as a ``Scratch``
factory) let every set-up outlive its use until a rare full collection (measured: 0.3-0.5 GB per set-up in
long-lived worker processes). With the collector disabled, nothing built and dropped here may survive."""

import gc
import weakref

import ngsolve as ngs
import numpy as np
import pytest

import ngsrng


@pytest.fixture(scope="module")
def fes():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=2)


def _build(fes):
    """Set up and draw from the usual objects; return weak references to them."""
    rng = ngsrng.default_rng(fes, seed=1, dtype="float64", correlation=ngsrng.matern_correlation(0.3, nu=2))
    prior = rng.gaussian()
    post = prior.condition(points=[(0.0, 0.5)], noise=1e-6, data=[0.2])
    ngsrng.pointwise_variance(post)
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left", correlation=post)
    d.sample(2)
    pair = ngsrng.correlated([d, d], 0.5)
    pair.sample(2)
    dense = pair.dense()
    dense.sample(2)
    stationary = ngsrng.StationaryMatern(fes, 0.3, nu=2, construction="torus", seed=2)
    stationary.sample(2)
    rng.normal(bnd="left", bnd_values=np.zeros(len(ngsrng.boundary_dofs(fes, "left"))), size=2)
    objects = [rng, prior, post, d.field, pair, dense, stationary, rng.bernstein_form(prior.space)._mass,
               prior.solver, prior.noise]
    return [weakref.ref(o) for o in objects]


def test_set_ups_are_freed_without_the_garbage_collector(fes):
    _build(fes)  # warm-up: module-level state (tunings, thread pools) is made once
    gc.collect()
    gc.disable()
    try:
        refs = _build(fes)
        alive = [type(r()).__name__ for r in refs if r() is not None]
    finally:
        gc.enable()
    assert not alive, f"kept alive by reference cycles: {alive}"
