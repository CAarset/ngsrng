"""In-place sampling: every sampler fills preallocated buffers.

Every sampler in ngsrng has an in-place form next to the allocating one, as torch has ``normal_`` next to ``normal``:
``rng.<family>(..., out=buffer)`` and ``FieldDistribution.sample(out=buffer)`` for the distributions (samples along
the first axis), ``Gaussian.sample(n, out=buffer)`` (also posteriors), ``LowRankGaussian.sample(n, out=buffer)`` and
``MatheronPosterior.sample(n, out=buffer)`` (an ndof x n buffer, the layout ``sample(n)`` returns). Resampling a
large batch then reuses one buffer.

The contract, for every sampler class, current and future (``tests/test_inplace.py`` checks it for every registered
family and every Gaussian type):

- ``sample_into(rows)`` fills ``rows`` in place: k samples along the first axis, each sample's values after it (any
  trailing shape with the right number of values); a torch tensor or a NumPy array (wrapped, not copied) of any
  floating dtype (integer for discrete families), on any device;
- no full-size intermediate: the work is done in chunks of at most ``work_values`` values (by default
  ``WORK_VALUES`` = 2^20 on the CPU, ``WORK_VALUES_CUDA`` = 2^24 on a GPU; set per generator with
  ``default_rng(..., work_values=...)``), so the transient memory does not grow with
  k; every chunk reaches the buffer in one copy (cast and device transfer included) or is written there directly
  (the CPU engine's worker threads write each sample into its row);
- the values are those ``sample`` returns for the same random stream (an identically seeded sampler, or the same
  torch seed for the default draw-time randomness: a call is one draw, :mod:`ngsrng.seeding`). Samplers draw their
  randomness sample by sample, so on the CPU engine the result does not depend on the chunking, ``work_values`` or
  the thread count: bit for bit where every sample is computed on its own (the Gaussian pipelines and everything
  built on their point values), to rounding where dense products or solves act on a whole chunk (low-rank models,
  Matheron, the Bernstein and projection coefficients). On the torch engine the random stream depends on the
  chunking, as torch's own generators do.

:class:`InPlace` implements the chunked fill on top of a sampler's ``sample``; samplers with a direct path override
``_fill`` or ``sample_into``.
"""

import numpy as np
import torch

from .backend import available_threads
from .seeding import draw

# Values per internal chunk of an in-place fill. The transient memory is a small multiple of a chunk (the point values
# plus the transform's temporaries: measured 2-10x in float64). On the CPU the threads are busy with far smaller
# chunks; on a GPU larger chunks batch better.
WORK_VALUES = 2**20
WORK_VALUES_CUDA = 2**24


def default_work_values(device) -> int:
    return WORK_VALUES_CUDA if getattr(device, "type", device) == "cuda" else WORK_VALUES


def as_tensor(out) -> torch.Tensor:
    """The buffer as a torch tensor sharing its memory (NumPy arrays are wrapped, not copied)."""
    t = torch.from_numpy(out) if isinstance(out, np.ndarray) else out
    if not isinstance(t, torch.Tensor):
        raise TypeError("out must be a NumPy array or a torch tensor")
    return t


def write(rows: torch.Tensor, vals):
    """Copy a chunk's values (..., k), sample axis last (NumPy or torch), into rows (k, ...): one copy, cast and
    device transfer included."""
    src = vals if isinstance(vals, torch.Tensor) else torch.from_numpy(np.asarray(vals))
    src = src.movedim(-1, 0)
    if tuple(src.shape) != tuple(rows.shape):
        if src[0].numel() != rows[0].numel():
            raise ValueError(f"out holds {rows[0].numel()} values per sample, the sampler gives {src[0].numel()}")
        rows = rows.view(src.shape)
    rows.copy_(src)


def chunks(n: int, per_sample: int, work_values=WORK_VALUES, minimum=1):
    """(start, stop) ranges covering range(n) with at most ``work_values`` internal values per chunk (but at least
    ``minimum`` samples)."""
    step = max(int(minimum), int(work_values // max(int(per_sample), 1)), 1)
    for k0 in range(0, n, step):
        yield k0, min(n, k0 + step)


class InPlace:
    """Mixin: ``sample_into(rows)`` as chunked ``sample`` calls (see the module docstring for the contract).

    Every ``sample`` and ``sample_into`` of a subclass is one draw (:func:`ngsrng.seeding.one_draw`): with draw-time
    randomness (seed None or a generator) the whole call takes one key, so its chunks continue one stream as with an
    explicit seed, and the values do not depend on the chunking on the CPU engine."""

    work_values = WORK_VALUES

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in ("sample", "sample_into"):
            fn = cls.__dict__.get(name)
            if fn is not None and not getattr(fn, "__ngsrng_draw__", False):
                setattr(cls, name, draw(fn))

    def _per_sample(self) -> int:
        """Values the sampler handles internally per sample (bounds the chunk size)."""
        space = getattr(self, "space", None)
        return getattr(space, "ndof", 1) + getattr(self, "nrand", 0)

    def _min_chunk(self) -> int:
        """Samples per chunk below which the CPU engine's threads (one sample each) would idle."""
        g = getattr(self, "gauss", None) or getattr(getattr(self, "points", None), "gauss", None) or self
        return available_threads() if getattr(g, "engine", None) == "ngsolve" else 1

    @draw
    def sample_into(self, rows):
        rows = as_tensor(rows)
        for k0, k1 in chunks(rows.shape[0], self._per_sample(), self.work_values, self._min_chunk()):
            self._fill(rows[k0:k1])
        return rows

    def _fill(self, rows):
        write(rows, self.sample(rows.shape[0]))

    def _sample_out(self, n, out):
        """``sample(n, out=buffer)`` for samplers whose ``sample(n)`` returns (..., n): fill that layout in place."""
        t = as_tensor(out)
        if n is not None and t.shape[-1] != int(n):
            raise ValueError(f"out holds {t.shape[-1]} samples (last axis), n is {n}")
        self.sample_into(t.movedim(-1, 0))
        return out
