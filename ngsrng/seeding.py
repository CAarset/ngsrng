"""Where ngsrng's randomness comes from: the ``seed`` argument of every sampler.

Two conventions, the ones torch and NumPy use:

- ``seed=None`` (default), or a generator object: **draw-time randomness**, as in ``torch.randn(...,
  generator=None)``. Every draw (one ``sample``/``sample_into`` call of any sampler, one ``rng.<family>(...)`` call)
  takes one 128-bit key from the generator at the time of the draw. Without a generator that is torch's default
  (CPU) generator, so ``torch.manual_seed`` and Lightning's ``seed_everything`` reproduce ngsrng whenever they are
  called (also after the sampler was built); an unseeded process differs from run to run (torch seeds itself from
  OS entropy at start-up); and samplers copied into DataLoader workers draw different samples in every worker,
  since torch reseeds each worker. A ``torch.Generator`` (any device) or a ``numpy.random.Generator`` passed as
  ``seed`` is used the same way: the sampler reads it at each draw, as torch and NumPy functions given a generator
  do. Each draw advances the generator by one key, whatever the number of samples.
- ``seed`` an int, a sequence of ints or a ``numpy.random.SeedSequence``: **an own stream**, as in
  ``numpy.random.default_rng(seed)``. The sampler continues its stream from draw to draw; no global state affects it
  or is advanced by it.

In a DataLoader worker, own streams and generator objects (copied into every worker) are re-keyed with the worker's
seed, base seed + worker id, so workers and epochs draw different samples, reproducibly under ``torch.manual_seed``
in the main process; torch reseeds its global generator per worker itself.

Samplers made from others inherit the convention: posteriors, low-rank and Karhunen-Loeve samplers, the fields of an
:class:`ngsrng.Generator` and ``Generator.spawn`` children of a seeded parent get independent child streams
(``SeedSequence.spawn``); those of a draw-time parent read the same generator. An explicit ``seed=`` overrides.

Inside one draw everything derives from the draw's key (or the own stream): white noise, mixing variables, KL
coefficients. So the CPU engine's samples do not depend on the number of threads, on how a batch is split into
chunks or on ``out=`` versus allocating calls: bit for bit given the same factorisation (a single NGSolve
``sparsecholesky`` factorisation rounds differently from one build to the next, ~1e-15 relative). On the torch
engine the samples depend on the chunk sizes, as torch's own generators do. Different engines, devices, solvers,
noise factors or library versions give different samples from the same seed, with the same law.

Randomness that only serves an algorithm (randomised range finders, trace and spectrum estimates) uses fixed
seeds of its own and leaves the samplers' streams alone: the results are deterministic.
"""

import threading
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch

KEY_WORDS = 2  # 63-bit words per draw key
WORKER_KEY = 2**32 + 1  # spawn-key tag of per-worker streams (SeedSequence.spawn indices never reach it)


def _worker_seed():
    """Inside a DataLoader worker: the worker's seed, base seed + worker id (torch draws the base seed from the main
    process's generator for every epoch's workers); None elsewhere.

    Samplers are copied into the workers: an own stream (explicit seed) or a generator object would continue
    identically in every worker and draw the same samples. Folding in the worker's seed gives each worker, and each
    epoch, its own stream, reproducible with ``torch.manual_seed`` in the main process, as torch's own generator
    is. torch's global generator is reseeded by the DataLoader itself."""
    data = __import__("sys").modules.get("torch.utils.data")
    info = data.get_worker_info() if data is not None else None
    return None if info is None else int(info.seed)


def _key(generator) -> np.random.SeedSequence:
    """One draw key from ``generator`` (None: torch's default CPU generator, which DataLoader workers reseed
    themselves; a generator object passed as seed gets the worker's seed folded in, see :func:`_worker_seed`)."""
    if isinstance(generator, np.random.Generator):
        words = generator.integers(0, 2**63, size=KEY_WORDS, dtype=np.int64).tolist()
    else:
        gen = torch.default_generator if generator is None else generator
        words = torch.empty(KEY_WORDS, dtype=torch.int64, device=gen.device).random_(generator=gen).tolist()
    wseed = _worker_seed() if generator is not None else None
    if wseed is not None:
        words = words + [WORKER_KEY, wseed]
    return np.random.SeedSequence([int(w) for w in words])


def _private_copy(seq: np.random.SeedSequence) -> np.random.SeedSequence:
    """An independent copy (spawning from it does not advance the caller's SeedSequence)."""
    return np.random.SeedSequence(seq.entropy, spawn_key=seq.spawn_key, pool_size=seq.pool_size,
                                  n_children_spawned=seq.n_children_spawned)


class Streams:
    """The random streams of one sampler, all derived from one SeedSequence: a NumPy generator, torch generators
    (one per device) and the keys of the CPU engine's normal stream (:class:`ngsrng.backend.NormalStream`)."""

    def __init__(self, seq: np.random.SeedSequence):
        self.seq_numpy, self.seq_torch, self.seq_normal = seq.spawn(3)
        self._numpy, self._torch, self.counter = None, {}, 0

    def numpy(self) -> np.random.Generator:
        if self._numpy is None:
            self._numpy = np.random.default_rng(self.seq_numpy)
        return self._numpy

    def torch(self, device) -> torch.Generator:
        device = torch.device(device)
        gen = self._torch.get(str(device))
        if gen is None:
            gen = self._torch[str(device)] = torch.Generator(device=device)
            gen.manual_seed(int(self.seq_torch.generate_state(1, np.uint64)[0] >> np.uint64(1)))
        return gen

    def reserve(self, n: int):
        """Keys of the next n samples of the normal stream: (SeedSequence, first key)."""
        first, self.counter = self.counter, self.counter + int(n)
        return self.seq_normal, first


class Source:
    """The randomness of one sampler (see the module docstring). ``seed``: None, an int (or sequence of ints), a
    ``numpy.random.SeedSequence``, a ``numpy.random.Generator``, a ``torch.Generator`` or another Source (shared)."""

    def __init__(self, seed=None):
        self.generator, self.seq, self._own = None, None, None
        if isinstance(seed, (torch.Generator, np.random.Generator)):
            self.generator = seed
        elif seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer, np.random.SeedSequence, list, tuple)):
                raise TypeError("seed must be None, an int, a sequence of ints, a numpy SeedSequence or Generator, "
                                f"or a torch.Generator; got {type(seed).__name__}")
            seq = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
            self.seq = _private_copy(seq)
            self._own = Streams(self.seq)

    @property
    def draw_time(self) -> bool:
        """Whether the randomness is read from a generator at each draw (seed None or a generator object)."""
        return self.seq is None

    def spawn(self) -> "Source":
        """A source for a derived sampler: an independent child stream, or the same draw-time generator."""
        return Source(self.seq.spawn(1)[0]) if self.seq is not None else Source(self.generator)

    def streams(self) -> Streams:
        """The streams for the current draw: the own ones, or ones keyed by the current draw's key. Inside a
        DataLoader worker an own stream is re-keyed with the worker's seed (see :func:`_worker_seed`)."""
        if self._own is None:
            return _draw_streams(self)
        wseed = _worker_seed()
        if wseed is None:
            return self._own
        cached = self.__dict__.get("_worker")
        if cached is None or cached[0] != wseed:
            seq = np.random.SeedSequence(self.seq.entropy, spawn_key=self.seq.spawn_key + (WORKER_KEY, wseed))
            cached = self._worker = (wseed, Streams(seq))
        return cached[1]

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_worker"}

    def describe(self) -> str:
        if self.seq is not None:
            return f"own stream (SeedSequence entropy {self.seq.entropy}, spawn key {self.seq.spawn_key})"
        if self.generator is None:
            return "torch's global generator, read at each draw"
        return f"{type(self.generator).__module__}.{type(self.generator).__name__}, read at each draw"

    def __repr__(self):
        return f"Source({self.describe()})"


def as_source(seed) -> Source:
    """The Source for a ``seed`` argument (a Source is returned as is)."""
    return seed if isinstance(seed, Source) else Source(seed)


def inherit(seed, parent) -> Source:
    """The source of a sampler derived from ``parent``: ``seed`` if given, else a child of the parent's source
    (draw-time with torch's global generator if the parent has none)."""
    if seed is not None:
        return as_source(seed)
    src = getattr(parent, "source", None)
    return src.spawn() if isinstance(src, Source) else Source()


# ------------------------------------------------------------------ draws

_LOCAL = threading.local()


def _draw_streams(source: Source) -> Streams:
    """Streams of a draw-time source for the current draw: all draw-time sources used in one draw share one key per
    generator, and get child streams of it in the order they are first used (deterministic for a given call)."""
    scope = getattr(_LOCAL, "scope", None)
    if scope is None:  # a draw outside any sampler call (a helper called directly): a key of its own
        return Streams(_key(source.generator))
    slot = scope["slots"].get(id(source))
    if slot is None:
        gid = id(source.generator)
        if gid not in scope["keys"]:
            scope["keys"][gid] = [source.generator, _key(source.generator), 0]
        entry = scope["keys"][gid]
        key, index = entry[1], entry[2]
        entry[2] += 1
        slot = scope["slots"][id(source)] = (source, Streams(np.random.SeedSequence(key.entropy, spawn_key=(index,))))
    return slot[1]


@contextmanager
def one_draw():
    """Everything inside is one draw: draw-time sources take one key per generator (the outermost call decides)."""
    if getattr(_LOCAL, "scope", None) is not None:
        yield
        return
    _LOCAL.scope = {"keys": {}, "slots": {}}
    try:
        yield
    finally:
        _LOCAL.scope = None


def draw(method):
    """Decorator: the method is one draw (see :func:`one_draw`)."""

    @wraps(method)
    def wrapper(*args, **kwargs):
        with one_draw():
            return method(*args, **kwargs)

    wrapper.__ngsrng_draw__ = True
    return wrapper
