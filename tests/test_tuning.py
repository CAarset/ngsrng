import json

import ngsolve as ngs
import pytest

import ngsrng
from ngsrng import tuning


def _recs(tmp_path):
    data = {
        "run": "test",
        "noise": {
            "cpu-ngsolve": {
                "2|H1|2|1000": {"method": "quadrature", "per_sample_s": 1e-4},
                "2|H1|2|1000000": {"method": "local-gram", "per_sample_s": 1e-2},
            }
        },
        "prior": {
            "cpu-ngsolve": {
                "2|H1|2|2|1000": {
                    "solver": "cg-local",
                    "method": {"solver": "cg", "precond": "local", "engine": "ngsolve"},
                    "per_sample_s": 1e-3,
                }
            }
        },
    }
    path = tmp_path / "recommendations.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_nearest_lookup(tmp_path):
    t = tuning.Tuning.load(_recs(tmp_path))
    assert t.noise("cpu-ngsolve", 2, "H1", 3, 2000) == "quadrature"
    assert t.noise("cpu-ngsolve", 2, "H1", 2, 5e5) == "local-gram"
    assert t.noise("cpu-ngsolve", 3, "H1", 2, 1000) is None
    assert t.prior("cpu-ngsolve", 2, "H1", 1, 3, 1e4)["solver"] == "cg-local"


def test_gaussian_follows_tuning(tmp_path):
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"tuning": _recs(tmp_path)})
    assert g.plan["noise"] == "quadrature" and g.plan["noise_choice"] == "tuning"
    assert g.plan["solver"] == "ngsolve-cg(local)" and g.plan["solver_choice"] == "tuning"
    plain = ngsrng.Gaussian(fes, ngsrng.matern(3.0))
    assert plain.plan["tuning"] == "built-in rules" and plain.plan["noise"] == "local-gram-explicit"
    assert g.plan["tuning"].startswith("noise: file calibration (run test)")


def test_install_and_active(tmp_path, monkeypatch):
    dest = tmp_path / "active.json"
    monkeypatch.setenv("NGSRNG_TUNING", str(dest))
    tuning.install(_recs(tmp_path))
    assert tuning.active().run == "test"


def test_tables_with_fractional_powers(tmp_path):
    """The calibration report writes the power as a float when a run had fractional powers ("2.0", "2.6")."""
    data = json.loads(open(_recs(tmp_path)).read())
    rec = data["prior"]["cpu-ngsolve"].pop("2|H1|2|2|1000")
    data["prior"]["cpu-ngsolve"]["2|H1|2|2.0|1000"] = rec
    data["prior"]["cpu-ngsolve"]["2|H1|2|2.6|1000"] = {**rec, "solver": "frac"}
    path = tmp_path / "float.json"
    path.write_text(json.dumps(data))
    t = tuning.Tuning.load(str(path))
    assert t.prior("cpu-ngsolve", 2, "H1", 2, 2, 1e3)["solver"] == "cg-local"
    assert t.prior("cpu-ngsolve", 2, "H1", 2, 2.5, 1e3)["solver"] == "frac"


def test_malformed_tables_are_rejected(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"prior": {"cpu-ngsolve": {"2|H1|2|1000": {"method": {}}}}}))
    with pytest.raises(ValueError, match="unreadable prior entry"):
        tuning.Tuning.load(str(path))
    monkeypatch.setenv("NGSRNG_TUNING", str(path))
    with pytest.warns(UserWarning, match="ignoring the installed tuning"):
        assert tuning.active() is None
    tuning._ACTIVE.clear()


def test_shipped_default_and_precedence(tmp_path, monkeypatch):
    """A fresh clone uses the shipped calibration; an installed calibration overrides it; NGSRNG_TUNING=none
    disables tuning (built-in rules only)."""
    monkeypatch.delenv("NGSRNG_TUNING", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    t = tuning.active()
    assert t is not None and t.source == tuning.SHIPPED and t.label == "shipped"
    gpu = "cuda-f64@NVIDIA A100-PCIE-40GB"  # a GPU table applies only on the device it was measured on
    assert {"cpu-ngsolve", gpu} <= set(t.data["prior"]) and {"cpu-ngsolve", gpu} <= set(t.data["noise"])
    assert t.covers("cpu-ngsolve") == "shipped" and t.covers("rocm-f64@AMD Radeon RX 7900 XTX") is None
    dest = tuning.install(_recs(tmp_path))
    t = tuning.active()
    assert dest == tuning.default_path() and t.run == "test" and t.label == "installed"
    assert t.covers("cpu-ngsolve") == "installed" and t.covers(gpu) == "shipped"  # layered over the shipped one
    monkeypatch.setenv("NGSRNG_TUNING", "none")
    assert tuning.active() is None
    tuning.install(_recs(tmp_path))  # "none" is not a file name: installs where calibrations go
    assert (tmp_path / "cache" / "ngsrng" / "recommendations.json").exists()


def test_shipped_default_is_usable():
    """Every shipped choice is one this version accepts (exact noise, offered solvers), and Gaussians follow it."""
    from ngsrng.gaussian import NOISE_METHODS, SOLVERS
    from ngsrng.solvers import INVERSES, PRECONDITIONERS

    t = tuning.Tuning.load(tuning.SHIPPED)
    for table in t.data["noise"].values():
        assert all(v["method"] in NOISE_METHODS and v["method"] not in ("chebyshev", "lumped") for v in table.values())
    for table in t.data["prior"].values():
        for v in table.values():
            m = v["method"]
            assert m["solver"] in SOLVERS and m.get("inverse", "umfpack") in INVERSES
            assert m.get("precond", "bddc") in PRECONDITIONERS
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.03)), order=1)  # within a decade of 1e4 dofs
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"tuning": tuning.SHIPPED})
    assert g.plan["noise_choice"] == "tuning" and g.plan["solver_choice"] == "tuning"
    assert g.plan["solver"] == "ngsolve-direct(sparsecholesky)"


def _table(tmp_path, noise=None, prior=None, name="t.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"run": "t", "noise": {"cpu-ngsolve": noise or {}},
                                "prior": {"cpu-ngsolve": prior or {}}}))
    return str(path)


def test_lookup_is_bounded(tmp_path):
    """An entry reaches one order and one decade of size beyond its calibrated case, no further: the built-in rules
    decide for cases nothing was calibrated near."""
    t = tuning.Tuning.load(_recs(tmp_path))
    assert t.noise("cpu-ngsolve", 2, "H1", 3, 2000) == "quadrature"  # order 3 from 2, size 2000 from 1000
    assert t.noise("cpu-ngsolve", 2, "H1", 4, 2000) is None  # two orders away
    assert t.noise("cpu-ngsolve", 2, "H1", 2, 5e4) is None  # more than a decade from 1e3 and from 1e6
    assert t.prior("cpu-ngsolve", 2, "H1", 2, 2, 2e4) is None


def test_tuned_choices_yield_to_the_rules(tmp_path, monkeypatch):
    """A calibration only suggests: a solver unavailable on this engine, or a sampler that fails to build, gives way
    to the built-in rule, and the plan says why."""
    from ngsrng import noise as nz

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)
    path = _table(tmp_path, noise={"2|H1|2|100": {"method": "quadrature"}},
                  prior={"2|H1|2|2|100": {"method": {"solver": "cudss", "engine": "ngsolve"}}})
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"tuning": path})
    assert g.plan["solver"] == "ngsolve-direct(sparsecholesky)" and "solver_choice" not in g.plan
    assert "cudss is not available here" in g.plan["tuning"]
    assert g.plan["noise"] == "quadrature" and g.plan["noise_choice"] == "tuning"

    def fail(*a, **k):
        raise MemoryError("quadrature operator would have too many nonzeros")

    monkeypatch.setattr(nz, "quadrature_noise", fail)
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"tuning": path})
    assert g.plan["noise"] == "local-gram-explicit" and "quadrature failed: MemoryError" in g.plan["tuning"]


def test_notice_once_per_session(tmp_path, monkeypatch):
    """An uncalibrated machine is told once per terminal session how to calibrate; never when NGSRNG_QUIET or
    NGSRNG_TUNING is set, or once this machine's calibration covers the platform."""
    import tempfile
    import warnings

    monkeypatch.delenv("NGSRNG_TUNING", raising=False)
    monkeypatch.delenv("NGSRNG_QUIET", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(tuning, "_NOTICED", False)
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=1)
    with pytest.warns(tuning.CalibrationNotice, match="python -m ngsrng.calibrate"):
        ngsrng.Gaussian(fes, ngsrng.matern(3.0))

    def quiet():
        with warnings.catch_warnings():
            warnings.simplefilter("error", tuning.CalibrationNotice)
            ngsrng.Gaussian(fes, ngsrng.matern(3.0))

    quiet()  # the same process
    monkeypatch.setattr(tuning, "_NOTICED", False)
    quiet()  # another process of the same session: the session marker
    if tuning._session_marker() is not None:
        for f in tmp_path.glob("ngsrng-notice-*"):
            f.unlink()
    monkeypatch.setattr(tuning, "_NOTICED", False)
    monkeypatch.setenv("NGSRNG_QUIET", "1")
    quiet()
    monkeypatch.delenv("NGSRNG_QUIET")
    tuning.install(_table(tmp_path, noise={"2|H1|1|100": {"method": "local-gram-explicit"}}))
    quiet()  # calibrated: nothing to say


@pytest.mark.skipif(not ngsrng.solvers.pyamg_available(), reason="needs pyamg")
def test_amg_pcg_matches_the_direct_solver():
    """AMG-preconditioned CG (the GPU solver where cuDSS does not exist) reproduces the direct samples, and is the
    torch engine's rule without cuDSS (Jacobi PCG stays for mass solves)."""
    import torch

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.05)), order=2)
    ref = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", method={"tuning": False})
    eta = torch.randn(ref.nrand, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    g = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64",
                        method={"engine": "torch", "tuning": False}, tol=1e-12)
    assert g.plan["solver"] == "torch-pcg(amg)"
    c, c0 = g.apply(eta).numpy(), ref.apply(eta).numpy()
    assert abs(c - c0).max() < 1e-9 * abs(c0).max()
    assert isinstance(g._make_solver("amg-pcg", ngsrng.space.mass_form, torch.float64), ngsrng.solvers.TorchPCG)


@pytest.mark.skipif(not ngsrng.solvers.pyamg_available(), reason="needs pyamg")
def test_amg_pcg_leaves_numpy_global_state_alone():
    """pyamg draws its spectral-radius start vectors from np.random: the AMG solver neither depends on the user's
    np.random.seed nor advances the user's np.random stream."""
    import numpy as np
    import torch

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.1)), order=2)

    def build(np_seed):
        np.random.seed(np_seed)
        g = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", method={"engine": "torch", "tuning": False})
        assert g.plan["solver"] == "torch-pcg(amg)"
        eta = torch.randn(g.nrand, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        return g.apply(eta).numpy(), np.random.rand()

    (x0, after0), (x1, _) = build(0), build(1)
    np.random.seed(0)
    assert after0 == np.random.rand()  # the user's stream did not move
    np.testing.assert_array_equal(x0, x1)  # and did not matter


def _fake_gpu_machine(monkeypatch, *, hip=None, cuda=None, amd=True, probes=None):
    """Simulate a machine with a GPU device torch cannot use: torch build, device nodes, fresh-process probes."""
    import os

    import torch

    monkeypatch.setattr("ngsrng.backend.cuda_usable", lambda: False)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    real = os.path.exists
    nodes = {"/dev/kfd": amd, "/dev/nvidiactl": not amd, "/dev/nvidia0": not amd}
    monkeypatch.setattr(os.path, "exists", lambda p: nodes[p] if p in nodes else real(p))
    if probes is not None:
        monkeypatch.setattr(tuning, "_probe", lambda *first: probes[bool(first)])


def test_gpu_diagnosis_names_the_wrong_torch_build(monkeypatch):
    _fake_gpu_machine(monkeypatch)
    assert "CPU-only build" in tuning.gpu_diagnosis()[-1] and "ROCm" in tuning.gpu_diagnosis()[-1]
    _fake_gpu_machine(monkeypatch, cuda="13.0")
    assert "NVIDIA (CUDA) build" in tuning.gpu_diagnosis()[-1]


def test_gpu_diagnosis_finds_the_import_order_conflict(monkeypatch):
    ok = {"available": True, "count": 1, "error": None}
    bad = {"available": False, "count": 0, "error": "RuntimeError: no HIP GPUs are available"}
    _fake_gpu_machine(monkeypatch, hip="7.0", probes={False: ok, True: bad})
    assert "not when NGSolve is imported first" in tuning.gpu_diagnosis()[-1]
    _fake_gpu_machine(monkeypatch, hip="7.0", probes={False: bad, True: bad})
    last = tuning.gpu_diagnosis()[-1]
    assert "even on its own" in last and "render and video" in last


def test_gpu_probe_runs():
    """The fresh-process probe itself (on this machine, whatever it finds)."""
    out = tuning._probe("ngsolve")
    assert set(out) == {"available", "count", "error"}
