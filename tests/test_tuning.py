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
    assert "tuning" not in plain.plan and plain.plan["noise"] == "local-gram-explicit"


def test_install_and_active(tmp_path, monkeypatch):
    dest = tmp_path / "active.json"
    monkeypatch.setenv("NGSRNG_TUNING", str(dest))
    tuning.install(_recs(tmp_path))
    assert tuning.active().run == "test"


def test_tables_with_fractional_powers(tmp_path):
    """bench/report.py writes the power as a float when the run had fractional powers ("2.0", "2.6")."""
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
