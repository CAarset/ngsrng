"""Import order: ngsolve imported before torch makes torch's batched CPU products crash;
``import ngsrng`` warns then, and is silent when torch (or ngsrng) came first. Each case runs in a fresh process."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("order, warns", [("import ngsolve, ngsrng", True), ("import ngsolve, torch, ngsrng", True),
                                          ("import torch, ngsolve, ngsrng", False), ("import ngsrng, ngsolve", False)])
def test_import_order_warning(order, warns):
    run = subprocess.run([sys.executable, "-W", "always", "-c", order], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    shadowed = subprocess.run(
        [sys.executable, "-c", "import ngsolve, torch; from ngsrng.backend import _batched_gemm_shadowed as s; "
                               "print(s())"], capture_output=True, text=True, timeout=300).stdout.strip() == "True"
    assert ("ImportOrderWarning" in run.stderr) == (warns and shadowed)
