"""torch's batched CPU matmul segfaults when ngsolve is imported before torch.

    python ngsolve_torch_crash.py                # ngsolve first: Segmentation fault (exit code 139)
    python ngsolve_torch_crash.py --torch-first  # torch first: prints "ok"
"""

import sys

if "--torch-first" in sys.argv:
    import torch  # noqa: I001  (the order is the point)
    import ngsolve
else:
    import ngsolve  # loads its bundled OpenBLAS with RTLD_GLOBAL
    import torch

print(f"ngsolve {ngsolve.__version__}, torch {torch.__version__}, Python {sys.version.split()[0]}", flush=True)
a = torch.randn(2, 16, 16, dtype=torch.float64)
torch.bmm(a, a)  # segmentation fault if ngsolve was imported first
print("ok")
