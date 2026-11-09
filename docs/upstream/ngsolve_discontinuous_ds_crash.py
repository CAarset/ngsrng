"""Assembling two ds integrals in one BilinearForm on a Discontinuous space segfaults.

    python ngsolve_discontinuous_ds_crash.py           # u*v*ds + u*v*ds: Segmentation fault (exit code 139)
    python ngsolve_discontinuous_ds_crash.py --one     # a single ds integral: prints "ok"
    python ngsolve_discontinuous_ds_crash.py --split   # the same two integrals in two BilinearForms: prints "ok"
    python ngsolve_discontinuous_ds_crash.py --h1      # the two integrals on the H1 space itself: prints "ok"
"""

import sys

import ngsolve as ngs

print(f"ngsolve {ngs.__version__}, Python {sys.version.split()[0]}", flush=True)
mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.4))
h1 = ngs.H1(mesh, order=1)
fes = h1 if "--h1" in sys.argv else ngs.Discontinuous(h1)
u, v = fes.TnT()

if "--one" in sys.argv:
    forms = [u * v * ngs.ds]
elif "--split" in sys.argv:
    forms = [u * v * ngs.ds, u * v * ngs.ds]
else:
    forms = [u * v * ngs.ds + u * v * ngs.ds]

for form in forms:
    a = ngs.BilinearForm(fes)
    a += form
    a.Assemble()  # segmentation fault for two ds integrals on Discontinuous(H1)
print("ok")
