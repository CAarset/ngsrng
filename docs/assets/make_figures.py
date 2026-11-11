"""The README's figures: random functions on the bundled test geometries (ngsrng.domains), rendered with ngsplot
(the same pictures as NGSolve's Draw), and the boundary-variance comparison taken from examples/demo.ipynb.

    python docs/assets/make_figures.py
"""

import ngsrng  # noqa: I001  (torch before NGSolve)

import base64
import os
import warnings

import matplotlib.pyplot as plt
import nbformat
import ngsolve as ngs
import numpy as np
from ngsplot import pyplot as ngsplot

from ngsrng import domains

HERE = os.path.dirname(os.path.abspath(__file__))
LOOK = {"settings": {"Objects": {"Wireframe": False}, "Colormap": {"ncolors": 32}}}
warnings.simplefilter("ignore")


def show(ax, field, mesh=None, title=None, width=520, **kw):
    extra = {} if isinstance(field, ngs.GridFunction) else {"mesh": mesh}
    if (field.space.mesh if isinstance(field, ngs.GridFunction) else mesh).dim == 2:
        extra["subdomains"] = False  # no lines between subdomains, only the boundary (2D meshes only)
    ngsplot.Plot(field, ax=ax, colorbar=False, width=width, height=None, fit="box", **extra, **LOOK, **kw)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, color="#24292f", pad=6)


def fields():
    """(field, mesh, title, options) for the banner and the gallery."""
    out = {}
    m = domains.NGSCircle(maxh=0.04).mesh
    r = ngsrng.default_rng(ngs.H1(m, order=2), seed=3, correlation=ngsrng.matern_correlation(0.25, nu=1.0))
    out["gaussian"] = (r.normal(), m, "Gaussian (Matérn)", {})
    m = domains.NGSDoubleHalfPipes(maxh=0.05).mesh
    r = ngsrng.default_rng(ngs.H1(m, order=2), seed=5, correlation=ngsrng.matern_correlation(0.4, nu=1.5))
    out["lognormal"] = (r.lognormal(0.0, 0.6, representation="bernstein"), m, "log-normal", {})
    m = domains.NGSPerforated(maxh=0.03).mesh
    r = ngsrng.default_rng(ngs.H1(m, order=2), seed=2, correlation=ngsrng.matern_correlation(0.2, nu=1.0))
    out["uniform"] = (r.uniform(representation="bernstein"), m, "uniform", {"vmin": 0, "vmax": 1})
    m = domains.NGSLShaped(maxh=0.03).mesh
    V = ngs.H1(m, order=2)
    prior = ngsrng.default_rng(V, seed=1, correlation=ngsrng.matern_correlation(0.25, nu=1.0)).gaussian()
    pts = np.array([(x, y) for x in np.linspace(0.07, 0.93, 6) for y in np.linspace(0.07, 0.93, 6)
                    if not (x > 0.5 and y > 0.5)])
    var = ngsrng.pointwise_variance(prior.condition(points=pts, noise=1e-3))
    out["posterior"] = (ngs.sqrt(ngs.IfPos(var, var, 0)), m, "posterior std (exact)", {})
    m = domains.NGSEllipseInPolygon(maxh=0.04).mesh
    r = ngsrng.default_rng(ngs.H1(m, order=2), seed=7, correlation=ngsrng.matern_correlation(0.3, nu=0.5))
    out["rough"] = (r.normal(), m, "rough (ν = 1/2)", {})
    r = ngsrng.default_rng(ngs.H1(m, order=2), seed=7, correlation=ngsrng.matern_correlation(0.3, nu=3.0))
    out["smooth"] = (r.normal(), m, "smooth (ν = 3)", {})
    m = domains.NGSDoubleSlit(maxh=0.04).mesh
    r = ngsrng.default_rng(ngs.H1(m, order=2), seed=4, correlation=ngsrng.matern_correlation(0.3, nu=1.0))
    out["gamma"] = (r.gamma(2.0, representation="bernstein"), m, "gamma(2)", {})
    cube = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.08))
    r = ngsrng.default_rng(ngs.H1(cube, order=1), seed=9, correlation=ngsrng.matern_correlation(0.3, nu=1.5))
    out["3d"] = (r.normal(), cube, "3D", {"elev": 25, "azim": -55})
    return out


def banner(f):
    fig = plt.figure(figsize=(16, 3.6), facecolor="white")
    grid = fig.add_gridspec(1, 5, width_ratios=[1.35, 1, 1, 1, 1], wspace=0.06)
    text = fig.add_subplot(grid[0])
    text.set_axis_off()
    text.text(0.02, 0.62, "ngsrng", fontsize=46, fontweight="bold", color="#24292f", transform=text.transAxes)
    text.text(0.03, 0.36, "random functions on\nfinite element spaces", fontsize=14, color="#57606a",
              linespacing=1.4, transform=text.transAxes)
    text.text(0.03, 0.14, "NGSolve · numpy API · CPU and GPU", fontsize=11, color="#8c959f",
              transform=text.transAxes)
    for k, key in enumerate(("gaussian", "lognormal", "uniform", "posterior")):
        field, mesh, title, opts = f[key]
        show(fig.add_subplot(grid[k + 1]), field, mesh, title, **opts)
    fig.savefig(os.path.join(HERE, "banner.png"), dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gallery(f):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), facecolor="white")
    for ax, key in zip(axes, ("rough", "smooth", "gamma", "3d")):
        field, mesh, title, opts = f[key]
        show(ax, field, mesh, title, **opts)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(os.path.join(HERE, "gallery.png"), dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def boundary_variance():
    """The demo's comparison of the pointwise variance up to the boundary (Neumann, Robin, torus, restrict)."""
    nb = nbformat.read(os.path.join(HERE, "..", "..", "examples", "demo.ipynb"), 4)
    for cell in nb.cells:
        if cell.cell_type == "code" and "variance_panels(" in cell.source:
            for out in cell.get("outputs", []):
                for mime, ext in (("image/png", "png"), ("image/jpeg", "jpg")):
                    if mime in out.get("data", {}):
                        path = os.path.join(HERE, f"boundary_variance.{ext}")
                        with open(path, "wb") as fh:
                            fh.write(base64.b64decode(out["data"][mime]))
                        return path
    raise RuntimeError("no variance figure found in examples/demo.ipynb")


if __name__ == "__main__":
    f = fields()
    banner(f)
    gallery(f)
    print(boundary_variance())
