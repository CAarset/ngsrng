"""Helpers for the constant-variance section of demo.ipynb: an irregular test domain and variance panels.

``irregular_mesh(maxh)``: a domain with a straight left edge (boundary name "left"), a wavy outer boundary with a bump
pushed in at the bottom and a bulge out at the top (name "outer"), and four non-circular holes -- a star, a rotated
ellipse, a triangle and a kidney shape (name "hole"). Curved to order 3.

``variance_panels(variances, titles, mesh)``: pointwise variance maps side by side on one diverging colour scale,
symmetric around the target 1 on [0.5, 1.5] and sensitive near it (colour intensity proportional to
|variance - 1|^(1/3), so that a 1% departure is already visible).
"""

import matplotlib.pyplot as plt
import ngsolve as ngs
import numpy as np
from matplotlib.colors import FuncNorm, LinearSegmentedColormap
from netgen.occ import Face, MoveTo, OCCGeometry, Pnt, Segment, SplineInterpolation, Wire, X

_CMAP = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4b0a8", "#e34948", "#9e1f1f"])
_CMAP.set_bad("white")


def _curve(xy, periodic=False):
    return SplineInterpolation([Pnt(float(x), float(y), 0) for x, y in xy], periodic=periodic)


def _polygon(xy):
    wp = MoveTo(*xy[0])
    for p in xy[1:]:
        wp = wp.LineTo(*p)
    return wp.Close().Face()


def irregular_mesh(maxh=0.06):
    n = 81
    taper = np.sin(np.pi * np.linspace(0, 1, n)) ** 2  # the waves vanish at the corners: right angles there
    x = np.linspace(0, 1.6, n)
    y = np.linspace(0, 1.0, n)
    bottom = np.c_[x, taper * 0.05 * np.sin(2 * np.pi * x / 0.4) + 0.13 * np.exp(-((x - 1.05) / 0.08) ** 2)]
    right = np.c_[1.6 + taper * 0.06 * np.sin(2 * np.pi * y / 0.25), y]
    top = np.c_[x, 1.0 + taper * 0.05 * np.sin(2 * np.pi * x / 0.4) + 0.16 * np.exp(-((x - 0.8) / 0.13) ** 2)][::-1]
    outer = Face(Wire([_curve(bottom), _curve(right), _curve(top), Segment(Pnt(0, 1, 0), Pnt(0, 0, 0))]))
    outer.edges.name = "outer"
    outer.edges.Min(X).name = "left"
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    a = np.pi / 2 + np.linspace(0, 2 * np.pi, 10, endpoint=False)
    star = _polygon([(0.45 + r * np.cos(b), 0.55 + r * np.sin(b)) for b, r in zip(a, [0.14, 0.06] * 5)])
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    ellipse = Face(Wire(_curve(np.c_[1.05 + c * 0.13 * np.cos(t) - s * 0.06 * np.sin(t),
                                     0.62 + s * 0.13 * np.cos(t) + c * 0.06 * np.sin(t)], periodic=True)))
    triangle = _polygon([(1.3, 0.25), (1.47, 0.3), (1.36, 0.43)])
    r = 0.09 * (1 + 0.35 * np.cos(2 * t) + 0.15 * np.sin(3 * t))
    kidney = Face(Wire(_curve(np.c_[0.8 + r * np.cos(t), 0.33 + r * np.sin(t)], periodic=True)))
    face = outer
    for hole in (star, ellipse, triangle, kidney):
        hole.edges.name = "hole"
        face = face - hole
    mesh = ngs.Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))
    mesh.Curve(3)
    return mesh


def _norm():
    def forward(v):
        d = np.asarray(v, dtype=float) - 1.0
        return np.sign(d) * np.abs(d) ** (1 / 3)

    def inverse(w):
        w = np.asarray(w, dtype=float)
        return 1.0 + np.sign(w) * np.abs(w) ** 3

    return FuncNorm((forward, inverse), vmin=0.5, vmax=1.5)


def _image(mesh, gf, n):
    xyz = np.array([v.point for v in mesh.vertices])
    lo, hi = xyz.min(0), xyz.max(0)
    ny = int(round(n * (hi[1] - lo[1]) / (hi[0] - lo[0])))
    X_, Y_ = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], ny))
    xs, ys = X_.ravel(), Y_.ravel()
    inside = np.asarray(mesh(xs, ys)["nr"]) >= 0
    vals = np.full(xs.size, np.nan)
    vals[inside] = gf(mesh(xs[inside], ys[inside])).ravel()  # evaluate only inside the mesh
    return vals.reshape(X_.shape), (lo[0], hi[0], lo[1], hi[1])


def variance_panels(variances, titles, mesh, n=400, size=3.6):
    variances, titles = list(variances), list(titles)
    fig, axes = plt.subplots(1, len(variances), figsize=(1.55 * size * len(variances), size), squeeze=False)
    norm = _norm()
    for ax, var, title in zip(axes[0], variances, titles):
        img, extent = _image(mesh, var, n)
        im = ax.imshow(img, origin="lower", extent=extent, cmap=_CMAP, norm=norm, interpolation="nearest")
        for el in mesh.Elements(ngs.BND):
            p = np.array([mesh.vertices[v.nr].point for v in el.vertices])
            ax.plot(p[:, 0], p[:, 1], color="#3d4450", lw=0.6)
        ax.set_title(f"{title}\nvariance in [{np.nanmin(img):.3f}, {np.nanmax(img):.3f}]", fontsize=10, loc="left")
        ax.set_aspect("equal")
        ax.axis("off")
    ticks = [0.5, 0.75, 0.9, 0.97, 0.99, 1.0, 1.01, 1.03, 1.1, 1.25, 1.5]
    cb = fig.colorbar(im, ax=list(axes[0]), fraction=0.012, pad=0.01, extend="both", ticks=ticks)
    cb.ax.set_yticklabels([f"{t:g}" for t in ticks])
    cb.set_label("pointwise variance (target 1)")
    plt.show()
