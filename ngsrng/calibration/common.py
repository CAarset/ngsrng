import platform
import time

import ngsolve as ngs
import numpy as np
import torch
from ngsolve.meshes import Make1DMesh, MakeStructured2DMesh, MakeStructured3DMesh


def structured_mesh(dim: int, n: int):
    if dim == 1:
        return Make1DMesh(n)
    if dim == 2:
        return MakeStructured2DMesh(quads=False, nx=n, ny=n)
    return MakeStructured3DMesh(hexes=False, nx=n, ny=n, nz=n)


def mesh_size_for(dim: int, kind: str, order: int, target_ndof: float) -> int:
    """Number of cells per direction giving roughly ``target_ndof`` dofs on a structured simplicial mesh."""
    if kind == "H1":
        return max(2, round(target_ndof ** (1 / dim) / order))
    simplices = {1: 1, 2: 2, 3: 6}[dim]
    local = {1: order + 1, 2: (order + 1) * (order + 2) // 2, 3: (order + 1) * (order + 2) * (order + 3) // 6}[dim]
    return max(2, round((target_ndof / (simplices * local)) ** (1 / dim)))


def make_space(dim, kind, order, target_ndof, **kw):
    mesh = structured_mesh(dim, mesh_size_for(dim, kind, order, target_ndof))
    return (ngs.H1 if kind == "H1" else ngs.L2)(mesh, order=order, **kw)


def timed(fn, repeat=3, sync=None):
    times, out = [], None
    for _ in range(repeat):
        if sync:
            sync()
        t = time.perf_counter()
        out = fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t)
    return float(np.median(times)), out


def machine_info() -> dict:
    return {
        "host": platform.node(),
        "torch_threads": torch.get_num_threads(),
        "ngsolve": ngs.__version__,
        "torch": torch.__version__,
    }
