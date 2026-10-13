#!/usr/bin/env python3
"""mesh_to_tower - turn a 3D model (.fbx / .obj / .glb ...) into ONE structure of Connectors on a red Base.

  1. the surface is sampled with the model's own colours (albedo texture through the UVs, or the
     material colour) and voxelised on a cubic grid whose cell is one Connector (0.4);
  2. the grid pitch is raised until the structure fits `--max` connectors (default 4900);
  3. loose pieces are tied together with the shortest face-to-face bridges (a tree needs a connected set);
  4. a spanning tree over face neighbours gives the attachments: every Connector is plugged by its ball into
     the cup of its parent that faces it (rtg_shape knows where every cube then sits);
  5. the lowest voxel is plugged on the top cup of a Base part, painted red; every other Connector gets the
     average colour of the model inside its voxel.

    python3 mesh_to_tower.py model.fbx -o tower_code.txt [--textures DIR] [--max 4900] [--up y]
"""
import argparse
import base64
import json
import os
from collections import deque

import numpy as np
from scipy import ndimage

from rtg_shape import CUPS, cup_normal, place

SKIN = True
NEIGH = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


# ------------------------------------------------------------------ model loading
def load_meshes(path, tex_dir):
    """-> list of (vertices (n,3), faces (m,3), uv (n,2) or None, texture array (h,w,3) or colour (3,))"""
    import assimp_py
    from PIL import Image
    sc = assimp_py.import_file(path, assimp_py.Process_Triangulate)
    out = []
    for m in sc.meshes:
        v = np.array(m.vertices, float).reshape(-1, 3)
        f = np.array(m.indices, int).reshape(-1, 3)
        uv = np.array(m.texcoords[0], float).reshape(-1, 2) if m.texcoords and len(m.texcoords[0]) else None
        mat = sc.materials[m.material_index]
        tex = np.array(mat.get("COLOR_DIFFUSE", [0.6, 0.6, 0.6])[:3], float) * 255
        name = mat.get("NAME", "")
        for cand in (f"{name}_albedo", f"{name}_diffuse", f"{name}_basecolor", f"{name}_BaseColor", name):
            for ext in (".jpg", ".png", ".jpeg"):
                p = os.path.join(tex_dir or "", cand + ext)
                if tex_dir and os.path.exists(p) and uv is not None:
                    tex = np.array(Image.open(p).convert("RGB"))
                    break
            else:
                continue
            break
        out.append((v, f, uv, tex, name))
    return out


def sample_surface(meshes, per_area, rng, cap=4_000_000):
    """Random points on every triangle with their colours -> (points (N,3), colours (N,3))."""
    P, C = [], []
    areas = []
    for v, f, *_ in meshes:
        a = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 2
        areas.append(a)
    total = sum(a.sum() for a in areas)
    scale = min(per_area, cap / total)
    for (v, f, uv, tex, _), a in zip(meshes, areas):
        n = rng.poisson(a * scale)                                   # points per triangle
        tri = np.repeat(np.arange(len(f)), n)
        r1, r2 = rng.random(len(tri)), rng.random(len(tri))
        flip = r1 + r2 > 1
        r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
        b = np.stack([1 - r1 - r2, r1, r2], 1)
        P.append(np.einsum("ij,ijk->ik", b, v[f[tri]]))
        if uv is not None and tex.ndim == 3:
            q = np.einsum("ij,ijk->ik", b, uv[f[tri]])
            h, w, _ = tex.shape
            x = np.clip((q[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
            y = np.clip((1 - (q[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(int)
            C.append(tex[y, x].astype(float))
        else:
            C.append(np.tile(np.asarray(tex, float).reshape(1, 3), (len(tri), 1)))
    return np.concatenate(P), np.concatenate(C)


# ------------------------------------------------------------------ voxels
def voxelise(P, C, pitch):
    lo = P.min(0)
    idx = np.floor((P - lo) / pitch).astype(int)
    shape = tuple(idx.max(0) + 1)
    flat = np.ravel_multi_index(idx.T, shape)
    uniq, inv = np.unique(flat, return_inverse=True)
    cnt = np.bincount(inv)
    col = np.stack([np.bincount(inv, C[:, k]) / cnt for k in range(3)], 1)
    cells = np.array(np.unravel_index(uniq, shape)).T
    return {tuple(c): col[i] for i, c in enumerate(cells)}, shape, lo


def skin(vox):
    """Sealed hollow skin of exactly one cube thickness.
    The sampled surface cells are closed (diagonal gaps sealed), the inside is filled, and only the solid
    cells that touch the outside through a face are kept: every row of the skin is one Connector wide and
    the surface has no holes.  New cells take the colour of the nearest sampled cell."""
    from scipy.spatial import cKDTree
    cells = np.array(list(vox))
    lo = cells.min(0) - 3
    shp = tuple(cells.max(0) - lo + 4)
    g = np.zeros(shp, bool)
    g[tuple((cells - lo).T)] = True
    cube = np.ones((3, 3, 3), bool)
    g = ndimage.binary_closing(g, cube, iterations=1, border_value=0)        # seal diagonal pin-holes
    cap = g.copy()
    cap[:, 0:2, :] = True                                                    # a temporary floor so open bottoms can fill
    solid = ndimage.binary_fill_holes(cap)
    solid[:, 0:2, :] = g[:, 0:2, :]
    shell = solid & ~ndimage.binary_erosion(solid, ndimage.generate_binary_structure(3, 1), border_value=0)
    new = np.argwhere(shell)
    tree = cKDTree(cells)
    out = {}
    for c in new:
        k = tuple(c + lo)
        out[k] = vox[k] if k in vox else vox[tuple(cells[tree.query(c + lo)[1]])]
    return out


def connect(vox):
    """Add the fewest bridging voxels so the set is one 6-connected piece (colour = nearest neighbour)."""
    st = ndimage.generate_binary_structure(3, 1)
    while True:
        cells = np.array(list(vox))
        lo, hi = cells.min(0) - 1, cells.max(0) + 2
        grid = np.zeros(tuple(hi - lo), bool)
        grid[tuple((cells - lo).T)] = True
        lab, n = ndimage.label(grid, st)
        if n == 1:
            return vox
        sizes = np.bincount(lab.ravel())[1:]
        main = int(np.argmax(sizes)) + 1
        # BFS through empty cells from the main piece until any other piece is reached
        prev = {}
        dq = deque()
        for c in map(tuple, np.argwhere(lab == main)):
            prev[c] = None
            dq.append(c)
        hit = None
        while dq and hit is None:
            c = dq.popleft()
            for d in NEIGH:
                nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if nb in prev or min(nb) < 0 or any(nb[i] >= grid.shape[i] for i in range(3)):
                    continue
                prev[nb] = c
                if lab[nb] not in (0, main):
                    hit = nb
                    break
                dq.append(nb)
        path = []
        c = prev[hit]
        while c is not None and lab[c] != main:
            path.append(c)
            c = prev[c]
        for c in path:
            g = tuple(np.array(c) + lo)
            near = min(vox, key=lambda k: sum((k[i] - g[i]) ** 2 for i in range(3)))
            vox[g] = vox[near]


def fit(P, C, limit, start, log=print):
    """Smallest pitch >= start whose connected voxel set has at most `limit` cells."""
    pitch = start
    while True:
        vox, _, lo = voxelise(P, C, pitch)
        if SKIN:
            vox = skin(vox)
        n0 = len(vox)
        if n0 <= limit:
            connect(vox)
            if len(vox) <= limit:
                log(f"pitch {pitch:.4f}: {n0} surface voxels, {len(vox)} after bridging")
                return vox, lo, pitch
        pitch *= 1.02


# ------------------------------------------------------------------ tree + code
def build(vox, base_rgb=(220, 30, 30)):
    cells = list(vox)
    ys = [c[1] for c in cells]
    low = [c for c in cells if c[1] == min(ys)]
    cx = np.mean([c[0] for c in cells]), np.mean([c[2] for c in cells])
    root = min(low, key=lambda c: (c[0] - cx[0]) ** 2 + (c[2] - cx[1]) ** 2)
    order, parent = [root], {root: None}
    dq = deque([root])
    while dq:                                    # breadth first spanning tree over face neighbours
        c = dq.popleft()
        for d in NEIGH:
            nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if nb in vox and nb not in parent:
                parent[nb] = (c, d)
                order.append(nb)
                dq.append(nb)
    assert len(order) == len(vox), "voxels not connected"
    index = {c: i + 2 for i, c in enumerate(order)}          # part 1 = the Base, parts 2.. = connectors
    R, p = {root: np.eye(3)}, {root: np.zeros(3)}
    parts = [["Base", [], {"RGB": list(base_rgb)}]]
    used = set()
    for c in order:
        rgb = [int(round(x)) for x in vox[c]]
        if parent[c] is None:
            parts.append(["Connector", [["5", "5", 1]], {"RGB": rgb}])           # ball 5 on the Base's top cup 5
            continue
        (pc, d) = parent[c]
        dvec = np.array(d, float)
        cup = max(CUPS, key=lambda k: float(cup_normal(R[pc], k) @ dvec))
        assert float(cup_normal(R[pc], cup) @ dvec) > 0.99 and (pc, cup) not in used
        used.add((pc, cup))
        R[c], p[c], _ = place(R[pc], p[pc], cup, {})
        parts.append(["Connector", [["5", str(cup), index[pc]]], {"RGB": rgb}])
    return parts, order, p


def write(parts, path):
    code = base64.b64encode(json.dumps(parts, separators=(",", ":")).encode()).decode()
    open(path, "w").write(code + "\n")
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-o", "--out", default="tower_code.txt")
    ap.add_argument("--textures", default=None)
    ap.add_argument("--max", type=int, default=4900)
    ap.add_argument("--up", default="y", help="axis of the model that points up: x, y or z")
    ap.add_argument("--rough", action="store_true", help="keep the raw sampled surface cells instead of the sealed 1-cube skin")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    global SKIN
    SKIN = not a.rough
    meshes = load_meshes(a.model, a.textures or os.path.join(os.path.dirname(a.model), "..", "textures"))
    perm = {"y": [0, 1, 2], "z": [0, 2, 1], "x": [1, 0, 2]}[a.up]
    meshes = [(v[:, perm], f, uv, t, n) for v, f, uv, t, n in meshes]
    rng = np.random.default_rng(a.seed)
    allv = np.concatenate([m[0] for m in meshes])
    size = allv.max(0) - allv.min(0)
    area = sum(np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum() / 2
               for v, f, *_ in meshes)
    start = max(float(np.sqrt(area / a.max)) * 0.8, float(size.max() / 200))   # first guess for the pitch
    pitch = start
    while True:
        P, C = sample_surface(meshes, 12 / pitch ** 2, rng)
        vox, lo, pitch = fit(P, C, a.max, pitch)
        break
    parts, order, pos = build(vox)
    write(parts, a.out)
    ext = np.array([max(c[i] for c in vox) - min(c[i] for c in vox) + 1 for i in range(3)])
    print(f"{len(parts) - 1} connectors + 1 Base, grid {ext[0]} x {ext[1]} x {ext[2]} cubes, "
          f"model size {size.round(1)}, pitch {pitch:.3f} model units per cube")
    return vox, parts


if __name__ == "__main__":
    main()
