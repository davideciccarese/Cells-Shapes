"""
render3d.py
===========

Render the 3D cube colonies as animated GIFs.

Cells are drawn as their spine segments (thin lines), so the colony reads as a
see-through wireframe of rods rather than a solid block: you can look straight
through it. Three views per interaction:

  strain    cells coloured blue (A) / green (B)
  nutrient  the substrate as a translucent 3D cloud (bright = high), with the
            colony carving out a low-nutrient cavity, cells faint on top
  lineage   cells coloured by founder lineage, so the floor founders and the
            columns they grow into are distinguishable

Everything uses matplotlib 3D so it has no heavy dependencies.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

A_COLOR = "#2563eb"
B_COLOR = "#16a34a"
# cell-shape colours, indexed like MORPHS = [cocci, chain, rod]
MORPH_COLORS = ["#2563eb", "#eab308", "#dc2626"]   # cocci blue, chain yellow, rod red
MORPH_NAMES = ["cocci", "long-rod chain", "short rod"]
_GREEN = LinearSegmentedColormap.from_list(
    "green_glow", ["#04130a", "#0c5a32", "#1f9e57", "#46d98a"])


def lineage_palette(n, seed=1):
    rng = np.random.default_rng(seed)
    base = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(n, 2)))
    rng.shuffle(base)
    return base


def _spines(snap):
    h = 0.5 * snap.L[:, None] * snap.ax
    p = snap.pos - h
    q = snap.pos + h
    return np.stack([p, q], axis=1)          # (n, 2, 3)


def rod_faces(snap, k=8, cap=3):
    """Build the polygon faces of every cell as a true spherocylinder (a rod):
    a k-sided cylinder body with a hemispherical cap of `cap` latitude bands at
    each end. Returns (polys, face_cell) with polys a list of (V,3) vertex
    arrays and face_cell mapping each face to its cell so colours follow."""
    n = snap.pos.shape[0]
    if n == 0:
        return [], np.zeros(0, dtype=int)
    R = np.asarray(snap.R, dtype=float)
    if R.ndim == 0:
        R = np.full(n, float(R))
    Rc = R[:, None, None]                       # per-cell radius (n,1,1)
    u = snap.ax / (np.linalg.norm(snap.ax, axis=1, keepdims=True) + 1e-9)
    ref = np.where(np.abs(u[:, 2:3]) < 0.9,
                   np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(u, ref)
    e1 /= (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
    e2 = np.cross(u, e1)
    th = np.linspace(0, 2 * np.pi, k, endpoint=False)
    rd = (np.cos(th)[None, :, None] * e1[:, None, :]
          + np.sin(th)[None, :, None] * e2[:, None, :])     # (n,k,3) unit radial
    half = 0.5 * snap.L[:, None] * u
    p0 = (snap.pos - half)[:, None, :]      # bottom cap centre
    p1 = (snap.pos + half)[:, None, :]      # top cap centre
    uu = u[:, None, :]

    def nxt(X):
        return np.roll(X, -1, axis=1)

    # cap latitude rings (per-cell radius); l = 0..cap (l = cap is the pole)
    phis = (np.arange(cap + 1) / cap) * (np.pi / 2)
    cosl = np.cos(phis)
    sinl = np.sin(phis)

    quads, tris = [], []
    qcell, tcell = [], []

    # cylinder body
    Rb = p0 + Rc * rd
    Rt = p1 + Rc * rd
    quads.append(np.stack([Rb, nxt(Rb), nxt(Rt), Rt], axis=2))   # (n,k,4,3)
    qcell.append(np.repeat(np.arange(n), k))

    for (centre, sgn) in ((p1, +1.0), (p0, -1.0)):
        prev = centre + (Rc * cosl[0]) * rd + sgn * (Rc * sinl[0]) * uu
        for l in range(1, cap + 1):
            if l < cap:
                cur = centre + (Rc * cosl[l]) * rd + sgn * (Rc * sinl[l]) * uu
                quads.append(np.stack([prev, nxt(prev), nxt(cur), cur], axis=2))
                qcell.append(np.repeat(np.arange(n), k))
                prev = cur
            else:
                pole = np.broadcast_to(centre + sgn * Rc * uu, (n, k, 3))
                tris.append(np.stack([prev, nxt(prev), pole], axis=2))
                tcell.append(np.repeat(np.arange(n), k))

    polys, face_cell = [], []
    if quads:
        Q = np.concatenate([q.reshape(-1, 4, 3) for q in quads], axis=0)
        polys.extend(list(Q))
        face_cell.append(np.concatenate(qcell))
    if tris:
        T = np.concatenate([t.reshape(-1, 3, 3) for t in tris], axis=0)
        polys.extend(list(T))
        face_cell.append(np.concatenate(tcell))
    return polys, np.concatenate(face_cell)


def _draw_rods(ax, snap, cell_colors, alpha=1.0, edge="#0f172a",
               edge_lw=0.25, k=6, shade=True):
    if snap.pos.shape[0] == 0:
        return
    polys, fc_idx = rod_faces(snap, k=k, cap=4)
    rgba = to_rgba_array(cell_colors)
    rgba = rgba[fc_idx]
    rgba[:, 3] = alpha
    if shade and len(polys):
        # ambient + diffuse + specular shading so the spherocylinders read as
        # glossy rounded solids with a soft highlight, not flat patches
        P = np.array([p[:3] for p in polys])               # first 3 verts/face
        nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        light = np.array([0.35, 0.2, 0.92])
        light /= np.linalg.norm(light)
        half = light + np.array([0.0, 0.0, 1.0])
        half /= np.linalg.norm(half)
        diff = np.abs(nrm @ light)
        spec = np.clip(np.abs(nrm @ half), 0, 1) ** 28
        bright = 0.55 + 0.42 * diff
        rgb = rgba[:, :3] * bright[:, None] + 0.30 * spec[:, None]
        rgba[:, :3] = np.clip(rgb, 0, 1)
    pc = Poly3DCollection(polys, facecolors=rgba, linewidths=edge_lw)
    if edge is not None:
        pc.set_edgecolor(edge)
    else:
        pc.set_edgecolor(rgba)
    ax.add_collection3d(pc)


def _cube_edges(ax, L, color="#9ca3af"):
    r = [0, L]
    pts = np.array([[x, y, z] for x in r for y in r for z in r])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    segs = [[pts[a], pts[b]] for a, b in edges]
    lc = Line3DCollection(segs, colors=color, linewidths=0.8, alpha=0.5)
    ax.add_collection3d(lc)


def _style_axes(ax, L, title):
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_facecolor("white")
    try:
        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((0.96, 0.97, 0.98, 1.0))
    except Exception:
        pass
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=12.5, color="#18181b", pad=2)


def _draw_cells(ax, snap, colors, lw=1.4, alpha=0.85):
    if snap.pos.shape[0] == 0:
        return
    segs = _spines(snap)
    lc = Line3DCollection(segs, colors=colors, linewidths=lw, alpha=alpha)
    ax.add_collection3d(lc)


def _draw_nutrient(ax, field, cfg, snap, norm=None, stride=2):
    """Show the substrate as a filled 3D cloud: every grid node is a translucent
    voxel whose colour and opacity follow its concentration on the fixed colour
    scale. Where the colony has eaten the substrate the cloud goes transparent,
    so a low-nutrient cavity opens up inside the green volume as it grows."""
    N, dx = cfg.N, cfg.dx
    if norm is None:
        norm = plt.Normalize(0.0, cfg.S0)
    sl = slice(0, N, stride)
    sub = field[sl, sl, sl]
    idx = np.arange(0, N, stride)
    X, Y, Z = np.meshgrid(idx * dx, idx * dx, idx * dx, indexing="ij")
    v = sub.ravel()
    vn = np.clip(norm(v), 0, 1)
    rgba = plt.get_cmap("magma")(vn)
    rgba[:, 3] = vn ** 1.2 * 0.42                 # opaque where rich, clear where consumed
    keep = vn > 0.05
    ax.scatter(X.ravel()[keep], Y.ravel()[keep], Z.ravel()[keep],
               c=rgba[keep], s=22, marker="s", edgecolors="none",
               depthshade=False)
    # colony faint, so you see it sitting in the cavity it carves
    if snap.pos.shape[0]:
        segs = _spines(snap)
        lc = Line3DCollection(segs, colors="#0b1220", linewidths=0.7, alpha=0.5)
        ax.add_collection3d(lc)


def multipanel3d_gif(results, cfg, path, fps=7, title=None):
    """results: list of (inter, frames, [(labelA, histA), (labelB, histB)]).
    Builds an N x 4 GIF: rows are interactions, columns are cells-by-strain,
    the field feeding species A, the field feeding species B, and
    cells-by-lineage. Each field is magma-coded relative to its own maximum."""
    import matplotlib as mpl
    nrows = len(results)
    n_frames = cfg.n_frames
    L = cfg.cube
    palettes = [lineage_palette(cfg.n_seed + 2, seed=2 + k)
                for k in range(nrows)]

    fig = plt.figure(figsize=(16.5, 4.3 * nrows), dpi=110)
    fig.patch.set_facecolor("white")
    axes = [[fig.add_subplot(nrows, 4, r * 4 + c + 1, projection="3d")
             for c in range(4)] for r in range(nrows)]
    col_titles = ["Cells by shape (solid)", "Field feeding species A",
                  "Field feeding species B", "Cells by lineage (see-through)"]
    if title:
        fig.suptitle(title, fontsize=17, color="#18181b", y=0.997)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.93, bottom=0.04,
                        wspace=0.0, hspace=0.08)
    import matplotlib.pyplot as _plt
    handles = [_plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                           markerfacecolor=MORPH_COLORS[k], markeredgecolor="none",
                           label=MORPH_NAMES[k]) for k in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.0))

    # each field is normalised to its own maximum over the run (relative scale)
    fnorms = []
    for (inter, frames, flist) in results:
        ns = []
        for (lab, hist) in flist:
            m = max((float(np.max(h)) for h in hist), default=1.0)
            ns.append(plt.Normalize(0.0, max(m, 1e-6)))
        fnorms.append(ns)

    cax = fig.add_axes([0.935, 0.36, 0.012, 0.30])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=plt.Normalize(0, 1),
                                            cmap="magma"), cax=cax)
    cb.set_label("relative concentration\n(dark low, bright high)",
                 fontsize=9, color="#3f3f46")
    cb.ax.tick_params(labelsize=8)

    imgs = []
    max_rf = int(getattr(cfg, "render_frames", 8))
    fidx = sorted(set(np.linspace(0, n_frames - 1,
                                  min(max_rf, n_frames)).astype(int).tolist()))
    for fi in fidx:
        azim = -60 + 80 * (fi / max(n_frames - 1, 1))
        elev = 16
        for r, (inter, frames, flist) in enumerate(results):
            snap = frames[fi]
            shape_cols = np.array(MORPH_COLORS)[snap.mtype]
            lin_cols = palettes[r][snap.lin % len(palettes[r])]
            ax0, ax1, ax2, ax3 = axes[r]
            for a in axes[r]:
                a.clear()

            _style_axes(ax0, L, col_titles[0] if r == 0 else None)
            _cube_edges(ax0, L)
            _draw_rods(ax0, snap, shape_cols, alpha=1.0, edge="#0f172a",
                       edge_lw=0.2, k=4)
            sa, sb = inter.shapes
            ax0.text2D(0.5, 0.02,
                       f"A: {MORPH_NAMES[sa]}   vs   B: {MORPH_NAMES[sb]}",
                       transform=ax0.transAxes, fontsize=9, color="#3f3f46",
                       ha="center", va="bottom")

            for col, ax in ((0, ax1), (1, ax2)):
                _style_axes(ax, L, col_titles[1 + col] if r == 0 else None)
                _cube_edges(ax, L)
                lab, hist = flist[col]
                _draw_nutrient(ax, hist[fi], cfg, snap, norm=fnorms[r][col])
                ax.text2D(0.5, 0.03, lab, transform=ax.transAxes, fontsize=9,
                          color="#3f3f46", ha="center", va="bottom")

            _style_axes(ax3, L, col_titles[3] if r == 0 else None)
            _cube_edges(ax3, L)
            _draw_rods(ax3, snap, lin_cols, alpha=0.42, edge=None, edge_lw=0.15,
                       k=4)

            ax0.text2D(-0.16, 0.5, inter.name, transform=ax0.transAxes,
                       fontsize=12, color="#18181b", rotation=90,
                       va="center", ha="right", weight="bold")
            ax0.view_init(elev=18, azim=azim)
            ax1.view_init(elev=18, azim=azim)
            ax2.view_init(elev=18, azim=azim)
            ax3.view_init(elev=18, azim=azim)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(Image.fromarray(buf[..., :3].copy()))

    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path


def shape_matrix_gif(inter, runs, cfg, path, fps=7):
    """One animated 3x3 panel for a single interaction: partner A's shape on the
    rows, partner B's shape on the columns, every cell coloured by its shape.
    runs is a dict {(a_idx, b_idx): frames}."""
    n_frames = cfg.n_frames
    L = cfg.cube
    fig = plt.figure(figsize=(11.5, 12.0), dpi=110)
    fig.patch.set_facecolor("white")
    axes = [[fig.add_subplot(3, 3, r * 3 + c + 1, projection="3d")
             for c in range(3)] for r in range(3)]
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.05,
                        wspace=0.0, hspace=0.0)
    fig.suptitle(f"{inter.name}: every shape pairing\n"
                 f"rows = partner A shape, columns = partner B shape",
                 fontsize=15, color="#18181b", y=0.985)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                          markerfacecolor=MORPH_COLORS[k], markeredgecolor="none",
                          label=MORPH_NAMES[k]) for k in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.01),
               title="colour = cell shape")

    imgs = []
    for fi in range(n_frames):
        azim = -60 + 80 * (fi / max(n_frames - 1, 1))
        for ra in range(3):           # A shape
            for cb in range(3):       # B shape
                ax = axes[ra][cb]
                ax.clear()
                _style_axes(ax, L, None)
                _cube_edges(ax, L)
                snap = runs[(ra, cb)][fi]
                if snap.pos.shape[0]:
                    cols = np.array(MORPH_COLORS)[snap.mtype]
                    _draw_rods(ax, snap, cols, alpha=1.0, edge="#0f172a",
                               edge_lw=0.15, k=5)
                ax.view_init(elev=15, azim=azim)
                if ra == 0:
                    ax.text2D(0.5, 1.02, f"B: {MORPH_NAMES[cb]}",
                              transform=ax.transAxes, fontsize=11,
                              color="#18181b", ha="center", va="bottom",
                              weight="bold")
                if cb == 0:
                    ax.text2D(-0.04, 0.5, f"A: {MORPH_NAMES[ra]}",
                              transform=ax.transAxes, fontsize=11,
                              color="#18181b", rotation=90, ha="right",
                              va="center", weight="bold")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(Image.fromarray(buf[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path
