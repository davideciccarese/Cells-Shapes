"""
render_re.py
============

Drawing for the range-expansion model. Every cell is drawn as its true
spherocylinder outline (a stadium shape) with a thin dark edge, so cells of
different radius read crisply and, when only outlined, an underlying field shows
through. Three panel types per interaction:

  strain    A blue, B green
  field     a chosen diffusion field as a heatmap with the cells outlined on top
  lineage   a fixed colour per founding lineage, so the radial sectors of the
            range expansion read as wedges spreading from the central seed

The cells-overlaid-with-contours requirement is met by the field panel: the
metabolite or substrate is shown as the heatmap and the cell capsule contours are
drawn over it.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

A_COLOR = "#2563eb"
B_COLOR = "#16a34a"
EDGE = "#0f172a"

# cell-shape colours, indexed like MORPHS = [cocci, chain, rod].
# Identical to MORPH_COLORS in render3d.py so the shape colour code is the same
# in the 3D cube model, the range-expansion model and the GIFs.
MORPH_COLORS = ["#2563eb", "#eab308", "#dc2626"]   # cocci blue, chain yellow, rod red
MORPH_NAMES = ["cocci", "chain", "rod"]

_GREEN_GLOW = LinearSegmentedColormap.from_list(
    "green_glow", ["#021a0e", "#0a5a30", "#28c172", "#c9ffd9"])
try:
    matplotlib.colormaps.register(_GREEN_GLOW)
except Exception:
    pass


def capsule_polys(snap, ncap=8):
    """Closed stadium outline for every cell, (n, 2*ncap, 2), per-cell radius."""
    n = snap.pos.shape[0]
    if n == 0:
        return np.zeros((0, 2 * ncap, 2))
    h = 0.5 * snap.L[:, None] * snap.ax
    p = snap.pos - h
    q = snap.pos + h
    R = snap.R[:, None]
    d = q - p
    phi = np.arctan2(d[:, 1], d[:, 0])[:, None]
    aq = phi + np.linspace(np.pi / 2, -np.pi / 2, ncap)[None, :]
    ap = phi + np.linspace(-np.pi / 2, -3 * np.pi / 2, ncap)[None, :]
    arc_q = np.stack([q[:, 0, None] + R * np.cos(aq),
                      q[:, 1, None] + R * np.sin(aq)], axis=2)
    arc_p = np.stack([p[:, 0, None] + R * np.cos(ap),
                      p[:, 1, None] + R * np.sin(ap)], axis=2)
    return np.concatenate([arc_q, arc_p], axis=1)


def lineage_colors(snap, seed=1):
    uniq = np.unique(snap.lin)
    rng = np.random.default_rng(seed)
    cmap = plt.get_cmap("tab20")
    cols = cmap((np.arange(len(uniq)) % 20) / 19.0)
    rng.shuffle(cols)
    lut = {int(v): cols[k] for k, v in enumerate(uniq)}
    return np.array([lut[int(v)] for v in snap.lin])


def _setup(ax, box):
    ax.set_xlim(0, box)
    ax.set_ylim(0, box)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d4d4d8")
        s.set_linewidth(0.8)


def draw_strain(ax, snap, cfg, lw=0.4):
    _setup(ax, cfg.box)
    if snap.pos.shape[0] == 0:
        return
    order = np.argsort(snap.R)            # draw small cells last so they stay visible
    polys = capsule_polys(snap)[order]
    cols = np.where(snap.sp[order] == 0, A_COLOR, B_COLOR)
    pc = PolyCollection(polys, facecolors=cols, edgecolors=EDGE,
                        linewidths=lw, alpha=0.95)
    ax.add_collection(pc)


def draw_shape(ax, snap, cfg, lw=0.4):
    """Colour every cell by its morphology, with the shared shape palette
    (cocci blue, chain yellow, rod red)."""
    _setup(ax, cfg.box)
    if snap.pos.shape[0] == 0:
        return
    order = np.argsort(snap.R)
    polys = capsule_polys(snap)[order]
    cols = np.array(MORPH_COLORS)[snap.mtype[order]]
    pc = PolyCollection(polys, facecolors=cols, edgecolors=EDGE,
                        linewidths=lw, alpha=0.95)
    ax.add_collection(pc)


def shape_legend_handles():
    from matplotlib.patches import Patch
    return [Patch(facecolor=MORPH_COLORS[k], edgecolor=EDGE, label=MORPH_NAMES[k])
            for k in range(3)]


def draw_lineage(ax, snap, cfg, lw=0.4, seed=1):
    _setup(ax, cfg.box)
    if snap.pos.shape[0] == 0:
        return
    cols = lineage_colors(snap, seed)
    polys = capsule_polys(snap)
    pc = PolyCollection(polys, facecolors=cols, edgecolors=EDGE,
                        linewidths=lw, alpha=0.97)
    ax.add_collection(pc)


def draw_field(ax, snap, cfg, field2d, cmap="magma", lw=0.35,
               outline="#e5e7eb"):
    _setup(ax, cfg.box)
    im = ax.imshow(field2d.T, origin="lower", extent=[0, cfg.box, 0, cfg.box],
                   cmap=cmap, aspect="equal")
    if snap.pos.shape[0]:
        polys = capsule_polys(snap)
        pc = PolyCollection(polys, facecolors="none", edgecolors=outline,
                            linewidths=lw, alpha=0.8)
        ax.add_collection(pc)
    return im


def fig_to_image(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf[:, :, :3].copy())


def save_gif(images, path, fps=8):
    if not images:
        return
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
