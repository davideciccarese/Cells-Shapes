"""
render_shapes.py
================

Animate, for one interaction, the full 3 by 3 matrix of cell-shape pairings:
the shape of partner A on the rows, the shape of partner B on the columns. Every
colony is coloured by cell shape (blue cocci, yellow long-rod chains, red short
rods), so the grid shows how shape and interaction together set colony form.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from render3d import _style_axes, _cube_edges, _draw_rods, MORPH_COLORS, MORPH_NAMES


def shape_matrix_gif(inter, grid, cfg, path, fps=7, k=4):
    """grid[a][b] is the list of frames for A-shape a, B-shape b."""
    L = cfg.cube
    nf = cfg.n_frames
    fig = plt.figure(figsize=(9.6, 10.2), dpi=92)
    fig.patch.set_facecolor("white")
    axes = [[fig.add_subplot(3, 3, a * 3 + b + 1, projection="3d")
             for b in range(3)] for a in range(3)]
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.075,
                        wspace=0.0, hspace=0.04)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                          markerfacecolor=MORPH_COLORS[c], markeredgecolor="none",
                          label=MORPH_NAMES[c]) for c in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.01),
               title="colour = cell shape")

    imgs = []
    for fi in range(nf):
        azim = -55 + 70 * (fi / max(nf - 1, 1))
        for a in range(3):
            for b in range(3):
                ax = axes[a][b]
                ax.clear()
                _style_axes(ax, L, None)
                _cube_edges(ax, L)
                s = grid[a][b][fi]
                cols = np.array(MORPH_COLORS)[s.mtype]
                _draw_rods(ax, s, cols, alpha=1.0, edge="#0f172a",
                           edge_lw=0.18, k=k)
                ax.view_init(elev=15, azim=azim)
                if a == 0:
                    ax.set_title("B: " + MORPH_NAMES[b], fontsize=10.5,
                                 color=MORPH_COLORS[b], weight="bold")
                if b == 0:
                    ax.text2D(-0.13, 0.5, "A: " + MORPH_NAMES[a],
                              transform=ax.transAxes, rotation=90,
                              va="center", ha="right", fontsize=10.5,
                              color=MORPH_COLORS[a], weight="bold")
        ra, rb = getattr(inter, "roles", ("A", "B"))
        fig.suptitle(inter.name
                     + "\nrows = A (" + ra + ")     columns = B (" + rb + ")",
                     fontsize=11.5, color="#18181b", y=0.98)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(Image.fromarray(buf[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path
