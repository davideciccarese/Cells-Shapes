"""
make_integrated.py
==================

The integrated view that puts the two models together. For every interaction it
runs both the 3D cube model and the 2D range-expansion model and draws them side
by side, so the same interaction is seen at once as a cube and as a flat colony.
Each is shown coloured by cell shape (every cell type) and coloured by founding
lineage (every lineage), so all interactions, all cell types and all lineages are
in one figure.

Outputs:
  integrated_panels.png    six interactions (rows) by four columns: cube by shape,
                           cube by lineage, range by shape, range by lineage
  integrated_rangeexp.gif  the two models growing together, cube on top and range
                           below, coloured by lineage, for all six interactions

    python scripts/make_integrated.py            # panels and gif
    python scripts/make_integrated.py panels      # panels only
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch

from config3d import Config3D
import cube3d as C
import render3d as R3

from config_re import ConfigRE
import interactions_re as IRE
import sim_re as SIM
import render_re as RR

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
SHAPE_NAMES = ["cocci", "chain", "rod"]
LEGEND = [Patch(facecolor=R3.MORPH_COLORS[k], edgecolor="#0f172a", label=SHAPE_NAMES[k])
          for k in range(3)]


def _cube_shape_cols(snap):
    return np.array(R3.MORPH_COLORS)[snap.mtype]


def _cube_lineage_cols(snap, pal):
    return pal[snap.lin % len(pal)]


def _draw_cube(ax, snap, L, cols):
    R3._style_axes(ax, L, None)
    R3._cube_edges(ax, L)
    R3._draw_rods(ax, snap, cols, alpha=1.0, edge="#0f172a", edge_lw=0.2, k=4)
    ax.view_init(elev=18, azim=-55)


def run_all():
    ccfg = Config3D(); ccfg.n_frames = 18
    rcfg = ConfigRE()
    pal = R3.lineage_palette(max(ccfg.n_seed, rcfg.n_seed) + 2, seed=2)
    data = []
    for inter3d in C.ALL3D:
        row = inter3d.row
        interre = [x for x in IRE.ALLRE if x.row == row][0]
        cf, _cfh, _cg = C.run(inter3d, ccfg)
        rf, _rfh, _rg = SIM.run(interre, rcfg)
        data.append((inter3d.name, cf, rf))
        print(f"  {row}: cube {cf[-1].pos.shape[0]} cells, range {rf[-1].pos.shape[0]} cells")
    return ccfg, rcfg, pal, data


def panels(ccfg, rcfg, pal, data):
    n = len(data)
    fig = plt.figure(figsize=(15, 3.7 * n), dpi=115)
    fig.patch.set_facecolor("white")
    for ri, (name, cf, rf) in enumerate(data):
        csnap, rsnap = cf[-1], rf[-1]
        ax = fig.add_subplot(n, 4, ri * 4 + 1, projection="3d")
        _draw_cube(ax, csnap, ccfg.cube, _cube_shape_cols(csnap))
        ax.set_title(f"{name}\ncube, by shape", fontsize=10, color="#18181b")
        ax = fig.add_subplot(n, 4, ri * 4 + 2, projection="3d")
        _draw_cube(ax, csnap, ccfg.cube, _cube_lineage_cols(csnap, pal))
        ax.set_title("cube, by lineage", fontsize=10, color="#18181b")
        axr = fig.add_subplot(n, 4, ri * 4 + 3)
        RR.draw_shape(axr, rsnap, rcfg)
        axr.set_title("range expansion, by shape", fontsize=10, loc="left")
        axr = fig.add_subplot(n, 4, ri * 4 + 4)
        RR.draw_lineage(axr, rsnap, rcfg)
        axr.set_title("range expansion, by lineage", fontsize=10, loc="left")
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("Integrated view: every interaction as a 3D cube and as a 2D "
                 "range expansion, coloured by cell shape and by founding lineage",
                 fontsize=14, y=0.998)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.965, bottom=0.03,
                        wspace=0.05, hspace=0.18)
    out = os.path.join(FIGDIR, "integrated_panels.png")
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print("wrote", out)


def gif(ccfg, rcfg, pal, data, nshow=9):
    ncf = len(data[0][1])
    nrf = len(data[0][2])
    cidx = np.linspace(0, ncf - 1, nshow).round().astype(int)
    ridx = np.linspace(0, nrf - 1, nshow).round().astype(int)
    images = []
    for f in range(nshow):
        fig = plt.figure(figsize=(15, 5.4), dpi=80)
        fig.patch.set_facecolor("white")
        for k, (name, cf, rf) in enumerate(data):
            ax = fig.add_subplot(2, 6, k + 1, projection="3d")
            cs = cf[cidx[f]]
            _draw_cube(ax, cs, ccfg.cube, _cube_lineage_cols(cs, pal))
            ax.set_title(name.split(" (")[0], fontsize=8, color="#18181b")
            axr = fig.add_subplot(2, 6, k + 7)
            RR.draw_lineage(axr, rf[ridx[f]], rcfg)
        fig.text(0.012, 0.74, "cube", rotation=90, va="center", fontsize=11)
        fig.text(0.012, 0.27, "range", rotation=90, va="center", fontsize=11)
        fig.suptitle(f"Integrated growth, cells by founding lineage  "
                     f"(step {f+1}/{nshow})", fontsize=13, y=0.99)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.02,
                            wspace=0.05, hspace=0.12)
        images.append(RR.fig_to_image(fig))
        plt.close(fig)
    out = os.path.join(FIGDIR, "integrated_rangeexp.gif")
    RR.save_gif(images, out, fps=4)
    print("wrote", out)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    ccfg, rcfg, pal, data = run_all()
    panels(ccfg, rcfg, pal, data)
    if mode != "panels":
        gif(ccfg, rcfg, pal, data)


if __name__ == "__main__":
    main()
