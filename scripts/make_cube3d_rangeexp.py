"""
make_cube3d_rangeexp.py
=======================

Top-down view of the 3D range expansion, the same object the metrics are computed
from. For each interaction three panels are drawn from the final simulation state:
cells coloured by strain, the metabolite field as a heatmap with the colony on
top, and cells coloured by founding lineage so the radial sectors are visible.

This is a static rendering of the model state (cell positions, strain, lineage,
and the diffusing field), not a separate model. The animated counterpart is the
3D panel GIF produced by make_cube3d.py.

    python scripts/make_cube3d_rangeexp.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config3d import Config3D
import cube3d as C

ROW_TITLE = {"neutralism": "Neutralism (0,0)", "commensalism": "Commensalism (0,+)",
             "amensalism": "Amensalism (0,-)", "public_good": "Public good (-,+)",
             "mutualism": "Mutualism (+,+)", "competition": "Competition (-,-)"}
STRAIN_COLORS = np.array([[0.15, 0.39, 0.92], [0.09, 0.64, 0.29]])  # A blue, B green


def main():
    cfg = Config3D(); cfg.n_frames = 26
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    rng = np.random.default_rng(7)

    n = len(C.ALL3D)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 3.2 * n), dpi=120)
    fig.patch.set_facecolor("white")

    for ri, inter in enumerate(C.ALL3D):
        flabel, fkey = inter.display_fields[0]
        frames, fh, gen = C.run(inter, cfg, capture_fields=[fkey])
        snap = frames[-1]
        x, y = snap.pos[:, 0], snap.pos[:, 1]
        sp = np.asarray(gen["sp"])[snap.cid]
        order = np.argsort(snap.pos[:, 2])           # draw low cells first
        size = (snap.R[order] * 14) ** 2 * 0.5

        # column 1: cells by strain
        ax = axes[ri, 0]
        ax.scatter(x[order], y[order], s=size, c=STRAIN_COLORS[sp[order]],
                   edgecolors="none", alpha=0.9)
        ax.set_title(f"{ROW_TITLE[inter.row]}\ncells by strain (A blue, B green)",
                     fontsize=10, loc="left")

        # column 2: metabolite field (mean over height) with colony outline
        ax = axes[ri, 1]
        field = fh[fkey][-1]                          # final 3D grid (N,N,N)
        f2d = field.mean(axis=2)
        im = ax.imshow(f2d.T, origin="lower", extent=[0, cfg.cube, 0, cfg.cube],
                       cmap="magma", aspect="equal")
        ax.scatter(x, y, s=2, c="white", alpha=0.35, edgecolors="none")
        ax.set_title(f"field: {flabel}\n(mean over height)", fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # column 3: cells by founding lineage (radial sectors)
        ax = axes[ri, 2]
        lin = snap.lin[order]
        uniq = {v: k for k, v in enumerate(np.unique(lin))}
        cidx = np.array([uniq[v] for v in lin])
        col = plt.cm.tab20((cidx % 20) / 19.0)
        ax.scatter(x[order], y[order], s=size, c=col, edgecolors="none", alpha=0.95)
        ax.set_title("cells by founding lineage\n(radial sectors)", fontsize=10, loc="left")

        for c in range(3):
            axes[ri, c].set_xlim(0, cfg.cube); axes[ri, c].set_ylim(0, cfg.cube)
            axes[ri, c].set_xticks([]); axes[ri, c].set_yticks([])
            axes[ri, c].set_aspect("equal")
        print(f"  {inter.row}: {snap.pos.shape[0]} cells")

    fig.suptitle("Top-down view of the 3D range expansion (final state), one row per "
                 "interaction\nthe radial expansion seeds at the centre and grows "
                 "outward; sectors widen at the front", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(figdir, "cube3d_rangeexp.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
