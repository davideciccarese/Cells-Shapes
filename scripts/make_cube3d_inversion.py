"""
make_cube3d_inversion.py
========================

Role inversion for the 3D cube model, the same experiment as
make_re_inversion.py but on the cube. Every interaction is run with its default
shape pairing and again with the two cell shapes swapped between the roles, so
each shape carries role A's sign once and role B's sign once. The cube is drawn
in 3D, coloured by shape, and the per-strain lineage success is compared between
the two role assignments.

Outputs:
  cube3d_inversion.png           the 3D cube, default and inverted, per
                                 interaction, cells coloured by shape
  cube3d_inversion_analysis.png  lineage success of each shape in role A and in
                                 role B, per interaction (distributions with
                                 points and a nonparametric test)

    python scripts/make_cube3d_inversion.py [n_rep]
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats as sps

from config3d import Config3D
import cube3d as C
import stats3d as S
import render3d as R3

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
DATA = os.path.join(FIGDIR, "cube3d_inversion_data.json")
SHAPE_NAMES = ["cocci", "chain", "rod"]
MORPH_COLORS = R3.MORPH_COLORS                      # cocci blue, chain yellow, rod red
from matplotlib.patches import Patch
LEGEND = [Patch(facecolor=MORPH_COLORS[k], edgecolor="#0f172a", label=SHAPE_NAMES[k])
          for k in range(3)]


def _violin_pts(ax, x, vals, color, w=0.6):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if vals.size > 1 and np.ptp(vals) > 0:
        parts = ax.violinplot([vals], positions=[x], widths=w,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("#27272a")
            pc.set_alpha(0.45)
        parts["cmedians"].set_edgecolor("#27272a")
    else:
        ax.plot([x - w * 0.4, x + w * 0.4], [vals[0], vals[0]],
                color="#27272a", lw=1.4, zorder=2)
    jit = np.random.default_rng(0).uniform(-w * 0.16, w * 0.16, vals.size)
    ax.scatter(x + jit, vals, s=30, color=color, edgecolor="#27272a",
               linewidth=0.4, alpha=0.8, zorder=3)


def _stars(p):
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def _bracket(ax, x1, x2, y, p, h=0.035):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#27272a", lw=1.1)
    ax.text((x1 + x2) / 2.0, y + h, _stars(p), ha="center", va="bottom",
            fontsize=13)


def compute(n_rep):
    cfg = Config3D()
    cfg.n_frames = 14
    cfg.n_seed = 40
    recs = []
    snaps = {}
    for inter in C.ALL3D:
        shA, shB = inter.shapes
        for tag, shapes in [("default", (shA, shB)), ("inverted", (shB, shA))]:
            for r in range(n_rep):
                cfg.seed = 5000 + r
                frames, _f, gen = C.run(inter, cfg, shapes=shapes)
                snap = frames[-1]
                out = S.strain_outcomes(snap, gen, cfg)
                recs.append({"row": inter.row, "tag": tag,
                             "shapeA": int(shapes[0]), "shapeB": int(shapes[1]),
                             "succA": out.get(0, (np.nan,))[0],
                             "succB": out.get(1, (np.nan,))[0]})
                if r == 0:
                    snaps[(inter.row, tag)] = (inter, snap)
        print(f"  {inter.row}: default and inverted done ({n_rep} reps each)")
    json.dump(recs, open(DATA, "w"))
    _plot_panels(cfg, snaps)
    _plot_analysis(recs)


def _plot_panels(cfg, snaps):
    rows = list(C.ALL3D)
    L = cfg.cube
    nrows = len(rows)
    fig = plt.figure(figsize=(9.5, 4.6 * nrows), dpi=110)
    fig.patch.set_facecolor("white")
    for ri, inter in enumerate(rows):
        sA, sB = inter.signs
        for ci, tag in enumerate(["default", "inverted"]):
            ax = fig.add_subplot(nrows, 2, ri * 2 + ci + 1, projection="3d")
            _, snap = snaps[(inter.row, tag)]
            R3._style_axes(ax, L, None)
            R3._cube_edges(ax, L)
            shape_cols = np.array(MORPH_COLORS)[snap.mtype]
            R3._draw_rods(ax, snap, shape_cols, alpha=1.0, edge="#0f172a",
                          edge_lw=0.2, k=4)
            ax.view_init(elev=18, azim=-55)
            shA, shB = (inter.shapes if tag == "default"
                        else (inter.shapes[1], inter.shapes[0]))
            ax.set_title(
                f"{inter.name}  [{tag}]\n"
                f"role A = {SHAPE_NAMES[shA]} (sign {sA}), "
                f"role B = {SHAPE_NAMES[shB]} (sign {sB})",
                fontsize=9.5, color="#18181b", pad=0)
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("3D cube: each interaction with the two cell shapes swapped "
                 "between the roles\nleft default, right inverted; cells coloured "
                 "by shape", fontsize=13, y=0.997)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.965, bottom=0.03,
                        wspace=0.02, hspace=0.12)
    out = os.path.join(FIGDIR, "cube3d_inversion.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def _plot_analysis(recs):
    rows = list(C.ALL3D)
    n_rep = max((sum(1 for d in recs if d["row"] == r.row and d["tag"] == "default")
                 for r in rows), default=0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), dpi=125)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    print("\n=== Role inversion: lineage success of each shape in role A vs role B "
          "(3D cube model) ===")
    for ax, inter in zip(axes, rows):
        shA, shB = inter.shapes
        sA, sB = inter.signs

        def vals(tag, key):
            return [d[key] for d in recs if d["row"] == inter.row and d["tag"] == tag
                    and np.isfinite(d[key])]
        a_roleA = vals("default", "succA")
        a_roleB = vals("inverted", "succB")
        b_roleB = vals("default", "succB")
        b_roleA = vals("inverted", "succA")
        groups = [(f"{SHAPE_NAMES[shA]}\nrole A (sign {sA})", a_roleA, "#2563eb"),
                  (f"{SHAPE_NAMES[shA]}\nrole B (sign {sB})", a_roleB, "#93c5fd"),
                  (f"{SHAPE_NAMES[shB]}\nrole A (sign {sA})", b_roleA, "#b91c1c"),
                  (f"{SHAPE_NAMES[shB]}\nrole B (sign {sB})", b_roleB, "#fca5a5")]
        for x, (lab, v, col) in enumerate(groups):
            _violin_pts(ax, x, v, col, w=0.7)
            k = int(np.isfinite(np.asarray(v, float)).sum())
            ax.text(x, -0.06, f"n={k}", ha="center", va="top", fontsize=8.5,
                    color="#71717a")

        def mw(u, w):
            u = np.asarray(u, float); u = u[np.isfinite(u)]
            w = np.asarray(w, float); w = w[np.isfinite(w)]
            if u.size >= 2 and w.size >= 2:
                return sps.mannwhitneyu(u, w).pvalue
            return float("nan")
        pA = mw(a_roleA, a_roleB)
        pB = mw(b_roleB, b_roleA)
        allv = [np.asarray(g[1], float) for g in groups]
        top = max((v[np.isfinite(v)].max() for v in allv
                   if np.isfinite(v).any()), default=1.0)
        ybr = min(top + 0.12, 0.92)
        _bracket(ax, 0, 1, ybr, pA)
        _bracket(ax, 2, 3, ybr, pB)
        ax.set_xticks(range(4))
        ax.set_xticklabels([g[0] for g in groups], fontsize=9)
        ax.set_ylabel("lineage success rate", fontsize=11)
        ax.set_ylim(-0.10, 1.18)
        ax.set_title(inter.name, fontsize=12, loc="left")
        ax.text(0.99, 0.97, f"N = {n_rep} replicates per group",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="#3f3f46")
        ax.grid(axis="y", color="#e4e4e7", lw=0.6)
        ax.set_axisbelow(True)
        print(f"  {inter.row:13s} {SHAPE_NAMES[shA]}: roleA {np.nanmean(a_roleA):.2f} "
              f"roleB {np.nanmean(a_roleB):.2f} ({_stars(pA)} p={pA:.2g}) | "
              f"{SHAPE_NAMES[shB]}: roleA {np.nanmean(b_roleA):.2f} "
              f"roleB {np.nanmean(b_roleB):.2f} ({_stars(pB)} p={pB:.2g})")
    fig.suptitle("Role inversion (3D cube model): lineage success of each cell "
                 "shape when it carries role A's sign and role B's sign\n"
                 f"N = {n_rep} replicates per group; brackets test role A against "
                 "role B for that shape (Mann-Whitney; ns, *, **, ***); a shape "
                 "whose two distributions match is insensitive to the sign",
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGDIR, "cube3d_inversion_analysis.png")
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print("wrote", out)


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    compute(n_rep)


if __name__ == "__main__":
    main()
