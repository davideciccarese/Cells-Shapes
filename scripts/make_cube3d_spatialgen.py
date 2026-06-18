"""
make_cube3d_spatialgen.py
=========================

Spatial-population-genetics companion to the compression metric. On the frontier
ring (the outermost cell per angle, explicit front tracking) of single-shape
colonies it measures, per cell shape and interaction:

  * correlation length xi   sector width  (length scale)
  * surviving sector count  number of clonal arcs at the rim (drift)
  * join-count mixing ratio observed boundaries / random expectation
                            (composition-free: <1 segregated, ~1 finely mixed)
  * composition entropy     Shannon entropy of rim lineage frequencies (control)

These separate how the lineages are arranged from how many survive, which a
single compression number cannot.

    python scripts/make_cube3d_spatialgen.py neutralism [n_rep]
    python scripts/make_cube3d_spatialgen.py all [n_rep]
    python scripts/make_cube3d_spatialgen.py --plot
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from config3d import Config3D
import cube3d as C
import stats3d as S

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
DATA = os.path.join(FIGDIR, "spatialgen_data.json")
MORPH_SHORT = ["cocci", "chain", "rod"]
MORPH_COLORS = ["#2563eb", "#eab308", "#dc2626"]
ORDER = ["neutralism", "commensalism", "amensalism",
         "public_good", "mutualism", "competition"]
METRICS = [("xi", "sector width  xi  (cell diameters)", False),
           ("S", "surviving sector count  (drift)", False),
           ("jc", "join-count mixing ratio\n(<1 segregated, ~1 mixed; composition-free)", False),
           ("Hcomp", "composition entropy of rim lineages\n(Shannon, aspatial control)", False)]


def _cfg():
    cfg = Config3D(); cfg.n_seed = 40; cfg.n_frames = 16
    return cfg


def run_one(row, n_rep):
    cfg = _cfg()
    inter = [i for i in C.ALL3D if i.row == row][0]
    default = tuple(inter.shapes)
    by_shape = {}
    for sh in (0, 1, 2):
        recs = {k: [] for k, _l, _i in METRICS}
        for r in range(n_rep):
            cfg.seed = 5000 + sh * 100 + r
            inter.shapes = (sh, sh)
            frames, _f, _g = C.run(inter, cfg)
            m = S.sector_metrics(frames[-1], cfg)
            if m is None:
                continue
            for k, _l, _i in METRICS:
                recs[k].append(float(m[k]))
        by_shape[sh] = recs
    inter.shapes = default
    data = json.load(open(DATA)) if os.path.exists(DATA) else {}
    data[row] = {"name": inter.name,
                 "by_shape": {str(s): by_shape[s] for s in by_shape}}
    json.dump(data, open(DATA, "w"))
    print(f"  {row}: {n_rep} replicates x 3 shapes stored")


def _violin_points(ax, x, vals, color, width=0.7):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if vals.size > 1 and np.ptp(vals) > 0:
        parts = ax.violinplot([vals], positions=[x], widths=width,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_edgecolor("#27272a"); pc.set_alpha(0.45)
        parts["cmedians"].set_edgecolor("#27272a"); parts["cmedians"].set_linewidth(1.4)
    jit = np.random.default_rng(0).normal(0, width * 0.05, vals.size)
    ax.scatter(x + jit, vals, s=14, color=color, edgecolor="#27272a",
               linewidth=0.3, zorder=3, alpha=0.8)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps
    plt.rcParams.update({"font.size": 16})

    data = json.load(open(DATA))
    order = [r for r in ORDER if r in data]

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), dpi=125)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    for ax, (key, label, _i) in zip(axes, METRICS):
        groups = []
        for sh in (0, 1, 2):
            pooled = []
            for row in order:
                pooled += data[row]["by_shape"][str(sh)][key]
            pooled = np.array(pooled, float); pooled = pooled[np.isfinite(pooled)]
            groups.append(pooled)
            _violin_points(ax, sh, pooled, MORPH_COLORS[sh], width=0.7)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(MORPH_SHORT, fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel(label, fontsize=15)
        try:
            H, p = sps.kruskal(*groups)
            star = ("***" if p < 1e-3 else "**" if p < 1e-2
                    else "*" if p < 5e-2 else "ns")
            ax.set_title(f"Kruskal-Wallis across shapes: {star}  p={p:.1e}",
                         fontsize=15, loc="left")
        except ValueError:
            pass
        ax.grid(axis="y", color="#e4e4e7", lw=0.6); ax.set_axisbelow(True)
    fig.suptitle("Spatial-genetics of the frontier ring, pooled over the six "
                 "interactions (single-shape colonies)\n"
                 "how the clonal pattern is organised, separated from how many "
                 "lineages survive", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGDIR, "cube3d_spatialgen.png")
    fig.savefig(out, dpi=125); plt.close(fig)
    print(f"wrote {out}")

    print("\nPer-interaction medians by shape (xi | sectors | jc | Hcomp):")
    for row in order:
        bs = data[row]["by_shape"]
        for sh in (0, 1, 2):
            r = bs[str(sh)]
            md = lambda k: np.nanmedian(r[k]) if r[k] else float("nan")
            print(f"  {row:13s} {MORPH_SHORT[sh]:6s} "
                  f"xi={md('xi'):.2f}  S={md('S'):.0f}  "
                  f"jc={md('jc'):.2f}  Hcomp={md('Hcomp'):.2f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    arg = sys.argv[1]
    if arg == "--plot":
        plot(); return
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if arg == "all":
        for inter in C.ALL3D:
            run_one(inter.row, n_rep)
    else:
        run_one(arg, n_rep)


if __name__ == "__main__":
    main()
