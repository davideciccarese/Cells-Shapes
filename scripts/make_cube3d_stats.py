"""
make_cube3d_stats.py
====================

Replicate-based statistics with significance tests, shown as violins with the
individual replicate points overlaid.

Run replicates for one interaction (append to the data file):
    python scripts/make_cube3d_stats.py neutralism [n_rep]
Run replicates for all interactions:
    python scripts/make_cube3d_stats.py all [n_rep]
Render the figure and print the tests:
    python scripts/make_cube3d_stats.py --plot

Two metrics:
  * lineage success rate per cell type (default pairing): fraction of a type's
    founders whose progeny reach the outer rim. Compares the two partners.
  * spatial lineage complexity per cell shape (same-shape colonies): a
    compression (Lempel-Ziv) proxy for the Kolmogorov complexity of the clonal
    arrangement, computed separately for cocci, long-rod chains and short rods,
    so the shapes are compared within every interaction.
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
DATA = os.path.join(FIGDIR, "stats_data.json")
MORPH_NAMES = ["cocci", "long-rod chain", "short rod"]
MORPH_COLORS = ["#2563eb", "#eab308", "#dc2626"]
MORPH_SHORT = ["cocci", "chain", "rod"]
SHORT_NAMES = {"neutralism": "Neutralism (0,0)", "commensalism": "Commensalism (0,+)",
               "amensalism": "Amensalism (0,-)", "public_good": "Public good (-,+)",
               "mutualism": "Mutualism (+,+)", "competition": "Competition (-,-)"}
ORDER = ["neutralism", "commensalism", "amensalism",
         "public_good", "mutualism", "competition"]


def _cfg():
    cfg = Config3D()
    cfg.n_seed = 40
    cfg.n_frames = 16          # light so the many replicates are affordable
    return cfg


def run_one(row, n_rep):
    cfg = _cfg()
    inter = [i for i in C.ALL3D if i.row == row][0]
    default_shapes = tuple(inter.shapes)

    # Lineage success rate. In the asymmetric interactions (commensalism,
    # amensalism, public good) the two strains hold different metabolic roles,
    # so a single shape-to-strain assignment would confound cell shape with
    # metabolic role. To separate them, every replicate is run in both
    # assignments: (shape1 in role A, shape2 in role B) and the swap. Each
    # shape is therefore measured the same number of times in each role, and
    # the pooled success rate of a shape is its average over both roles.
    succ = {}
    s1, s2 = default_shapes
    orderings = [(s1, s2), (s2, s1)]
    for r in range(n_rep):
        for j, order in enumerate(orderings):
            cfg.seed = 2000 + j * 1000 + r
            inter.shapes = order
            frames, _f, gen = C.run(inter, cfg)
            ls = S.lineage_success(frames[-1], gen, cfg)
            for mo, (_n, _s, rate) in ls.items():
                succ.setdefault(int(mo), []).append(rate)

    # spatial complexity per cell shape, from single-shape colonies so each
    # shape is isolated under the same interaction. Same-shape colonies already
    # balance the two metabolic roles within a shape, so this part is reused
    # unchanged if it was computed before.
    existing = json.load(open(DATA)) if os.path.exists(DATA) else {}
    prev = existing.get(row, {}).get("complexity_by_shape")
    if prev is not None:
        comp_shape = {int(k): v for k, v in prev.items()}
    else:
        comp_shape = {}
        for sh in (0, 1, 2):
            vals = []
            for r in range(n_rep):
                cfg.seed = 3000 + sh * 100 + r
                inter.shapes = (sh, sh)
                frames, _f, _g = C.run(inter, cfg)
                vals.append(S.lineage_complexity(frames[-1], cfg))
            comp_shape[sh] = vals
    inter.shapes = default_shapes

    data = json.load(open(DATA)) if os.path.exists(DATA) else {}
    data[row] = {
        "name": inter.name,
        "roles": list(getattr(inter, "roles", ("A", "B"))),
        "success_rates": {str(k): v for k, v in succ.items()},
        "complexity_by_shape": {str(k): v for k, v in comp_shape.items()},
    }
    json.dump(data, open(DATA, "w"))
    print(f"  {row}: {n_rep} replicates (success + complexity-by-shape) stored")


def _violin_points(ax, x, vals, color, width=0.7):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if vals.size > 1 and np.ptp(vals) > 0:
        parts = ax.violinplot([vals], positions=[x], widths=width,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("#27272a")
            pc.set_alpha(0.45)
        parts["cmedians"].set_edgecolor("#27272a")
        parts["cmedians"].set_linewidth(1.2)
    jit = np.random.default_rng(0).normal(0, width * 0.06, vals.size)
    ax.scatter(x + jit, vals, s=18, color=color, edgecolor="#27272a",
               linewidth=0.4, zorder=3, alpha=0.92)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    from scipy import stats as sps

    plt.rcParams.update({"font.size": 20})
    data = json.load(open(DATA))
    order = [r for r in ORDER if r in data]

    # ============ figure 1: lineage success rate by cell type ============
    figA, axA = plt.subplots(figsize=(19.5, 8.5), dpi=130)
    figA.patch.set_facecolor("white")
    blendA = blended_transform_factory(axA.transData, axA.transAxes)
    x = 0.0
    ticks, labels, group_centres = [], [], []
    for row in order:
        sr = data[row]["success_rates"]
        morphs = sorted(int(m) for m in sr)
        xs, tops = [], []
        for mo in morphs:
            v = np.array(sr[str(mo)])
            _violin_points(axA, x, v, MORPH_COLORS[mo])
            ticks.append(x); labels.append(MORPH_SHORT[mo])
            xs.append(x); tops.append(v.max())
            x += 2.0
        if len(morphs) == 2:
            a = np.array(sr[str(morphs[0])]); b = np.array(sr[str(morphs[1])])
            try:
                _, p = sps.wilcoxon(a, b)
            except ValueError:
                _, p = sps.mannwhitneyu(a, b)
            star = ("***" if p < 1e-3 else "**" if p < 1e-2
                    else "*" if p < 5e-2 else "ns")
            y = max(tops) + 0.07
            axA.plot([xs[0], xs[0], xs[1], xs[1]],
                     [y - 0.025, y, y, y - 0.025], color="#27272a", lw=1.2)
            axA.text(np.mean(xs), y + 0.012, f"{star}  p={p:.3f}", ha="center",
                     va="bottom", fontsize=18, color="#18181b")
        group_centres.append((np.mean(xs), SHORT_NAMES.get(row, data[row]["name"])))
        x += 2.9
    for gx, nm in group_centres:
        axA.text(gx, -0.17, nm, transform=blendA, ha="center", va="top",
                 fontsize=18, fontweight="bold", color="#18181b")
    axA.set_xticks(ticks); axA.set_xticklabels(labels, fontsize=17)
    axA.tick_params(axis="y", labelsize=17)
    axA.set_ylim(-0.05, 1.28)
    axA.set_ylabel("lineage success rate\n(founders reaching the rim)", fontsize=21)
    axA.set_title("Which cell type's lineages reach the front\n"
                  "violin + replicate points, paired Wilcoxon between cell types",
                  fontsize=23, loc="left")
    axA.grid(axis="y", color="#e4e4e7", lw=0.7); axA.set_axisbelow(True)
    figA.tight_layout(rect=[0, 0.05, 1, 1])
    outA = os.path.join(FIGDIR, "cube3d_stats_success.png")
    figA.savefig(outA, dpi=130); plt.close(figA)
    print(f"wrote {outA}")

    # ============ figure 2: spatial complexity by cell shape ============
    figB, axB = plt.subplots(figsize=(19.5, 8.5), dpi=130)
    figB.patch.set_facecolor("white")
    trB = blended_transform_factory(axB.transData, axB.transAxes)
    x = 0.0
    ticksB, labelsB, centresB = [], [], []
    for row in order:
        cs = data[row]["complexity_by_shape"]
        xs, groups = [], []
        for sh in (0, 1, 2):
            v = np.array(cs[str(sh)])
            _violin_points(axB, x, v, MORPH_COLORS[sh])
            ticksB.append(x); labelsB.append(MORPH_SHORT[sh])
            xs.append(x); groups.append(v[np.isfinite(v)])
            x += 2.0
        try:
            H, p = sps.kruskal(*groups)
            star = ("***" if p < 1e-3 else "**" if p < 1e-2
                    else "*" if p < 5e-2 else "ns")
            axB.text(np.mean(xs), 0.95, f"{star} p={p:.3f}", transform=trB,
                     ha="center", va="bottom", fontsize=17, color="#18181b")
        except ValueError:
            pass
        centresB.append((np.mean(xs), SHORT_NAMES.get(row, data[row]["name"])))
        x += 2.4
    for gx, nm in centresB:
        axB.text(gx, -0.18, nm, transform=trB, ha="center", va="top",
                 fontsize=18, fontweight="bold", color="#18181b")
    axB.set_xticks(ticksB); axB.set_xticklabels(labelsB, fontsize=17)
    axB.tick_params(axis="y", labelsize=17)
    axB.set_ylabel("spatial lineage complexity\n(LZ proxy for Kolmogorov)", fontsize=21)
    axB.set_title("How each cell shape organises in space, per interaction\n"
                  "single-shape colonies, Kruskal-Wallis across the three shapes",
                  fontsize=23, loc="left")
    axB.grid(axis="y", color="#e4e4e7", lw=0.7); axB.set_axisbelow(True)
    figB.tight_layout(rect=[0, 0.05, 1, 1])
    outB = os.path.join(FIGDIR, "cube3d_stats_complexity.png")
    figB.savefig(outB, dpi=130); plt.close(figB)
    print(f"wrote {outB}")

    # printed summary
    print("\nLineage success rate (mean over replicates), paired test:")
    for row in order:
        sr = data[row]["success_rates"]; morphs = sorted(int(m) for m in sr)
        txt = ", ".join(f"{MORPH_NAMES[m]} {np.mean(sr[str(m)]):.2f}" for m in morphs)
        line = f"  {row:13s} {txt}"
        if len(morphs) == 2:
            a = np.array(sr[str(morphs[0])]); b = np.array(sr[str(morphs[1])])
            try:
                _, p = sps.wilcoxon(a, b)
            except ValueError:
                _, p = sps.mannwhitneyu(a, b)
            line += f"   p={p:.4f}"
        print(line)
    print("\nSpatial complexity by shape (mean), Kruskal-Wallis across shapes:")
    for row in order:
        cs = data[row]["complexity_by_shape"]
        txt = ", ".join(f"{MORPH_NAMES[s]} {np.nanmean(cs[str(s)]):.3f}"
                        for s in (0, 1, 2))
        groups = [np.array(cs[str(s)])[np.isfinite(cs[str(s)])] for s in (0, 1, 2)]
        try:
            H, p = sps.kruskal(*groups); tail = f"   H={H:.1f} p={p:.4f}"
        except ValueError:
            tail = ""
        print(f"  {row:13s} {txt}{tail}")


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    arg = sys.argv[1]
    if arg == "--plot":
        plot(); return
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if arg == "all":
        for inter in C.ALL3D:
            run_one(inter.row, n_rep)
    else:
        run_one(arg, n_rep)


if __name__ == "__main__":
    main()
