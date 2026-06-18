"""
make_re_metrics.py
==================

Spatial-organization metrics of the front, computed from the 2D range-expansion
model. For each interaction the colony is run with each cell shape on both
strains (a symmetric shape pairing), over replicates, and the frontier sector
statistics of metrics_re are recorded. This isolates the effect of cell shape on
the clonal organisation of the expanding front within each interaction.

Reported per interaction and shape (median over replicates):
  jc    join-count mixing ratio (< 1 clean segregated sectors, ~ 1 finely mixed)
  xi    sector width at the rim (arc length)
  S     surviving sector count
  D     surviving lineages at the rim
  frag  sectors per surviving lineage

Across the three shapes a Kruskal-Wallis test gives the significance of the shape
effect on each metric, within each interaction.

    python scripts/make_re_metrics.py [n_rep]
    python scripts/make_re_metrics.py --plot
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from config_re import ConfigRE
import interactions_re as I
import sim_re as SIM
import metrics_re as M

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
DATA = os.path.join(FIGDIR, "re_metrics_data.json")

ROW_ORDER = ["neutralism", "commensalism", "amensalism",
             "public_good", "mutualism", "competition"]
ROW_TITLE = {"neutralism": "Neutralism (0,0)", "commensalism": "Commensalism (0,+)",
             "amensalism": "Amensalism (0,-)", "public_good": "Public good (-,+)",
             "mutualism": "Mutualism (+,+)", "competition": "Competition (-,-)"}
SHAPE_NAMES = ["cocci", "chain", "rod"]
SHAPE_COLORS = ["#2563eb", "#eab308", "#dc2626"]
KEYS = [("jc", "mixing ratio (join-count)"),
        ("xi", "sector width at rim"),
        ("S", "surviving sectors"),
        ("D", "surviving lineages"),
        ("frag", "sectors per lineage")]


def compute(n_rep):
    cfg = ConfigRE()
    cfg.n_frames = 22
    cfg.n_seed = 60
    recs = []
    for inter in I.ALLRE:
        for sh in (0, 1, 2):
            for r in range(n_rep):
                cfg.seed = 4000 + sh * 100 + r
                frames, _f, gen = SIM.run(inter, cfg, shapes=(sh, sh))
                sm = M.sector_metrics(frames[-1], cfg)
                if sm is None:
                    continue
                rec = {"row": inter.row, "shape": sh}
                rec.update({k: sm[k] for k, _ in KEYS})
                recs.append(rec)
        print(f"  {inter.row}: done ({n_rep} reps x 3 shapes)")
    json.dump(recs, open(DATA, "w"))
    print(f"wrote {DATA} ({len(recs)} records)")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps
    plt.rcParams.update({"font.size": 13})

    recs = json.load(open(DATA))
    n_rep = max(1, round(len(recs) / (6 * 3)))

    def _stars(p):
        return ("***" if p < 1e-3 else "**" if p < 1e-2
                else "*" if p < 5e-2 else "ns")

    # two metrics on the front that carry the clearest spatial reading
    for key, label, fname in [("jc", "mixing ratio (join-count)", "re_mixing_byinteraction.png"),
                              ("xi", "sector width at rim", "re_sectorwidth_byinteraction.png")]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=125)
        fig.patch.set_facecolor("white")
        axes = axes.ravel()
        print(f"\n=== {label.upper()} by interaction and cell shape "
              f"(range-expansion model; Kruskal-Wallis across shapes) ===")
        for ax, row in zip(axes, ROW_ORDER):
            groups = []
            tops = []
            for sh in (0, 1, 2):
                vals = [d[key] for d in recs if d["row"] == row and d["shape"] == sh
                        and np.isfinite(d[key])]
                groups.append(np.array(vals, float))
                v = np.asarray(vals, float)
                v = v[np.isfinite(v)]
                if v.size:
                    if v.size > 1 and np.ptp(v) > 0:
                        parts = ax.violinplot([v], positions=[sh], widths=0.7,
                                              showmedians=True, showextrema=False)
                        for pc in parts["bodies"]:
                            pc.set_facecolor(SHAPE_COLORS[sh])
                            pc.set_edgecolor("#27272a")
                            pc.set_alpha(0.4)
                        parts["cmedians"].set_edgecolor("#27272a")
                    jit = np.random.default_rng(0).normal(0, 0.04, v.size)
                    ax.scatter(sh + jit, v, s=14, color=SHAPE_COLORS[sh],
                               edgecolor="#27272a", linewidth=0.2, alpha=0.7, zorder=3)
                    ax.text(sh, np.median(v), f" {np.median(v):.2f}",
                            va="center", ha="left", fontsize=11)
                    tops.append(np.max(v))
            try:
                H, p = sps.kruskal(*groups)
                ktxt = f"Kruskal-Wallis {_stars(p)}  p={p:.1e}"
            except ValueError:
                p = np.nan
                ktxt = "Kruskal-Wallis n/a"
            top = max(tops) if tops else 1.0
            ax.set_ylim(top=top * 1.25)
            ax.set_xticks(range(3))
            ax.set_xticklabels(SHAPE_NAMES, fontsize=13)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f"{ROW_TITLE[row]}\n{ktxt}", fontsize=12.5, loc="left")
            ax.grid(axis="y", color="#e4e4e7", lw=0.6)
            ax.set_axisbelow(True)
            meds = ", ".join(f"{SHAPE_NAMES[s]} {np.median(groups[s]):.3f}"
                             if groups[s].size else f"{SHAPE_NAMES[s]} na"
                             for s in (0, 1, 2))
            print(f"  {row:13s} medians: {meds}  | KW p={p:.2e}")
        if key == "jc":
            note = ("mixing ratio below 1 means clean segregated sectors, "
                    "near 1 means finely mixed lineages")
        else:
            note = "sector width is the angular correlation length times the rim radius"
        fig.suptitle(
            f"{label} by interaction and cell shape (range-expansion model)\n"
            f"{note}; Kruskal-Wallis across shapes (ns, *, **, ***); "
            f"N = {n_rep} replicates per shape", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(FIGDIR, fname)
        fig.savefig(out, dpi=125)
        plt.close(fig)
        print(f"wrote {out}")

    _print_full_table(recs)


def _print_full_table(recs):
    print("\n=== Full sector-metric table (median over replicates), "
          "range-expansion model ===")
    header = "  interaction    shape   " + "".join(f"{lab.split('(')[0].strip():>16s}"
                                                   for _, lab in KEYS)
    print(header)
    for row in ROW_ORDER:
        for sh in (0, 1, 2):
            vals = {}
            for k, _ in KEYS:
                v = [d[k] for d in recs if d["row"] == row and d["shape"] == sh
                     and np.isfinite(d[k])]
                vals[k] = np.median(v) if v else float("nan")
            print(f"  {ROW_TITLE[row]:14s} {SHAPE_NAMES[sh]:6s} "
                  + "".join(f"{vals[k]:>16.3f}" for k, _ in KEYS))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        plot()
        return
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    compute(n_rep)
    plot()


if __name__ == "__main__":
    main()
