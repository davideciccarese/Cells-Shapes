"""
make_cube3d_factorial.py
========================

Full factorial separating cell shape from interaction sign.

Each strain in a colony is defined by two factors: its cell shape (cocci, chain,
rod) and the interaction sign it experiences (0 neutral, + positive, - negative).
The six interactions are the six sign pairs, so role A and role B of each
interaction carry a fixed sign:

    neutralism  (0,0)    commensalism (0,+)   amensalism  (0,-)
    public_good (-,+)    mutualism    (+,+)   competition (-,-)

For every interaction we run all nine ordered shape pairs (shape of strain A by
shape of strain B) over replicates, and for each run we record the per-strain
outcome (lineage success rate and median travel distance) together with that
strain's own shape and sign. Pooling these records gives, for each cell type, a
3x3 table of outcome indexed by (shape, sign), so the effect of shape (variation
down the columns), the effect of sign (variation across the rows) and the effect
of the combination (the interaction term) can be read separately.

    python scripts/make_cube3d_factorial.py neutralism [n_rep]
    python scripts/make_cube3d_factorial.py all [n_rep]
    python scripts/make_cube3d_factorial.py --plot
"""

import os
import sys
import json
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from config3d import Config3D
import cube3d as C
import stats3d as S

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
DATA = os.path.join(FIGDIR, "factorial_data.json")

# sign carried by (role A, role B) of each interaction; 0 neutral, 1 +, -1 -
SIGNS = {"neutralism": (0, 0), "commensalism": (0, 1), "amensalism": (0, -1),
         "public_good": (-1, 1), "mutualism": (1, 1), "competition": (-1, -1)}
SHAPE_NAMES = ["cocci", "chain", "rod"]
SIGN_NAMES = {-1: "negative (-)", 0: "neutral (0)", 1: "positive (+)"}
SIGN_ORDER = [-1, 0, 1]


def run_one(row, n_rep):
    cfg = Config3D(); cfg.n_seed = 40; cfg.n_frames = 14
    inter = [i for i in C.ALL3D if i.row == row][0]
    sgA, sgB = SIGNS[row]
    recs = json.load(open(DATA)) if os.path.exists(DATA) else []
    for shA, shB in itertools.product((0, 1, 2), repeat=2):
        for r in range(n_rep):
            cfg.seed = 9000 + (shA * 3 + shB) * 50 + r
            inter.shapes = (shA, shB)
            frames, _f, gen = C.run(inter, cfg)
            out = S.strain_outcomes(frames[-1], gen, cfg)
            if 0 in out:
                recs.append({"row": row, "shape": shA, "sign": sgA,
                             "psh": shB, "psg": sgB,
                             "succ": out[0][0], "trav": out[0][1]})
            if 1 in out:
                recs.append({"row": row, "shape": shB, "sign": sgB,
                             "psh": shA, "psg": sgA,
                             "succ": out[1][0], "trav": out[1][1]})
    inter.shapes = tuple(inter.shapes)
    json.dump(recs, open(DATA, "w"))
    print(f"  {row}: {n_rep} reps x 9 shape pairs stored "
          f"(total records now {len(recs)})")


def _grid(recs, key):
    """mean outcome and n for each (shape, sign) cell, pooled over partner."""
    M = np.full((3, 3), np.nan)
    N = np.zeros((3, 3), int)
    raw = {}
    for sh in (0, 1, 2):
        for si, sg in enumerate(SIGN_ORDER):
            vals = [d[key] for d in recs if d["shape"] == sh and d["sign"] == sg
                    and np.isfinite(d[key])]
            raw[(sh, si)] = vals
            if vals:
                M[sh, si] = np.mean(vals); N[sh, si] = len(vals)
    return M, N, raw


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps
    plt.rcParams.update({"font.size": 14})

    recs = json.load(open(DATA))
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5), dpi=125)
    fig.patch.set_facecolor("white")

    for col, (key, label, cmap, vlim) in enumerate([
            ("succ", "lineage success rate", "viridis", (0, 1)),
            ("trav", "median travel distance", "magma", None)]):
        M, N, raw = _grid(recs, key)
        # ---- heatmap: shape (rows) x sign (columns) ----
        ax = axes[col, 0]
        vmin, vmax = vlim if vlim else (np.nanmin(M), np.nanmax(M))
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels([SIGN_NAMES[s] for s in SIGN_ORDER], fontsize=12)
        ax.set_yticks(range(3)); ax.set_yticklabels(SHAPE_NAMES, fontsize=13)
        ax.set_xlabel("interaction sign", fontsize=13)
        ax.set_ylabel("cell shape", fontsize=13)
        ax.set_title(f"{label}\nby shape and sign (mean over partner)", fontsize=13, loc="left")
        for i in range(3):
            for j in range(3):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}\nn={N[i, j]}", ha="center",
                            va="center", color="white", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # ---- marginal effect of shape (averaged over sign) ----
        ax = axes[col, 1]
        for sh in (0, 1, 2):
            vals = [d[key] for d in recs if d["shape"] == sh and np.isfinite(d[key])]
            _violin_pts(ax, sh, vals, ["#2563eb", "#eab308", "#dc2626"][sh])
        ax.set_xticks(range(3)); ax.set_xticklabels(SHAPE_NAMES, fontsize=13)
        ax.set_ylabel(label, fontsize=13)
        try:
            groups = [[d[key] for d in recs if d["shape"] == sh and np.isfinite(d[key])] for sh in (0, 1, 2)]
            H, p = sps.kruskal(*groups)
            ax.set_title(f"marginal effect of SHAPE\n(pooled over sign)  KW p={p:.1e}", fontsize=13, loc="left")
        except ValueError:
            ax.set_title("marginal effect of SHAPE", fontsize=13, loc="left")
        ax.grid(axis="y", color="#e4e4e7", lw=0.6); ax.set_axisbelow(True)
        # ---- marginal effect of sign (averaged over shape) ----
        ax = axes[col, 2]
        for si, sg in enumerate(SIGN_ORDER):
            vals = [d[key] for d in recs if d["sign"] == sg and np.isfinite(d[key])]
            _violin_pts(ax, si, vals, ["#0ea5e9", "#64748b", "#16a34a"][si])
        ax.set_xticks(range(3)); ax.set_xticklabels([SIGN_NAMES[s] for s in SIGN_ORDER], fontsize=12)
        ax.set_ylabel(label, fontsize=13)
        try:
            groups = [[d[key] for d in recs if d["sign"] == sg and np.isfinite(d[key])] for sg in SIGN_ORDER]
            H, p = sps.kruskal(*groups)
            ax.set_title(f"marginal effect of SIGN\n(pooled over shape)  KW p={p:.1e}", fontsize=13, loc="left")
        except ValueError:
            ax.set_title("marginal effect of SIGN", fontsize=13, loc="left")
        ax.grid(axis="y", color="#e4e4e7", lw=0.6); ax.set_axisbelow(True)

    fig.suptitle("Full factorial: cell shape (3) by interaction sign (3), per-strain outcome pooled over the partner strain\n"
                 "left: the shape-by-sign table; middle: outcome by shape with sign averaged out; right: outcome by sign with shape averaged out",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGDIR, "cube3d_factorial.png")
    fig.savefig(out, dpi=125); plt.close(fig)
    print(f"wrote {out}")
    _print_tables(recs)


def _violin_pts(ax, x, vals, color, w=0.6):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if vals.size > 1 and np.ptp(vals) > 0:
        parts = ax.violinplot([vals], positions=[x], widths=w, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_edgecolor("#27272a"); pc.set_alpha(0.4)
        parts["cmedians"].set_edgecolor("#27272a")
    jit = np.random.default_rng(0).normal(0, w * 0.05, vals.size)
    ax.scatter(x + jit, vals, s=10, color=color, edgecolor="#27272a", linewidth=0.2, alpha=0.6, zorder=3)


def _print_tables(recs):
    for key, label in [("succ", "LINEAGE SUCCESS RATE"), ("trav", "MEDIAN TRAVEL DISTANCE")]:
        M, N, _ = _grid(recs, key)
        print(f"\n{label}: rows = shape, columns = sign (mean over partner)")
        print("            " + "".join(f"{SIGN_NAMES[s]:>16s}" for s in SIGN_ORDER))
        for sh in (0, 1, 2):
            print(f"  {SHAPE_NAMES[sh]:8s} " + "".join(
                f"{M[sh, si]:>10.3f} (n={N[sh, si]:3d})" for si in range(3)))
        print(f"  marginal by shape: " + ", ".join(
            f"{SHAPE_NAMES[sh]} {np.nanmean(M[sh]):.3f}" for sh in (0, 1, 2)))
        print(f"  marginal by sign:  " + ", ".join(
            f"{SIGN_NAMES[SIGN_ORDER[si]]} {np.nanmean(M[:, si]):.3f}" for si in range(3)))


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    if sys.argv[1] == "--plot":
        plot(); plot_byinteraction(); return
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if sys.argv[1] == "all":
        for inter in C.ALL3D:
            run_one(inter.row, n_rep)
    else:
        run_one(sys.argv[1], n_rep)




ROW_ORDER = ["neutralism", "commensalism", "amensalism",
             "public_good", "mutualism", "competition"]
ROW_TITLE = {"neutralism": "Neutralism (0,0)", "commensalism": "Commensalism (0,+)",
             "amensalism": "Amensalism (0,-)", "public_good": "Public good (-,+)",
             "mutualism": "Mutualism (+,+)", "competition": "Competition (-,-)"}
SHAPE_COLORS = ["#2563eb", "#eab308", "#dc2626"]


def _stars(p):
    return ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns")


def plot_byinteraction():
    """One panel per interaction, the three cell shapes side by side, for the
    lineage success rate and again for the travel distance. Within each panel a
    Kruskal-Wallis test across the three shapes gives the overall p-value, and
    pairwise Mann-Whitney tests give the significance brackets between shapes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps
    plt.rcParams.update({"font.size": 14})

    recs = json.load(open(DATA))
    for key, label, fname in [
            ("succ", "lineage success rate", "cube3d_success_byinteraction.png"),
            ("trav", "median travel distance", "cube3d_travel_byinteraction.png")]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=125)
        fig.patch.set_facecolor("white")
        axes = axes.ravel()
        print(f"\n=== {label.upper()} by interaction and cell shape "
              f"(Kruskal-Wallis across shapes; pairwise Mann-Whitney) ===")
        for ax, row in zip(axes, ROW_ORDER):
            groups, tops = [], []
            for sh in (0, 1, 2):
                vals = [d[key] for d in recs if d["row"] == row and d["shape"] == sh
                        and np.isfinite(d[key])]
                groups.append(np.array(vals, float))
                _violin_pts(ax, sh, vals, SHAPE_COLORS[sh], w=0.7)
                if vals:
                    ax.text(sh, np.median(vals), f" {np.median(vals):.2f}",
                            va="center", ha="left", fontsize=11)
                    tops.append(max(vals))
            try:
                H, p = sps.kruskal(*groups)
                ktxt = f"Kruskal-Wallis {_stars(p)}  p={p:.1e}"
            except ValueError:
                p = np.nan; ktxt = "Kruskal-Wallis n/a"
            # pairwise brackets
            top = max(tops) if tops else 1.0
            pairs = [(0, 1), (1, 2), (0, 2)]
            line = []
            for k, (a, b) in enumerate(pairs):
                if groups[a].size >= 2 and groups[b].size >= 2:
                    _, pp = sps.mannwhitneyu(groups[a], groups[b])
                    y = top * (1.06 + 0.10 * k)
                    ax.plot([a, a, b, b], [y - top*0.02, y, y, y - top*0.02],
                            color="#27272a", lw=1.0)
                    ax.text((a + b) / 2.0, y, _stars(pp), ha="center",
                            va="bottom", fontsize=11)
                    line.append(f"{SHAPE_NAMES[a]}-{SHAPE_NAMES[b]} p={pp:.1e}")
            ax.set_ylim(top=top * 1.45)
            ax.set_xticks(range(3)); ax.set_xticklabels(SHAPE_NAMES, fontsize=13)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f"{ROW_TITLE[row]}\n{ktxt}", fontsize=12.5, loc="left")
            ax.grid(axis="y", color="#e4e4e7", lw=0.6); ax.set_axisbelow(True)
            meds = ", ".join(f"{SHAPE_NAMES[s]} {np.median(groups[s]):.3f}"
                             if groups[s].size else f"{SHAPE_NAMES[s]} na" for s in (0, 1, 2))
            print(f"  {row:13s} medians: {meds}  | KW p={p:.2e}  | "
                  + "; ".join(line))
        fig.suptitle(f"{label} by interaction and cell shape, per-strain, "
                     "counterbalanced over metabolic role\n"
                     "each shape pooled over both roles of the interaction and over the "
                     "partner shape; tests are nonparametric", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = os.path.join(FIGDIR, fname)
        fig.savefig(out, dpi=125); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
