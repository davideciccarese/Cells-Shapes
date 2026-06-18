"""
make_cube3d_control.py
======================

Control experiment: is the spatial organisation driven by the mechanics, or just
by the cell-type label? Everything is held fixed except one mechanical knob.

Common garden: neutralism (both strains identical, independent metabolism), so
metabolism is symmetric and the same for every condition. Growth is biomass-fair
throughout. We then sweep the nematic-alignment strength k_nematic for a rod body
from 0 upward, with a cocci colony as the no-elongation baseline.

If the frontier metrics move smoothly with k_nematic while the total biomass (the
growth check) stays flat, the spatial advantage comes from the alignment
mechanics, not from the bare fact that the cell type is different.

    python scripts/make_cube3d_control.py [n_rep]
    python scripts/make_cube3d_control.py --plot
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
DATA = os.path.join(FIGDIR, "control_data.json")

# (label, shape index, k_nematic override or None to keep default)
CONDITIONS = [
    ("cocci\nbaseline", 0, None, False),
    ("rod\nno align\n(k_nem=0)", 2, 0.0, False),
    ("rod\nalign\n(k_nem=0.02)", 2, 0.02, False),
    ("rod\nalign +\niso-division", 2, 0.02, True),
]
COND_COLOR = ["#2563eb", "#fca5a5", "#dc2626", "#7c3aed"]


def total_biomass(snap):
    R = snap.R; L = snap.L; m = snap.mtype
    vol = np.where(m == 0, 4.0/3.0*np.pi*R**3, np.pi*R**2*L + 4.0/3.0*np.pi*R**3)
    return float(np.nansum(vol))


def nematic_order(snap):
    el = snap.mtype != 0
    if el.sum() < 5:
        return float("nan")
    u = snap.ax[el]
    Q = (u[:, :, None] * u[:, None, :]).mean(0) * 1.5 - 0.5 * np.eye(3)
    return float(np.linalg.eigvalsh(Q).max())


def run(n_rep):
    inter = [i for i in C.ALL3D if i.row == "neutralism"][0]
    out = []
    for label, sh, knem, iso in CONDITIONS:
        rec = {"xi": [], "S": [], "jc": [], "order": [], "biomass": [], "ncell": []}
        for r in range(n_rep):
            cfg = Config3D(); cfg.n_seed = 40; cfg.n_frames = 16
            cfg.seed = 7000 + r
            if knem is not None:
                cfg.k_nematic = knem
            cfg.iso_division = iso
            inter.shapes = (sh, sh)
            frames, _f, _g = C.run(inter, cfg)
            snap = frames[-1]
            m = S.sector_metrics(snap, cfg)
            if m is not None:
                rec["xi"].append(m["xi"]); rec["S"].append(m["S"]); rec["jc"].append(m["jc"])
            rec["order"].append(nematic_order(snap))
            rec["biomass"].append(total_biomass(snap))
            rec["ncell"].append(int(snap.pos.shape[0]))
        out.append({"label": label, "rec": rec})
        print(f"  {label.replace(chr(10),' '):18s} done")
    inter.shapes = (2, 0)
    json.dump(out, open(DATA, "w"))


def _vp(ax, x, vals, color, w=0.6):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if vals.size > 1 and np.ptp(vals) > 0:
        parts = ax.violinplot([vals], positions=[x], widths=w, showmedians=True,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_edgecolor("#27272a"); pc.set_alpha(0.45)
        parts["cmedians"].set_edgecolor("#27272a")
    jit = np.random.default_rng(0).normal(0, w * 0.05, vals.size)
    ax.scatter(x + jit, vals, s=16, color=color, edgecolor="#27272a",
               linewidth=0.3, zorder=3, alpha=0.85)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15})
    data = json.load(open(DATA))
    labels = [d["label"] for d in data]
    panels = [("order", "nematic order parameter\n(the knob actually works)"),
              ("xi", "sector width  xi  (cell diameters)"),
              ("jc", "join-count mixing ratio\n(<1 segregated, ~1 mixed)"),
              ("S", "surviving sector count"),
              ("biomass", "total biomass at end\n(per-cell growth law is identical;\ndifference is emergent from geometry)"),
              ("ncell", "cell count at end")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=125)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    for ax, (key, label) in zip(axes, panels):
        for x, d in enumerate(data):
            _vp(ax, x, d["rec"][key], COND_COLOR[x])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel(label, fontsize=14)
        ax.grid(axis="y", color="#e4e4e7", lw=0.6); ax.set_axisbelow(True)
        if key == "biomass":
            ax.set_ylim(bottom=0)
    fig.suptitle("Control (neutralism common garden, identical metabolism, "
                 "biomass-fair growth): a mechanism ladder on a fixed body\n"
                 "axial division placement, not the alignment torque, sets the sectors "
                 "(making rod division isotropic reverts the metrics toward cocci); "
                 "the per-cell growth law is identical throughout",
                 fontsize=15.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGDIR, "cube3d_control.png")
    fig.savefig(out, dpi=125); plt.close(fig)
    print(f"wrote {out}")
    print("\nmedians by condition:")
    for d in data:
        r = d["rec"]
        md = lambda k: np.nanmedian(r[k]) if r[k] else float("nan")
        print(f"  {d['label'].replace(chr(10),' '):18s} order={md('order'):.2f} "
              f"xi={md('xi'):.2f} jc={md('jc'):.2f} S={md('S'):.0f} "
              f"biomass={md('biomass'):.0f} n={md('ncell'):.0f}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        plot(); return
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run(n_rep)


if __name__ == "__main__":
    main()
