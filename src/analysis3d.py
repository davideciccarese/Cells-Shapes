"""
analysis3d.py
=============

Analyses on the 3D cube runs.

1. Spatial lineage trees. For each interaction and each species we pick a mother
   cell near the centre of the founding patch (central founders are hemmed in,
   so their clones are the ones that get shoved upward) and draw that clone's
   spatial tree: each division an edge from parent to daughter birth position,
   coloured by time of birth, with the longest internal route highlighted.

2. Individual clones. The largest few clones near the centre, one species per
   row of cubes, either one clone per cube (static) or several clones per cube
   each in its own colour (animated).

3. Growth and spatial organisation: correlation length of growth, growth versus
   height (a lucky place), and growth over height and time.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.distance import pdist, squareform
from PIL import Image

from render3d import _cube_edges, MORPH_COLORS, MORPH_NAMES

A_COLOR = "#2563eb"
B_COLOR = "#16a34a"


# ----------------------------------------------------------------------
# genealogy helpers
# ----------------------------------------------------------------------
def depth_array(gen):
    """Generation depth of every cell (parents always precede children)."""
    parent = gen["parent"]
    M = parent.shape[0]
    d = np.zeros(M, dtype=int)
    for c in range(M):
        p = parent[c]
        d[c] = 0 if p < 0 else d[p] + 1
    return d


def founder_stats(gen, cfg, depth):
    """Per-founder summary: size, species, radial distance of the mother cell
    from the cube centre, deepest tip and generation count."""
    founder = gen["founder"]
    sp = gen["sp"]
    bpos = gen["bpos"]
    c = cfg.center
    stats = {}
    for f in np.unique(founder):
        f = int(f)
        mem = np.where(founder == f)[0]
        fx, fy = bpos[f][0], bpos[f][1]
        tip = int(mem[np.argmax(depth[mem])])
        stats[f] = dict(size=int(mem.size), sp=int(sp[f]),
                        morph=int(gen["mtype"][f]),
                        rad=float(np.hypot(fx - c, fy - c)),
                        tip=tip, gen=int(depth[mem].max()), members=mem)
    return stats


def path_to_root(gen, tip):
    parent = gen["parent"]
    p, c = [], int(tip)
    while c >= 0:
        p.append(c)
        c = int(parent[c])
    return p[::-1]


def central_founders(stats, species, n, min_size=2):
    """The n founders of a species nearest the centre (with a real clone)."""
    cand = [f for f, s in stats.items()
            if s["sp"] == species and s["size"] >= min_size]
    cand.sort(key=lambda f: stats[f]["rad"])
    return cand[:n]


def central_founders_morph(stats, morph, n, min_size=2):
    """The n founders of a given cell type (morphology) nearest the centre."""
    cand = [f for f, st in stats.items()
            if st["morph"] == morph and st["size"] >= min_size]
    cand.sort(key=lambda f: stats[f]["rad"])
    return cand[:n]


def central_deep_founder(stats, species, cfg, min_size=2):
    """A central founder with a long route: among founders inside the inner part
    of the patch, the one whose clone reached the most generations."""
    inner = 0.6 * cfg.seed_radius
    cand = [f for f, s in stats.items() if s["sp"] == species
            and s["size"] >= min_size and s["rad"] <= inner]
    if not cand:
        cand = central_founders(stats, species, 1, min_size)
        return cand[0] if cand else None
    cand.sort(key=lambda f: -stats[f]["gen"])
    return cand[0]


def _clone_edges(gen, members):
    parent, bpos, bframe = gen["parent"], gen["bpos"], gen["bframe"]
    segs, bf = [], []
    for c in members:
        p = parent[c]
        if p < 0:
            continue
        segs.append([bpos[p], bpos[c]])
        bf.append(int(bframe[c]))
    return segs, (np.array(bf) if bf else np.zeros(0, int))


# ----------------------------------------------------------------------
# 1. lineage trees, species A and B, central founders
# ----------------------------------------------------------------------
def _setup_cube(ax, L):
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    _cube_edges(ax, L)


def _lineage_choices(runs, cfg, nfew=8):
    """For each interaction, the central mother cells of each cell type present
    (cocci, chain, rod). Columns are organised by cell type; every edge also
    carries its generation (division depth) so it can be colour-coded by
    division. Returns (data, gmax)."""
    out = []
    gmax = 1
    for inter, gen in runs:
        depth = depth_array(gen)
        stats = founder_stats(gen, cfg, depth)
        bpos, parent = gen["bpos"], gen["parent"]
        bframe = gen["bframe"]
        morphs_present = sorted(set(int(m) for m in gen["mtype"]))
        columns = []
        for m in morphs_present:
            fs = central_founders_morph(stats, m, nfew, min_size=2)
            clones = []
            for fdr in fs:
                mem = stats[fdr]["members"]
                segs, eg, ebf = [], [], []
                for c in mem:
                    p = parent[c]
                    if p < 0:
                        continue
                    segs.append([bpos[p], bpos[c]])
                    eg.append(int(depth[c]))
                    ebf.append(int(bframe[c]))
                eg = np.array(eg)
                if eg.size:
                    gmax = max(gmax, int(eg.max()))
                clones.append(dict(f=fdr, segs=segs, eg=eg, ebf=np.array(ebf),
                                   fpos=bpos[fdr], gen=stats[fdr]["gen"]))
            columns.append(dict(morph=m, clones=clones))
        out.append((inter, columns))
    return out, gmax


def _draw_lineage_panel(ax, clones, L, color, norm, cmap, upto=None):
    _setup_cube(ax, L)
    for cl in clones:
        fp = cl["fpos"]
        if len(cl["segs"]):
            m = (np.ones(len(cl["segs"]), bool) if upto is None
                 else (cl["ebf"] <= upto))
            if np.any(m):
                segs = [seg for seg, mm in zip(cl["segs"], m) if mm]
                cols = [cmap(norm(g)) for g, mm in zip(cl["eg"], m) if mm]
                ax.add_collection3d(Line3DCollection(segs, colors=cols,
                                                     linewidths=1.6, alpha=0.95))
        # founder marker in the cell-type colour, so the column still reads as a type
        ax.scatter([fp[0]], [fp[1]], [fp[2]], color=color, s=44,
                   marker="s", edgecolor="#111827", linewidth=0.5)


def _lineage_colorbar(fig, norm, cmap):
    cax = fig.add_axes([0.92, 0.32, 0.014, 0.36])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("generation (number of divisions)", fontsize=10,
                 color="#3f3f46")


def _lineage_typenote(fig):
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                          markerfacecolor=MORPH_COLORS[k], markeredgecolor="#111827",
                          label=MORPH_NAMES[k]) for k in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005),
               title="founder marker = cell type   (edges coloured by division)")


def _max_cols(data):
    return max(len(cols) for _inter, cols in data)


def travel_distance_figure(data, cfg, path):
    """How far cells of each type are carried within each interaction. For every
    cell alive at the end we take the distance from its birth position to its
    final position (how far the expansion advected it). The violins show the full
    per-cell distribution pooled over replicates; the significance test is a
    paired Wilcoxon on the per-replicate median travel of the two cell types, so
    the replicate structure is respected and individual cells are not treated as
    independent samples.

    data is a list of (inter, pooled_by_morph, medians_by_morph):
      pooled_by_morph  = {morph: 1D array of per-cell travel over all replicates}
      medians_by_morph = {morph: list of per-replicate median travel}
    """
    from scipy import stats as sps
    n = len(data)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.9 * nrow),
                             dpi=120)
    fig.patch.set_facecolor("white")
    axes = np.atleast_1d(axes).ravel()
    summary = {}
    for ax, (inter, pooled, medians) in zip(axes, data):
        morphs = sorted(int(m) for m in pooled)
        dvals, labels, colors, tops = [], [], [], []
        for m in morphs:
            t = np.asarray(pooled[m], float)
            if t.size < 3:
                continue
            dvals.append(t)
            labels.append(MORPH_NAMES[m])
            colors.append(MORPH_COLORS[m])
            tops.append(np.percentile(t, 97))
            summary.setdefault(inter.row, {})[MORPH_NAMES[m]] = float(np.median(t))
        if not dvals:
            ax.set_axis_off()
            continue
        parts = ax.violinplot(dvals, showmedians=True, widths=0.8)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_edgecolor("#27272a")
            pc.set_alpha(0.78)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_edgecolor("#27272a")
                parts[key].set_linewidth(1.0)
        for i, t in enumerate(dvals):
            ax.text(i + 1, np.median(t), f" {np.median(t):.1f}", va="center",
                    ha="left", fontsize=8.5, color="#18181b")
        # replicate-based significance between the two cell types
        if len(morphs) == 2:
            a = np.asarray(medians[morphs[0]], float)
            b = np.asarray(medians[morphs[1]], float)
            a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            p = np.nan
            if a.size >= 2 and b.size >= 2:
                try:
                    _, p = sps.wilcoxon(a, b)
                except ValueError:
                    _, p = sps.mannwhitneyu(a, b)
            star = ("***" if p < 1e-3 else "**" if p < 1e-2
                    else "*" if p < 5e-2 else "ns") if np.isfinite(p) else "ns"
            y = max(tops) * 1.04
            ax.plot([1, 1, 2, 2], [y - 0.05 * y, y, y, y - 0.05 * y],
                    color="#27272a", lw=1.0)
            lbl = f"{star}  p={p:.3f}" if np.isfinite(p) else star
            ax.text(1.5, y, lbl, ha="center", va="bottom", fontsize=8.5,
                    color="#18181b")
            ax.set_ylim(top=y * 1.18)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(inter.name, fontsize=11, color="#18181b")
        ax.set_ylabel("travel distance (birth to final)", fontsize=9)
        ax.grid(axis="y", color="#e4e4e7", linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle("How far each cell type is carried by the range expansion, "
                 "per interaction\n(violins = per-cell distribution pooled over "
                 "replicates; stars = paired Wilcoxon on per-replicate medians)",
                 fontsize=12.5, color="#18181b")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, summary


def lineage_tree_figure(runs, cfg, path):
    data, gmax = _lineage_choices(runs, cfg)
    n = len(data)
    ncol = _max_cols(data)
    L = cfg.cube
    norm = plt.Normalize(0, gmax)
    cmap = plt.get_cmap("magma")
    fig = plt.figure(figsize=(4.1 * ncol, 4.3 * n), dpi=120)
    fig.patch.set_facecolor("white")
    fig.suptitle("Lineage trees of the central mother cells: columns are cell "
                 "types, edges coloured by division (generation)", fontsize=13,
                 color="#18181b", y=0.997)
    for r, (inter, columns) in enumerate(data):
        for ci, col in enumerate(columns):
            ax = fig.add_subplot(n, ncol, r * ncol + ci + 1, projection="3d")
            color = MORPH_COLORS[col["morph"]]
            _draw_lineage_panel(ax, col["clones"], L, color, norm, cmap)
            ax.view_init(elev=16, azim=-50)
            ngen = max([c["gen"] for c in col["clones"]], default=0)
            ax.set_title(f"{MORPH_NAMES[col['morph']]}: "
                         f"{len(col['clones'])} clones, up to {ngen} gen",
                         fontsize=10.5, color=color, weight="bold")
            if ci == 0:
                ax.text2D(-0.08, 0.5, inter.name, transform=ax.transAxes,
                          fontsize=11.5, color="#18181b", rotation=90,
                          va="center", ha="right", weight="bold")
    fig.subplots_adjust(left=0.06, right=0.9, top=0.94, bottom=0.05,
                        wspace=0.02, hspace=0.16)
    _lineage_colorbar(fig, norm, cmap)
    _lineage_typenote(fig)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def lineage_tree_gif(runs, cfg, path, fps=7):
    data, gmax = _lineage_choices(runs, cfg)
    n = len(data)
    ncol = _max_cols(data)
    nf = cfg.n_frames
    L = cfg.cube
    norm = plt.Normalize(0, gmax)
    cmap = plt.get_cmap("magma")
    fig = plt.figure(figsize=(4.1 * ncol, 4.3 * n), dpi=115)
    fig.patch.set_facecolor("white")
    axes = [[fig.add_subplot(n, ncol, r * ncol + c + 1, projection="3d")
             for c in range(ncol)] for r in range(n)]
    fig.subplots_adjust(left=0.06, right=0.9, top=0.93, bottom=0.05,
                        wspace=0.02, hspace=0.16)
    _lineage_colorbar(fig, norm, cmap)
    _lineage_typenote(fig)

    imgs = []
    for fi in range(nf):
        azim = -55 + 70 * (fi / max(nf - 1, 1))
        for r, (inter, columns) in enumerate(data):
            for ci in range(ncol):
                ax = axes[r][ci]
                ax.clear()
                if ci < len(columns):
                    col = columns[ci]
                    color = MORPH_COLORS[col["morph"]]
                    _draw_lineage_panel(ax, col["clones"], L, color, norm, cmap,
                                        upto=fi)
                    ax.view_init(elev=16, azim=azim)
                    ax.set_title(f"{MORPH_NAMES[col['morph']]}: "
                                 f"{len(col['clones'])} central clones",
                                 fontsize=10.5, color=color, weight="bold")
                    if ci == 0:
                        ax.text2D(-0.08, 0.5, inter.name,
                                  transform=ax.transAxes, fontsize=11.5,
                                  color="#18181b", rotation=90, va="center",
                                  ha="right", weight="bold")
                else:
                    ax.set_axis_off()
        fig.subplots_adjust(left=0.06, right=0.9, top=0.93, bottom=0.05,
                            wspace=0.02, hspace=0.16)
        fig.suptitle("Lineage trees of central mother cells: columns = cell "
                     "type, edges coloured by division, over time",
                     fontsize=12.5, color="#18181b", y=0.985)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(Image.fromarray(buf[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path


def _central_per_type(gen, stats, n_per_type):
    """The n_per_type most central founders (with a real clone) of EACH cell
    type present, so every cell type is represented by the same number of
    founders in the matrix panels."""
    morphs = sorted(set(int(gen["mtype"][f]) for f in stats))
    chosen = []
    for m in morphs:
        cand = [f for f in stats
                if int(gen["mtype"][f]) == m and stats[f]["size"] >= 2]
        cand.sort(key=lambda f: stats[f]["rad"])
        chosen.extend(cand[:n_per_type])
    return chosen


def _combo_clones(gen, cfg, nfew, depth=None):
    """Central founders of a single run, the same number per cell type, with
    edges tagged by generation, for one cell of the matrix."""
    if depth is None:
        depth = depth_array(gen)
    stats = founder_stats(gen, cfg, depth)
    chosen = _central_per_type(gen, stats, nfew)
    clones = []
    gmax = 1
    for f in chosen:
        mem = stats[f]["members"]
        segs, eg, ebf = [], [], []
        for c in mem:
            p = gen["parent"][c]
            if p < 0:
                continue
            segs.append([gen["bpos"][p], gen["bpos"][c]])
            eg.append(int(depth[c]))
            ebf.append(int(gen["bframe"][c]))
        eg = np.array(eg)
        if eg.size:
            gmax = max(gmax, int(eg.max()))
        clones.append(dict(segs=segs, eg=eg, ebf=np.array(ebf),
                           fpos=gen["bpos"][f], morph=int(gen["mtype"][f])))
    return clones, gmax


def _combo_clones_colored(gen, cfg, nfew):
    """Central clones of one run, the same number per cell type, each tagged with
    a distinct palette colour and its cell type, for the clones matrix."""
    depth = depth_array(gen)
    stats = founder_stats(gen, cfg, depth)
    chosen = _central_per_type(gen, stats, nfew)
    palette = list(plt.get_cmap("tab10").colors)
    clones = []
    for i, f in enumerate(chosen):
        segs, bf = _clone_edges(gen, stats[f]["members"])
        clones.append(dict(segs=segs, bf=np.array(bf), color=palette[i % 10],
                           fpos=gen["bpos"][f], morph=int(gen["mtype"][f])))
    return clones


def clones_matrix_gif(inter, gens, cfg, path, fps=7, nfew=5):
    """One animated 3 by 3 clones grid for an interaction, the same layout as the
    shape and lineage matrices: A shape on the rows, B shape on the columns. Each
    cell shows the central clones of that pairing, one colour per clone, so the
    clonal sectors of the range expansion are visible per shape combination."""
    L = cfg.cube
    nf = cfg.n_frames
    cells = [[_combo_clones_colored(gens[a][b], cfg, nfew) for b in range(3)]
             for a in range(3)]

    fig = plt.figure(figsize=(9.8, 10.4), dpi=92)
    fig.patch.set_facecolor("white")
    axes = [[fig.add_subplot(3, 3, a * 3 + b + 1, projection="3d")
             for b in range(3)] for a in range(3)]
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.075,
                        wspace=0.0, hspace=0.04)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                          markerfacecolor=MORPH_COLORS[k], markeredgecolor="#111827",
                          label=MORPH_NAMES[k]) for k in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.005),
               title="founder marker = cell type   (one colour per clone)")

    imgs = []
    for fi in range(nf):
        azim = -55 + 70 * (fi / max(nf - 1, 1))
        for a in range(3):
            for b in range(3):
                ax = axes[a][b]
                ax.clear()
                _draw_clone_panel(ax, cells[a][b], L, upto=fi)
                # founder markers in the cell-type colour
                for c in cells[a][b]:
                    fp = c["fpos"]
                    ax.scatter([fp[0]], [fp[1]], [fp[2]],
                               color=MORPH_COLORS[c["morph"]], s=34, marker="s",
                               edgecolors="#111827", linewidths=0.4)
                ax.view_init(elev=15, azim=azim)
                if a == 0:
                    ax.set_title("B: " + MORPH_NAMES[b], fontsize=10.5,
                                 color=MORPH_COLORS[b], weight="bold")
                if b == 0:
                    ax.text2D(-0.13, 0.5, "A: " + MORPH_NAMES[a],
                              transform=ax.transAxes, rotation=90, va="center",
                              ha="right", fontsize=10.5,
                              color=MORPH_COLORS[a], weight="bold")
        fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.075,
                            wspace=0.0, hspace=0.04)
        ra, rb = getattr(inter, "roles", ("A", "B"))
        fig.suptitle(inter.name + " clones: rows = A (" + ra
                     + ")     columns = B (" + rb + ")",
                     fontsize=11.5, color="#18181b", y=0.98)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(Image.fromarray(buf[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path


def lineage_matrix_gif(inter, gens, cfg, path, fps=7, nfew=5):
    """One animated 3 by 3 lineage grid for an interaction: A shape on the rows,
    B shape on the columns (matching the shape matrix). Each cell draws the
    central mother cells of that shape pairing, founder markers coloured by cell
    type, edges coloured by division (generation)."""
    L = cfg.cube
    nf = cfg.n_frames
    cells = [[None] * 3 for _ in range(3)]
    gmax = 1
    for a in range(3):
        for b in range(3):
            clones, gm = _combo_clones(gens[a][b], cfg, nfew)
            cells[a][b] = clones
            gmax = max(gmax, gm)
    norm = plt.Normalize(0, gmax)
    cmap = plt.get_cmap("magma")

    fig = plt.figure(figsize=(9.8, 10.4), dpi=92)
    fig.patch.set_facecolor("white")
    axes = [[fig.add_subplot(3, 3, a * 3 + b + 1, projection="3d")
             for b in range(3)] for a in range(3)]
    fig.subplots_adjust(left=0.07, right=0.9, top=0.9, bottom=0.075,
                        wspace=0.0, hspace=0.04)
    _lineage_colorbar(fig, norm, cmap)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                          markerfacecolor=MORPH_COLORS[k], markeredgecolor="#111827",
                          label=MORPH_NAMES[k]) for k in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.005),
               title="founder marker = cell type   (edges = division)")

    imgs = []
    for fi in range(nf):
        azim = -55 + 70 * (fi / max(nf - 1, 1))
        for a in range(3):
            for b in range(3):
                ax = axes[a][b]
                ax.clear()
                _draw_lineage_panel(ax, [dict(segs=c["segs"], eg=c["eg"],
                                              ebf=c["ebf"], fpos=c["fpos"],
                                              f=0, gen=0)
                                         for c in cells[a][b]], L, "#111827",
                                    norm, cmap, upto=fi)
                # recolour founder markers by their own cell type
                for c in cells[a][b]:
                    fp = c["fpos"]
                    ax.scatter([fp[0]], [fp[1]], [fp[2]],
                               color=MORPH_COLORS[c["morph"]], s=42, marker="s",
                               edgecolor="#111827", linewidth=0.5)
                ax.view_init(elev=15, azim=azim)
                if a == 0:
                    ax.set_title("B: " + MORPH_NAMES[b], fontsize=10.5,
                                 color=MORPH_COLORS[b], weight="bold")
                if b == 0:
                    ax.text2D(-0.13, 0.5, "A: " + MORPH_NAMES[a],
                              transform=ax.transAxes, rotation=90, va="center",
                              ha="right", fontsize=10.5,
                              color=MORPH_COLORS[a], weight="bold")
        fig.subplots_adjust(left=0.07, right=0.9, top=0.9, bottom=0.075,
                            wspace=0.0, hspace=0.04)
        ra, rb = getattr(inter, "roles", ("A", "B"))
        fig.suptitle(inter.name + " lineages: rows = A (" + ra
                     + ")     columns = B (" + rb + ")",
                     fontsize=11.5, color="#18181b", y=0.98)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(Image.fromarray(buf[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path


# ----------------------------------------------------------------------
# 2. individual clones (central), species A top row, B bottom row
# ----------------------------------------------------------------------
def _clone_panels(runs, cfg, per_sp=5):
    """For every interaction and species, a few central clones, each tagged with
    its own colour and edge birth frames."""
    palette = list(plt.get_cmap("tab10").colors)
    data = []
    for inter, gen in runs:
        depth = depth_array(gen)
        stats = founder_stats(gen, cfg, depth)
        per = {}
        for spv in (0, 1):
            fs = central_founders(stats, spv, per_sp, min_size=2)
            clones = []
            for i, f in enumerate(fs):
                segs, bf = _clone_edges(gen, stats[f]["members"])
                clones.append(dict(segs=segs, bf=bf, color=palette[i % 10],
                                   fpos=gen["bpos"][f]))
            per[spv] = clones
        data.append((inter, per))
    return data


def _draw_clone_panel(ax, clones, L, upto=None):
    _setup_cube(ax, L)
    for cl in clones:
        if len(cl["segs"]):
            m = (np.ones(len(cl["segs"]), bool) if upto is None
                 else (cl["bf"] <= upto))
            if np.any(m):
                segs = [s for s, mm in zip(cl["segs"], m) if mm]
                ax.add_collection3d(Line3DCollection(
                    segs, colors=[cl["color"]] * len(segs), linewidths=1.6,
                    alpha=0.95))
        fp = cl["fpos"]
        ax.scatter([fp[0]], [fp[1]], [fp[2]], color=cl["color"], s=34,
                   marker="s", edgecolors="#111827", linewidths=0.4)


def clones_all_figure(runs, cfg, path, per_sp=5):
    data = _clone_panels(runs, cfg, per_sp)
    n = len(data)
    L = cfg.cube
    heads = [("Species A", A_COLOR), ("Species B", B_COLOR)]
    fig = plt.figure(figsize=(8.2, 4.3 * n), dpi=120)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Central clones in 3D, per interaction and species "
                 f"({per_sp} clones each, one colour per clone)",
                 fontsize=12.5, color="#18181b", y=0.997)
    for r, (inter, per) in enumerate(data):
        for cset in (0, 1):
            ax = fig.add_subplot(n, 2, r * 2 + cset + 1, projection="3d")
            _draw_clone_panel(ax, per[cset], L)
            ax.view_init(elev=16, azim=-50)
            label, scol = heads[cset]
            ax.set_title(f"{label}: {len(per[cset])} clones", fontsize=10.5,
                         color=scol)
            if cset == 0:
                ax.text2D(-0.08, 0.5, inter.name, transform=ax.transAxes,
                          fontsize=11.5, color="#18181b", rotation=90,
                          va="center", ha="right", weight="bold")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.02,
                        wspace=0.02, hspace=0.16)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def clones_all_gif(runs, cfg, path, per_sp=5, fps=7):
    data = _clone_panels(runs, cfg, per_sp)
    n = len(data)
    nf = cfg.n_frames
    L = cfg.cube
    heads = [("Species A", A_COLOR), ("Species B", B_COLOR)]
    fig = plt.figure(figsize=(8.2, 4.3 * n), dpi=115)
    fig.patch.set_facecolor("white")
    axes = [[fig.add_subplot(n, 2, r * 2 + c + 1, projection="3d")
             for c in range(2)] for r in range(n)]
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.02, hspace=0.16)
    imgs = []
    for fi in range(nf):
        azim = -55 + 70 * (fi / max(nf - 1, 1))
        for r, (inter, per) in enumerate(data):
            for cset in (0, 1):
                ax = axes[r][cset]
                ax.clear()
                _draw_clone_panel(ax, per[cset], L, upto=fi)
                ax.view_init(elev=16, azim=azim)
                label, scol = heads[cset]
                ax.set_title(f"{label}: {len(per[cset])} central clones",
                             fontsize=10.5, color=scol)
                if cset == 0:
                    ax.text2D(-0.08, 0.5, inter.name, transform=ax.transAxes,
                              fontsize=11.5, color="#18181b", rotation=90,
                              va="center", ha="right", weight="bold")
        fig.suptitle("Central clones in 3D, per interaction and species "
                     "(one colour per clone), growing over time",
                     fontsize=12.5, color="#18181b", y=0.985)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(Image.fromarray(buf[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path


def species_clones_figure(inter, gen, cfg, path, ncols=4):
    depth = depth_array(gen)
    stats = founder_stats(gen, cfg, depth)
    nf = cfg.n_frames
    norm = plt.Normalize(0, nf - 1)
    cmap = plt.get_cmap("plasma")
    L = cfg.cube
    fig = plt.figure(figsize=(3.3 * ncols, 7.2), dpi=120)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{inter.name}: central clones in 3D (top species A, "
                 "bottom species B; edges coloured by time of birth)",
                 fontsize=12.5, color="#18181b", y=0.985)
    rows = [("Species A", A_COLOR, 0), ("Species B", B_COLOR, 1)]
    for r, (label, scol, spv) in enumerate(rows):
        clones = central_founders(stats, spv, ncols)
        for k in range(ncols):
            ax = fig.add_subplot(2, ncols, r * ncols + k + 1, projection="3d")
            _setup_cube(ax, L)
            if k < len(clones):
                f = clones[k]
                s = stats[f]
                segs, bf = _clone_edges(gen, s["members"])
                if segs:
                    ax.add_collection3d(Line3DCollection(
                        segs, colors=[cmap(norm(b)) for b in bf],
                        linewidths=1.4, alpha=0.9))
                fp = gen["bpos"][f]
                ax.scatter([fp[0]], [fp[1]], [fp[2]], color=scol, s=45,
                           marker="s")
                ax.set_title(f"clone {f}: {s['size']} cells, {s['gen']} gen",
                             fontsize=9.5, color="#18181b")
            ax.view_init(elev=15, azim=-50)
            if k == 0:
                ax.text2D(-0.12, 0.5, label, transform=ax.transAxes,
                          fontsize=12, color=scol, rotation=90,
                          va="center", ha="right", weight="bold")
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.02,
                        wspace=0.05, hspace=0.18)
    cax = fig.add_axes([0.92, 0.3, 0.013, 0.4])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("time of birth (frame)", fontsize=10, color="#3f3f46")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def species_clones_gif(inter, gen, cfg, path, ncubes=1, per_cube=4, fps=7):
    """Animated clones: one cube per species (A on top, B below), each holding a
    few central clones in distinct colours, growing over time."""
    depth = depth_array(gen)
    stats = founder_stats(gen, cfg, depth)
    nf = cfg.n_frames
    L = cfg.cube
    palette = list(plt.get_cmap("tab10").colors)
    nshow = ncubes * per_cube

    def organise(spv):
        flist = central_founders(stats, spv, nshow)
        cubes = [[] for _ in range(ncubes)]
        for i, f in enumerate(flist):
            segs, bf = _clone_edges(gen, stats[f]["members"])
            cubes[i // per_cube].append(
                dict(segs=segs, bf=bf, color=palette[i % per_cube],
                     fpos=gen["bpos"][f]))
        return cubes

    rows = [("Species A", A_COLOR, organise(0)),
            ("Species B", B_COLOR, organise(1))]

    fig = plt.figure(figsize=(4.8 * ncubes, 8.4), dpi=115)
    fig.patch.set_facecolor("white")
    axes = [[fig.add_subplot(2, ncubes, r * ncubes + c + 1, projection="3d")
             for c in range(ncubes)] for r in range(2)]
    fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.02,
                        wspace=0.05, hspace=0.12)

    imgs = []
    for fi in range(nf):
        azim = -55 + 70 * (fi / max(nf - 1, 1))
        for r, (label, scol, cubes) in enumerate(rows):
            for c in range(ncubes):
                ax = axes[r][c]
                ax.clear()
                _setup_cube(ax, L)
                for cl in cubes[c]:
                    m = cl["bf"] <= fi
                    if np.any(m):
                        segs = [s for s, mm in zip(cl["segs"], m) if mm]
                        ax.add_collection3d(Line3DCollection(
                            segs, colors=[cl["color"]] * len(segs),
                            linewidths=1.5, alpha=0.95))
                    fp = cl["fpos"]
                    ax.scatter([fp[0]], [fp[1]], [fp[2]], color=cl["color"],
                               s=30, marker="s")
                ax.view_init(elev=15, azim=azim)
                if c == 0:
                    ax.text2D(-0.1, 0.5, label, transform=ax.transAxes,
                              fontsize=12, color=scol, rotation=90,
                              va="center", ha="right", weight="bold")
        fig.suptitle(f"{inter.name}\ncentral clones (top A, bottom B)",
                     fontsize=12, color="#18181b", y=0.985)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(Image.fromarray(buf[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path


# ----------------------------------------------------------------------
# 3. growth and spatial organisation
# ----------------------------------------------------------------------
def _growth_correlation(P, g, nbins=22, rmax=None, maxpts=700):
    nlive = P.shape[0]
    if nlive < 30 or g.std() < 1e-9:
        return None
    if nlive > maxpts:
        sel = np.random.default_rng(0).choice(nlive, maxpts, replace=False)
        P, g = P[sel], g[sel]
    dg = g - g.mean()
    var = (dg * dg).mean()
    D = squareform(pdist(P))
    prod = np.outer(dg, dg) / (var + 1e-12)
    iu = np.triu_indices(P.shape[0], k=1)
    r, c = D[iu], prod[iu]
    if rmax is None:
        rmax = np.percentile(r, 80)
    bins = np.linspace(0, rmax, nbins + 1)
    which = np.digitize(r, bins) - 1
    rr, cc = [], []
    for b in range(nbins):
        m = which == b
        if m.sum() > 5:
            rr.append(0.5 * (bins[b] + bins[b + 1]))
            cc.append(c[m].mean())
    rr, cc = np.array(rr), np.array(cc)
    xi = np.nan
    below = np.where(cc <= 0)[0]
    if below.size and below[0] > 0:
        i = below[0]
        x0, x1, y0, y1 = rr[i - 1], rr[i], cc[i - 1], cc[i]
        xi = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
    return rr, cc, xi


def growth_analysis_figure(runs, cfg, path):
    n = len(runs)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.0 * n), dpi=120)
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = axes[None, :]
    L = cfg.cube
    nzb = 18
    zedges = np.linspace(0, L, nzb + 1)
    zc = 0.5 * (zedges[:-1] + zedges[1:])

    for r, (inter, frames) in enumerate(runs):
        last = frames[-1]
        ax = axes[r][0]
        out = _growth_correlation(last.pos, last.g)
        if out:
            rr, cc, xi = out
            ax.axhline(0, color="#9ca3af", lw=0.8)
            ax.plot(rr, cc, "-o", color="#2563eb", ms=3)
            if np.isfinite(xi):
                ax.axvline(xi, color="#dc2626", lw=1.5)
                ax.text(xi, 0.82, f"  xi = {xi:.1f}", color="#dc2626",
                        fontsize=10)
            ax.set_ylim(-0.35, 1.05)
        ax.set_xlabel("distance r", fontsize=10)
        ax.set_ylabel("growth correlation C(r)", fontsize=10)
        if r == 0:
            ax.set_title("Correlation length of growth\n(zero crossing = xi)",
                         fontsize=12)

        ax = axes[r][1]
        z, g = last.pos[:, 2], last.g
        idx = np.clip(np.digitize(z, zedges) - 1, 0, nzb - 1)
        mean_g = np.array([g[idx == b].mean() if np.any(idx == b) else np.nan
                           for b in range(nzb)])
        count = np.array([(idx == b).sum() for b in range(nzb)], dtype=float)
        ax.plot(mean_g, zc, "-o", color="#16a34a", ms=3)
        ax.set_xlabel("mean growth rate", fontsize=10, color="#16a34a")
        ax.set_ylabel("height z", fontsize=10)
        ax.tick_params(axis="x", colors="#16a34a")
        axb = ax.twiny()
        axb.plot(count / max(count.max(), 1), zc, color="#9ca3af", lw=1.2,
                 ls="--")
        axb.set_xlabel("relative biomass", fontsize=9, color="#6b7280")
        if np.any(np.isfinite(mean_g)):
            zbest = zc[np.nanargmax(mean_g)]
            ax.axhline(zbest, color="#dc2626", lw=1.2, ls=":")
            ax.text(ax.get_xlim()[1] * 0.5, zbest, f"  best z ~ {zbest:.1f}",
                    color="#dc2626", fontsize=9, va="bottom")
        if r == 0:
            ax.set_title("Is there a lucky height to grow?", fontsize=12)

        ax = axes[r][2]
        nf = len(frames)
        H = np.full((nzb, nf), np.nan)
        for fi, fr in enumerate(frames):
            if fr.pos.shape[0] == 0:
                continue
            zi = np.clip(np.digitize(fr.pos[:, 2], zedges) - 1, 0, nzb - 1)
            for b in range(nzb):
                m = zi == b
                if np.any(m):
                    H[b, fi] = fr.g[m].mean()
        im = ax.imshow(H, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, nf, 0, L], interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(
            "mean growth", fontsize=9)
        ax.set_xlabel("time (frame)", fontsize=10)
        ax.set_ylabel("height z", fontsize=10)
        if r == 0:
            ax.set_title("Growth over time and height", fontsize=12)

        axes[r][0].text(-0.28, 0.5, inter.name, transform=axes[r][0].transAxes,
                        fontsize=12, color="#18181b", rotation=90,
                        va="center", ha="right", weight="bold")

    fig.suptitle("Growth and spatial organisation in the 3D colony",
                 fontsize=15, color="#18181b", y=0.997)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.07,
                        wspace=0.32, hspace=0.30)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
