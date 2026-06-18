"""
make_cube3d_travel.py
=====================

For each interaction, measure how far every cell alive at the end was carried
from its birth position to its final position, and compare the cell types.

The violins show the full per-cell distribution pooled over replicates; the
significance star is a paired Wilcoxon on the per-replicate median travel of the
two cell types, so the replicate structure is respected.

    python scripts/make_cube3d_travel.py [n_rep]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from config3d import Config3D
import cube3d as C
import analysis3d as A


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cfg = Config3D()
    cfg.n_seed = 40
    cfg.n_frames = 16          # light so replicates are affordable
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "figures"))
    os.makedirs(figdir, exist_ok=True)

    data = []
    for inter in C.ALL3D:
        pooled = {}            # morph -> list of per-cell travel (all reps)
        medians = {}           # morph -> list of per-replicate medians
        s1, s2 = tuple(inter.shapes)
        orderings = [(s1, s2), (s2, s1)]   # counterbalance shape against role
        for r in range(n_rep):
            for j, order in enumerate(orderings):
                cfg.seed = 4000 + j * 1000 + r
                inter.shapes = order
                frames, _f, gen = C.run(inter, cfg)
                snap = frames[-1]
                travel = np.linalg.norm(snap.pos - gen["bpos"][snap.cid], axis=1)
                for m in sorted(set(int(x) for x in snap.mtype)):
                    t = travel[snap.mtype == m]
                    if t.size == 0:
                        continue
                    pooled.setdefault(m, []).append(t)
                    medians.setdefault(m, []).append(float(np.median(t)))
        inter.shapes = (s1, s2)
        pooled = {m: np.concatenate(v) for m, v in pooled.items()}
        data.append((inter, pooled, medians))
        print(f"  ran {inter.row}: {n_rep} replicates x 2 role assignments")

    out = os.path.join(figdir, "cube3d_travel.png")
    _, summary = A.travel_distance_figure(data, cfg, out)
    print(f"wrote {out}")
    print("median travel distance by cell type (pooled):")
    for row, d in summary.items():
        print("  %-13s %s" % (row, ", ".join(f"{k} {v:.1f}" for k, v in d.items())))


if __name__ == "__main__":
    main()
