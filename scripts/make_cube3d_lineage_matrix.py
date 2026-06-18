"""
make_cube3d_lineage_matrix.py
=============================

For each interaction, render one animated 3 by 3 lineage grid that matches the
shape matrix: shape of partner A on the rows, shape of partner B on the columns,
all nine pairings. Each cell shows the central mother cells of that pairing,
founder markers coloured by cell type, edges coloured by division (generation).

    python scripts/make_cube3d_lineage_matrix.py            # all three
    python scripts/make_cube3d_lineage_matrix.py mutualism  # just one
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config3d import Config3D
import cube3d as C
import analysis3d as A


def main():
    cfg = Config3D()
    cfg.cube = 10.0
    cfg.seed_radius = 1.8
    cfg.n_seed = 45
    cfg.n_frames = 18

    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "figures"))
    os.makedirs(figdir, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else None

    for inter in C.ALL3D:
        if which and inter.row != which:
            continue
        t0 = time.time()
        gens = [[None] * 3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                inter.shapes = (a, b)
                _frames, _f, gen = C.run(inter, cfg)
                gens[a][b] = gen
        out = os.path.join(figdir, f"cube3d_lineage_{inter.row}.gif")
        A.lineage_matrix_gif(inter, gens, cfg, out)
        print(f"  wrote {out}  ({time.time() - t0:.1f}s)")
        outc = os.path.join(figdir, f"cube3d_clones_{inter.row}.gif")
        A.clones_matrix_gif(inter, gens, cfg, outc)
        print(f"  wrote {outc}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
