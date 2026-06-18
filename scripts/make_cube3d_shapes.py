"""
make_cube3d_shapes.py
=====================

For each interaction, render one animated 3 by 3 grid of cell-shape pairings
(shape of partner A on the rows, shape of partner B on the columns), so all nine
shape combinations per interaction (27 in total) can be compared.

    python scripts/make_cube3d_shapes.py            # all three interactions
    python scripts/make_cube3d_shapes.py mutualism  # just one
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config3d import Config3D
import cube3d as C
from render_shapes import shape_matrix_gif


def main():
    cfg = Config3D()
    cfg.cube = 10.0
    cfg.seed_radius = 1.8
    cfg.n_seed = 45
    cfg.n_frames = 15

    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "figures"))
    os.makedirs(figdir, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else None

    for inter in C.ALL3D:
        if which and inter.row != which:
            continue
        t0 = time.time()
        grid = [[None] * 3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                inter.shapes = (a, b)
                frames, _f, _g = C.run(inter, cfg)
                grid[a][b] = frames
        out = os.path.join(figdir, f"cube3d_shapes_{inter.row}.gif")
        shape_matrix_gif(inter, grid, cfg, out)
        print(f"  wrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
