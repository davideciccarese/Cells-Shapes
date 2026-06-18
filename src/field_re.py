"""
field_re.py
===========

A scalar diffusion field on a 2D grid for the range-expansion model, with the
same coupling interface as field3d.Field3D (sample, deposit, step) so the six
interaction classes can be shared between the two models with no change to their
chemistry.

Two boundary regimes:
  reservoir  the substrate is held at a fixed value on the border, as on a plate
             fed from outside, so the colony draws a real radial gradient as it
             consumes from the centre outward.
  zeroflux   secreted molecules use no flux on every edge, so they stay local and
             build up next to the cells that make them. This is the localising
             effect of spatial structure: in a structured habitat diffusion keeps
             secreted molecules concentrated next to their producers.

Grid index order is [ix, iy].
"""

import numpy as np


class FieldRE:
    def __init__(self, N, dx, D, dt, c0=0.0, boundary="reservoir",
                 reservoir=1.0, decay=0.0):
        self.N = N
        self.dx = dx
        self.D = D
        self.dt = dt
        self.boundary = boundary
        self.reservoir = reservoir
        self.decay = decay
        self.c = np.full((N, N), c0, dtype=np.float64)
        # explicit-diffusion substep count for 2D stability: D dt / dx^2 < 1/4
        r = D * dt / (dx * dx)
        self.sub = max(1, int(np.ceil(r / 0.20)))
        self.rs = r / self.sub
        self._apply_bc()

    def _apply_bc(self):
        c = self.c
        if self.boundary == "reservoir":
            R = self.reservoir
            c[0, :] = R
            c[-1, :] = R
            c[:, 0] = R
            c[:, -1] = R
        else:  # zeroflux on all edges
            c[0, :] = c[1, :]
            c[-1, :] = c[-2, :]
            c[:, 0] = c[:, 1]
            c[:, -1] = c[:, -2]

    def step(self):
        c = self.c
        for _ in range(self.sub):
            lap = (
                np.roll(c, 1, 0) + np.roll(c, -1, 0)
                + np.roll(c, 1, 1) + np.roll(c, -1, 1)
                - 4.0 * c
            )
            c += self.rs * lap
            if self.decay:
                c -= self.decay * self.dt / self.sub * c
            self._apply_bc()
        np.clip(c, 0.0, None, out=c)

    # ---- cell coupling: nearest-node sampling and deposition ----
    def _idx(self, pts):
        i = np.clip((pts / self.dx).astype(int), 0, self.N - 1)
        return i[:, 0], i[:, 1]

    def sample(self, pts):
        ix, iy = self._idx(pts)
        return self.c[ix, iy]

    def deposit(self, pts, amounts):
        ix, iy = self._idx(pts)
        np.add.at(self.c, (ix, iy), amounts)
        np.clip(self.c, 0.0, None, out=self.c)
