"""
cube3d.py
=========

A 3D range expansion inside a cube. Cells are spherocylinders that start as a
small patch of founders on the floor and grow upward and outward. Only the
exposed surface of the colony grows; buried cells freeze. Cells whose centre
leaves the cube are dropped: they have simply grown out of the box.

Three interactions are provided, the three the brief asks for:
  commensalism      A leaks a by-product M for free; B scavenges it (0, +)
  public_good       A pays to secrete a public good P; B free-rides   (-, +)
  mutualism         facultative reciprocal by-product exchange         (+, +)

State is held in plain arrays so the whole thing stays readable:
  pos (n,3) centre, ax (n,3) unit axis, L (n,) cylinder length, R radius,
  sp (n,) strain 0/1, lin (n,) founder lineage id, alive (n,) bool.
"""

import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree

from field3d import Field3D


def vol_sphere(R):
    return (4.0 / 3.0) * np.pi * R ** 3


def vol_capsule(R, L):
    # cylinder of length L plus two hemispherical caps of radius R
    return np.pi * R * R * L + (4.0 / 3.0) * np.pi * R ** 3


def monod(c, k):
    return c / (k + c)


# ----------------------------------------------------------------------
# segment to segment closest distance in 3D, vectorised over pairs
# (clamped, after Ericson, Real-Time Collision Detection)
# ----------------------------------------------------------------------
def seg_seg(P1, Q1, P2, Q2):
    d1 = Q1 - P1
    d2 = Q2 - P2
    r = P1 - P2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-9, np.clip((b * f - c * e) / np.where(denom > 1e-9, denom, 1.0), 0, 1), 0.0)
    t = (b * s + f) / np.where(e > 1e-9, e, 1.0)
    t = np.clip(t, 0, 1)
    s = np.clip((b * t - c) / np.where(a > 1e-9, a, 1.0), 0, 1)
    c1 = P1 + d1 * s[:, None]
    c2 = P2 + d2 * t[:, None]
    diff = c1 - c2
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff) + 1e-12)
    return c1, c2, dist



# ----------------------------------------------------------------------
# cell morphologies: a strain's shape is one of these
# ----------------------------------------------------------------------
@dataclass
class MorphType:
    key: str
    name: str
    R: float            # radius
    L_birth: float      # cylinder length at birth (0 for cocci)
    L_div: float        # division length (rods/chains); cocci use a timer
    chaining: bool      # daughters stay bonded (a chain)?
    iscocci: bool       # round cell (grows by a maturation timer, splits any way)


COCCI = MorphType("cocci", "cocci", 0.367, 0.0, 0.333, False, True)
CHAIN = MorphType("chain", "long-rod chain", 0.253, 1.133, 2.533, True, False)
ROD = MorphType("rod", "short rod", 0.333, 0.60, 1.333, False, False)
MORPHS = [COCCI, CHAIN, ROD]


class Colony3D:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.pos = np.zeros((0, 3))
        self.ax = np.zeros((0, 3))
        self.L = np.zeros(0)
        self.R = np.zeros(0)                   # per-cell radius (shape)
        self.V = np.zeros(0)                   # per-cell current biomass (volume)
        self.Vb = np.zeros(0)                  # per-cell birth biomass (volume)
        self.Lb = np.zeros(0)                  # per-cell birth length
        self.Ld = np.zeros(0)                  # per-cell division length
        self.mtype = np.zeros(0, dtype=int)    # index into MORPHS
        self.chain = np.zeros(0, dtype=bool)
        self.iscocci = np.zeros(0, dtype=bool)
        self.mat = np.zeros(0)                 # cocci maturation clock
        self.matdiv = np.zeros(0)
        self.sp = np.zeros(0, dtype=int)
        self.lin = np.zeros(0, dtype=int)
        self.cid = np.zeros(0, dtype=int)      # unique id of each living cell
        self.alive = np.zeros(0, dtype=bool)
        self.geff = np.zeros(0)                # effective growth at last step
        self.links = set()                     # {(cid_a, cid_b)} chain bonds
        # persistent genealogy, indexed by cid (never culled), for lineage trees
        self.par = []        # parent cid (-1 for founders)
        self.bpos = []       # birth position
        self.bframe = []     # birth frame
        self.founder = []    # root founder id
        self.csp = []        # strain of each cid
        self.cmorph = []     # morphology (MORPHS index) of each cid

    @property
    def n(self):
        return self.pos.shape[0]

    def seed_floor(self, inter):
        """Seed founders on the floor. Each strain (A = 0, B = 1) is given a
        cell shape from inter.shapes = (A_morph_index, B_morph_index)."""
        cfg = self.cfg
        c = cfg.cube * 0.5
        m = cfg.n_seed
        shapes = getattr(inter, "shapes", (2, 2))

        # biomass-balanced seeding. seed_frac is the target biomass share of A,
        # not a head-count share, so a strain made of big rods gets fewer but
        # larger founders and neither shape starts with a biomass advantage.
        def _morph_vb(mi):
            mo = MORPHS[mi]
            if mo.iscocci:
                return vol_sphere(cfg.cocci_birth_R)
            return vol_capsule(mo.R, mo.L_birth)
        VbA = _morph_vb(int(shapes[0]))
        VbB = _morph_vb(int(shapes[1]))
        fa = float(getattr(inter, "seed_frac", 0.5))
        wA = VbB * fa
        wB = VbA * (1.0 - fa)
        pA = wA / (wA + wB + 1e-12)
        nA = int(round(m * pA))
        sp = np.ones(m, dtype=int)
        sp[:nA] = 0
        self.rng.shuffle(sp)
        midx = np.where(sp == 0, shapes[0], shapes[1]).astype(int)
        R = np.array([MORPHS[k].R for k in midx])
        isc0 = np.array([MORPHS[k].iscocci for k in midx])
        R = np.where(isc0, cfg.cocci_birth_R, R)   # cocci start at birth radius
        Lb = np.array([MORPHS[k].L_birth for k in midx])
        Ld = np.array([MORPHS[k].L_div for k in midx])
        chain = np.array([MORPHS[k].chaining for k in midx])
        isc = np.array([MORPHS[k].iscocci for k in midx])

        ang = self.rng.uniform(0, 2 * np.pi, m)
        rad = cfg.seed_radius * np.sqrt(self.rng.uniform(0, 1, m))
        x = c + rad * np.cos(ang)
        y = c + rad * np.sin(ang)
        z = R + 0.05
        self.pos = np.column_stack((x, y, z))
        theta = self.rng.uniform(0, 2 * np.pi, m)
        tilt = self.rng.uniform(0.0, 0.35, m)
        self.ax = np.column_stack((np.cos(theta) * np.cos(tilt),
                                   np.sin(theta) * np.cos(tilt), np.sin(tilt)))
        self._normalize()
        self.R = R
        self.Lb = Lb
        self.Ld = Ld
        self.chain = chain
        self.iscocci = isc
        self.mtype = midx
        self.L = Lb.copy()
        # biomass bookkeeping: every cell doubles its own birth volume per cycle
        self.Vb = np.where(isc, vol_sphere(R), vol_capsule(R, Lb))
        self.V = self.Vb.copy()
        self.mat = np.zeros(m)
        self.matdiv = np.full(m, 0.9)
        self.sp = sp
        self.lin = np.arange(m)
        self.cid = np.arange(m)
        self.alive = np.ones(m, dtype=bool)
        self.geff = np.zeros(m)
        self.links = set()
        self.par = [-1] * m
        self.bpos = [self.pos[i].copy() for i in range(m)]
        self.bframe = [0] * m
        self.founder = list(range(m))
        self.csp = [int(v) for v in self.sp]
        self.cmorph = [int(v) for v in self.mtype]

    def _normalize(self):
        n = np.linalg.norm(self.ax, axis=1, keepdims=True)
        n[n == 0] = 1.0
        self.ax = self.ax / n

    def endpoints(self):
        h = 0.5 * self.L[:, None] * self.ax
        return self.pos - h, self.pos + h

    def exposure(self):
        """Per cell surface-exposure score in [0,1]: 1 = sticking out, 0 = buried.

        Built from the resultant of unit vectors to nearby cells. A buried cell
        has neighbours all around so the resultant is small; a surface cell has
        neighbours mostly on one side so the resultant is large. A small upward
        bonus lets the top of the colony grow into the open medium.
        """
        cfg = self.cfg
        if self.n == 0:
            return np.zeros(0)
        tree = cKDTree(self.pos)
        nbrs = tree.query_ball_point(self.pos, cfg.front_radius)
        score = np.zeros(self.n)
        for i, nb in enumerate(nbrs):
            nb = [j for j in nb if j != i]
            if not nb:
                score[i] = 1.0
                continue
            d = self.pos[nb] - self.pos[i]
            dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
            res = np.linalg.norm(dn.mean(axis=0))
            score[i] = res
        return score

    def front_factor(self):
        s = self.exposure()
        cfg = self.cfg
        phi = np.clip((s - cfg.front_lo) / (cfg.front_hi - cfg.front_lo), 0, 1)
        return phi

    def support_factor(self):
        """Confinement: a cell grows in proportion to how crowded its neighbourhood
        is. A well-supported cell in the pack grows fully; an isolated cell poking
        into empty space gets little support and barely grows, so cells cannot fly
        off as free spikes (cf. Bhattacharjee et al. 2024, growth within a confining
        crowd). Combined with the nutrient field this keeps growth a supported,
        diffusion-limited front rather than runaway protrusions."""
        cfg = self.cfg
        n = self.n
        if n < 2:
            return np.ones(n)
        tree = cKDTree(self.pos)
        nb = tree.query_ball_point(self.pos, cfg.support_radius)
        cnt = np.array([len(x) - 1 for x in nb], dtype=float)
        return np.clip(cnt / cfg.support_full, 0.0, 1.0)

    def grow(self, eff, dt):
        """Biomass-based growth. Every cell increases its volume at rate eff so
        it doubles its own birth volume in time 1/eff, regardless of shape, then
        the geometry is read back from the volume: cocci grow their radius, rods
        and chains grow their length at fixed radius. This removes any biomass
        advantage that a particular shape would otherwise enjoy."""
        eff = np.maximum(eff, 0.0)
        self.V = self.V + eff * self.Vb * dt
        cm = self.iscocci
        # cocci: radius set by the sphere volume
        self.R[cm] = (3.0 * self.V[cm] / (4.0 * np.pi)) ** (1.0 / 3.0)
        # rods and chains: length set by the capsule volume at fixed radius
        rm = ~cm
        Lc = (self.V[rm] - vol_sphere(self.R[rm])) / (np.pi * self.R[rm] ** 2)
        self.L[rm] = np.maximum(Lc, 0.0)

    def divide(self, frame=0):
        rng = self.rng
        cfg = self.cfg
        # a cell divides once it has doubled its birth biomass, the same rule for
        # every shape; the two daughters split the biomass evenly
        ready = self.V >= 2.0 * self.Vb
        idx = np.where(self.alive & ready)[0]
        if idx.size == 0:
            return

        child_pos = np.zeros((idx.size, 3))
        child_ax = np.zeros((idx.size, 3))
        child_L = np.zeros(idx.size)
        for kk, i in enumerate(idx):
            if self.iscocci[i]:
                d = rng.normal(size=3)
                d /= np.linalg.norm(d) + 1e-9
                child_pos[kk] = self.pos[i] + 1.15 * self.R[i] * d
                child_ax[kk] = rng.normal(size=3)
                child_L[kk] = 0.0
                self.R[i] = cfg.cocci_birth_R
                self.V[i] = self.Vb[i]
            else:
                if getattr(cfg, "iso_division", False):
                    dd = rng.normal(size=3)
                    dd /= np.linalg.norm(dd) + 1e-9
                    off = 0.25 * self.L[i] * dd
                else:
                    off = 0.25 * self.L[i] * self.ax[i]
                child_pos[kk] = self.pos[i] + off
                self.pos[i] = self.pos[i] - off
                self.L[i] = self.Lb[i]
                self.V[i] = self.Vb[i]
                child_ax[kk] = self.ax[i] + rng.normal(0, 0.05, 3)
                child_L[kk] = self.Lb[i]

        base = len(self.par)
        new_cid = np.arange(base, base + idx.size)
        parent_cid = self.cid[idx]
        for kk in range(idx.size):
            pc = int(parent_cid[kk])
            self.par.append(pc)
            self.bpos.append(child_pos[kk].copy())
            self.bframe.append(frame)
            self.founder.append(self.founder[pc])
            self.csp.append(int(self.sp[idx[kk]]))
            self.cmorph.append(int(self.mtype[idx[kk]]))
            if self.chain[idx[kk]]:
                self.links.add((pc, int(new_cid[kk])))

        self.pos = np.vstack((self.pos, child_pos))
        self.ax = np.vstack((self.ax, child_ax))
        self.L = np.concatenate((self.L, child_L))
        child_R = self.R[idx].copy()
        child_R[self.iscocci[idx]] = cfg.cocci_birth_R
        self.R = np.concatenate((self.R, child_R))
        child_Vb = self.Vb[idx].copy()
        self.Vb = np.concatenate((self.Vb, child_Vb))
        self.V = np.concatenate((self.V, child_Vb.copy()))
        self.Lb = np.concatenate((self.Lb, self.Lb[idx].copy()))
        self.Ld = np.concatenate((self.Ld, self.Ld[idx].copy()))
        self.mtype = np.concatenate((self.mtype, self.mtype[idx].copy()))
        self.chain = np.concatenate((self.chain, self.chain[idx].copy()))
        self.iscocci = np.concatenate((self.iscocci, self.iscocci[idx].copy()))
        self.mat = np.concatenate((self.mat, np.zeros(idx.size)))
        self.matdiv = np.concatenate((self.matdiv, self.matdiv[idx].copy()))
        self.sp = np.concatenate((self.sp, self.sp[idx].copy()))
        self.lin = np.concatenate((self.lin, self.lin[idx].copy()))
        self.cid = np.concatenate((self.cid, new_cid))
        self.alive = np.concatenate((self.alive, np.ones(idx.size, dtype=bool)))
        self.geff = np.concatenate((self.geff, np.zeros(idx.size)))
        self._normalize()

    def spine_points(self, m=5):
        """m points sampled evenly along each cell's spine, shape (n, m, 3)."""
        t = np.linspace(-0.5, 0.5, m)
        return self.pos[:, None, :] + (t[None, :, None] * self.L[:, None, None]) * self.ax[:, None, :]

    def relax(self, iters=8):
        """Overdamped rigid-rod mechanics in 3D.

        Each contact applies a repulsive force along the contact normal at the
        contact point. The force translates a cell (drag grows with length) and,
        because it acts off the centre, also torques it (rotational drag grows
        with length cubed). The floor pushes up on any part of a cell dipping
        below it. There is no scripted upward bias: cells lie down, and the
        vertical structure that appears is buckling, the colony being shoved up
        out of a crowded basal layer. Per-iteration moves are clamped so the
        explicit solver stays stable.
        """
        cfg = self.cfg
        for _ in range(iters):
            n = self.n
            if n < 1:
                break
            R = self.R
            F = np.zeros((n, 3))
            Tq = np.zeros((n, 3))

            # local support (confinement): unsupported cells are the ones poking
            # into empty space; they get pulled flat and down so nothing flies,
            # while cells embedded in the dense mass grow up freely and fill the cube
            tree = cKDTree(self.pos)
            cnt = np.array([len(x) - 1
                            for x in tree.query_ball_point(self.pos,
                                                           cfg.support_radius)])
            unsup = 1.0 - np.clip(cnt / cfg.support_full, 0.0, 1.0)

            # cell to cell contacts, per-cell radius
            if n >= 2:
                P, Q = self.endpoints()
                cutoff = self.L.max() + 2 * R.max() + 0.5
                pairs = tree.query_pairs(cutoff, output_type="ndarray")
                if pairs.size:
                    i, j = pairs[:, 0], pairs[:, 1]
                    c1, c2, dist = seg_seg(P[i], Q[i], P[j], Q[j])
                    ov = (R[i] + R[j]) - dist
                    hit = ov > 0
                    if np.any(hit):
                        i, j = i[hit], j[hit]
                        c1, c2 = c1[hit], c2[hit]
                        dist, ov = dist[hit], ov[hit]
                        nrm = (c1 - c2) / dist[:, None]
                        f = cfg.k_contact * ov[:, None] * nrm
                        np.add.at(F, i, f)
                        np.add.at(F, j, -f)
                        np.add.at(Tq, i, np.cross(c1 - self.pos[i], f))
                        np.add.at(Tq, j, np.cross(c2 - self.pos[j], -f))

                        # collective nematic alignment: two elongated cells in
                        # contact rotate toward a common axis, so longer cells
                        # build aligned microdomains and gain the front advantage
                        # that longer cells show in range expansions
                        # (Smith et al. 2017 PNAS; Rahbar et al. 2024)
                        ee = (~self.iscocci[i]) & (~self.iscocci[j])
                        if cfg.k_nematic > 0 and np.any(ee):
                            ie, je = i[ee], j[ee]
                            ui, uj = self.ax[ie], self.ax[je]
                            sgn = np.sign(np.sum(ui * uj, axis=1))
                            sgn[sgn == 0] = 1.0
                            uj_s = sgn[:, None] * uj
                            np.add.at(Tq, ie,
                                      cfg.k_nematic * np.cross(ui, uj_s))
                            np.add.at(Tq, je,
                                      cfg.k_nematic * np.cross(uj, sgn[:, None] * ui))
            # chain bonds: a spring to end-to-end spacing plus gentle alignment
            if self.links:
                pos_of = {int(cid): k for k, cid in enumerate(self.cid)}
                for (ca, cb) in self.links:
                    a = pos_of.get(ca)
                    b = pos_of.get(cb)
                    if a is None or b is None:
                        continue
                    dvec = self.pos[b] - self.pos[a]
                    dist = np.linalg.norm(dvec) + 1e-9
                    dhat = dvec / dist
                    rest = 0.5 * (self.L[a] + self.L[b]) + 0.7 * (R[a] + R[b])
                    fl = cfg.k_link * (dist - rest) * dhat
                    F[a] += fl
                    F[b] -= fl
                    for kk in (a, b):
                        u = self.ax[kk]
                        sgn = np.sign(np.dot(u, dhat)) or 1.0
                        Tq[kk] += cfg.k_align * np.cross(u, sgn * dhat)

            # floor contact, per-cell radius
            sp = self.spine_points(5)
            pen = R[:, None] - sp[:, :, 2]
            below = pen > 0
            if np.any(below):
                fz = np.zeros_like(sp)
                fz[:, :, 2] = cfg.k_floor * np.where(below, pen, 0.0)
                F += fz.sum(axis=1)
                Tq += np.cross(sp - self.pos[:, None, :], fz).sum(axis=1)

            # settling: unsupported (protruding) cells are pulled down hard,
            # supported cells in the mass feel only a touch, so the colony fills
            # the cube as a dense front instead of throwing up free spikes
            size = self.L + 2 * R
            F[:, 2] -= cfg.gravity * size * (0.2 + unsup)

            # lay-flat torque, applied only to poorly-supported rods and chains:
            # a protruding filament is laid back into the colony, an embedded one
            # is left free to point any way the packing dictates
            h = self.ax.copy()
            h[:, 2] = 0.0
            hn = np.linalg.norm(h, axis=1, keepdims=True)
            flat = np.where(hn > 1e-6, h / np.where(hn > 1e-6, hn, 1.0), self.ax)
            lay = cfg.k_lay * (~self.iscocci)
            Tq += lay[:, None] * np.cross(self.ax, flat)

            zt = np.maximum(size, 1.0)[:, None]
            zr = np.maximum(size ** 3 * cfg.rot_drag, 1e-3)[:, None]
            dpos = F / zt
            mag = np.linalg.norm(dpos, axis=1, keepdims=True)
            dpos *= np.minimum(1.0, (0.4 * R[:, None]) / (mag + 1e-9))
            self.pos = self.pos + dpos

            omega = Tq / zr
            wn = np.linalg.norm(omega, axis=1, keepdims=True)
            omega *= np.minimum(1.0, 0.25 / (wn + 1e-9))
            self.ax = self.ax + np.cross(omega, self.ax)
            # rotational noise: real rod colonies never settle into one perfect
            # crystal, they break into local microdomains with curvature and
            # defects, so a little orientational jitter is added each step
            if cfg.ax_noise > 0.0 and self.n:
                jit = self.rng.normal(0.0, cfg.ax_noise, self.ax.shape)
                jit[self.iscocci] = 0.0
                self.ax = self.ax + jit
            self._normalize()
            self._floor()

    def _floor(self):
        zlow = self.pos[:, 2] - 0.5 * self.L * np.abs(self.ax[:, 2])
        below = zlow < self.R
        self.pos[below, 2] += (self.R[below] - zlow[below])

    def cull_outside(self):
        """Drop cells whose centre has left the cube (they grew out of the box)."""
        c = self.cfg.cube
        p = self.pos
        keep = (
            (p[:, 0] >= 0) & (p[:, 0] <= c)
            & (p[:, 1] >= 0) & (p[:, 1] <= c)
            & (p[:, 2] >= 0) & (p[:, 2] <= c)
            & self.alive
        )
        if keep.all():
            return
        for a in ("pos", "ax", "L", "R", "V", "Vb", "Lb", "Ld", "mtype",
                  "chain", "iscocci", "mat", "matdiv", "sp", "lin", "cid",
                  "geff", "alive"):
            setattr(self, a, getattr(self, a)[keep])
        self._cull_floaters()

    def _cull_floaters(self):
        """Remove cells that cannot be in contact with any other cell (true
        floaters). A cell can only touch a neighbour whose centre is within its
        own reach (half length plus radius) plus the neighbour's; using the
        largest cell as the neighbour bound gives a safe, generous test, so only
        genuinely detached cells are dropped."""
        n = self.n
        if n < 3:
            return
        reach = 0.5 * self.L + self.R
        bound = reach + reach.max() + 0.25
        tree = cKDTree(self.pos)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            nb = tree.query_ball_point(self.pos[i], bound[i])
            if len(nb) <= 1:
                keep[i] = False
        if keep.all():
            return
        for a in ("pos", "ax", "L", "R", "V", "Vb", "Lb", "Ld", "mtype",
                  "chain", "iscocci", "mat", "matdiv", "sp", "lin", "cid",
                  "geff", "alive"):
            setattr(self, a, getattr(self, a)[keep])


# ----------------------------------------------------------------------
# interactions: each returns mu (elongation rate) and edits the fields
# ----------------------------------------------------------------------
class Inter3D:
    name = "base"
    row = "base"
    signs = ("0", "0")
    seed_frac = 0.5
    shapes = (2, 2)          # (A_morph, B_morph) indices into MORPHS
    roles = ("A", "B")

    def fields(self, cfg):
        return {}

    def step(self, col, F, cfg, dt):
        raise NotImplementedError


def _S(cfg):
    return Field3D(cfg.N, cfg.dx, cfg.D_S, cfg.dt, c0=cfg.S0,
                   boundary="top", reservoir=cfg.S0)


def _deposit_body(field, col, per_cell, msample=3):
    """Deposit a per-cell amount spread over the cell body (several spine
    points), which is more realistic than a point sink and couples the cells to
    the field strongly enough to draw a real gradient."""
    pts = col.spine_points(msample)                  # (n, m, 3)
    flat = pts.reshape(-1, 3)
    amt = np.repeat(per_cell / msample, msample)
    field.deposit(flat, amt)


class Commensalism3D(Inter3D):
    name = "Commensalism (0, +)"
    row = "commensalism"
    signs = ("0", "+")
    shapes = (2, 0)          # A short rod, B cocci
    roles = ("producer", "commensal consumer")
    display_fields = [("Substrate S (feeds A)", "S"),
                      ("Metabolite M (feeds B)", "M")]

    def fields(self, cfg):
        return {"S": _S(cfg),
                "M": Field3D(cfg.N, cfg.dx, cfg.D_M, cfg.dt, c0=0.0,
                             boundary="zeroflux", decay=cfg.decay_M)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        S = F["S"].sample(p)
        M = F["M"].sample(p)
        a = col.sp == 0
        b = ~a
        mu = np.zeros(col.n)
        mu[a] = cfg.g_max * monod(S[a], cfg.K_S)
        mu[b] = cfg.g_max * monod(M[b], 0.4 * cfg.K_M)
        _deposit_body(F["S"], col, np.where(a, -cfg.Y_consume * mu, 0.0) * dt)
        F["M"].deposit(p, np.where(a, 1.8 * cfg.Y_produce * mu,
                                   -cfg.Y_consume * mu) * dt)
        return mu


class PublicGood3D(Inter3D):
    name = "Public good (-, +)"
    row = "public_good"
    signs = ("-", "+")
    shapes = (1, 0)          # A long-rod chain (producer), B cocci (free-rider)
    roles = ("producer (pays cost)", "free-rider")
    display_fields = [("Substrate S (feeds A & B)", "S"),
                      ("Public good P (feeds B)", "P")]

    def fields(self, cfg):
        return {"S": _S(cfg),
                "P": Field3D(cfg.N, cfg.dx, cfg.D_P, cfg.dt, c0=0.0,
                             boundary="zeroflux", decay=cfg.decay_P)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        S = F["S"].sample(p)
        P = F["P"].sample(p)
        a = col.sp == 0
        b = ~a
        base = cfg.g_max * monod(S, cfg.K_S)
        mu = np.zeros(col.n)
        mu[a] = base[a] * (1.0 - cfg.cost_public_good)
        mu[b] = base[b] * (1.0 + cfg.pg_gain * monod(P[b], cfg.K_P))
        _deposit_body(F["S"], col, -cfg.Y_consume * mu * dt)
        F["P"].deposit(p, np.where(a, cfg.Y_produce * base, 0.0) * dt)
        return mu


class Mutualism3D(Inter3D):
    name = "Facultative mutualism (+, +)"
    row = "mutualism"
    signs = ("+", "+")
    shapes = (2, 1)          # A short rod, B long-rod chain
    roles = ("cross-feeder", "cross-feeder")
    display_fields = [("Cross-fed Mb (feeds A)", "Mb"),
                      ("Cross-fed Ma (feeds B)", "Ma")]

    def fields(self, cfg):
        return {"S": _S(cfg),
                "Ma": Field3D(cfg.N, cfg.dx, cfg.D_M, cfg.dt, c0=0.0,
                              boundary="zeroflux", decay=cfg.decay_M),
                "Mb": Field3D(cfg.N, cfg.dx, cfg.D_M, cfg.dt, c0=0.0,
                              boundary="zeroflux", decay=cfg.decay_M)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        S = F["S"].sample(p)
        Ma = F["Ma"].sample(p)
        Mb = F["Mb"].sample(p)
        a = col.sp == 0
        b = ~a
        fb = cfg.fac_base
        s = monod(S, cfg.K_S)
        mu = np.zeros(col.n)
        mu[a] = cfg.g_max * s[a] * (fb + (1 - fb) * monod(Mb[a], cfg.K_M))
        mu[b] = cfg.g_max * s[b] * (fb + (1 - fb) * monod(Ma[b], cfg.K_M))
        _deposit_body(F["S"], col, -cfg.Y_consume * mu * dt)
        F["Ma"].deposit(p, np.where(a, cfg.Y_produce * mu,
                                    -0.5 * cfg.Y_consume * mu) * dt)
        F["Mb"].deposit(p, np.where(b, cfg.Y_produce * mu,
                                    -0.5 * cfg.Y_consume * mu) * dt)
        return mu


class Neutralism3D(Inter3D):
    name = "Neutralism (0, 0)"
    row = "neutralism"
    signs = ("0", "0")
    shapes = (2, 0)          # A short rod, B cocci
    roles = ("independent", "independent")
    display_fields = [("Substrate Sa (feeds A)", "Sa"),
                      ("Substrate Sb (feeds B)", "Sb")]

    def fields(self, cfg):
        return {"Sa": _S(cfg), "Sb": _S(cfg)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        Sa = F["Sa"].sample(p)
        Sb = F["Sb"].sample(p)
        a = col.sp == 0
        b = ~a
        mu = np.zeros(col.n)
        mu[a] = cfg.g_max * monod(Sa[a], cfg.K_S)
        mu[b] = cfg.g_max * monod(Sb[b], cfg.K_S)
        _deposit_body(F["Sa"], col, np.where(a, -cfg.Y_consume * mu, 0.0) * dt)
        _deposit_body(F["Sb"], col, np.where(b, -cfg.Y_consume * mu, 0.0) * dt)
        return mu


class Amensalism3D(Inter3D):
    name = "Amensalism (0, -)"
    row = "amensalism"
    signs = ("0", "-")
    shapes = (2, 0)          # A short rod (inhibitor maker), B cocci (inhibited)
    roles = ("inhibitor maker", "inhibited")
    display_fields = [("Substrate Sa (feeds A)", "Sa"),
                      ("Inhibitor I from A (harms B)", "I")]

    def fields(self, cfg):
        # A and B live on independent substrates, so the only cross-effect is
        # the inhibitor A leaks onto B; A stays genuinely unaffected (a clean 0)
        return {"Sa": _S(cfg), "Sb": _S(cfg),
                "I": Field3D(cfg.N, cfg.dx, cfg.D_P, cfg.dt, c0=0.0,
                             boundary="zeroflux", decay=cfg.decay_P)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        Sa = F["Sa"].sample(p)
        Sb = F["Sb"].sample(p)
        I = F["I"].sample(p)
        a = col.sp == 0
        b = ~a
        mua = cfg.g_max * monod(Sa, cfg.K_S)
        mub = cfg.g_max * monod(Sb, cfg.K_S)
        mu = np.zeros(col.n)
        mu[a] = mua[a]                                   # A on its own resource
        mu[b] = mub[b] / (1.0 + I[b] / cfg.K_I)           # B inhibited by A's I
        _deposit_body(F["Sa"], col, np.where(a, -cfg.Y_consume * mu, 0.0) * dt)
        _deposit_body(F["Sb"], col, np.where(b, -cfg.Y_consume * mu, 0.0) * dt)
        F["I"].deposit(p, np.where(a, 3.0 * cfg.Y_produce * mua, 0.0) * dt)
        return mu


class Competition3D(Inter3D):
    name = "Competition (-, -)"
    row = "competition"
    signs = ("-", "-")
    shapes = (0, 2)          # A cocci, B short rod, both drawing one substrate
    roles = ("competitor", "competitor")
    display_fields = [("Shared substrate S (feeds A)", "S"),
                      ("Shared substrate S (feeds B)", "S")]

    def fields(self, cfg):
        return {"S": _S(cfg)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        S = F["S"].sample(p)
        mu = cfg.g_max * monod(S, cfg.K_S)               # both grow on S
        # both draw the single shared substrate hard, so each depletes it for
        # the other: the (-, -) competition is emergent, not imposed
        _deposit_body(F["S"], col, -cfg.comp_draw * cfg.Y_consume * mu * dt)
        return mu


ALL3D = [Neutralism3D(), Commensalism3D(), Amensalism3D(),
         PublicGood3D(), Mutualism3D(), Competition3D()]


class Snapshot3D:
    __slots__ = ("pos", "ax", "L", "sp", "lin", "cid", "g", "R", "mtype")

    def __init__(self, col):
        self.pos = col.pos.copy()
        self.ax = col.ax.copy()
        self.L = col.L.copy()
        self.sp = col.sp.copy()
        self.lin = col.lin.copy()
        self.cid = col.cid.copy()
        self.g = col.geff.copy()       # effective growth rate at this frame
        self.R = col.R.copy()
        self.mtype = col.mtype.copy()


def run(inter, cfg, capture_field=None, capture_fields=None, shapes=None):
    """Grow one interaction.

    Returns (frames, field_history, genealogy). capture_field captures one field
    by name; capture_fields captures several. Each frame carries per-cell growth
    g (= mu * front factor). genealogy holds, indexed by unique cell id: parent
    cid, birth position, birth frame, root founder and species, including cells
    that later left the cube, so a full lineage tree can be rebuilt.
    """
    rng = np.random.default_rng(cfg.seed)
    col = Colony3D(cfg, rng)
    saved_shapes = getattr(inter, "shapes", (2, 2))
    if shapes is not None:
        inter.shapes = shapes
    try:
        col.seed_floor(inter)
    finally:
        inter.shapes = saved_shapes
    F = inter.fields(cfg)

    keys = list(capture_fields) if capture_fields else []
    if capture_field and capture_field not in keys:
        keys.append(capture_field)

    def record_growth():
        mu = inter.step(col, F, cfg, 0.0)      # dt=0: read mu, no field change
        col.geff = mu * col.front_factor()          # frontier-driven (range expansion)

    def grab():
        for k in keys:
            fhist[k].append(F[k].c.copy())

    record_growth()
    frames = [Snapshot3D(col)]
    fhist = {k: [] for k in keys}
    grab()

    for fr in range(1, cfg.n_frames):
        for _ in range(cfg.steps_per_frame):
            for f in F.values():
                f.step()
            mu = inter.step(col, F, cfg, cfg.dt)
            eff = mu * col.front_factor()            # frontier-driven (range expansion)
            col.geff = eff
            col.grow(eff, cfg.dt)
            col.divide(frame=fr)
            col.relax(cfg.relax_iters)
            col.cull_outside()
        record_growth()
        frames.append(Snapshot3D(col))
        grab()

    genealogy = {
        "parent": np.array(col.par),
        "bpos": np.array(col.bpos),
        "bframe": np.array(col.bframe),
        "founder": np.array(col.founder),
        "sp": np.array(col.csp),
        "mtype": np.array(col.cmorph),
    }
    return frames, fhist, genealogy
