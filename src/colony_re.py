"""
colony_re.py
============

A 2D range expansion of spherocylinder cells, the flat companion to cube3d.py.
Cells start as a small mixed disk of founders at the centre of a square plate and
grow outward; only the exposed rim grows and the buried core freezes, so a few
founding lineages surf the moving front into sectors while the interior mosaic is
locked in place.

The three cell shapes are identical to the 3D model:
  cocci   round cell, grows by inflating its radius, divides in a random direction
  chain   slender capsule whose daughters stay mechanically bonded
  rod     short capsule, divides along its long axis into separate cells

Growth is defined on biomass, so a cell of any shape doubles its own birth volume
in the same time and no shape begins with a size advantage. State is held in plain
arrays:
  pos (n,2) centre, ax (n,2) unit axis, L (n,) cylinder length, R (n,) radius,
  sp (n,) strain 0/1, lin (n,) founder lineage id, cid (n,) unique id.

A persistent genealogy (parent cid, birth position, birth frame, founder, strain,
morphology), never culled, lets a full lineage tree and the spatial-genetics
metrics be rebuilt after the run.
"""

import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree


def area_disk(R):
    return np.pi * R ** 2


def area_capsule(R, L):
    # 2D footprint of a stadium: central rectangle L x 2R plus a disk of radius R
    return 2.0 * R * L + np.pi * R ** 2


def monod(c, k):
    return c / (k + c)


# ----------------------------------------------------------------------
# closest points between two stacks of 2D segments (clamped, Ericson)
# ----------------------------------------------------------------------
def seg_seg_2d(p1, q1, p2, q2):
    eps = 1e-9
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    a = np.maximum(a, eps)
    e = np.maximum(e, eps)
    denom = a * e - b * b
    s = np.where(denom > eps, (b * f - c * e) / np.where(denom > eps, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = (b * s + f) / e
    lo = t < 0.0
    t = np.where(lo, 0.0, t)
    s = np.where(lo, np.clip(-c / a, 0.0, 1.0), s)
    hi = t > 1.0
    t = np.where(hi, 1.0, t)
    s = np.where(hi, np.clip((b - c) / a, 0.0, 1.0), s)
    c1 = p1 + d1 * s[:, None]
    c2 = p2 + d2 * t[:, None]
    diff = c1 - c2
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff) + 1e-12)
    return c1, c2, dist


# ----------------------------------------------------------------------
# cell morphologies (identical set to the 3D model)
# ----------------------------------------------------------------------
@dataclass
class MorphType:
    key: str
    name: str
    R: float
    L_birth: float
    L_div: float
    chaining: bool
    iscocci: bool


COCCI = MorphType("cocci", "cocci", 0.367, 0.0, 0.333, False, True)
CHAIN = MorphType("chain", "long-rod chain", 0.253, 1.133, 2.533, True, False)
ROD = MorphType("rod", "short rod", 0.333, 0.60, 1.333, False, False)
MORPHS = [COCCI, CHAIN, ROD]


class ColonyRE:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.pos = np.zeros((0, 2))
        self.ax = np.zeros((0, 2))
        self.L = np.zeros(0)
        self.R = np.zeros(0)
        self.V = np.zeros(0)
        self.Vb = np.zeros(0)
        self.Lb = np.zeros(0)
        self.Ld = np.zeros(0)
        self.mtype = np.zeros(0, dtype=int)
        self.chain = np.zeros(0, dtype=bool)
        self.iscocci = np.zeros(0, dtype=bool)
        self.sp = np.zeros(0, dtype=int)
        self.lin = np.zeros(0, dtype=int)
        self.cid = np.zeros(0, dtype=int)
        self.alive = np.zeros(0, dtype=bool)
        self.geff = np.zeros(0)
        self.links = set()
        self._frozen = None
        # persistent genealogy, indexed by cid (never culled)
        self.par = []
        self.bpos = []
        self.bframe = []
        self.founder = []
        self.csp = []
        self.cmorph = []

    @property
    def n(self):
        return self.pos.shape[0]

    def seed_disk(self, inter):
        """Seed founders as a small central disk. Each strain (A=0, B=1) is given
        a cell shape from inter.shapes = (A_morph_index, B_morph_index). Seeding is
        biomass-balanced, so neither shape starts with a footprint advantage."""
        cfg = self.cfg
        c = cfg.center
        m = cfg.n_seed
        shapes = getattr(inter, "shapes", (2, 2))

        def _morph_vb(mi):
            mo = MORPHS[mi]
            if mo.iscocci:
                return area_disk(cfg.cocci_birth_R)
            return area_capsule(mo.R, mo.L_birth)
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
        isc = np.array([MORPHS[k].iscocci for k in midx])
        R = np.where(isc, cfg.cocci_birth_R, R)
        Lb = np.array([MORPHS[k].L_birth for k in midx])
        Ld = np.array([MORPHS[k].L_div for k in midx])
        chain = np.array([MORPHS[k].chaining for k in midx])

        ang = self.rng.uniform(0, 2 * np.pi, m)
        rad = cfg.seed_radius * np.sqrt(self.rng.uniform(0, 1, m))
        x = c + rad * np.cos(ang)
        y = c + rad * np.sin(ang)
        self.pos = np.column_stack((x, y))
        theta = self.rng.uniform(0, 2 * np.pi, m)
        self.ax = np.column_stack((np.cos(theta), np.sin(theta)))
        self._normalize()
        self.R = R
        self.Lb = Lb
        self.Ld = Ld
        self.chain = chain
        self.iscocci = isc
        self.mtype = midx
        self.L = Lb.copy()
        self.Vb = np.where(isc, area_disk(R), area_capsule(R, Lb))
        self.V = self.Vb.copy()
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

    def spine_points(self, m=3):
        t = np.linspace(-0.5, 0.5, m)
        return self.pos[:, None, :] + (t[None, :, None] * self.L[:, None, None]) * self.ax[:, None, :]

    # -- range-expansion front (exposure isotropy score) -------------
    def front_factor(self):
        """Per-cell growth multiplier in [0,1] and a deep-core freeze mask, from
        the resultant of unit vectors to nearby cells: near 1 for an exposed rim
        cell, near 0 for a buried one. Growth ramps from zero in the interior to
        full at the rim, so only the margin advances (radial range expansion). The
        factor is smoothed over neighbours to damp single-cell fingering."""
        cfg = self.cfg
        n = self.n
        if n < 3:
            return np.ones(n), np.zeros(n, dtype=bool)
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(cfg.front_radius, output_type="ndarray")
        sumx = np.zeros(n)
        sumy = np.zeros(n)
        cnt = np.zeros(n)
        if pairs.size:
            i = pairs[:, 0]
            j = pairs[:, 1]
            v = self.pos[j] - self.pos[i]
            d = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)
            ux = v[:, 0] / d
            uy = v[:, 1] / d
            np.add.at(sumx, i, ux)
            np.add.at(sumy, i, uy)
            np.add.at(cnt, i, 1.0)
            np.add.at(sumx, j, -ux)
            np.add.at(sumy, j, -uy)
            np.add.at(cnt, j, 1.0)
        safe = np.maximum(cnt, 1.0)
        res = np.hypot(sumx, sumy) / safe
        res = np.where(cnt <= 2, 1.0, res)
        factor = np.clip((res - cfg.front_lo) / (cfg.front_hi - cfg.front_lo), 0.0, 1.0)
        if pairs.size and cfg.front_smooth_iters > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            for _ in range(cfg.front_smooth_iters):
                acc = factor.copy()
                num = np.ones(n)
                np.add.at(acc, i, factor[j])
                np.add.at(acc, j, factor[i])
                np.add.at(num, i, 1.0)
                np.add.at(num, j, 1.0)
                factor = acc / num
        frozen = np.zeros(n, dtype=bool)
        if cfg.freeze_core:
            frozen = (res < cfg.freeze_res) & (cnt >= cfg.freeze_count)
        return factor, frozen

    def support_factor(self):
        """Confinement gate in [0,1]. A cell grows in proportion to how crowded
        its neighbourhood is: a cell embedded in a dense crowd is fully supported
        and grows, an isolated cell poking into empty space gets little support
        and barely grows. Multiplying the front factor by this keeps growth a
        supported, diffusion-limited front and stops single cells from extending
        free spikes and detaching (cf. Bhattacharjee et al. 2024)."""
        cfg = self.cfg
        n = self.n
        if n < 2:
            return np.ones(n)
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(cfg.support_radius, output_type="ndarray")
        cnt = np.zeros(n)
        if pairs.size:
            cnt = np.bincount(pairs.ravel(), minlength=n).astype(float)
        return np.clip(cnt / cfg.support_full, 0.0, 1.0)

    # -- biomass growth and division ---------------------------------
    def grow(self, eff, dt):
        eff = np.maximum(eff, 0.0)
        self.V = self.V + eff * self.Vb * dt
        cm = self.iscocci
        # cocci: radius from disk area
        self.R[cm] = np.sqrt(self.V[cm] / np.pi)
        # rods and chains: length from capsule area at fixed radius
        rm = ~cm
        Lc = (self.V[rm] - np.pi * self.R[rm] ** 2) / (2.0 * self.R[rm])
        self.L[rm] = np.maximum(Lc, 0.0)

    def divide(self, frame=0):
        rng = self.rng
        cfg = self.cfg
        ready = self.V >= 2.0 * self.Vb
        room = cfg.max_cells - self.n
        idx = np.where(self.alive & ready)[0]
        if idx.size == 0:
            return
        if room <= 0:
            self.V[idx] = np.minimum(self.V[idx], 2.0 * self.Vb[idx])
            return
        if idx.size > room:
            idx = idx[:room]

        child_pos = np.zeros((idx.size, 2))
        child_ax = np.zeros((idx.size, 2))
        child_L = np.zeros(idx.size)
        for kk, i in enumerate(idx):
            if self.iscocci[i]:
                a = rng.uniform(0, 2 * np.pi)
                d = np.array([np.cos(a), np.sin(a)])
                child_pos[kk] = self.pos[i] + 1.15 * self.R[i] * d
                child_ax[kk] = d
                child_L[kk] = 0.0
                self.R[i] = cfg.cocci_birth_R
                self.V[i] = self.Vb[i]
            else:
                # split along the current long axis, then turn the mother and the
                # daughter by independent small angles so successive divisions do
                # not lay descendants in a straight radial line; the colony front
                # builds curved, locally aligned domains instead of spokes
                u = self.ax[i]
                if cfg.iso_division:
                    a = rng.uniform(0, 2 * np.pi)
                    u = np.array([np.cos(a), np.sin(a)])
                off = 0.25 * self.L[i] * u
                child_pos[kk] = self.pos[i] + off
                self.pos[i] = self.pos[i] - off
                self.L[i] = self.Lb[i]
                self.V[i] = self.Vb[i]
                th = np.arctan2(u[1], u[0])
                dm = rng.normal(0, cfg.div_angle_noise)
                dd = rng.normal(0, cfg.div_angle_noise)
                self.ax[i] = np.array([np.cos(th + dm), np.sin(th + dm)])
                child_ax[kk] = np.array([np.cos(th + dd), np.sin(th + dd)])
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
        self.sp = np.concatenate((self.sp, self.sp[idx].copy()))
        self.lin = np.concatenate((self.lin, self.lin[idx].copy()))
        self.cid = np.concatenate((self.cid, new_cid))
        self.alive = np.concatenate((self.alive, np.ones(idx.size, dtype=bool)))
        self.geff = np.concatenate((self.geff, np.zeros(idx.size)))
        self._normalize()

    # -- mechanics ----------------------------------------------------
    def relax(self, iters=None):
        cfg = self.cfg
        iters = cfg.relax_iters if iters is None else iters
        for _ in range(iters):
            n = self.n
            if n < 2:
                break
            R = self.R
            frozen = self._frozen if (self._frozen is not None
                                      and self._frozen.shape[0] == n) else None
            if frozen is not None and frozen.any():
                pos0 = self.pos.copy()
                ax0 = self.ax.copy()
            dpos = np.zeros((n, 2))
            dth = np.zeros(n)

            tree = cKDTree(self.pos)
            P, Q = self.endpoints()
            cutoff = self.L.max() + 2 * R.max() + 0.5
            pairs = tree.query_pairs(cutoff, output_type="ndarray")
            if pairs.size:
                i, j = pairs[:, 0], pairs[:, 1]
                c1, c2, dist = seg_seg_2d(P[i], Q[i], P[j], Q[j])
                ov = (R[i] + R[j]) - dist
                hit = ov > 0
                if np.any(hit):
                    i, j = i[hit], j[hit]
                    c1, c2 = c1[hit], c2[hit]
                    dist, ov = dist[hit], ov[hit]
                    nrm = (c1 - c2) / dist[:, None]
                    corr = 0.5 * cfg.relax_stiff * ov
                    f = corr[:, None] * nrm
                    np.add.at(dpos, i, f)
                    np.add.at(dpos, j, -f)
                    # off-centre contact -> 2D torque (scalar z of lever x normal)
                    ri = c1 - self.pos[i]
                    rj = c2 - self.pos[j]
                    ti = ri[:, 0] * nrm[:, 1] - ri[:, 1] * nrm[:, 0]
                    tj = rj[:, 0] * (-nrm[:, 1]) - rj[:, 1] * (-nrm[:, 0])
                    np.add.at(dth, i, cfg.torque_gain * corr * ti)
                    np.add.at(dth, j, cfg.torque_gain * corr * tj)
                    # nematic alignment between contacting elongated cells
                    ee = (~self.iscocci[i]) & (~self.iscocci[j])
                    if cfg.k_nematic > 0 and np.any(ee):
                        ie, je = i[ee], j[ee]
                        ui, uj = self.ax[ie], self.ax[je]
                        sgn = np.sign(np.sum(ui * uj, axis=1))
                        sgn[sgn == 0] = 1.0
                        uj_s = sgn[:, None] * uj
                        cz_i = ui[:, 0] * uj_s[:, 1] - ui[:, 1] * uj_s[:, 0]
                        ui_s = sgn[:, None] * ui
                        cz_j = uj[:, 0] * ui_s[:, 1] - uj[:, 1] * ui_s[:, 0]
                        np.add.at(dth, ie, cfg.k_nematic * cz_i)
                        np.add.at(dth, je, cfg.k_nematic * cz_j)

            # chain bonds: spring to end-to-end spacing plus gentle alignment
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
                    dpos[a] += fl
                    dpos[b] -= fl
                    for kk in (a, b):
                        u = self.ax[kk]
                        sgn = np.sign(np.dot(u, dhat)) or 1.0
                        cz = u[0] * (sgn * dhat[1]) - u[1] * (sgn * dhat[0])
                        dth[kk] += cfg.k_align * cz

            # overdamped update with per-step clamp for stability
            size = self.L + 2 * R
            zt = np.maximum(size, 1.0)[:, None]
            step = dpos / zt
            mag = np.linalg.norm(step, axis=1, keepdims=True)
            step *= np.minimum(1.0, (0.4 * R[:, None]) / (mag + 1e-9))
            self.pos = self.pos + step

            dth = np.clip(dth, -0.25, 0.25)
            cth = np.cos(dth)
            sth = np.sin(dth)
            ax_new = np.column_stack((
                cth * self.ax[:, 0] - sth * self.ax[:, 1],
                sth * self.ax[:, 0] + cth * self.ax[:, 1]))
            self.ax = ax_new
            if cfg.ax_noise > 0.0 and n:
                jit = self.rng.normal(0.0, cfg.ax_noise, self.ax.shape)
                jit[self.iscocci] = 0.0
                self.ax = self.ax + jit
            self._normalize()

            if frozen is not None and frozen.any():
                self.pos[frozen] = pos0[frozen]
                self.ax[frozen] = ax0[frozen]

    def cull_outside(self):
        """Drop cells whose centre has left the plate (grown out of the box)."""
        b = self.cfg.box
        p = self.pos
        keep = ((p[:, 0] >= 0) & (p[:, 0] <= b)
                & (p[:, 1] >= 0) & (p[:, 1] <= b) & self.alive)
        if not keep.all():
            self._apply_keep(keep)
        self._cull_floaters()

    def _cull_floaters(self):
        """Remove any cell whose nearest neighbour is beyond touching range, so a
        cell can only persist while it stays connected to the colony. The bound
        uses each cell's own reach (half length plus radius) plus the largest
        cell's reach, which only ever drops genuinely detached cells. Vectorised
        with one nearest-neighbour query."""
        n = self.n
        if n < 3:
            return
        reach = 0.5 * self.L + self.R
        tree = cKDTree(self.pos)
        dist, _ = tree.query(self.pos, k=2)
        nn = dist[:, 1]
        bound = reach + reach.max() + self.cfg.floater_gap
        keep = nn <= bound
        if not keep.all():
            self._apply_keep(keep)

    def _apply_keep(self, keep):
        for a in ("pos", "ax", "L", "R", "V", "Vb", "Lb", "Ld", "mtype",
                  "chain", "iscocci", "sp", "lin", "cid", "geff", "alive"):
            setattr(self, a, getattr(self, a)[keep])


class SnapshotRE:
    __slots__ = ("pos", "ax", "L", "sp", "lin", "cid", "g", "R", "mtype")

    def __init__(self, col):
        self.pos = col.pos.copy()
        self.ax = col.ax.copy()
        self.L = col.L.copy()
        self.sp = col.sp.copy()
        self.lin = col.lin.copy()
        self.cid = col.cid.copy()
        self.g = col.geff.copy()
        self.R = col.R.copy()
        self.mtype = col.mtype.copy()
