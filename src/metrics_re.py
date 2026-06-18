"""
metrics_re.py
=============

Spatial-genetics metrics for the range-expansion model. Every quantity here is
computed from the cell state of the 2D range-expansion simulation (positions,
founding lineages, strains), not from the 3D cube and not from any rendered
image. The repository uses this model, and only this model, for the metrics and
the statistical comparisons.

The clonal organisation of a radial range expansion is read on the moving front
as a function of angle. For each angular bin the outermost cell (largest radius)
is taken, giving a one-dimensional circular sequence of lineage labels l(theta).
All sector statistics are computed on this ring, so the radius at which they are
measured is explicit (the front) and there is no raster, no scan direction and no
compression window to bias the result.

Two further per-strain outcomes (lineage success at the rim, median travel
distance) drive the shape-by-sign factorial, because when both strains share a
shape a per-shape statistic would merge the two metabolic roles into one value.
"""

import zlib
import numpy as np


def frontier_ring(snap, cfg, n_bins=180):
    """Outermost cell per angular bin around the colony centre. Returns the
    circular sequence of lineage labels at occupied bins and the rim radii."""
    c = cfg.center
    dx = snap.pos[:, 0] - c
    dy = snap.pos[:, 1] - c
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    b = ((th + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    ring_lin = np.full(n_bins, -1, dtype=np.int64)
    ring_r = np.full(n_bins, -1.0)
    order = np.argsort(r)
    ring_lin[b[order]] = snap.lin[order]
    ring_r[b[order]] = r[order]
    occ = ring_lin >= 0
    return ring_lin[occ], ring_r[occ]


def sector_metrics(snap, cfg, n_bins=180):
    """Spatial-genetics descriptors of the clonal pattern at the front.

    Returns a dict with:
      S      surviving sector count (contiguous lineage arcs round the ring)
      D      number of surviving lineages at the rim (richness)
      xi     sector width: angular correlation length times rim radius (length)
      jc     join-count mixing ratio: observed boundaries / random expectation
             (< 1 segregated clean sectors, ~ 1 finely mixed; composition-free)
      Hcomp  Shannon entropy of rim lineage frequencies (aspatial composition)
      frag   sectors per surviving lineage S/D (configuration given composition)
      Rrim   median rim radius
    """
    seq, rr = frontier_ring(snap, cfg, n_bins)
    n = seq.size
    if n < 8:
        return None
    diff = seq != np.roll(seq, -1)
    S = int(diff.sum())
    vals, counts = np.unique(seq, return_counts=True)
    D = int(vals.size)
    p = counts / counts.sum()
    Hcomp = float(-(p * np.log(p)).sum())
    simpson = float((p ** 2).sum())
    E_S = n * (1.0 - simpson)
    jc = float(S / E_S) if E_S > 0 else float("nan")
    frag = float(S / max(D, 1))
    Rrim = float(np.median(rr))
    maxlag = max(2, n // 3)
    lags = np.arange(1, maxlag)
    H = np.array([np.mean(seq != np.roll(seq, -int(k))) for k in lags])
    Hinf = 1.0 - simpson
    xi = float("nan")
    if Hinf > 1e-6:
        target = (1.0 - 1.0 / np.e) * Hinf
        hit = np.where(H >= target)[0]
        if hit.size:
            k = hit[0]
            if k == 0:
                lag_c = 1.0
            else:
                h0, h1 = H[k - 1], H[k]
                frac = 0.0 if h1 == h0 else (target - h0) / (h1 - h0)
                lag_c = k + frac
            theta_c = lag_c * (2 * np.pi / n)
            xi = float(theta_c * Rrim)
    return dict(S=S, D=D, xi=xi, jc=jc, Hcomp=Hcomp, frag=frag, Rrim=Rrim)


def lineage_success(snap, gen, cfg, frontier_q=0.70):
    """Fraction of founders of each cell type whose lineage still holds a cell in
    the outer frontier rim. Returns {morph: (n_founders, n_success, rate)}."""
    pos = snap.pos
    c = cfg.center
    rad = np.hypot(pos[:, 0] - c, pos[:, 1] - c)
    if rad.size == 0:
        return {}
    thr = np.quantile(rad, frontier_q)
    front_lins = set(int(x) for x in snap.lin[rad >= thr].tolist())
    founders = np.where(gen["parent"] < 0)[0]
    fmorph = gen["mtype"][founders]
    out = {}
    for mo in sorted(set(int(x) for x in fmorph)):
        fids = founders[fmorph == mo]
        n = int(fids.size)
        succ = int(sum(1 for f in fids if int(f) in front_lins))
        out[mo] = (n, succ, succ / max(n, 1))
    return out


def strain_outcomes(snap, gen, cfg, frontier_q=0.70):
    """Per-strain lineage success rate and median travel distance, returned as
    {strain: (success_rate, median_travel)} for strain in {0, 1}."""
    pos = snap.pos
    c = cfg.center
    rad = np.hypot(pos[:, 0] - c, pos[:, 1] - c)
    if rad.size == 0:
        return {}
    thr = np.quantile(rad, frontier_q)
    front_lins = set(int(x) for x in snap.lin[rad >= thr].tolist())
    sp_founder = np.asarray(gen["sp"])
    parent = np.asarray(gen["parent"])
    founders = np.where(parent < 0)[0]
    bpos = np.asarray(gen["bpos"])
    travel = np.linalg.norm(pos - bpos[snap.cid], axis=1)
    sp_cell = sp_founder[snap.cid]
    out = {}
    for sp in (0, 1):
        fids = founders[sp_founder[founders] == sp]
        n = int(fids.size)
        succ = int(sum(1 for f in fids if int(f) in front_lins))
        tvals = travel[sp_cell == sp]
        med = float(np.median(tvals)) if tvals.size else float("nan")
        out[sp] = (succ / max(n, 1), med)
    return out


def lineage_complexity(snap, cfg, grid=64):
    """Compression complexity of the clonal raster, symmetrised over the raster
    and its transpose to cancel the row-major scan bias. Kept as a quick
    aggregate; the sector metrics above carry the spatial interpretation."""
    pos = snap.pos
    if pos.shape[0] < 8:
        return float("nan")
    c = cfg.box
    gx = np.clip((pos[:, 0] / c * grid).astype(int), 0, grid - 1)
    gy = np.clip((pos[:, 1] / c * grid).astype(int), 0, grid - 1)
    order = np.argsort(np.hypot(pos[:, 0] - cfg.center, pos[:, 1] - cfg.center))
    raster = np.full((grid, grid), -1, dtype=np.int64)
    raster[gy[order], gx[order]] = snap.lin[order]
    _, inv = np.unique(raster, return_inverse=True)
    relab = inv.reshape(raster.shape).astype(np.uint16)
    raw = relab.tobytes()
    rawT = np.ascontiguousarray(relab.T).tobytes()
    k = 0.5 * (len(zlib.compress(raw, 9)) + len(zlib.compress(rawT, 9)))
    return k / len(raw)
