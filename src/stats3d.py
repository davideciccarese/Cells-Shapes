"""
stats3d.py
==========

Robust, replicate-based statistics for the range-expansion model. Two metrics
that are sturdier than a raw travel distance:

1. Lineage success rate. In a range expansion the founders that matter are the
   ones whose progeny reach the moving frontier (the surviving lineages of
   Blanchard and Lu 2015). For each cell type we report the fraction of its
   founders that still hold a cell in the outer rim at the end. Comparing the
   types across replicates gives a significance test of which shape wins.

2. Spatial lineage complexity. The clonal pattern is rasterised from the top and
   compressed; the normalised compressed size is a Lempel-Ziv proxy for the
   Kolmogorov complexity of the spatial arrangement. Clean wide sectors compress
   well (low complexity); finely mixed or fragmented patterns compress poorly
   (high complexity). This separates mixing interactions from segregating ones.

Both are computed per run so that replicate runs yield distributions and proper
tests (Wilcoxon, Mann-Whitney, Kruskal-Wallis) rather than single numbers.
"""

import zlib
import numpy as np


def lineage_success(snap, gen, cfg, frontier_q=0.70):
    """Fraction of founders of each cell type whose lineage still holds a cell in
    the outer frontier rim. Returns {morph: (n_founders, n_success, rate)}."""
    pos = snap.pos
    c = cfg.cube * 0.5
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


def lineage_complexity(snap, cfg, grid=64):
    """Lempel-Ziv proxy for the Kolmogorov complexity of the clonal arrangement.
    The top-down view is rasterised by lineage (highest cell wins each pixel),
    relabelled to compact ids, then zlib-compressed. Returns the normalised
    compressed size in [0, 1]; higher means a more complex, more mixed pattern."""
    pos = snap.pos
    if pos.shape[0] < 8:
        return float("nan")
    c = cfg.cube
    gx = np.clip((pos[:, 0] / c * grid).astype(int), 0, grid - 1)
    gy = np.clip((pos[:, 1] / c * grid).astype(int), 0, grid - 1)
    order = np.argsort(pos[:, 2])           # ascending z, higher z overwrites
    raster = np.full((grid, grid), -1, dtype=np.int64)
    raster[gy[order], gx[order]] = snap.lin[order]
    # compact relabelling so the raw byte alphabet is comparable across runs
    _, inv = np.unique(raster, return_inverse=True)
    relab = inv.reshape(raster.shape).astype(np.uint16)
    raw = relab.tobytes()
    comp = zlib.compress(raw, 9)
    return len(comp) / len(raw)


def run_replicates(inter, cfg, run_fn, n_rep=12, base_seed=0):
    """Run n_rep independent replicates of one interaction and collect both
    metrics per replicate. run_fn(inter, cfg) -> (frames, fields, gen)."""
    succ_rates = {}          # morph -> list of per-replicate rates
    succ_counts = {}         # morph -> list of (n, success)
    complexity = []
    for r in range(n_rep):
        cfg.seed = base_seed + r
        frames, _f, gen = run_fn(inter, cfg)
        snap = frames[-1]
        ls = lineage_success(snap, gen, cfg)
        for mo, (n, s, rate) in ls.items():
            succ_rates.setdefault(mo, []).append(rate)
            succ_counts.setdefault(mo, []).append((n, s))
        complexity.append(lineage_complexity(snap, cfg))
    return {
        "row": inter.row,
        "name": inter.name,
        "roles": list(getattr(inter, "roles", ("A", "B"))),
        "success_rates": {int(k): v for k, v in succ_rates.items()},
        "success_counts": {int(k): v for k, v in succ_counts.items()},
        "complexity": complexity,
    }


# ---------------------------------------------------------------------------
# Spatial-population-genetics metrics on the frontier ring
#
# The clonal organisation of a radial range expansion is read on the moving
# front as a function of angle. For each angular bin we take the OUTERMOST cell
# (largest radius), giving a one-dimensional circular sequence of lineage labels
# l(theta). All sector statistics are computed on this ring, so the radius at
# which we measure is explicit (the front) and there is no 2D raster, no scan
# direction and no compression window to bias the result.
# ---------------------------------------------------------------------------

def frontier_ring(snap, cfg, n_bins=180):
    """Outermost cell per angular bin around the colony axis. Returns the
    circular sequence of lineage labels at occupied bins (in angular order) and
    the median rim radius."""
    c = cfg.cube * 0.5
    dx = snap.pos[:, 0] - c
    dy = snap.pos[:, 1] - c
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)                      # -pi .. pi
    b = ((th + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    ring_lin = np.full(n_bins, -1, dtype=np.int64)
    ring_r = np.full(n_bins, -1.0)
    order = np.argsort(r)                         # ascending; largest r wins bin
    ring_lin[b[order]] = snap.lin[order]
    ring_r[b[order]] = r[order]
    occ = ring_lin >= 0
    return ring_lin[occ], ring_r[occ]


def sector_metrics(snap, cfg, n_bins=180):
    """Spatial-genetics descriptors of the clonal pattern at the front.

    Returns a dict with:
      S      surviving sector count  (number of contiguous lineage arcs, drift)
      D      number of surviving lineages at the rim (richness)
      xi     sector width: angular correlation length times rim radius (length)
      jc     join-count mixing ratio: observed boundaries / random expectation
             (< 1 segregated clean sectors, ~1 finely mixed; composition-free)
      Hcomp  Shannon entropy of rim lineage frequencies (aspatial composition)
      frag   sectors per surviving lineage S/D (configuration given composition)
    """
    seq, rr = frontier_ring(snap, cfg, n_bins)
    n = seq.size
    if n < 8:
        return None
    # circular boundaries between unlike neighbours = surviving sector count
    diff = seq != np.roll(seq, -1)
    S = int(diff.sum())
    vals, counts = np.unique(seq, return_counts=True)
    D = int(vals.size)
    p = counts / counts.sum()
    Hcomp = float(-(p * np.log(p)).sum())
    simpson = float((p ** 2).sum())              # prob two cells share a lineage
    # expected boundaries if the same labels were arranged at random round the
    # ring: each adjacent pair differs with prob (1 - simpson). Dividing by this
    # removes the richness (composition) contribution from the mixing measure.
    E_S = n * (1.0 - simpson)
    jc = float(S / E_S) if E_S > 0 else float("nan")
    frag = float(S / max(D, 1))
    Rrim = float(np.median(rr))
    # angular heterozygosity H(lag) = P(labels differ at angular separation lag),
    # treating occupied bins as evenly spaced round the circle. The sector width
    # is the angle at which H reaches (1 - 1/e) of its plateau (1 - simpson).
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
            # linear interpolation between lag k and k-1 for a smooth crossing
            if k == 0:
                lag_c = 1.0
            else:
                h0, h1 = H[k - 1], H[k]
                frac = 0.0 if h1 == h0 else (target - h0) / (h1 - h0)
                lag_c = k + frac
            theta_c = lag_c * (2 * np.pi / n)    # mean sector angular width
            xi = float(theta_c * Rrim)           # arc length at the rim
    return dict(S=S, D=D, xi=xi, jc=jc, Hcomp=Hcomp, frag=frag, Rrim=Rrim)


def lineage_complexity_sym(snap, cfg, grid=64):
    """Compression complexity kept only as a quick aggregate, symmetrised over
    the raster and its transpose to cancel the row-major scan bias. Note this
    still inherits the finite zlib window, which is why the sector metrics above,
    not this number, carry the spatial interpretation."""
    pos = snap.pos
    if pos.shape[0] < 8:
        return float("nan")
    c = cfg.cube
    gx = np.clip((pos[:, 0] / c * grid).astype(int), 0, grid - 1)
    gy = np.clip((pos[:, 1] / c * grid).astype(int), 0, grid - 1)
    order = np.argsort(pos[:, 2])
    raster = np.full((grid, grid), -1, dtype=np.int64)
    raster[gy[order], gx[order]] = snap.lin[order]
    _, inv = np.unique(raster, return_inverse=True)
    relab = inv.reshape(raster.shape).astype(np.uint16)
    raw = relab.tobytes()
    rawT = np.ascontiguousarray(relab.T).tobytes()
    k = 0.5 * (len(zlib.compress(raw, 9)) + len(zlib.compress(rawT, 9)))
    return k / len(raw)


def strain_outcomes(snap, gen, cfg, frontier_q=0.70):
    """Per-strain (not per-shape) lineage success rate and median travel distance.
    Returns {strain: (success_rate, median_travel)} for strain in {0, 1}. Working
    per strain is necessary when both strains share a cell shape, because then a
    per-shape statistic would merge the two metabolic roles into one value."""
    pos = snap.pos
    c = cfg.cube * 0.5
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
