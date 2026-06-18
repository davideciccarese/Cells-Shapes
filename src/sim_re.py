"""
sim_re.py
=========

Wire a ColonyRE to its diffusion fields and one interaction, advance the system
as a frontier-gated range expansion, and capture a snapshot per frame.

Macro step order:
  1. diffuse every field
  2. interaction.step -> per-cell metabolic rate mu, and field edits
  3. gate mu by the front factor (only the rim grows: range expansion)
  4. grow biomass, divide, freeze the jammed core, relax contacts
  5. drop cells that grew out of the plate

run() returns (frames, field_history, genealogy), the same shape the 3D model's
run() returns, so the renderers and metrics treat the two models uniformly.
"""

import numpy as np

from colony_re import ColonyRE, SnapshotRE


def run(inter, cfg, capture_field=None, capture_fields=None, shapes=None):
    rng = np.random.default_rng(cfg.seed)
    col = ColonyRE(cfg, rng)
    saved = getattr(inter, "shapes", (2, 2))
    if shapes is not None:
        inter.shapes = shapes
    try:
        col.seed_disk(inter)
    finally:
        inter.shapes = saved
    F = inter.fields(cfg)

    col.relax(iters=10)

    keys = list(capture_fields) if capture_fields else []
    if capture_field and capture_field not in keys:
        keys.append(capture_field)
    fhist = {k: [] for k in keys}

    def record_growth():
        mu = inter.step(col, F, cfg, 0.0)
        fac, _ = col.front_factor()
        col.geff = mu * fac * col.support_factor()

    def grab():
        for k in keys:
            fhist[k].append(F[k].c.copy())

    record_growth()
    frames = [SnapshotRE(col)]
    grab()

    for fr in range(1, cfg.n_frames):
        for _ in range(cfg.steps_per_frame):
            for f in F.values():
                f.step()
            mu = inter.step(col, F, cfg, cfg.dt)
            fac, frozen = col.front_factor()
            eff = mu * fac * col.support_factor()
            col.geff = eff
            col._frozen = None
            col.grow(eff, cfg.dt)
            col.divide(frame=fr)
            _, frozen = col.front_factor()
            col._frozen = frozen
            col.relax(cfg.relax_iters)
            col.cull_outside()
        record_growth()
        frames.append(SnapshotRE(col))
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
