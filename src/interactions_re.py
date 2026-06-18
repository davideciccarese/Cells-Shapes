"""
interactions_re.py
==================

The six pairwise interaction sign classes for the 2D range-expansion model. The
chemistry is identical to the 3D model (cube3d.py): each class returns a per-cell
metabolic rate mu and edits the shared diffusion fields, and carries a default
(shape_A, shape_B) pairing and the sign of each role. Only the field geometry
differs, the plate border supplies the substrate so the colony draws a radial
gradient as it consumes from the centre outward.

    Neutralism   (0, 0)   independent substrates, no shared chemical
    Commensalism (0, +)   A leaks metabolite M for free; B scavenges it
    Amensalism   (0, -)   A leaks an inhibitor I that slows B; A unaffected
    Public good  (-, +)   A pays to secrete public good P; B free-rides
    Mutualism    (+, +)   reciprocal facultative cross-feeding
    Competition  (-, -)   one shared substrate, drawn down by both
"""

import numpy as np

from field_re import FieldRE
from colony_re import monod


def _S(cfg):
    return FieldRE(cfg.N, cfg.dx, cfg.D_S, cfg.dt, c0=cfg.S0,
                   boundary="reservoir", reservoir=cfg.S0)


def _deposit_body(field, col, per_cell, msample=3):
    pts = col.spine_points(msample)
    flat = pts.reshape(-1, 2)
    amt = np.repeat(per_cell / msample, msample)
    field.deposit(flat, amt)


class InterRE:
    name = "base"
    row = "base"
    signs = ("0", "0")
    seed_frac = 0.5
    shapes = (2, 2)
    roles = ("A", "B")
    display_fields = []

    def fields(self, cfg):
        return {}

    def step(self, col, F, cfg, dt):
        raise NotImplementedError


class NeutralismRE(InterRE):
    name = "Neutralism (0, 0)"
    row = "neutralism"
    signs = ("0", "0")
    shapes = (2, 0)
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
        mu = np.zeros(col.n)
        mu[a] = cfg.g_max * monod(Sa[a], cfg.K_S)
        mu[~a] = cfg.g_max * monod(Sb[~a], cfg.K_S)
        _deposit_body(F["Sa"], col, np.where(a, -cfg.Y_consume * mu, 0.0) * dt)
        _deposit_body(F["Sb"], col, np.where(~a, -cfg.Y_consume * mu, 0.0) * dt)
        return mu


class CommensalismRE(InterRE):
    name = "Commensalism (0, +)"
    row = "commensalism"
    signs = ("0", "+")
    shapes = (2, 0)
    roles = ("producer", "commensal consumer")
    display_fields = [("Substrate S (feeds A)", "S"),
                      ("Metabolite M (feeds B)", "M")]

    def fields(self, cfg):
        return {"S": _S(cfg),
                "M": FieldRE(cfg.N, cfg.dx, cfg.D_M, cfg.dt, c0=0.0,
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


class AmensalismRE(InterRE):
    name = "Amensalism (0, -)"
    row = "amensalism"
    signs = ("0", "-")
    shapes = (2, 0)
    roles = ("inhibitor maker", "inhibited")
    display_fields = [("Substrate Sa (feeds A)", "Sa"),
                      ("Inhibitor I from A (harms B)", "I")]

    def fields(self, cfg):
        return {"Sa": _S(cfg), "Sb": _S(cfg),
                "I": FieldRE(cfg.N, cfg.dx, cfg.D_P, cfg.dt, c0=0.0,
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
        mu[a] = mua[a]
        mu[b] = mub[b] / (1.0 + I[b] / cfg.K_I)
        _deposit_body(F["Sa"], col, np.where(a, -cfg.Y_consume * mu, 0.0) * dt)
        _deposit_body(F["Sb"], col, np.where(b, -cfg.Y_consume * mu, 0.0) * dt)
        F["I"].deposit(p, np.where(a, 3.0 * cfg.Y_produce * mua, 0.0) * dt)
        return mu


class PublicGoodRE(InterRE):
    name = "Public good (-, +)"
    row = "public_good"
    signs = ("-", "+")
    shapes = (1, 0)
    roles = ("producer (pays cost)", "free-rider")
    display_fields = [("Substrate S (feeds A & B)", "S"),
                      ("Public good P (feeds B)", "P")]

    def fields(self, cfg):
        return {"S": _S(cfg),
                "P": FieldRE(cfg.N, cfg.dx, cfg.D_P, cfg.dt, c0=0.0,
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


class MutualismRE(InterRE):
    name = "Facultative mutualism (+, +)"
    row = "mutualism"
    signs = ("+", "+")
    shapes = (2, 1)
    roles = ("cross-feeder", "cross-feeder")
    display_fields = [("Cross-fed Mb (feeds A)", "Mb"),
                      ("Cross-fed Ma (feeds B)", "Ma")]

    def fields(self, cfg):
        return {"S": _S(cfg),
                "Ma": FieldRE(cfg.N, cfg.dx, cfg.D_M, cfg.dt, c0=0.0,
                              boundary="zeroflux", decay=cfg.decay_M),
                "Mb": FieldRE(cfg.N, cfg.dx, cfg.D_M, cfg.dt, c0=0.0,
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


class CompetitionRE(InterRE):
    name = "Competition (-, -)"
    row = "competition"
    signs = ("-", "-")
    shapes = (0, 2)
    roles = ("competitor", "competitor")
    display_fields = [("Shared substrate S (feeds A)", "S"),
                      ("Shared substrate S (feeds B)", "S")]

    def fields(self, cfg):
        return {"S": _S(cfg)}

    def step(self, col, F, cfg, dt):
        p = col.pos
        S = F["S"].sample(p)
        mu = cfg.g_max * monod(S, cfg.K_S)
        _deposit_body(F["S"], col, -cfg.comp_draw * cfg.Y_consume * mu * dt)
        return mu


ALLRE = [NeutralismRE(), CommensalismRE(), AmensalismRE(),
         PublicGoodRE(), MutualismRE(), CompetitionRE()]
