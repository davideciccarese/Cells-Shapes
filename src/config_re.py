"""
config_re.py
============

All tunable numbers for the 2D range-expansion model in one dataclass.

This is the flat companion to the 3D cube model. The same two strains, the same
three cell shapes and the same six interaction sign classes are used, but the
colony grows in a plane as a top-down radial range expansion from a small central
inoculum. The metrics and the shape-by-sign factorial in the repository are
computed from this model, not from the 3D cube. The cube is the qualitative
spatial picture in three dimensions; this is the quantitative range-expansion
model the statistics read.

Kinetic constants (g_max, K_S, costs, yields, diffusivities) are kept equal to
config3d.py so the two models differ only in dimension and boundary geometry.
"""

from dataclasses import dataclass


@dataclass
class ConfigRE:
    # domain and grid (a square plate seen from above)
    box: float = 40.0           # side of the square the colony expands into
    N: int = 240                # field grid nodes per side
    dt: float = 0.05
    steps_per_frame: int = 3
    n_frames: int = 28
    relax_iters: int = 6
    seed: int = 0
    max_cells: int = 9000       # population cap so a run stays bounded

    # inoculum (range-expansion start: a small mixed disk at the centre)
    n_seed: int = 90
    seed_radius: float = 1.8

    # cell mechanics (overdamped contact relaxation, per-cell radius)
    relax_stiff: float = 0.9    # cell-cell repulsion stiffness
    torque_gain: float = 0.16   # off-centre contact torque (rod alignment)
    k_link: float = 0.42        # chain link spring stiffness
    k_align: float = 0.015      # chain alignment: low so chains stay floppy and buckle
    k_nematic: float = 0.02     # contact nematic alignment: low, so order stays local
    ax_noise: float = 0.10      # orientational jitter -> microdomains, not a crystal
    div_angle_noise: float = 0.20  # turn at each division so daughters do not line up
    cocci_grow: float = 0.5

    # morphology radii (2D scale; matched in proportion to the 3D shapes)
    cocci_birth_R: float = 0.30
    cocci_div_R: float = 0.41

    # range-expansion front (only the exposed rim grows; the core freezes)
    front_radius: float = 3.2
    front_lo: float = 0.10
    front_hi: float = 0.40
    front_smooth_iters: int = 3
    freeze_core: bool = True
    freeze_res: float = 0.07
    freeze_count: int = 7

    # confinement (a cell grows only where it is supported by a crowd of
    # neighbours, so an isolated cell poking into empty space barely grows and
    # cannot run off the front as a free spike)
    support_radius: float = 2.2
    support_full: float = 5.0
    # a cell with no neighbour within this distance has detached and is dropped
    floater_gap: float = 0.45

    # growth kinetics (identical to config3d.py)
    g_max: float = 5.0
    K_S: float = 0.15
    K_M: float = 0.12
    K_P: float = 0.14

    # active-interaction costs and public-good gain (identical to config3d.py)
    cost_public_good: float = 0.30
    pg_gain: float = 1.5
    fac_base: float = 0.55

    # stoichiometry (identical to config3d.py)
    Y_consume: float = 1.80
    Y_produce: float = 0.055

    # field transport (identical to config3d.py)
    D_S: float = 0.4
    D_M: float = 4.0
    D_P: float = 4.0
    S0: float = 1.0
    decay_M: float = 0.02
    decay_P: float = 0.02
    K_I: float = 0.05
    comp_draw: float = 1.6

    # division geometry control (used by the control experiment)
    iso_division: bool = False

    @property
    def dx(self):
        return self.box / self.N

    @property
    def center(self):
        return self.box * 0.5
