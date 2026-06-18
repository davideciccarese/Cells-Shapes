"""
config3d.py
===========

All tunable numbers for the 3D cube model in one place.
"""

from dataclasses import dataclass


@dataclass
class Config3D:
    # cube and grid
    cube: float = 12.0          # small cube the colony grows to fill
    N: int = 30                 # field grid nodes per side
    dt: float = 0.05
    steps_per_frame: int = 4
    n_frames: int = 30
    relax_iters: int = 6
    seed: int = 0

    # cells
    R: float = 0.5              # rod radius
    L_birth: float = 1.3        # length at birth
    L_div: float = 2.6          # length triggering division
    n_seed: int = 50           # founders in the small central patch
    seed_radius: float = 2.0    # small central inoculum (radial range expansion)

    # 3D contact mechanics (overdamped, force and torque)
    k_contact: float = 0.50     # cell-cell repulsion stiffness
    k_floor: float = 0.70       # floor push-up stiffness
    rot_drag: float = 0.020     # rotational drag coefficient (x length cubed)
    k_link: float = 0.45        # chain link spring stiffness (chaining cells)
    k_align: float = 0.015      # chain alignment: low so chains stay floppy, not rigid
    gravity: float = 0.035      # lets the radial expansion fill more of the cube
    support_radius: float = 1.9 # neighbourhood radius for the confinement/support gate
    support_full: float = 6.0   # neighbour count at which growth is fully supported
    cocci_birth_R: float = 0.30    # 2/3 scale (cells a third smaller)
    cocci_div_R: float = 0.41
    cocci_grow: float = 0.5     # how fast cocci inflate their radius
    k_lay: float = 0.18         # lay-flat on protruding rods/chains

    # range-expansion front (only the exposed surface grows)
    front_radius: float = 2.2
    front_lo: float = 0.18
    front_hi: float = 0.60

    # growth kinetics
    g_max: float = 5.0
    K_S: float = 0.15
    K_M: float = 0.12
    K_P: float = 0.14

    # active-interaction costs and public-good gain
    cost_public_good: float = 0.30
    pg_gain: float = 1.5
    fac_base: float = 0.55

    # stoichiometry
    Y_consume: float = 1.80     # strong uptake so consumption is clearly visible
    Y_produce: float = 0.055

    # field transport
    D_S: float = 0.4            # slow diffusion -> strong consumption gradient
    D_M: float = 4.0
    D_P: float = 4.0
    S0: float = 1.0
    decay_M: float = 0.02
    decay_P: float = 0.02
    K_I: float = 0.05           # inhibitor half-saturation (amensalism)
    comp_draw: float = 1.6
    k_nematic: float = 0.020    # gentle nematic alignment (local order only)
    ax_noise: float = 0.10      # orientational jitter so rods form microdomains, not a crystal

    @property
    def dx(self):
        return self.cube / self.N

    @property
    def center(self):
        return self.cube * 0.5
