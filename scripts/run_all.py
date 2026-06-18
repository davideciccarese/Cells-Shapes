"""
run_all.py
==========

Regenerate the full figure set in the order the repository is meant to be read:

  1. 3D cube model        cell-shape interactions in three dimensions (qualitative
                          spatial structure)
  2. range-expansion model  the same interactions and shapes as a 2D radial range
                          expansion (strain, nutrient field with cell contours,
                          lineage sectors)
  3. spatial metrics      sector statistics and the shape-by-sign factorial,
                          computed from the range-expansion model only

The 3D step calls the existing cube scripts unchanged. The metrics and factorial
steps read the 2D range-expansion model. Pass --quick to use fewer replicates.

    python scripts/run_all.py
    python scripts/run_all.py --quick
"""

import os
import sys
import subprocess

HERE = os.path.dirname(__file__)
QUICK = "--quick" in sys.argv

# replicate counts
F_REP = "2" if QUICK else "4"      # factorial replicates
M_REP = "3" if QUICK else "5"      # sector-metric replicates
I_REP = "2" if QUICK else "4"      # role-inversion replicates


def run(args):
    print("\n>>>", " ".join(args))
    subprocess.run([sys.executable] + args, cwd=os.path.dirname(HERE), check=True)


def main():
    # ---- 1. 3D cube model (unchanged) ----
    run([os.path.join("scripts", "make_cube3d.py")])
    run([os.path.join("scripts", "make_cube3d_shapes.py")])
    run([os.path.join("scripts", "make_cube3d_lineage_matrix.py")])
    run([os.path.join("scripts", "make_cube3d_inversion.py"), I_REP])

    # ---- 2. range-expansion model ----
    run([os.path.join("scripts", "make_re_panels.py")])
    run([os.path.join("scripts", "make_re_inversion.py"), I_REP])

    # ---- 3. spatial metrics, from the range-expansion model ----
    run([os.path.join("scripts", "make_re_factorial.py"), "all", F_REP])
    run([os.path.join("scripts", "make_re_factorial.py"), "--plot"])
    run([os.path.join("scripts", "make_re_metrics.py"), M_REP])
    run([os.path.join("scripts", "make_stats_report.py")])

    # ---- integrated view of both models ----
    run([os.path.join("scripts", "make_integrated.py")])

    print("\nAll figures regenerated.")
    print("3D cube model      -> figures/cube3d_*.png|gif")
    print("cube inversion     -> figures/cube3d_inversion*.png")
    print("range expansion    -> figures/re_panels.png, figures/re_rangeexp.gif")
    print("range inversion    -> figures/re_inversion*.png")
    print("range-exp metrics  -> figures/re_factorial.png, figures/re_*_byinteraction.png")


if __name__ == "__main__":
    main()
