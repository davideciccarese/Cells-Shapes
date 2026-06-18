"""
make_re_panels.py
=================

Static and animated panels of the 2D range-expansion model, one row per
interaction. Three columns are drawn from the final state:

  1. cells coloured by shape (cocci blue, chain yellow, rod red), the same shape
     colour code used in the 3D cube model and the GIFs
  2. the first display field as a heatmap with the cell contours drawn on top
     (the nutrient or metabolite the colony shapes, cells overlaid as contours)
  3. cells coloured by founding lineage, so the radial sectors are visible

It also writes the essential range-expansion GIF: the six interactions side by
side, coloured by shape, growing outward frame by frame.

    python scripts/make_re_panels.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_re import ConfigRE
import interactions_re as I
import sim_re as SIM
import render_re as RR

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def main():
    cfg = ConfigRE()
    os.makedirs(FIGDIR, exist_ok=True)

    n = len(I.ALLRE)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 3.4 * n), dpi=120)
    fig.patch.set_facecolor("white")

    all_frames = []

    for ri, inter in enumerate(I.ALLRE):
        flabel, fkey = inter.display_fields[0]
        frames, fh, gen = SIM.run(inter, cfg, capture_fields=[fkey])
        all_frames.append((inter, frames))
        snap = frames[-1]

        sa, sb = inter.shapes
        RR.draw_shape(axes[ri, 0], snap, cfg)
        axes[ri, 0].set_title(
            f"{inter.name}\ncells by shape  (A {RR.MORPH_NAMES[sa]}, "
            f"B {RR.MORPH_NAMES[sb]})", fontsize=10, loc="left")

        f2d = fh[fkey][-1]
        im = RR.draw_field(axes[ri, 1], snap, cfg, f2d)
        axes[ri, 1].set_title(f"field: {flabel}\ncells overlaid as contours",
                              fontsize=10, loc="left")
        fig.colorbar(im, ax=axes[ri, 1], fraction=0.046, pad=0.04)

        RR.draw_lineage(axes[ri, 2], snap, cfg)
        axes[ri, 2].set_title("cells by founding lineage\n(radial sectors)",
                              fontsize=10, loc="left")
        print(f"  {inter.row}: {snap.pos.shape[0]} cells")

    fig.legend(handles=RR.shape_legend_handles(), loc="lower center",
               ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.18, 0.995))
    fig.suptitle(
        "Range-expansion model (2D): final state, one row per interaction\n"
        "the colony seeds at the centre and expands outward; sectors widen at the front",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(FIGDIR, "re_panels.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)

    # ---- essential GIF: six interactions, cells by shape, growing ----
    nfr = cfg.n_frames
    images = []
    for fr in range(0, nfr, 2):
        gfig, gax = plt.subplots(2, 3, figsize=(12, 8.4), dpi=90)
        gfig.patch.set_facecolor("white")
        gax = gax.ravel()
        for k, (inter, frames) in enumerate(all_frames):
            RR.draw_shape(gax[k], frames[fr], cfg)
            gax[k].set_title(inter.name, fontsize=11, loc="left")
        gfig.legend(handles=RR.shape_legend_handles(), loc="lower center",
                    ncol=3, frameon=False, fontsize=11)
        gfig.suptitle(f"Range expansion, cells by shape  (frame {fr+1}/{nfr})",
                      fontsize=13)
        gfig.tight_layout(rect=[0, 0.03, 1, 0.96])
        images.append(RR.fig_to_image(gfig))
        plt.close(gfig)
    gif = os.path.join(FIGDIR, "re_rangeexp.gif")
    RR.save_gif(images, gif, fps=6)
    images[-1].save(os.path.join(FIGDIR, "re_rangeexp.png"))
    print("wrote", gif)

    # ---- companion still: the same final states coloured by lineage ----
    lfig, lax = plt.subplots(2, 3, figsize=(12, 8.4), dpi=110)
    lfig.patch.set_facecolor("white")
    lax = lax.ravel()
    for k, (inter, frames) in enumerate(all_frames):
        RR.draw_lineage(lax[k], frames[-1], cfg)
        lax[k].set_title(inter.name, fontsize=11, loc="left")
    lfig.suptitle("Range expansion, cells by founding lineage (radial sectors)",
                  fontsize=13)
    lfig.tight_layout(rect=[0, 0, 1, 0.96])
    lfig.savefig(os.path.join(FIGDIR, "re_lineage_panels.png"), dpi=110)
    plt.close(lfig)
    print("wrote", os.path.join(FIGDIR, "re_lineage_panels.png"))


if __name__ == "__main__":
    main()
