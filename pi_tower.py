#!/usr/bin/env python3
"""A Pi (pi) symbol made of Connectors only: one flat rigid structure, tilted cubes make the curve.
Positions follow rtg_shape (the rule learned from a real build); this file only DESCRIBES the shape.

    python3 pi_tower.py        -> writes pi_code.txt (creation code) and pi_preview.png
"""
from rtg_shape import Shape, positions_from_code

# ---------------------------------------------------------------- SETTINGS
BAR = (227, 84, 25)            # orange top bar
LEG = (48, 40, 42)             # black legs
LEG_H = 6                      # cubes in each leg below the bar
BAR_LEFT, BAR_MID, BAR_RIGHT = 2, 4, 2   # cubes: overhang left of the left leg / between the legs / overhang right
FOOT = 4                       # curved foot of the right leg: cubes, each turned a further STEP
STEP = 22.5                    # degrees (the game turns parts in steps of 22.5)
UP, DOWN, LEFT, RIGHT = (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0)


def pi():
    s = Shape(LEG)
    # left leg: from its foot straight up, then the first bar cube on top of it
    c = s.root
    for _ in range(LEG_H - 1):
        c = s.add(c, UP)
    left_junction = s.add(c, UP, color=BAR)
    # bar: overhang to the left, and to the right as far as the right leg
    c = left_junction
    for _ in range(BAR_LEFT):
        c = s.add(c, LEFT)
    c = left_junction
    for _ in range(BAR_MID):
        c = s.add(c, RIGHT)
    right_junction = c
    for _ in range(BAR_RIGHT):                     # right overhang
        c = s.add(c, RIGHT)
    # right leg: straight down from the bar, then a foot curving to the right (22.5 more degrees per cube)
    c = right_junction
    c = s.add(c, DOWN, color=LEG)
    for _ in range(LEG_H - FOOT):
        c = s.add(c, None)
    for _ in range(FOOT):
        c = s.add(c, None, turn=STEP)
    return s


def preview(s, path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (p, R) in positions_from_code(s.code()).items():      # recomputed from the CODE, not from the builder
        pts = np.array([p + R @ (np.array(v) * 0.2) for v in __import__("itertools").product([-1, 1], repeat=3)])[:, :2]
        col = np.array(s.cubes[i - 1]["color"]) / 255
        ax.add_patch(Polygon(pts[ConvexHull(pts).vertices], fc=col, ec="k", lw=.8))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("seen from +z (x right, y up)")
    plt.savefig(path, dpi=80)


if __name__ == "__main__":
    s = pi()
    code = s.code()
    open("pi_code.txt", "w").write(code + "\n")
    print(len(s.cubes), "connectors; overlaps between non-neighbours:", s.overlaps())
    print({k: v for k, v in s.cubes[-1]["props"].items()})
    preview(s, "pi_preview.png")
