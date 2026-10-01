"""QA: physical sanity of the 3D ball (ball3d.npz).

  uv run python -m tracking.ball_check [--ball3d data/ball3d.npz]

Prints: joins between consecutive segments (should be ~0), bounce positions
(should be on or near the court), net-crossing heights (should clear 0.86–0.91 m),
and distances from hits to the hitter.
"""
import argparse

import numpy as np

from .ball3d import LINK_GAP, simulate
from .camera import HALF_L, HALF_W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball3d", default="data/ball3d.npz")
    a = ap.parse_args()
    R = np.load(a.ball3d)
    S, E = R["segs"], R["events"]
    joins, crossings = [], []
    for i in range(len(S)):
        s = S[i]
        T = s[1] - s[0]
        taus = np.linspace(0, T, max(int(T * 240), 2))
        P, _ = simulate(s[2:5], s[5:8], taus)
        sx = np.sign(P[:, 0])
        for k in np.flatnonzero(sx[:-1] * sx[1:] < 0):
            crossings.append(P[k, 2])
        if i + 1 < len(S) and S[i + 1][0] - s[1] <= LINK_GAP:
            pa, _ = simulate(s[2:5], s[5:8], [S[i + 1][0] - s[0]])
            joins.append(np.linalg.norm(pa[0] - S[i + 1][2:5]))
    joins, crossings = np.array(joins), np.array(crossings)
    print(f"segments {len(S)}, reprojection median {np.median(S[:, 8]):.2f} px, p90 {np.percentile(S[:, 8], 90):.2f} px")
    if len(joins):
        print(f"joins between consecutive segments: median {np.median(joins) * 100:.1f} cm, p90 {np.percentile(joins, 90) * 100:.1f} cm")
    if len(crossings):
        print(f"net crossings: {len(crossings)}, height median {np.median(crossings):.2f} m, "
              f"below 0.86 m: {(crossings < 0.86).mean() * 100:.0f}%")
    b = E[E[:, 1] == 0]
    if len(b):
        inside = (np.abs(b[:, 2]) < HALF_L + 0.5) & (np.abs(b[:, 3]) < HALF_W + 0.5)
        print(f"bounces: {len(b)}, within 0.5 m of the court: {inside.mean() * 100:.0f}%")
    h = E[E[:, 1] == 1]
    print(f"hits: {len(h)}, contact height median {np.median(h[:, 4]):.2f} m")


if __name__ == "__main__":
    main()
