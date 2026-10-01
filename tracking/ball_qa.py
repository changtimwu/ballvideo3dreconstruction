"""QA: crops around tracked 2D ball positions (red ring = observed, magenta = interpolated).

  uv run python -m tracking.ball_qa data/ball_qa.jpg [--ball data/ball2d.npz] [--n 18] [--seed 0]
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ball", default="data/ball2d.npz")
    ap.add_argument("--video", default="match.mp4")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    B = np.load(a.ball)
    xy, obs, fps, f0 = B["xy"], B["observed"], float(B["fps"]), int(B["f0"])
    have = np.flatnonzero(np.isfinite(xy[:, 0]))
    pick = sorted(np.random.default_rng(a.seed).choice(have, min(a.n, len(have)), replace=False))
    cap = cv2.VideoCapture(a.video)
    tiles = []
    for k in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0 + int(k))
        _, img = cap.read()
        x, y = int(xy[k, 0]), int(xy[k, 1])
        crop = cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_CONSTANT)[y:y + 160, x:x + 160].copy()
        cv2.circle(crop, (80, 80), 16, (0, 0, 255) if obs[k] else (255, 0, 255), 1)
        cv2.putText(crop, f"{(f0 + k) / fps:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(cv2.resize(crop, (200, 200), interpolation=cv2.INTER_NEAREST))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
