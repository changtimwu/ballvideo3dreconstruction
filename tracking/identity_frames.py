"""QA: per-tracker boxes at chosen timestamps, tiled into one image.

  uv run python -m tracking.identity_frames data/identity.jpg 10 60 175 360 640 880 \
      [--players data/players.npz] [--det data/detections.npz] [--video match.mp4]

T0..T3 are tracker identities from players.npz (not the tracking_data.json order).
Check that each identity stays on the same person across the match, including
after the teams switch ends. This is the check that actually catches swaps.
"""
import argparse

import cv2
import numpy as np

COLS = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 0, 255), 3: (255, 200, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("--players", default="data/players.npz")
    ap.add_argument("--det", default="data/detections.npz")
    ap.add_argument("--video", default="match.mp4")
    a = ap.parse_args()

    P, D = np.load(a.players), np.load(a.det)
    t = P["t"]
    cap = cv2.VideoCapture(a.video)
    tiles = []
    for tt in a.times:
        i = int(np.clip(np.searchsorted(t, tt), 0, len(t) - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(P["frames"][i]))
        _, img = cap.read()
        for k in range(P["det_of"].shape[1]):
            j = P["det_of"][i, k]
            if j < 0:
                continue
            b = D["box"][j].astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), COLS[k], 5)
            cv2.putText(img, f"T{k}", (b[0], b[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 2, COLS[k], 5)
        cv2.putText(img, f"{tt:.0f}s", (1650, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
        tiles.append(cv2.resize(img, (640, 360)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.out, np.vstack([np.hstack(tiles[r:r + 3]) for r in range(0, len(tiles), 3)]))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
