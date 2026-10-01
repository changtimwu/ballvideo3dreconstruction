"""Milestone 2, stage B1: ball candidates in 2D, every frame (60 fps).

  uv run python -m tracking.ball_detect match.mp4 [--start S --end E] [--out data/ball_candidates.npz]

The ball in match.mp4 is a neon yellow-green disc (≈20–250 px). Per frame:
  1. colour mask (HSV hue ≈ 25–50, high S and V),
  2. minus a static mask of pixels that are ball-coloured most of the time
     (net-post sticker, logos), learned from a sample of frames,
  3. keep components that are moving (difference to 2 frames earlier),
  4. record shape/colour features and the distance to the nearest ankle
     (yellow shoe stripes are the main false positive) for the tracker to weigh.
No learned model: these candidates are what ball_track links into trajectories.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, line_score

H_LO, H_HI, S_MIN, V_MIN = 25, 50, 110, 120
AREA_MIN, AREA_MAX = 3, 900
SCOREBOARD = (0, 0, 300, 165)


def ball_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return ((h >= H_LO) & (h <= H_HI) & (s >= S_MIN) & (v >= V_MIN)).astype(np.uint8)


def static_mask(cap, n_total, samples=120):
    """Pixels that are ball-coloured in >20% of sampled frames never belong to the ball."""
    acc = None
    for f in np.linspace(0, n_total - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        m = ball_mask(img).astype(np.float32)
        acc = m if acc is None else acc + m
    st = (acc / samples > 0.2).astype(np.uint8)
    return cv2.dilate(st, np.ones((9, 9), np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="camera.json from tracking.calibrate")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--det", default="data/detections.npz", help="person detections (for ankle distances)")
    ap.add_argument("--out", default="data/ball_candidates.npz")
    a = ap.parse_args()

    cam = Camera.load(a.camera)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(round(a.start * fps))
    f1 = min(n_total, int(round(a.end * fps)) if a.end else n_total)

    print("learning static mask…")
    static = static_mask(cap, n_total)
    x0, y0, x1, y1 = SCOREBOARD
    static[y0:y1, x0:x1] = 1
    keep_region = (1 - static).astype(np.uint8)

    # ankles per (even) detection frame, for shoe rejection
    ankles = {}
    if Path(a.det).exists():
        D = np.load(a.det)
        for fr, kp in zip(D["frame"], D["kp"]):
            ankles.setdefault(int(fr), []).extend([kp[j, :2] for j in (15, 16) if kp[j, 2] > 0.3])

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0 - 2))
    hist = []                      # last 2 grey frames
    rows = []                      # frame, x, y, area, w, h, fill, sat, val, motion, ankle_d, hue
    court = np.zeros(f1 - f0, np.float32)
    t_start = time.time()
    f = max(0, f0 - 2)
    while f < f1:
        ok, img = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if f >= f0 and len(hist) == 2:
            if (f - f0) % 30 == 0:
                court[f - f0] = line_score(img, cam)
            m = ball_mask(img) & keep_region
            diff = cv2.absdiff(grey, hist[0]) > 18
            n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n > 1:
                hsv = None
                near = ankles.get(f) or ankles.get(f - 1) or []
                near = np.array(near) if near else None
                for i in range(1, n):
                    area = st[i, cv2.CC_STAT_AREA]
                    if not AREA_MIN <= area <= AREA_MAX:
                        continue
                    bx, by, bw, bh = st[i, :4]
                    comp = lab[by:by + bh, bx:bx + bw] == i
                    motion = diff[by:by + bh, bx:bx + bw][comp].mean()
                    if motion < 0.2:
                        continue
                    if hsv is None:
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    pix = hsv[by:by + bh, bx:bx + bw][comp]
                    cx, cy = cen[i]
                    ad = float(np.min(np.linalg.norm(near - [cx, cy], axis=1))) if near is not None else 1e4
                    rows.append((f, cx, cy, area, bw, bh, area / (bw * bh), pix[:, 1].mean(), pix[:, 2].mean(), motion, ad,
                                 pix[:, 0].mean()))
        hist = (hist + [grey])[-2:]
        f += 1
        if (f - f0) % 1800 == 0:
            el = time.time() - t_start
            print(f"\rframe {f}/{f1}  {(f - f0) / el:5.0f} fps  candidates {len(rows)}", end="", flush=True)
    print()
    R = np.array(rows, np.float32).reshape(-1, 12)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, fps=fps, f0=f0, f1=f1, cand=R,
                        cols=np.array(["frame", "x", "y", "area", "w", "h", "fill", "sat", "val", "motion", "ankle_d", "hue"]))
    per = np.bincount((R[:, 0] - f0).astype(int), minlength=f1 - f0) if len(R) else np.zeros(1)
    print(f"wrote {out}: {len(R)} candidates; frames with 0/1/2/3+ candidates: "
          f"{[(per == 0).mean().round(3), (per == 1).mean().round(3), (per == 2).mean().round(3), (per >= 3).mean().round(3)]}")


if __name__ == "__main__":
    main()
