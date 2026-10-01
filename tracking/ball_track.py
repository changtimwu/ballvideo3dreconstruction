"""Milestone 2, stage B2: link ball candidates into 2D trajectories.

  uv run python -m tracking.ball_track [--cand data/ball_candidates.npz] [--out data/ball2d.npz]

Between contacts the ball's image path is close to a parabola, so:
  1. seed a track from three candidates in nearby frames with consistent velocity,
  2. extend it both ways using a quadratic fit of its recent points, gating the
     search radius by speed and allowing short gaps (net, player occlusion),
  3. drop tracks that aren't ball-coloured (shoe stripes, paddle grips), are
     static, or whose size-implied depth puts them over a neighbouring court,
  4. keep the best set of tracks with at most one ball per frame (weighted
     interval scheduling by track length).
Hits and bounces end a track; the next flight starts a new one.
"""
import argparse
from pathlib import Path

import numpy as np

from .camera import Camera

BALL_D = 0.074      # pickleball diameter, m
MAX_GAP = 6          # frames a track may skip
MIN_LEN = 8          # observations to keep a track
SPEED_MAX = 90.0     # px/frame


class Track:
    def __init__(self, pts):
        self.pts = list(pts)          # (frame, x, y, cand_index)

    def frames(self):
        return [p[0] for p in self.pts]

    def predict(self, f, side):
        """Position at frame f extrapolated from the nearest ≤8 points on that side."""
        pts = self.pts[-8:] if side > 0 else self.pts[:8]
        F = np.array([p[0] for p in pts], float)
        X = np.array([[p[1], p[2]] for p in pts])
        deg = 2 if len(pts) >= 5 else 1
        if len(pts) < 2:
            return X[0], 0.0
        cx = np.polyfit(F - F[0], X[:, 0], deg)
        cy = np.polyfit(F - F[0], X[:, 1], deg)
        p = np.array([np.polyval(cx, f - F[0]), np.polyval(cy, f - F[0])])
        v = np.hypot(*(X[-1] - X[-2])) / max(F[-1] - F[-2], 1) if side > 0 else np.hypot(*(X[1] - X[0])) / max(F[1] - F[0], 1)
        return p, v


def build_tracks(cand, by_frame, frames_sorted):
    used = np.zeros(len(cand), bool)
    tracks = []
    order = np.argsort(-cand[:, 3])            # seed from larger, clearer blobs first
    for i in order:
        if used[i]:
            continue
        f, x, y = int(cand[i, 0]), cand[i, 1], cand[i, 2]
        # seed: one candidate in each of the next two occupied frames within reach
        seed = [(f, x, y, i)]
        for _ in range(2):
            lf, lx, ly, _li = seed[-1]
            best = None
            for df in range(1, 4):
                for j in by_frame.get(lf + df, []):
                    if used[j] or any(j == s[3] for s in seed):
                        continue
                    d = np.hypot(cand[j, 1] - lx, cand[j, 2] - ly) / df
                    if d > SPEED_MAX or d < 0.5:
                        continue
                    if len(seed) == 2:
                        v1 = np.array([lx - seed[0][1], ly - seed[0][2]]) / (lf - seed[0][0])
                        v2 = np.array([cand[j, 1] - lx, cand[j, 2] - ly]) / df
                        if np.linalg.norm(v2 - v1) > 6 + 0.35 * np.linalg.norm(v1):
                            continue
                    if best is None or d < best[0]:
                        best = (d, (lf + df, cand[j, 1], cand[j, 2], j))
                if best:
                    break
            if not best:
                break
            seed.append(best[1])
        if len(seed) < 3:
            continue
        tr = Track(seed)
        for side in (1, -1):
            miss = 0
            while miss <= MAX_GAP:
                edge = tr.pts[-1][0] if side > 0 else tr.pts[0][0]
                f_next = edge + side * (miss + 1)
                pred, v = tr.predict(f_next, side)
                gate = 5 + 0.45 * v * (miss + 1)
                best = None
                for j in by_frame.get(f_next, []):
                    if used[j]:
                        continue
                    d = np.hypot(cand[j, 1] - pred[0], cand[j, 2] - pred[1])
                    if d < gate and (best is None or d < best[0]):
                        best = (d, j)
                if best:
                    j = best[1]
                    p = (f_next, cand[j, 1], cand[j, 2], j)
                    if side > 0:
                        tr.pts.append(p)
                    else:
                        tr.pts.insert(0, p)
                    miss = 0
                else:
                    miss += 1
        if len(tr.pts) >= MIN_LEN:
            for p in tr.pts:
                used[p[3]] = True
            tracks.append(tr)
    return tracks


def in_our_airspace(cam, cand, idx):
    """Fraction of a track's points whose size-implied depth puts them above this court.

    Pixel diameter gives depth ≈ f·0.074/diam; blur and thresholding make that rough,
    so a point counts if any depth in [0.6, 2.2]× the estimate lands inside the
    court's airspace (|x| < 9, |y| < 5.5, 0 ≤ z < 8 m). Balls on neighbouring courts
    look too small for where their ray crosses this court.
    """
    px = cand[idx][:, 1:3]
    diam = 2 * np.sqrt(cand[idx, 3] / np.pi)
    d0 = cam.K[0, 0] * BALL_D / np.maximum(diam, 1)
    rays = cam.rays(px)
    c = cam.center
    inside = np.zeros(len(idx), bool)
    for s in (0.6, 0.8, 1.0, 1.3, 1.7, 2.2):
        P = c + rays * (d0 * s)[:, None]
        inside |= (np.abs(P[:, 0]) < 9) & (np.abs(P[:, 1]) < 5.5) & (P[:, 2] > -0.3) & (P[:, 2] < 8)
    return inside.mean()


def plausible(tr, cand, cam):
    idx = [p[3] for p in tr.pts]
    P = np.array([[p[1], p[2]] for p in tr.pts])
    F = np.array([p[0] for p in tr.pts], float)
    span = F[-1] - F[0]
    speed = np.linalg.norm(P[-1] - P[0]) / max(span, 1)
    path = np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / max(span, 1)
    if path < 1.0:                       # static or nearly so
        return False
    # Colour separates the ball cleanly in match.mp4: ball hue 43–50 / sat > 190;
    # yellow shoe stripes hue 34–37 / sat 150–190; the red player's paddle grip
    # and far-court balls sat < 140. (Distance to ankles is not used: low balls
    # bounce right by players' feet.)
    if np.median(cand[idx, 11]) < 40 or np.median(cand[idx, 7]) < 165:
        return False
    if in_our_airspace(cam, cand, idx) < 0.5:
        return False
    if np.median(cand[idx, 6]) < 0.35:   # fill ratio: ragged blobs aren't a ball
        return False
    return True


def select_non_overlapping(tracks):
    """Weighted interval scheduling: max total observations, one track per frame
    (allowing a 3-frame overlap at hand-overs)."""
    iv = sorted(((t.pts[0][0], t.pts[-1][0], len(t.pts), t) for t in tracks), key=lambda z: z[1])
    ends = [z[1] for z in iv]
    best = [0] * (len(iv) + 1)
    take = [False] * len(iv)
    prev = []
    for k, (s, e, w, _) in enumerate(iv):
        # tracks may share up to 3 frames at a hand-over (one flight ends as the next begins)
        j = np.searchsorted(ends, s + 3, side="left")
        prev.append(j)
        best[k + 1] = max(best[k], best[j] + w)
    out, k = [], len(iv)
    while k > 0:
        s, e, w, t = iv[k - 1]
        if best[prev[k - 1]] + w >= best[k - 1]:
            out.append(t); k = prev[k - 1]
        else:
            k -= 1
    return sorted(out, key=lambda t: t.pts[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="data/ball_candidates.npz")
    ap.add_argument("--out", default="data/ball2d.npz")
    a = ap.parse_args()

    B = np.load(a.cand)
    cand, fps, f0, f1 = B["cand"], float(B["fps"]), int(B["f0"]), int(B["f1"])
    by_frame = {}
    for i, f in enumerate(cand[:, 0].astype(int)):
        by_frame.setdefault(f, []).append(i)
    tracks = build_tracks(cand, by_frame, sorted(by_frame))
    cam = Camera.load()
    good = [t for t in tracks if plausible(t, cand, cam)]
    chosen = select_non_overlapping(good)
    print(f"{len(tracks)} tracks, {len(good)} plausible, {len(chosen)} selected")

    n = f1 - f0
    xy = np.full((n, 2), np.nan)
    tid = np.full(n, -1)
    observed = np.zeros(n, bool)
    for k, t in enumerate(chosen):
        F = np.array([p[0] for p in t.pts]) - f0
        P = np.array([[p[1], p[2]] for p in t.pts])
        full = np.arange(F[0], F[-1] + 1)
        xy[full, 0] = np.interp(full, F, P[:, 0])
        xy[full, 1] = np.interp(full, F, P[:, 1])
        tid[full] = k
        observed[F] = True
    out = Path(a.out)
    np.savez_compressed(out, fps=fps, f0=f0, xy=xy, track=tid, observed=observed)
    print(f"wrote {out}: ball in {np.isfinite(xy[:, 0]).mean() * 100:.1f}% of frames "
          f"({observed.mean() * 100:.1f}% directly observed), median track {np.median([len(t.pts) for t in chosen]):.0f} obs")


if __name__ == "__main__":
    main()
