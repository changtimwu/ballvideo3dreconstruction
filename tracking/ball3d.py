"""Milestone 3: lift the 2D ball track to 3D and find hits and bounces.

  uv run python -m tracking.ball3d [--ball data/ball2d.npz] [--players data/players.npz] [--out data/ball3d.npz]

One camera can't see depth, but between contacts the ball flies ballistically:
p(t) = p0 + v0 t + ½ g t², six unknowns per flight segment, while a segment has
tens of observations through the calibrated camera. Per 2D track:
  1. initialise from the ball's pixel size (74 mm ball → rough depth along the ray),
  2. fit p0, v0 by robust least squares on reprojection error (plus a weak
     size-depth prior for conditioning),
  3. if one parabola doesn't explain the track, split at the sharpest 2D kink
     (a bounce or a hit) and recurse.
Segments are then chained into rallies and refit jointly with air drag
(a = g − k|v|v) and soft constraints: consecutive segments meet at one point, which
lies on the floor at a bounce. That stops short segments drifting in depth.
Boundaries are classified as bounces (on the floor, vertical velocity flips) or
hits (credited to the nearest player).
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .camera import Camera

G = np.array([0.0, 0.0, -9.81])
BALL_D = 0.074
MIN_SEG = 7            # observations needed to fit a segment
RMS_OK = 2.5           # px
MAX_OK = 8.0           # px
LINK_GAP = 0.3         # s: segments closer than this are consecutive (bounce/hit between them)
DRAG_K = 0.04          # 1/m: ½ρ·Cd·A/m for a 26 g, 74 mm ball with Cd≈0.4
SIM_DT = 1 / 120


def simulate_batch(P0, V0, T, k=DRAG_K):
    """Integrate B balls at once (RK4, gravity + quadratic drag) over [0, T].
    Returns (ts, P[n+1, B, 3], V[n+1, B, 3])."""
    n = int(np.ceil(max(T, 0.0) / SIM_DT)) + 1
    P = np.empty((n + 1,) + P0.shape); V = np.empty_like(P)
    p, v = np.array(P0, float), np.array(V0, float)

    def acc(v_):
        return G - k * np.linalg.norm(v_, axis=-1, keepdims=True) * v_
    h = SIM_DT
    for i in range(n + 1):
        P[i], V[i] = p, v
        k1v = acc(v); k2v = acc(v + 0.5 * h * k1v); k3v = acc(v + 0.5 * h * k2v); k4v = acc(v + h * k3v)
        p = p + h / 6 * (v + 2 * (v + 0.5 * h * k1v) + 2 * (v + 0.5 * h * k2v) + (v + h * k3v))
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return np.arange(n + 1) * h, P, V


def sample(ts, P, b, taus):
    """Linear interpolation of trajectory b at times taus."""
    f = np.clip(np.asarray(taus) / SIM_DT, 0, len(ts) - 1.000001)
    i = f.astype(int); u = (f - i)[:, None]
    return P[i, b] * (1 - u) + P[i + 1, b] * u


def simulate(p0, v0, taus, k=DRAG_K):
    """Positions and velocities of one ball at times taus (≥ 0)."""
    taus = np.atleast_1d(np.asarray(taus, float))
    ts, P, V = simulate_batch(np.asarray(p0, float)[None], np.asarray(v0, float)[None], float(taus.max()), k)
    return sample(ts, P, 0, taus), sample(ts, V, 0, taus)


def project(cam, P):
    return cam.project(P)


def model(theta, tau):
    p0, v0 = theta[:3], theta[3:]
    return p0 + v0 * tau[:, None] + 0.5 * G * (tau ** 2)[:, None]


def size_depth(cam, area):
    diam = 2 * np.sqrt(np.maximum(area, 1) / np.pi)
    return np.clip(cam.K[0, 0] * BALL_D / diam, 2.0, 45.0)


def fit(cam, tau, U, area):
    """Fit one ballistic segment. Returns (theta, rms_px, residual_px per obs)."""
    rays = cam.rays(U)
    c = cam.center
    d0 = size_depth(cam, area)
    A = np.c_[np.ones_like(tau), tau]
    best = None
    for s in (0.75, 1.0, 1.35):
        P = c + rays * (d0 * s)[:, None] - 0.5 * G * (tau ** 2)[:, None]
        coef, *_ = np.linalg.lstsq(A, P, rcond=None)      # rows: p0, v0
        theta0 = np.r_[coef[0], coef[1]]

        def resid(th):
            Pm = model(th, tau)
            r_px = (project(cam, Pm) - U).ravel()
            depth = np.linalg.norm(Pm - c, axis=1)
            r_sz = 0.6 * np.log(depth / (d0 * s)) / 0.35   # weak: size depth is ±35%
            return np.r_[r_px, r_sz]

        sol = least_squares(resid, theta0, loss="soft_l1", f_scale=3.0, max_nfev=200)
        px = np.linalg.norm((project(cam, model(sol.x, tau)) - U), axis=1)
        if best is None or sol.cost < best[0]:
            best = (sol.cost, sol.x, px)
    _, th, px = best
    return th, float(np.sqrt(np.mean(px ** 2))), px


def plausible(theta, tau):
    P = model(theta, tau)
    v = np.linalg.norm(theta[3:])
    return (np.abs(P[:, 0]).max() < 10 and np.abs(P[:, 1]).max() < 6.5 and
            P[:, 2].min() > -0.35 and P[:, 2].max() < 9 and v < 35)


def kinks(F, U):
    """Indices ranked by 2D direction/speed change (bounce or hit candidates)."""
    v = np.diff(U, axis=0) / np.diff(F)[:, None]
    dv = np.linalg.norm(v[2:] - v[:-2], axis=1)      # centred change over 2 steps
    return np.argsort(-dv) + 2                         # index into U of the turning point


def segment(cam, F, U, area, fps, lo, hi, depth=0):
    """Recursively fit [lo, hi) of one track. Returns list of (lo, hi, theta, rms)."""
    n = hi - lo
    if n < MIN_SEG:
        return []
    tau = (F[lo:hi] - F[lo]) / fps
    th, rms, px = fit(cam, tau, U[lo:hi], area[lo:hi])
    if rms < RMS_OK and px.max() < MAX_OK and plausible(th, tau):
        return [(lo, hi, th, rms)]
    if depth > 6 or n < 2 * MIN_SEG - 1:
        return [(lo, hi, th, rms)] if rms < 2 * RMS_OK and plausible(th, tau) else []
    best = None
    ks = [k for k in kinks(F[lo:hi], U[lo:hi]) if MIN_SEG - 1 <= k <= n - MIN_SEG][:4]
    for k in ks:
        ta = (F[lo:lo + k + 1] - F[lo]) / fps
        tb = (F[lo + k:hi] - F[lo + k]) / fps
        _, ra, _ = fit(cam, ta, U[lo:lo + k + 1], area[lo:lo + k + 1])
        _, rb, _ = fit(cam, tb, U[lo + k:hi], area[lo + k:hi])
        score = ra * (k + 1) + rb * (n - k)
        if best is None or score < best[0]:
            best = (score, k)
    if best is None:
        return []
    k = best[1]
    return (segment(cam, F, U, area, fps, lo, lo + k + 1, depth + 1) +
            segment(cam, F, U, area, fps, lo + k, hi, depth + 1))


def is_bounce(pa, va, pb, vb):
    return 0.5 * (pa[2] + pb[2]) < 0.4 and va[2] < 0 and vb[2] > 0


def refit_chain(cam, segs, seg_obs, ch, w_join=0.05, w_floor=0.03, eps=1e-4):
    """Joint least squares over a chain of segments: reprojection of every segment
    (gravity + drag), a weak size-depth prior, soft continuity where each segment
    meets the next (σ = w_join m), and bounce points on the floor (σ = w_floor m).

    All trajectories needed for one Jacobian (each segment and its 6 perturbed
    variants) are integrated together in one batch. Updates segs in place."""
    m = len(ch)
    c = cam.center
    t0 = np.array([segs[si][0] for si in ch])
    obs_t = [seg_obs[si][0] - t0[j] for j, si in enumerate(ch)]
    U = [seg_obs[si][1] for si in ch]
    d0 = [size_depth(cam, seg_obs[si][2]) for si in ch]
    # where segment j must meet j+1: at j+1's start time
    meet = [t0[j + 1] - t0[j] for j in range(m - 1)]
    bounce = []
    for j in range(m - 1):
        sa, sb = segs[ch[j]], segs[ch[j + 1]]
        tau = meet[j]
        pa, va = sa[2:5] + sa[5:8] * tau + 0.5 * G * tau ** 2, sa[5:8] + G * tau
        bounce.append(is_bounce(pa, va, sb[2:5], sb[5:8]))
    need = [np.r_[obs_t[j], meet[j]] if j < m - 1 else obs_t[j] for j in range(m)]
    T = max(float(x.max()) for x in need)

    # residual layout
    n_obs = [len(o) for o in obs_t]
    seg_rows = []; r = 0
    for j in range(m):
        seg_rows.append((r, r + 3 * n_obs[j])); r += 3 * n_obs[j]
    knot_rows = []
    for j in range(m - 1):
        k_ = 5 if bounce[j] else 3
        knot_rows.append((r, r + k_)); r += k_
    R = r

    def blocks(j, Pj):
        """Residuals of segment j's own block given its sampled positions (obs + meet)."""
        P = Pj[:n_obs[j]]
        rp = (cam.project(P) - U[j]).ravel()
        rs = 0.6 * np.log(np.linalg.norm(P - c, axis=1) / d0[j]) / 0.35
        return np.r_[rp, rs]

    def knot(j, Pa_meet, Pb_start):
        out = (Pa_meet - Pb_start) / w_join
        if bounce[j]:
            out = np.r_[out, Pa_meet[2] / w_floor, Pb_start[2] / w_floor]
        return out

    def positions(TH):
        """TH: (B, 6) states, B = m (base) or m*7 (base + perturbations)."""
        ts, P, _ = simulate_batch(TH[:, :3], TH[:, 3:], T)
        return ts, P

    def assemble(samples):
        r = np.empty(R)
        for j in range(m):
            a0, a1 = seg_rows[j]; r[a0:a1] = blocks(j, samples[j])
        for j in range(m - 1):
            a0, a1 = knot_rows[j]; r[a0:a1] = knot(j, samples[j][n_obs[j]], samples[j + 1][0])
        return r

    def fun(x):
        TH = x.reshape(m, 6)
        ts, P = positions(TH)
        return assemble([sample(ts, P, j, need[j]) for j in range(m)])

    def jac(x):
        TH = x.reshape(m, 6)
        X = np.repeat(TH, 7, axis=0)                     # per segment: base + 6 perturbations
        for j in range(m):
            for q in range(6):
                X[j * 7 + 1 + q, q] += eps
        ts, P = positions(X)
        base = [sample(ts, P, j * 7, need[j]) for j in range(m)]
        J = np.zeros((R, 6 * m))
        for j in range(m):
            b0 = blocks(j, base[j])
            for q in range(6):
                Sj = sample(ts, P, j * 7 + 1 + q, need[j])
                col = j * 6 + q
                a0, a1 = seg_rows[j]
                J[a0:a1, col] = (blocks(j, Sj) - b0) / eps
                if j < m - 1:            # j's meet point enters knot j
                    k0, k1 = knot_rows[j]
                    J[k0:k1, col] = (knot(j, Sj[n_obs[j]], base[j + 1][0]) - knot(j, base[j][n_obs[j]], base[j + 1][0])) / eps
                if j > 0:                # j's start enters knot j-1
                    k0, k1 = knot_rows[j - 1]
                    J[k0:k1, col] = (knot(j - 1, base[j - 1][n_obs[j - 1]], Sj[0]) - knot(j - 1, base[j - 1][n_obs[j - 1]], base[j][0])) / eps
        return J

    x0 = np.concatenate([segs[si][2:8] for si in ch])
    sol = least_squares(fun, x0, jac=jac, loss="soft_l1", f_scale=3.0, max_nfev=40)
    TH = sol.x.reshape(m, 6)
    ts, P = positions(TH)
    for j, si in enumerate(ch):
        Pj = sample(ts, P, j, obs_t[j])
        segs[si][2:8] = TH[j]
        segs[si][8] = float(np.sqrt(np.mean(np.sum((cam.project(Pj) - U[j]) ** 2, axis=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="camera.json from tracking.calibrate")
    ap.add_argument("--ball", default="data/ball2d.npz")
    ap.add_argument("--players", default="data/players.npz")
    ap.add_argument("--out", default="data/ball3d.npz")
    a = ap.parse_args()

    cam = Camera.load(a.camera)
    B = np.load(a.ball)
    fps, f0 = float(B["fps"]), int(B["f0"])
    raw, tid, area = B["raw"], B["track"], B["area"]
    obs = np.isfinite(raw[:, 0])

    segs = []      # t0, t1, p0(3), v0(3), rms, n
    seg_obs = []   # per segment: (times, pixels, areas)
    for k in np.unique(tid[tid >= 0]):
        idx = np.flatnonzero((tid == k) & obs)
        if len(idx) < MIN_SEG:
            continue
        F = (idx + f0).astype(float)
        U = raw[idx]
        for lo, hi, th, rms in segment(cam, F, U, area[idx], fps, 0, len(idx)):
            segs.append([F[lo] / fps, F[hi - 1] / fps, *th, rms, hi - lo])
            seg_obs.append((F[lo:hi] / fps, U[lo:hi], area[idx][lo:hi]))
    order = np.argsort([s_[0] for s_ in segs])
    segs = np.array(segs)[order]
    seg_obs = [seg_obs[i] for i in order]
    print(f"{len(segs)} ballistic segments (gravity-only), median reprojection {np.median(segs[:, 8]):.2f} px")

    def at(s, t):
        p, v = simulate(s[2:5], s[5:8], [t - s[0]])
        return p[0], v[0]

    # Chain consecutive segments and refit each chain jointly with drag + continuity.
    chains, cur = [], [0]
    for i in range(1, len(segs)):
        if segs[i][0] - segs[i - 1][1] <= LINK_GAP:
            cur.append(i)
        else:
            chains.append(cur); cur = [i]
    chains.append(cur)
    for ch in chains:
        refit_chain(cam, segs, seg_obs, ch)
    print(f"refit {len(chains)} chains with drag; median reprojection {np.median(segs[:, 8]):.2f} px")

    # Events at segment boundaries.
    P = np.load(a.players)
    pt, ppos = P["t"], P["pos"]

    def nearest_player(t, p):
        i = int(np.clip(np.searchsorted(pt, t), 0, len(pt) - 1))
        d = np.linalg.norm(ppos[i] - p[:2], axis=1)
        if not np.isfinite(d).any():
            return -1, np.inf
        k = int(np.nanargmin(d))
        return k, float(d[k])

    events = []    # t, type(0 bounce, 1 hit), x, y, z, player
    for i in range(len(segs)):
        s = segs[i]
        prev = segs[i - 1] if i > 0 and s[0] - segs[i - 1][1] <= LINK_GAP else None
        if prev is not None:
            tb = 0.5 * (prev[1] + s[0])
            pa, va = at(prev, tb)
            pb, vb = at(s, tb)
            pm = 0.5 * (pa + pb)
            if is_bounce(pa, va, pb, vb):
                events.append([tb, 0, pm[0], pm[1], 0.0, -1])
                continue
        # a flight that starts without a bounce: someone hit it
        p, _ = at(s, s[0])
        k, d = nearest_player(s[0], p)
        if d < 2.2 and p[2] > 0.15:
            events.append([s[0], 1, *p, k])
    events = np.array(events).reshape(-1, 6)
    print(f"events: {int((events[:, 1] == 0).sum())} bounces, {int((events[:, 1] == 1).sum())} hits")
    out = Path(a.out)
    np.savez_compressed(out, segs=segs, events=events, fps=fps)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
