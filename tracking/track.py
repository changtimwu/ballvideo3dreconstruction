"""Stage 3: assign detections to the 4 players and build smooth floor tracks.

  uv run python -m tracking.track [--det data/detections.npz] [--out data/players.npz]

The match has exactly four players with distinct shirts, so instead of a
generic multi-object tracker we:
  1. link detections into tracklets by floor-position continuity only, splitting
     whenever two people get close enough to be confused,
  2. take each tracklet's median shirt colour (stable over hundreds of frames,
     unlike per-frame samples), learn four colour prototypes, and assign
     tracklets to players longest-first so no player is in two places at once,
  3. estimate each player's hip height and use it to place the feet when the
     ankles are occluded or out of frame,
  4. reject outliers, fill short gaps and low-pass filter.
Identity is colour-based, so it survives the teams switching ends. Frames where
the court is hidden (edited-in overlays) are treated as missing.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter

from .camera import Camera, HALF_L, HALF_W

N_PLAYERS = 4
COURT_OK = 0.6         # camera.line_score below this = court hidden (overlay/cut)
ANKLE_Z = 0.08          # ankle joint height above the floor (m)
DEFAULT_HIP_Z = 0.92
MAX_SPEED = 8.0         # m/s gate for re-association
MAX_GAP_S = 2.0         # interpolate gaps up to this long
CUTOFF_HZ = 2.5


def ok(kp, idx, thr=0.45):
    return (kp[idx, 2] > thr).all()


def floor_estimates(cam: Camera, box, kp, size, hip_z=DEFAULT_HIP_Z):
    """(xy, source) — source 0: ankles, 1: hips (feet hidden), 2: bbox bottom."""
    W, H = size
    at_edge = box[3] > H - 6 or box[0] < 3 or box[2] > W - 3
    if ok(kp, [15, 16]) and not at_edge:
        return cam.to_plane(kp[[15, 16], :2].mean(0)[None], ANKLE_Z)[0, :2], 0
    if ok(kp, [11, 12]):
        return cam.to_plane(kp[[11, 12], :2].mean(0)[None], hip_z)[0, :2], 1
    return cam.to_plane(np.array([[(box[0] + box[2]) / 2, box[3]]]), 0.0)[0, :2], 2


def hip_height(cam: Camera, kp) -> float:
    """Height of the hip midpoint above the ankle-derived floor point."""
    foot = cam.to_plane(kp[[15, 16], :2].mean(0)[None], ANKLE_Z)[0]
    d = cam.rays(kp[[11, 12], :2].mean(0)[None])[0]
    c = cam.center
    # point on the hip ray nearest the vertical line through the foot
    s = np.dot(foot[:2] - c[:2], d[:2]) / np.dot(d[:2], d[:2])
    return float(c[2] + d[2] * s)


def lab_to_hex(lab):
    px = np.uint8([[np.clip(lab, 0, 255)]])
    r, g, b = cv2.cvtColor(px, cv2.COLOR_LAB2RGB)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def colour_name(lab) -> str:
    """Rough Traditional Chinese shirt-colour name from an OpenCV Lab triple."""
    L = lab[0] * 100 / 255
    a, b = lab[1] - 128, lab[2] - 128
    chroma = np.hypot(a, b)
    # Thresholds tuned on match.mp4, where shirts read ~15 L darker than they look.
    if chroma < 14:
        return "白衣" if L > 55 else "灰衣" if L > 25 else "黑衣"
    hue = np.degrees(np.arctan2(b, a)) % 360
    if hue < 50 or hue > 330:
        return "紅衣" if L < 70 else "粉衣"
    if hue < 75:
        return "橘衣"
    if hue < 110:
        return "黃衣"
    if hue < 170:
        return "綠衣"
    return "淺藍衣" if L > 55 else "藍衣"


CROWD_M = 0.7          # people closer than this can swap identities


def dedupe_and_split(by_frame, box, conf, xy):
    """Drop duplicate boxes on one person; split off people standing within CROWD_M of another.

    Returns (clean, crowded): per-frame detection lists. Crowded detections are kept
    out of tracklet building, where they would shatter tracklets, and are filled in
    after identities are known (fill_crowded).
    """
    clean, crowded = {}, {}
    for f, ids in by_frame.items():
        keep = []
        for i in sorted(ids, key=lambda i: -conf[i]):
            if all(iou(box[i], box[j]) < 0.5 and np.linalg.norm(xy[i] - xy[j]) > 0.3 for j in keep):
                keep.append(i)
        close = {i for i in keep for j in keep if i != j and np.linalg.norm(xy[i] - xy[j]) < CROWD_M}
        clean[f] = [i for i in keep if i not in close]
        if close:
            crowded[f] = sorted(close)
    return clean, crowded


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def fill_crowded(det_of, crowded, frames, xy, t_of, max_span=1.5, gate=1.2):
    """Give crowded detections to the free players whose track interpolates to them."""
    fidx = {int(f): k for k, f in enumerate(frames)}
    filled = 0
    known = [np.flatnonzero(det_of[:, k] >= 0) for k in range(N_PLAYERS)]
    for f, ids in crowded.items():
        fi = fidx[int(f)]
        cand, preds = [], []
        for k in range(N_PLAYERS):
            if det_of[fi, k] >= 0 or len(known[k]) == 0:
                continue
            j = np.searchsorted(known[k], fi)
            if j == 0 or j == len(known[k]):
                continue
            a, b = known[k][j - 1], known[k][j]
            if t_of[b] - t_of[a] > max_span:
                continue
            u = (t_of[fi] - t_of[a]) / (t_of[b] - t_of[a])
            cand.append(k); preds.append(xy[det_of[a, k]] * (1 - u) + xy[det_of[b, k]] * u)
        if not cand:
            continue
        C = np.linalg.norm(xy[ids][:, None] - np.array(preds)[None], axis=2)
        for r, c in zip(*linear_sum_assignment(C)):
            if C[r, c] < gate:
                det_of[fi, cand[c]] = ids[r]
                filled += 1
    return filled


def build_tracklets(frames, by_frame, xy, fps):
    """Link detections frame-to-frame by floor distance (Hungarian, speed-gated)."""
    active, done = [], []
    for fi, f in enumerate(frames):
        t = f / fps
        ids = by_frame.get(f, [])
        # close tracklets that have been unmatched too long
        still = []
        for tr in active:
            (still if t - tr["t"] < 0.35 else done).append(tr)
        active = still
        matched = set()
        if active and ids:
            C = np.full((len(active), len(ids)), 1e6)
            for r, tr in enumerate(active):
                dt = t - tr["t"]
                pred = tr["xy"] + tr["v"] * dt
                for c, i in enumerate(ids):
                    d = np.linalg.norm(xy[i] - pred)
                    if d < 0.5 + MAX_SPEED * dt:
                        C[r, c] = d
            rr, cc = linear_sum_assignment(C)
            for r, c in zip(rr, cc):
                if C[r, c] >= 1e6:
                    continue
                tr, i = active[r], ids[c]
                dt = t - tr["t"]
                tr["v"] = 0.5 * tr["v"] + 0.5 * (xy[i] - tr["xy"]) / max(dt, 1e-3)
                tr["xy"], tr["t"] = xy[i], t
                tr["dets"].append(i); tr["fis"].append(fi)
                matched.add(i)
        for i in ids:
            if i in matched:
                continue
            active.append({"xy": xy[i], "t": t, "v": np.zeros(2), "dets": [i], "fis": [fi]})
    done += active
    for tr in done:
        tr["dets"] = np.array(tr["dets"]); tr["fis"] = np.array(tr["fis"])
    return done


def merge_unambiguous(tracklets, xy, t_of, max_gap=1.0, radius=1.5):
    """Join tracklet A→B when B starts soon after A ends, nearby, and neither has another candidate."""
    while True:
        ends = [(tr["fis"][-1], xy[tr["dets"][-1]]) for tr in tracklets]
        starts = [(tr["fis"][0], xy[tr["dets"][0]]) for tr in tracklets]
        cand = {}
        for a, (fa, pa) in enumerate(ends):
            for b, (fb, pb) in enumerate(starts):
                dt = t_of[fb] - t_of[fa]
                if a != b and 0 < dt <= max_gap and np.linalg.norm(pb - pa) < radius + 2.0 * dt:
                    cand.setdefault(("a", a), []).append(b)
                    cand.setdefault(("b", b), []).append(a)
        pairs = [(a, bs[0]) for (kind, a), bs in cand.items()
                 if kind == "a" and len(bs) == 1 and len(cand[("b", bs[0])]) == 1]
        if not pairs:
            return tracklets
        nxt = dict(pairs)
        heads = set(nxt) - set(nxt.values())
        merged, used = [], set()
        for h in heads:
            chain, c = [h], h
            while c in nxt:
                c = nxt[c]; chain.append(c)
            used.update(chain)
            merged.append({"dets": np.concatenate([tracklets[i]["dets"] for i in chain]),
                           "fis": np.concatenate([tracklets[i]["fis"] for i in chain])})
        tracklets = merged + [tr for i, tr in enumerate(tracklets) if i not in used]


def greedy_assign(tracklets, tlen, cost, n):
    """Longest tracklets first; each takes its cheapest player who is free for its whole span."""
    occupied = np.zeros((N_PLAYERS, n), bool)
    det_of = np.full((n, N_PLAYERS), -1)
    owner = np.full(len(tracklets), -1)
    for ti in np.argsort(-tlen):
        if not np.isfinite(cost[ti]).any():
            continue
        fis = tracklets[ti]["fis"]
        for k in np.argsort(cost[ti]):
            if np.isfinite(cost[ti, k]) and not occupied[k, fis].any():
                occupied[k, fis] = True
                det_of[fis, k] = tracklets[ti]["dets"]
                owner[ti] = k
                break
    return det_of, owner


def team_two_colour(tracklets, tlen, cost, teams, n, team_of):
    """Joint within-team assignment.

    Each tracklet joins the team whose players it fits best. Inside a team, two
    tracklets that overlap in time must be different players: a two-colouring of the
    overlap graph, kept consistent with a parity union-find. Tracklets are added
    longest first; one that contradicts the colouring is dropped (spectators,
    duplicates). Each connected component is then oriented as a whole by its
    length-weighted colour evidence, so one ambiguous tracklet can't flip its neighbours.
    """
    det_of = np.full((n, N_PLAYERS), -1)
    owner = np.full(len(tracklets), -1)
    span = np.array([(tr["fis"][0], tr["fis"][-1]) for tr in tracklets])
    for ti_team, (pa, pb) in enumerate(teams):
        members = [i for i in np.argsort(-tlen) if team_of[i] == ti_team and np.isfinite(cost[i, [pa, pb]]).all()]
        parent, parity, added = {}, {}, []

        def find(x):
            if parent[x] == x:
                return x, 0
            r, p = find(parent[x])
            parent[x], parity[x] = r, parity[x] ^ p
            return r, parity[x]

        for i in members:
            over = [j for j in added if span[i, 0] <= span[j, 1] and span[j, 0] <= span[i, 1]
                    and np.intersect1d(tracklets[i]["fis"], tracklets[j]["fis"], assume_unique=True).size]
            parent[i], parity[i] = i, 0
            ok_ = True
            trial = dict(parent), dict(parity)
            for j in over:
                ri, pi = find(i); rj, pj = find(j)
                if ri == rj:
                    if pi == pj:          # would need i and j to be the same player while both on court
                        ok_ = False
                        break
                else:
                    parent[ri], parity[ri] = rj, pi ^ pj ^ 1
            if not ok_:
                parent, parity = trial
                del parent[i], parity[i]
                continue
            added.append(i)
        comps = {}
        for i in added:
            r, p = find(i)
            comps.setdefault(r, []).append((i, p))
        for items in comps.values():
            # orientation A: parity 0 -> pa ; orientation B: parity 0 -> pb
            ca = sum(tlen[i] * cost[i, pa if p == 0 else pb] for i, p in items)
            cb = sum(tlen[i] * cost[i, pb if p == 0 else pa] for i, p in items)
            for i, p in items:
                k = (pa if p == 0 else pb) if ca <= cb else (pb if p == 0 else pa)
                owner[i] = k
                det_of[tracklets[i]["fis"], k] = tracklets[i]["dets"]
    return det_of, owner


def pair_teams(det_of, xy):
    """The two player pairs that most often stand on the same half."""
    side = np.where(det_of >= 0, np.sign(xy[np.maximum(det_of, 0), 0]), 0)
    def agree(a, b):
        both = (side[:, a] != 0) & (side[:, b] != 0)
        return (side[both, a] == side[both, b]).mean() if both.any() else 0
    best = max(((0, 1), (0, 2), (0, 3)), key=lambda p: agree(*p))
    rest = tuple(k for k in range(N_PLAYERS) if k not in best)
    return [best, rest]


def partner_relative(by_frame, xy, colour):
    """colour minus the colour of the one other person on the same half, else NaN."""
    rel = np.full_like(colour, np.nan)
    for ids in by_frame.values():
        for half in (-1, 1):
            h = [i for i in ids if np.sign(xy[i, 0]) == half and np.isfinite(colour[i]).all()]
            if len(h) == 2:
                rel[h[0]] = colour[h[0]] - colour[h[1]]
                rel[h[1]] = -rel[h[0]]
    return rel


def nanmedian_rows(X):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(X, axis=0)


def nan_dist(X, p):
    """RMS difference over the dimensions both have; NaN if none."""
    d = (X - p) ** 2
    m = np.isfinite(d)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(m, d, 0).sum(-1) / m.sum(-1))


def anchor_side(det_of, xy, t_of, protos, teams, half_window=15.0):
    """Which half the most distinctive player (highest shirt chroma, e.g. red) is on over time.

    Partners share a half, and the ends switch only between games, so this tells
    every tracklet its team more reliably than shirt colour does.
    Returns (side(t) -> -1/0/+1, index of the anchor's team).
    """
    chroma = np.hypot(protos[:, 1] - 128, protos[:, 2] - 128)
    k = int(np.argmax(chroma))
    fis = np.flatnonzero(det_of[:, k] >= 0)
    ts, xs = t_of[fis], xy[det_of[fis, k], 0]

    def side(t):
        lo, hi = np.searchsorted(ts, [t - half_window, t + half_window])
        if hi - lo < 20:
            return 0
        m = np.median(xs[lo:hi])
        return int(np.sign(m)) if abs(m) > 0.5 else 0
    team = next(i for i, tm in enumerate(teams) if k in tm)
    return side, team


def team_by_side(tracklets, xy, t_of, side_fn, anchor_team, dist, teams):
    by_colour = np.argmin([np.minimum(dist[:, x], dist[:, y]) for x, y in teams], axis=0)
    out = by_colour.copy()
    for i, tr in enumerate(tracklets):
        mx = np.median(xy[tr["dets"], 0])
        s = side_fn(np.median(t_of[tr["fis"]]))
        if s != 0 and abs(mx) > 0.5:
            out[i] = anchor_team if np.sign(mx) == s else 1 - anchor_team
    return out


def split_on_appearance(tracklets, parts, team_of, teams, team_protos, win=31, min_run=45):
    """Cut tracklets where the person changes (the motion linker can swap two players
    during a crossing). Per detection, score which teammate prototype it resembles;
    smooth over ~1 s; cut between consecutive runs of opposite sign that each last
    at least min_run samples (~1.5 s)."""
    from scipy.ndimage import median_filter
    out = []
    for ti, tr in enumerate(tracklets):
        if len(tr["dets"]) < 2 * min_run:
            out.append(tr); continue
        pa, pb = team_protos[teams[team_of[ti]]]
        sc = nan_dist(parts[tr["dets"]], pa) - nan_dist(parts[tr["dets"]], pb)
        ok_ = np.isfinite(sc)
        if ok_.sum() < 2 * min_run:
            out.append(tr); continue
        idx = np.flatnonzero(ok_)
        sm = np.sign(median_filter(np.interp(np.arange(len(sc)), idx, sc[idx]), size=win, mode="nearest"))
        runs, start = [], 0                 # (start, end, sign) runs of the smoothed sign
        for k in range(1, len(sm) + 1):
            if k == len(sm) or sm[k] != sm[start]:
                runs.append((start, k, sm[start])); start = k
        long_runs = [r for r in runs if r[1] - r[0] >= min_run and r[2] != 0]
        cuts = [(p[1] + q[0]) // 2 for p, q in zip(long_runs, long_runs[1:]) if p[2] != q[2]]
        if not cuts:
            out.append(tr); continue
        for lo, hi in zip([0] + cuts, cuts + [len(sm)]):
            out.append({"dets": tr["dets"][lo:hi], "fis": tr["fis"][lo:hi]})
    return out


def two_means_seeded(tracklets, desc, tlen, members, iters=20):
    """2-means over a team's tracklet descriptors, seeded by the longest pair of
    tracklets that overlap in time (necessarily two different people)."""
    best, seed = -1, None
    mem = sorted(members, key=lambda i: -tlen[i])[:60]
    for x in range(len(mem)):
        for y in range(x + 1, len(mem)):
            i, j = mem[x], mem[y]
            ov = np.intersect1d(tracklets[i]["fis"], tracklets[j]["fis"], assume_unique=True).size
            if ov and min(tlen[i], tlen[j]) > best and np.isfinite(desc[i]).sum() > 6 and np.isfinite(desc[j]).sum() > 6:
                best, seed = min(tlen[i], tlen[j]), (i, j)
    pa, pb = desc[seed[0]].copy(), desc[seed[1]].copy()
    for _ in range(iters):
        da, db = nan_dist(desc[members], pa), nan_dist(desc[members], pb)
        ga = members[np.nan_to_num(da, nan=1e9) <= np.nan_to_num(db, nan=1e9)]
        gb = members[np.nan_to_num(da, nan=1e9) > np.nan_to_num(db, nan=1e9)]
        for g, which in ((ga, 0), (gb, 1)):
            if len(g):
                W = np.where(np.isfinite(desc[g]), tlen[g, None], 0)
                v = (np.nan_to_num(desc[g]) * W).sum(0) / np.maximum(W.sum(0), 1e-9)
                v[W.sum(0) == 0] = np.nan
                if which == 0: pa = v
                else: pb = v
    return pa, pb


def weighted_kmeans(X, w, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(10):
        C = X[rng.choice(len(X), k, replace=False, p=w / w.sum())]
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C = np.array([np.average(X[lab == j], axis=0, weights=w[lab == j]) if (lab == j).any() else C[j]
                          for j in range(k)])
        cost = (w * ((X - C[lab]) ** 2).sum(-1)).sum()
        if best is None or cost < best[0]:
            best = (cost, C)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="data/detections.npz")
    ap.add_argument("--app", default=None, help="appearance.npz (default: next to --det)")
    ap.add_argument("--out", default="data/players.npz")
    a = ap.parse_args()

    cam = Camera.load()
    D = np.load(a.det)
    fps, step = float(D["fps"]), int(D["step"])
    order = np.argsort(D["frames"])
    frames = D["frames"][order]
    court_ok = (D["court_score"][order] >= COURT_OK) if "court_score" in D else np.ones(len(frames), bool)
    hidden = set(frames[~court_ok].tolist())
    print(f"court hidden in {len(hidden)} frames ({len(hidden) / len(frames) * 100:.1f}%)")
    det_frame, box, kp, colour = D["frame"], D["box"], D["kp"], D["colour"]
    n_det = len(det_frame)
    print(f"{n_det} detections, {len(frames)} frames")

    xy = np.zeros((n_det, 2)); src = np.zeros(n_det, int)
    for i in range(n_det):
        xy[i], src[i] = floor_estimates(cam, box[i], kp[i], cam.size)
    strict = (np.abs(xy[:, 0]) < HALF_L + 1.5) & (np.abs(xy[:, 1]) < HALF_W + 1.0)

    by_frame = {}
    for i, f in enumerate(det_frame):
        if strict[i] and f not in hidden:
            by_frame.setdefault(f, []).append(i)
    n = len(frames)
    t_of = frames / fps

    # 1. Tracklets from motion alone. People closer than CROWD_M to another person
    #    are left out, so colour decides who is who after a crossing.
    by_frame, crowded = dedupe_and_split(by_frame, box, D["conf"], xy)
    print(f"crowded detections in {len(crowded)} frames")
    tracklets = build_tracklets(frames, by_frame, xy, fps)
    n0 = len(tracklets)
    tracklets = merge_unambiguous(tracklets, xy, t_of)
    print(f"{n0} tracklets → {len(tracklets)} after unambiguous merges")

    # 2. Colour per tracklet (median over its detections), prototypes by
    #    length-weighted k-means, then greedy assignment longest-first with no
    #    player in two places at once.
    tcol = np.array([np.nanmedian(colour[tr["dets"]], axis=0) for tr in tracklets])
    tlen = np.array([len(tr["dets"]) for tr in tracklets], float)
    good = np.isfinite(tcol).all(1) & (tlen >= 8)
    protos = weighted_kmeans(tcol[good], tlen[good], N_PLAYERS)
    for k, p in enumerate(protos):
        print(f"  player {k}: Lab {p.round(1)}  {colour_name(p)} {lab_to_hex(p)}")

    dist = np.linalg.norm(tcol[:, None] - protos[None], axis=2)
    det_of, owner = greedy_assign(tracklets, tlen, dist, n)

    # 2b. Teammates can wear near-identical shirts (the near pair here: light grey
    #     vs white), so who-is-who within a team uses a multi-region descriptor
    #     (hair, torso, shorts, elbows — see appearance.py) when available, else the
    #     shirt colour relative to the partner on the same half. Team membership
    #     still comes from absolute shirt colour, which separates the pairs well.
    teams = pair_teams(det_of, xy)
    side_fn, anchor_team = anchor_side(det_of, xy, t_of, protos, teams)
    app_path = Path(a.app) if a.app else Path(a.det.replace("detections", "appearance"))
    cost = None
    if app_path.exists():
        parts = np.load(app_path)["parts"].reshape(n_det, -1)
        print(f"within-team identity from {app_path.name} ({parts.shape[1] // 3} regions)")
        team_protos = {}
        for _ in range(2):  # learn teammate prototypes, split tracklets that change person, relearn
            tcol = np.array([nanmedian_rows(colour[tr["dets"]]) for tr in tracklets])
            tlen = np.array([len(tr["dets"]) for tr in tracklets], float)
            dist = np.linalg.norm(tcol[:, None] - protos[None], axis=2)
            dist[~np.isfinite(dist)] = np.inf
            team_of = team_by_side(tracklets, xy, t_of, side_fn, anchor_team, dist, teams)
            tdesc = np.array([nanmedian_rows(parts[tr["dets"]]) for tr in tracklets])
            for ti_, team in enumerate(teams):
                team_protos[team] = two_means_seeded(tracklets, tdesc, tlen, np.flatnonzero(team_of == ti_))
            if _ == 0:
                n_before = len(tracklets)
                tracklets = split_on_appearance(tracklets, parts, team_of, teams, team_protos)
                print(f"split on appearance change: {n_before} → {len(tracklets)} tracklets")
        cost = np.full_like(dist, np.inf)
        for a_, b_ in teams:
            team_d = np.minimum(dist[:, a_], dist[:, b_])
            pa, pb = team_protos[(a_, b_)]
            da, db = nan_dist(tdesc, pa), nan_dist(tdesc, pb)
            cost[:, a_] = team_d / 20 + np.nan_to_num(da - db, nan=0) / 15
            cost[:, b_] = team_d / 20 + np.nan_to_num(db - da, nan=0) / 15
    else:
        cost = np.full_like(dist, np.inf)
        rel = partner_relative(by_frame, xy, colour)
        trel = np.array([np.nanmedian(rel[tr["dets"]], axis=0) if np.isfinite(rel[tr["dets"]]).all(1).sum() >= 5
                         else np.full(3, np.nan) for tr in tracklets])
        for a_, b_ in teams:
            team_d = np.minimum(dist[:, a_], dist[:, b_])
            for k, other in ((a_, b_), (b_, a_)):
                u = protos[k] - protos[other]
                u /= np.linalg.norm(u)
                within = np.where(np.isfinite(trel).all(1), -np.clip(trel @ u / 20, -3, 3), (dist[:, k] - team_d) / 20)
                cost[:, k] = team_d / 20 + within
    # team membership from court side (colour as fallback); the within-team score must not move a tracklet across teams
    team_of = team_by_side(tracklets, xy, t_of, side_fn, anchor_team, dist, teams)
    det_of, owner = team_two_colour(tracklets, tlen, cost, teams, n, team_of)
    if os.environ.get("TRACK_DEBUG"):
        np.savez("data/tracklets_debug.npz", start=[t_of[tr["fis"][0]] for tr in tracklets],
                 end=[t_of[tr["fis"][-1]] for tr in tracklets], owner=owner, tlen=tlen, tcol=tcol,
                 medx=[np.median(xy[tr["dets"], 0]) for tr in tracklets], cost=cost)
    kept = owner >= 0
    print(f"teams {teams}; assigned {kept.sum()}/{len(tracklets)} tracklets "
          f"({tlen[kept].sum() / tlen.sum() * 100:.1f}% of detections)")

    print(f"filled {fill_crowded(det_of, crowded, frames, xy, t_of)} crowded detections")

    # 3. Per-player hip height, then recompute positions that relied on hips.
    pos = np.full((n, N_PLAYERS, 2), np.nan)
    hip_z = []
    for k in range(N_PLAYERS):
        hs = [hip_height(cam, kp[i]) for i in det_of[:, k] if i >= 0 and src[i] == 0 and ok(kp[i], [11, 12, 15, 16])]
        hs = [h for h in hs if 0.6 < h < 1.2]
        hip_z.append(float(np.median(hs)) if hs else DEFAULT_HIP_Z)
    for fi in range(n):
        for k in range(N_PLAYERS):
            i = det_of[fi, k]
            if i >= 0:
                pos[fi, k], _ = floor_estimates(cam, box[i], kp[i], cam.size, hip_z[k])
    print("hip heights:", [round(h, 3) for h in hip_z])

    # 4. Clean-up: outliers vs a running median, gap fill, zero-phase low-pass.
    t = frames / fps
    raw = pos.copy()
    smooth = np.full_like(pos, np.nan)
    b, a_ = butter(2, CUTOFF_HZ / (fps / step / 2))
    for k in range(N_PLAYERS):
        v = np.isfinite(pos[:, k, 0])
        if v.sum() < 10:
            continue
        idx = np.flatnonzero(v)
        for d in range(2):
            series = np.interp(np.arange(n), idx, pos[idx, k, d])
            med = median_filter(series, size=9, mode="nearest")
            bad = v & (np.abs(series - med) > 0.6)
            v[bad] = False
        idx = np.flatnonzero(v)
        filled = np.stack([np.interp(t, t[idx], pos[idx, k, d]) for d in range(2)], 1)
        # mark long gaps as missing so the viewer doesn't show a confident slide
        gap = np.zeros(n, bool)
        nxt = np.searchsorted(idx, np.arange(n))
        prev_t = np.where(nxt > 0, t[idx[np.maximum(nxt - 1, 0)]], -np.inf)
        next_t = np.where(nxt < len(idx), t[idx[np.minimum(nxt, len(idx) - 1)]], np.inf)
        gap = ((next_t - prev_t > MAX_GAP_S) & ~v) | ~court_ok
        f2 = filled.copy()
        if n > 12:
            f2 = np.stack([filtfilt(b, a_, filled[:, d]) for d in range(2)], 1)
        f2[gap] = np.nan
        smooth[:, k] = f2
        print(f"player {k}: observed {v.mean() * 100:5.1f}% of frames, "
              f"{np.mean(src[det_of[det_of[:, k] >= 0, k]] == 0) * 100:4.1f}% from ankles, long gaps {gap.mean() * 100:.1f}%")

    # identity ↔ colour can differ from the initial clusters after within-team
    # orientation, so name/colour each identity from the detections it was given
    final_protos = np.array([nanmedian_rows(colour[det_of[det_of[:, k] >= 0, k]]) for k in range(N_PLAYERS)])
    for k, p in enumerate(final_protos):
        print(f"  identity {k}: shirt Lab {p.round(0)} {colour_name(p)}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t=t, frames=frames, fps=fps, step=step, pos=smooth, raw=raw, det_of=det_of, court_ok=court_ok,
                        protos=final_protos, hip_z=np.array(hip_z))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
