"""Stage 1: calibrate the fixed camera from the court lines, then check it never moves.

  uv run python -m tracking.calibrate VIDEO --out work/<id>/camera.json [--click] [--no-gui]

Automatic path:
  1. median of ~30 frames across the video → an empty court (players averaged away),
  2. white-line mask → Hough segments → merged lines → two direction families,
  3. every pair-of-lines × pair-of-lines × court-model labelling is a homography
     hypothesis; score each by how much of the projected court model lands on
     painted lines (vectorised over tens of thousands of hypotheses),
  4. best homography → camera (solvePnP with a focal-length sweep), snapped to the
     lines (camera.refine), oriented so the near baseline is the one by the camera.
If that doesn't validate, an OpenCV window asks you to click known court points
(--click forces this; --no-gui fails instead).
Then the drift check re-snaps the model every 30 s and reports how far it moves.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from .camera import (HALF_L, HALF_W, KITCHEN, MODEL_LINES, MODEL_POINTS, Camera, draw_court, line_score,
                     refine, solve_pnp, white_mask)

X_LEVELS = [-HALF_L, -KITCHEN, KITCHEN, HALF_L]     # across-court painted lines (x = const)
Y_LEVELS = [-HALF_W, 0.0, HALF_W]                   # along-court lines (y = const)
OK_SCORE = 0.55                                      # line_score of a validated calibration
OK_ERR = 4.0                                         # px


def median_background(cap: cv2.VideoCapture, n: int = 31) -> np.ndarray:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in np.linspace(total * 0.03, total * 0.97, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            frames.append(img)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def detect_lines(mask: np.ndarray, max_lines: int = 16):
    """Hough segments merged into infinite lines. Returns [(theta, rho, length)]."""
    H, W = mask.shape
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 720, threshold=60,
                           minLineLength=int(0.05 * W), maxLineGap=25)
    if segs is None:
        return []
    lines = []
    for x0, y0, x1, y1 in np.asarray(segs).reshape(-1, 4):
        th = np.arctan2(y1 - y0, x1 - x0) % np.pi
        n = np.array([-np.sin(th), np.cos(th)])
        rho = n @ [x0, y0]
        L = np.hypot(x1 - x0, y1 - y0)
        for ln in lines:      # merge with an existing line of the same angle and offset
            dth = abs((th - ln[0] + np.pi / 2) % np.pi - np.pi / 2)
            if dth < np.radians(1.5) and abs(rho - ln[1]) < 10:
                w = ln[2] + L
                ln[0] = (ln[0] * ln[2] + th * L) / w if abs(th - ln[0]) < np.pi / 2 else ln[0]
                ln[1] = (ln[1] * ln[2] + rho * L) / w
                ln[2] = w
                break
        else:
            lines.append([th, rho, L])
    lines.sort(key=lambda l: -l[2])
    return lines[:max_lines]


def intersect(l1, l2):
    n1 = np.array([-np.sin(l1[0]), np.cos(l1[0])]); n2 = np.array([-np.sin(l2[0]), np.cos(l2[0])])
    A = np.array([n1, n2])
    if abs(np.linalg.det(A)) < 1e-3:
        return None
    return np.linalg.solve(A, [l1[1], l2[1]])


PER_LINE = 24


def model_samples(per_line: int = PER_LINE, side: float = 0.2):
    """Points along every model line, plus points `side` metres either side of it.
    A painted line is white at the centre and floor-coloured beside it; walls and
    other clutter are white on both."""
    c, l, r = [], [], []
    for (x0, y0), (x1, y1) in MODEL_LINES.values():
        s = np.linspace(0.03, 0.97, per_line)[:, None]
        p = np.array([x0, y0]) + s * np.array([x1 - x0, y1 - y0])
        d = np.array([x1 - x0, y1 - y0]) / np.hypot(x1 - x0, y1 - y0)
        n = np.array([-d[1], d[0]])
        c.append(p); l.append(p + side * n); r.append(p - side * n)
    return np.vstack(c), np.vstack(l), np.vstack(r)


def auto_homography(mask: np.ndarray):
    """Best floor homography (court x, y → pixels) by exhaustive line-pair hypotheses."""
    H_img, W_img = mask.shape
    lines = detect_lines(mask)
    if len(lines) < 4:
        return None, 0.0
    ang = np.array([l[0] for l in lines])
    # two direction families: k-means on the doubled angle
    v = np.c_[np.cos(2 * ang), np.sin(2 * ang)].astype(np.float32)
    _, lab, _ = cv2.kmeans(v, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3), 5,
                           cv2.KMEANS_PP_CENTERS)
    fam = [[l for l, k in zip(lines, lab.ravel()) if k == c][:8] for c in (0, 1)]
    if min(len(f) for f in fam) < 2:
        return None, 0.0
    dil = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    centre, left, right = model_samples()
    hom = lambda q: np.c_[q, np.ones(len(q))]
    src_c, src_l, src_r = hom(centre), hom(left), hom(right)
    best = (None, 0.0)
    x_pairs = list(itertools.permutations(X_LEVELS, 2))
    y_pairs = list(itertools.permutations(Y_LEVELS, 2))
    for fa, fb in ((0, 1), (1, 0)):            # which family holds the x = const lines
        for a1, a2 in itertools.combinations(fam[fa], 2):
            for b1, b2 in itertools.combinations(fam[fb], 2):
                quad = [intersect(a, b) for a in (a1, a2) for b in (b1, b2)]
                if any(q is None for q in quad):
                    continue
                quad = np.array(quad)
                if np.abs(quad).max() > 4 * max(W_img, H_img):
                    continue
                Hs = []
                for (xa, xb), (ya, yb) in itertools.product(x_pairs, y_pairs):
                    court = np.float32([[xa, ya], [xa, yb], [xb, ya], [xb, yb]])
                    Hs.append(cv2.getPerspectiveTransform(court, quad.astype(np.float32)))
                Hs = np.array(Hs)                                   # (h, 3, 3)
                def lookup(src, m):
                    proj = np.einsum("hij,nj->hni", Hs, src)
                    w = proj[..., 2]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        px = proj[..., :2] / w[..., None]
                    ins = (w > 0) & (px[..., 0] >= 0) & (px[..., 0] < W_img) & (px[..., 1] >= 0) & (px[..., 1] < H_img)
                    xi = np.clip(np.nan_to_num(px[..., 0]), 0, W_img - 1).astype(int)
                    yi = np.clip(np.nan_to_num(px[..., 1]), 0, H_img - 1).astype(int)
                    return ins, m[yi, xi] > 0
                inside, on = lookup(src_c, dil)
                _, wl = lookup(src_l, mask)
                _, wr = lookup(src_r, mask)
                hit = inside & on & ~(wl & wr)
                nl = len(MODEL_LINES)
                hit_l = hit.reshape(len(Hs), nl, PER_LINE).sum(2)
                in_l = inside.reshape(len(Hs), nl, PER_LINE).sum(2)
                frac_l = np.where(in_l > 0, hit_l / np.maximum(in_l, 1), 0)
                # each visible model line must land on paint along its length; count how
                # many do, weighted by how much of the line is visible
                good_l = (in_l >= PER_LINE // 3) & (frac_l > 0.6)
                score = (good_l * in_l * frac_l).sum(1) * (good_l.sum(1) >= 4)
                # reject collapsed/tiny projections: the court's outer quad must be sizeable and convex
                corners = np.einsum("hij,nj->hni", Hs, np.array([[-HALF_L, HALF_W, 1], [-HALF_L, -HALF_W, 1],
                                                                  [HALF_L, -HALF_W, 1], [HALF_L, HALF_W, 1]]))
                cz = corners[..., 2]
                cxy = corners[..., :2] / np.where(np.abs(cz) < 1e-9, 1e-9, cz)[..., None]
                e1 = np.roll(cxy, -1, axis=1) - cxy
                cross = e1[..., 0] * np.roll(e1, -1, axis=1)[..., 1] - e1[..., 1] * np.roll(e1, -1, axis=1)[..., 0]
                convex = (cz > 0).all(1) & ((cross > 0).all(1) | (cross < 0).all(1))
                area = 0.5 * np.abs((cxy[..., 0] * np.roll(cxy[..., 1], -1, 1) - np.roll(cxy[..., 0], -1, 1) * cxy[..., 1]).sum(1))
                score = np.where(convex & (area > 0.03 * W_img * H_img), score, 0)
                k = int(np.argmax(score))
                if score[k] > best[1]:
                    best = (Hs[k], float(score[k]))
    return best


def camera_from_homography(Hm: np.ndarray, size) -> Camera:
    W, H = size
    pts, court = [], []
    for _, xy in MODEL_POINTS.values():
        p = Hm @ [xy[0], xy[1], 1.0]
        if p[2] <= 0:
            continue
        p = p[:2] / p[2]
        if -0.1 * W < p[0] < 1.1 * W and -0.1 * H < p[1] < 1.1 * H:
            pts.append(p); court.append(xy)
    cam, _ = solve_pnp(np.array(pts), np.array(court), size)
    return cam


def orient_near(cam: Camera) -> Camera:
    """Make the half nearer the camera the "near" half (x < 0) by rotating the court 180°."""
    if cam.center[0] <= 0:
        return cam
    # rotating the world by 180° about z: x → −x, y → −y
    Rz = np.diag([-1.0, -1.0, 1.0])
    R = cam.R @ Rz
    return Camera(cam.K, cv2.Rodrigues(R)[0], cam.tvec, cam.size)


def validate(img: np.ndarray, cam: Camera) -> float:
    return line_score(img, cam)


def auto_calibrate(bg: np.ndarray):
    mask = white_mask(bg)
    Hm, score = auto_homography(mask)
    if Hm is None:
        return None, "no court lines found"
    try:
        cam = orient_near(camera_from_homography(Hm, (bg.shape[1], bg.shape[0])))
        cam, err, used = refine(bg, cam)
        cam = orient_near(cam)
    except ValueError as e:
        return None, str(e)
    s = validate(bg, cam)
    if s < OK_SCORE or err > OK_ERR:
        return None, f"best fit didn't validate (line score {s:.2f}, error {err:.1f} px)"
    return cam, f"auto: {used} points, error {err:.2f} px, line score {s:.2f}"


def click_calibrate(bg: np.ndarray) -> Camera:
    """Ask the user to click named court points on the empty-court frame."""
    names = list(MODEL_POINTS)
    scale = min(1.0, 1600 / bg.shape[1])
    shown = cv2.resize(bg, None, fx=scale, fy=scale)
    picked: dict[str, tuple[float, float]] = {}
    state = {"i": 0}
    win = "calibrate: click the named point (s = skip, u = undo, Enter = done)"

    def redraw():
        im = shown.copy()
        for nm, (x, y) in picked.items():
            cv2.circle(im, (int(x * scale), int(y * scale)), 6, (0, 0, 255), 2)
            cv2.putText(im, nm, (int(x * scale) + 8, int(y * scale) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        prompt = names[state["i"]] if state["i"] < len(names) else "all points done — press Enter"
        cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(im, f"Click: {prompt}   ({len(picked)} picked, need 4+)", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, im)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and state["i"] < len(names):
            picked[names[state["i"]]] = (x / scale, y / scale)
            state["i"] += 1
            redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (13, 10) and len(picked) >= 4:
            break
        if k == ord("s") and state["i"] < len(names):
            state["i"] += 1; redraw()
        if k == ord("u") and state["i"] > 0:
            state["i"] -= 1; picked.pop(names[state["i"]], None); redraw()
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("calibration cancelled")
    cv2.destroyAllWindows()
    img_pts = np.array(list(picked.values()))
    court = np.array([MODEL_POINTS[n][1] for n in picked])
    cam, _ = solve_pnp(img_pts, court, (bg.shape[1], bg.shape[0]))
    cam, _, _ = refine(bg, cam)
    return orient_near(cam)


def drift_check(cap: cv2.VideoCapture, cam: Camera, every: float = 30.0):
    """Re-snap the model at intervals; returns [(t, median px shift of the court points)]."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    court = np.array([[*xy, 0] for _, xy in MODEL_POINTS.values()])
    ref = cam.project(court)
    W, H = cam.size
    vis = (ref[:, 0] > 0) & (ref[:, 0] < W) & (ref[:, 1] > 0) & (ref[:, 1] < H)
    out = []
    for t in np.arange(every / 2, total / fps - 1, every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, img = cap.read()
        if not ok or line_score(img, cam) < 0.3:
            continue                      # overlay / cut: not a camera move
        try:
            c2, _, _ = refine(img, cam, iters=1)
        except ValueError:
            continue
        out.append((float(t), float(np.median(np.linalg.norm(c2.project(court)[vis] - ref[vis], axis=1)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--click", action="store_true", help="skip auto-detection and click points")
    ap.add_argument("--no-gui", action="store_true", help="fail instead of opening the click window")
    ap.add_argument("--overlay", default=None, help="write a court-overlay check image here")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    bg = median_background(cap)
    cam, how = (None, "click requested") if a.click else auto_calibrate(bg)
    print(f"calibration: {how}")
    if cam is None:
        if a.no_gui:
            raise SystemExit("automatic calibration failed; re-run without --no-gui to click court points")
        cam = click_calibrate(bg)
        print(f"calibration: clicked, line score {validate(bg, cam):.2f}")
    print(f"camera at {cam.center.round(2)} m, vertical FOV {cam.vfov_deg:.1f}°")

    drift = drift_check(cap, cam)
    if drift:
        d = np.array([x[1] for x in drift])
        print(f"drift check over {len(d)} samples: median {np.median(d):.1f} px, worst {d.max():.1f} px")
        if np.median(d) > 8:
            print("WARNING: the camera seems to move; tracking assumes a fixed camera")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cam.to_json(), indent=2) + "\n")
    print(f"wrote {a.out}")
    if a.overlay:
        cv2.imwrite(a.overlay, cv2.resize(draw_court(bg.copy(), cam), None, fx=0.5, fy=0.5))


if __name__ == "__main__":
    main()
