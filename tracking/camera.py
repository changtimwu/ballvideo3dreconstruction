"""Court model and a fixed camera looking at it.

Court coordinates (metres) match index.html: origin at the net centre on the
floor, +x toward the far baseline, +y to the left as seen from behind the near
baseline, z up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HALF_L, HALF_W, KITCHEN = 6.7056, 3.048, 2.1336
L_, W_, K_ = HALF_L, HALF_W, KITCHEN

# Painted lines of the court model: name -> ((x0, y0), (x1, y1)). "n" = near half
# (x < 0, the half closer to the camera), "f" = far half. Sidelines are split at the
# net (and stop 0.3 m short of it): fitting one straight line through the net mesh
# over the full court length biases it.
NET_GAP = 0.3
MODEL_LINES = {
    "n_base": ((-L_, W_), (-L_, -W_)), "f_base": ((L_, W_), (L_, -W_)),
    "n_kitchen": ((-K_, W_), (-K_, -W_)), "f_kitchen": ((K_, W_), (K_, -W_)),
    "nl_side": ((-L_, W_), (-NET_GAP, W_)), "fl_side": ((NET_GAP, W_), (L_, W_)),
    "nr_side": ((-L_, -W_), (-NET_GAP, -W_)), "fr_side": ((NET_GAP, -W_), (L_, -W_)),
    "n_center": ((-L_, 0), (-K_, 0)), "f_center": ((K_, 0), (L_, 0)),
}
# Named court points (line intersections), in the order the click fallback asks for them.
MODEL_POINTS = {
    "near-left corner": (("n_base", "nl_side"), (-L_, W_)),
    "near-right corner": (("n_base", "nr_side"), (-L_, -W_)),
    "far-right corner": (("f_base", "fr_side"), (L_, -W_)),
    "far-left corner": (("f_base", "fl_side"), (L_, W_)),
    "near kitchen, left end": (("n_kitchen", "nl_side"), (-K_, W_)),
    "near kitchen, right end": (("n_kitchen", "nr_side"), (-K_, -W_)),
    "far kitchen, right end": (("f_kitchen", "fr_side"), (K_, -W_)),
    "far kitchen, left end": (("f_kitchen", "fl_side"), (K_, W_)),
    "near baseline, centre mark": (("n_base", "n_center"), (-L_, 0)),
    "near kitchen, centre": (("n_kitchen", "n_center"), (-K_, 0)),
    "far kitchen, centre": (("f_kitchen", "f_center"), (K_, 0)),
    "far baseline, centre mark": (("f_base", "f_center"), (L_, 0)),
}


@dataclass
class Camera:
    K: np.ndarray        # 3x3 intrinsics
    rvec: np.ndarray     # world -> camera rotation (Rodrigues)
    tvec: np.ndarray     # world -> camera translation
    size: tuple[int, int]  # (width, height) in pixels

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def center(self) -> np.ndarray:
        return (-self.R.T @ self.tvec.reshape(3)).reshape(3)

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.size[1] / 2 / self.K[1, 1])))

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Court points (N,3) -> pixels (N,2)."""
        out, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 3), self.rvec, self.tvec, self.K, None)
        return out.reshape(-1, 2)

    def rays(self, px: np.ndarray) -> np.ndarray:
        """Pixels (N,2) -> unit ray directions in court coordinates (N,3)."""
        px = np.asarray(px, np.float64).reshape(-1, 2)
        h = np.c_[px, np.ones(len(px))]
        d = (self.R.T @ (np.linalg.inv(self.K) @ h.T)).T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_plane(self, px: np.ndarray, z: float | np.ndarray = 0.0) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane at height z -> court (N,3)."""
        d = self.rays(px)
        c = self.center
        s = (np.asarray(z, np.float64) - c[2]) / d[:, 2]
        return c + d * s[:, None]

    def to_json(self) -> dict:
        c = self.center
        fwd = self.R.T @ np.array([0, 0, 1.0])
        look = c + fwd * (c[2] / -fwd[2])
        return {
            "position": c.round(4).tolist(),
            "look_at": look.round(4).tolist(),
            "vfov_deg": round(self.vfov_deg, 3),
            "aspect": round(self.size[0] / self.size[1], 6),
            "K": self.K.tolist(), "rvec": self.rvec.reshape(3).tolist(), "tvec": self.tvec.reshape(3).tolist(),
            "size": list(self.size),
        }

    @classmethod
    def from_json(cls, d: dict) -> "Camera":
        return cls(np.array(d["K"]), np.array(d["rvec"]).reshape(3, 1), np.array(d["tvec"]).reshape(3, 1), tuple(d["size"]))

    @classmethod
    def load(cls, path) -> "Camera":
        return cls.from_json(json.loads(Path(path).read_text()))


def solve_pnp(img_pts: np.ndarray, court_xy: np.ndarray, size: tuple[int, int]) -> tuple[Camera, float]:
    """Camera from ≥4 floor correspondences (pixels ↔ court x, y), sweeping the focal
    length (square pixels, centred principal point, no distortion). Returns (camera, mean px error)."""
    W, H = size
    obj = np.c_[np.asarray(court_xy, np.float64), np.zeros(len(court_xy))]
    img_pts = np.asarray(img_pts, np.float64)
    best = None
    for f in np.geomspace(0.35, 4.0, 160) * W:
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        ok, r, t = cv2.solvePnP(obj, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        cam = Camera(K, r, t, (W, H))
        if cam.center[2] <= 0.3:          # camera must be above the floor
            continue
        e = float(np.linalg.norm(cam.project(obj) - img_pts, axis=1).mean())
        if best is None or e < best[1]:
            best = (cam, e)
    if best is None:
        raise ValueError("no camera fits these points")
    return best


def white_mask(img: np.ndarray) -> np.ndarray:
    """Painted-line pixels: unsaturated and brighter than the floor around them
    (white top-hat keeps bright structures narrower than ~41 px, i.e. lines, not walls)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    return ((hsv[..., 1] < 80) & (hsv[..., 2] > 120) & (tophat > 25)).astype(np.uint8)


def fit_line_near(mask: np.ndarray, p0, p1, band: float = 12.0):
    """Total-least-squares line through mask pixels within `band` px of segment p0–p1.
    Returns (normal, offset) or None."""
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(float)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    if L < 30:
        return None
    d = (p1 - p0) / L
    n = np.array([-d[1], d[0]]); c0 = n @ p0
    t = (P - p0) @ d
    near = (t > 0.04 * L) & (t < 0.96 * L)
    tol = band
    for _ in range(3):
        m = near & (np.abs(P @ n - c0) < tol)
        Q = P[m]
        if len(Q) < max(40, 0.3 * L):
            return None
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c)
        n = np.array([-vt[0][1], vt[0][0]]); c0 = n @ c
        tol = 4.0
    return n, c0


def visible_segment(cam: Camera, line, margin: int = 4, n: int = 200):
    """The part of a model line that projects inside the image, as two pixel endpoints."""
    (x0, y0), (x1, y1) = line
    s = np.linspace(0, 1, n)[:, None]
    pts = np.c_[np.array([x0, y0]) + s * (np.array([x1 - x0, y1 - y0])), np.zeros(n)]
    px = cam.project(pts)
    W, H = cam.size
    inside = (px[:, 0] > margin) & (px[:, 0] < W - margin) & (px[:, 1] > margin) & (px[:, 1] < H - margin)
    if inside.sum() < 10:
        return None
    k = np.flatnonzero(inside)
    return px[k[0]], px[k[-1]]


def refine(img: np.ndarray, cam: Camera, iters: int = 2) -> tuple[Camera, float, int]:
    """Snap the court model to the painted lines: fit each visible model line to white
    pixels near its projection, intersect, and re-solve. Returns (camera, px error, #points)."""
    mask = white_mask(img)
    used = 0
    err = np.inf
    for _ in range(iters):
        fits = {}
        for name, line in MODEL_LINES.items():
            seg = visible_segment(cam, line)
            if seg is not None:
                f = fit_line_near(mask, *seg)
                if f is not None:
                    fits[name] = f
        img_pts, court = [], []
        W, H = cam.size
        for (la, lb), xy in MODEL_POINTS.values():
            if la in fits and lb in fits:
                A = np.array([fits[la][0], fits[lb][0]])
                if abs(np.linalg.det(A)) < 1e-3:
                    continue
                p = np.linalg.solve(A, [fits[la][1], fits[lb][1]])
                if -20 < p[0] < W + 20 and -20 < p[1] < H + 20:
                    img_pts.append(p); court.append(xy)
        if len(img_pts) < 4:
            break
        img_pts, court = np.array(img_pts), np.array(court)
        # Near-half intersections are large and well measured; far ones are tiny,
        # foreshortened and sit next to the neighbouring court's lines. Solve from the
        # near half, then admit far points only where they agree with it.
        near = court[:, 0] < 0
        base = near if near.sum() >= 4 else np.ones(len(court), bool)
        cam, err = solve_pnp(img_pts[base], court[base], cam.size)
        res = np.linalg.norm(cam.project(np.c_[court, np.zeros(len(court))]) - img_pts, axis=1)
        keep = base & (res < max(3.0, 3 * np.median(res[base]))) | (~base & (res < 6.0))
        if keep.sum() >= 4 and not (keep == base).all():
            img_pts, court = img_pts[keep], court[keep]
            cam, err = solve_pnp(img_pts, court, cam.size)
        used = len(img_pts)
    return cam, err, used


def court_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return list(MODEL_LINES.values())


def draw_court(img: np.ndarray, cam: Camera, color=(0, 0, 255), thickness=2) -> np.ndarray:
    for a, b in court_segments():
        p = cam.project(np.array([[*a, 0], [*b, 0]])).astype(int)
        cv2.line(img, tuple(p[0]), tuple(p[1]), color, thickness, cv2.LINE_AA)
    return img


def line_score(img: np.ndarray, cam: Camera, n: int = 400) -> float:
    """Fraction of points on the projected near-half court lines that look like white paint.

    ~0.5–0.9 on normal frames (players occlude some), near 0 when the court is
    covered by an overlay, a replay from another angle, or a cut.
    """
    if not hasattr(line_score, "_pts") or line_score._key != (id(cam), img.shape):
        L, W, K = HALF_L, HALF_W, KITCHEN
        segs = [((-L, W), (-L, -W)), ((-L, W), (-K, W)), ((-L, -W), (-K, -W)), ((-K, W), (-K, -W)), ((-L, 0), (-K, 0))]
        pts = []
        for a, b in segs:
            s = np.linspace(0.05, 0.95, n // len(segs))[:, None]
            pts.append(np.c_[np.array(a) + s * (np.array(b) - np.array(a)), np.zeros(len(s))])
        px = cam.project(np.vstack(pts))
        H, Wd = img.shape[:2]
        px = px[(px[:, 0] > 2) & (px[:, 0] < Wd - 3) & (px[:, 1] > 2) & (px[:, 1] < H - 3)]
        line_score._pts, line_score._key = np.round(px).astype(int), (id(cam), img.shape)
    px = line_score._pts
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 160)).astype(np.uint8)
    white = cv2.dilate(white, np.ones((5, 5), np.uint8))
    return float(white[px[:, 1], px[:, 0]].mean())
