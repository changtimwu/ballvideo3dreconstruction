"""Court model and the fixed match camera.

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
CAMERA_JSON = Path(__file__).with_name("camera.json")

# Rough guides for the near-half court lines, in 960x540 image coordinates for
# the fixed camera in match.mp4. The fit snaps them to the actual white pixels.
ROUGH_LINES = {
    "base": [(20, 268), (488, 528)],
    "lside": [(20, 260), (372, 189)],
    "rside": [(492, 528), (800, 276)],
    "kitch": [(372, 189), (800, 276)],
    "center": [(165, 348), (530, 222)],
}
# Court points as intersections of fitted lines.
CORNERS = {
    "NL": (("base", "lside"), (-HALF_L, HALF_W, 0)),
    "NR": (("base", "rside"), (-HALF_L, -HALF_W, 0)),
    "KL": (("kitch", "lside"), (-KITCHEN, HALF_W, 0)),
    "KR": (("kitch", "rside"), (-KITCHEN, -HALF_W, 0)),
    "CB": (("base", "center"), (-HALF_L, 0, 0)),
    "CK": (("kitch", "center"), (-KITCHEN, 0, 0)),
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
    def load(cls, path: Path = CAMERA_JSON) -> "Camera":
        return cls.from_json(json.loads(Path(path).read_text()))


def _fit_lines(img: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
    H, W = img.shape[:2]
    sx, sy = W / 960, H / 540
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 60) & (hsv[..., 2] > 170)
    ys, xs = np.nonzero(white)
    P = np.stack([xs, ys], 1).astype(float)
    fit = {}
    for k, (a, b) in ROUGH_LINES.items():
        a = np.array(a) * (sx, sy); b = np.array(b) * (sx, sy)
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]]); c0 = n @ a
        t = (P - a) @ d
        band = (t > 40) & (t < np.linalg.norm(b - a) - 40)
        tol = 10.0
        for _ in range(3):  # shrink the band around the refit line to drop players/shadows
            m = band & (np.abs(P @ n - c0) < tol)
            Q = P[m]
            if len(Q) < 50:
                raise ValueError(f"court line {k!r} not found")
            c = Q.mean(0)
            _, _, vt = np.linalg.svd(Q - c)
            n = np.array([-vt[0][1], vt[0][0]]); c0 = n @ c
            tol = 4.0
        fit[k] = (n, c0)
    return fit


def solve(img: np.ndarray) -> tuple[Camera, float, dict[str, np.ndarray]]:
    """Solve the camera from one frame. Returns (camera, mean reprojection px, image points)."""
    H, W = img.shape[:2]
    fit = _fit_lines(img)

    def cross(a, b):
        A = np.array([fit[a][0], fit[b][0]])
        return np.linalg.solve(A, [fit[a][1], fit[b][1]])

    names = list(CORNERS)
    img_pts = np.array([cross(*CORNERS[k][0]) for k in names], np.float64)
    obj = np.array([CORNERS[k][1] for k in names], np.float64)
    best = None
    for f in np.linspace(800, 3000, 221) * (H / 1080):
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        _, r, t = cv2.solvePnP(obj, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        pr, _ = cv2.projectPoints(obj, r, t, K, None)
        e = float(np.linalg.norm(pr.reshape(-1, 2) - img_pts, axis=1).mean())
        if best is None or e < best[0]:
            best = (e, Camera(K, r, t, (W, H)))
    return best[1], best[0], dict(zip(names, img_pts))


def court_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    L, W, K = HALF_L, HALF_W, KITCHEN
    return [((-L, W), (L, W)), ((-L, -W), (L, -W)), ((-L, W), (-L, -W)), ((L, W), (L, -W)),
            ((-K, W), (-K, -W)), ((K, W), (K, -W)), ((-L, 0), (-K, 0)), ((K, 0), (L, 0))]


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
