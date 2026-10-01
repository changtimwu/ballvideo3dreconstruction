"""Stage 2b: multi-region appearance per detection.

  uv run python -m tracking.appearance match.mp4 [--det data/detections.npz] [--out data/appearance.npz]

Shirt colour alone can't separate players in similar shirts (the near pair in
match.mp4 both wear light grey/white). This samples the median Lab colour of
several keypoint-anchored regions: hair, torso, shorts and each elbow (one
player wears a black elbow sleeve). Re-reads the video but doesn't re-run the
detector. Regions that aren't visible are NaN.
"""
import argparse
import time

import cv2
import numpy as np

PARTS = ["hair", "torso", "shorts", "l_elbow", "r_elbow"]
THR = 0.4


def _median(lab, mask_poly=None, circle=None):
    """Median Lab inside a polygon or circle, rasterised only over its bounding box."""
    H, W = lab.shape[:2]
    if mask_poly is not None:
        pts = mask_poly.astype(np.int32)
    else:
        (cx, cy), r = circle
        r = max(2, int(r))
        pts = np.array([[cx - r, cy - r], [cx + r, cy + r]], np.int32)
    x0, y0 = np.clip(pts.min(0), 0, [W - 1, H - 1])
    x1, y1 = np.clip(pts.max(0) + 1, 1, [W, H])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.full(3, np.nan)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    if mask_poly is not None:
        cv2.fillConvexPoly(m, pts - [x0, y0], 1)
    else:
        cv2.circle(m, (int(cx) - x0, int(cy) - y0), r, 1, -1)
    px = lab[y0:y1, x0:x1][m.astype(bool)]
    return np.median(px, axis=0) if len(px) >= 12 else np.full(3, np.nan)


def part_colours(lab: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """(len(PARTS), 3) Lab medians for one person; NaN where the region isn't visible."""
    out = np.full((len(PARTS), 3), np.nan, np.float32)
    v = kp[:, 2] > THR
    p = kp[:, :2]
    sh_w = np.linalg.norm(p[5] - p[6]) if v[5] and v[6] else None
    # hair: above the head centre, scaled by ear (or shoulder) spacing
    head = [j for j in (1, 2, 3, 4) if v[j]]
    if head and sh_w:
        c = p[head].mean(0)
        s = max(np.linalg.norm(p[3] - p[4]) if v[3] and v[4] else 0.45 * sh_w, 6)
        out[0] = _median(lab, mask_poly=np.array([[c[0] - 0.45 * s, c[1] - 1.1 * s], [c[0] + 0.45 * s, c[1] - 1.1 * s],
                                                  [c[0] + 0.45 * s, c[1] - 0.45 * s], [c[0] - 0.45 * s, c[1] - 0.45 * s]]))
    if v[[5, 6, 11, 12]].all():
        q = np.array([p[5], p[6], p[12], p[11]])
        out[1] = _median(lab, mask_poly=q.mean(0) + (q - q.mean(0)) * 0.7)
    if v[[11, 12, 13, 14]].all():
        top = np.array([p[11], p[12]])
        bot = top + (np.array([p[13], p[14]]) - top) * 0.45
        q = np.array([top[0], top[1], bot[1], bot[0]])
        out[2] = _median(lab, mask_poly=q.mean(0) + (q - q.mean(0)) * 0.8)
    for k, (s_, e, w) in enumerate(((5, 7, 9), (6, 8, 10))):
        if v[s_] and v[e]:
            r = 0.18 * np.linalg.norm(p[e] - p[s_])
            out[3 + k] = _median(lab, circle=(p[e], r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--det", default="data/detections.npz")
    ap.add_argument("--out", default="data/appearance.npz")
    a = ap.parse_args()

    D = np.load(a.det)
    det_frame, kp = D["frame"], D["kp"]
    order = np.argsort(det_frame, kind="stable")
    parts = np.full((len(det_frame), len(PARTS), 3), np.nan, np.float32)
    cap = cv2.VideoCapture(a.video)
    first, last = int(det_frame[order[0]]), int(det_frame[order[-1]])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    f, k, t0 = first, 0, time.time()
    while f <= last and k < len(order):
        if not cap.grab():
            break
        if det_frame[order[k]] == f:
            _, img = cap.retrieve()
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            while k < len(order) and det_frame[order[k]] == f:
                i = order[k]
                parts[i] = part_colours(lab, kp[i])
                k += 1
            if f % 600 == 0:
                print(f"\r{k}/{len(order)} detections  {(f - first) / max(time.time() - t0, 1e-6):.0f} frames/s",
                      end="", flush=True)
        f += 1
    print()
    np.savez_compressed(a.out, parts=parts, names=np.array(PARTS))
    vis = np.isfinite(parts[..., 0]).mean(0)
    print(f"wrote {a.out}: visible " + ", ".join(f"{n} {v * 100:.0f}%" for n, v in zip(PARTS, vis)))


if __name__ == "__main__":
    main()
