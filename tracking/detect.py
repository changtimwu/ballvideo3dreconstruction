"""Stage 2: person detection + 2D keypoints on the match court.

  uv run python -m tracking.detect match.mp4 --start 290 --end 320

Runs a YOLO pose model on every --step-th frame, keeps people whose feet land on
or near this court (dropping neighbouring courts and spectators), and samples a
torso colour for identity matching. Each frame also gets a court-visibility
score (see camera.line_score) so edited-in overlays — match.mp4 has a
full-screen ad at ~312.5–316.4 s — can be treated as missing data.
Writes data/detections.npz.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .camera import Camera, HALF_L, HALF_W, line_score

MARGIN_X, MARGIN_Y = 2.6, 1.8      # metres beyond baselines / sidelines that still count as "on court"
SCOREBOARD = (0, 0, 300, 165)       # x0, y0, x1, y1 at 1080p


def foot_pixel(box: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Best guess of the floor contact pixel: ankle midpoint, else bbox bottom centre."""
    ank = kp[[15, 16]]
    good = ank[:, 2] > 0.4
    if good.any():
        p = ank[good, :2].mean(0)
        return np.array([p[0], max(p[1], box[3] - 6)]) if good.all() else p
    return np.array([(box[0] + box[2]) / 2, box[3]])


def torso_colour(img_lab: np.ndarray, box: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Median Lab colour of the shirt (shoulder→hip quad, shrunk), or of the bbox's upper half."""
    sh, hp = kp[[5, 6]], kp[[11, 12]]
    if (sh[:, 2] > 0.4).all() and (hp[:, 2] > 0.4).all():
        quad = np.array([sh[0, :2], sh[1, :2], hp[1, :2], hp[0, :2]])
        c = quad.mean(0)
        quad = c + (quad - c) * 0.7
    else:
        x0, y0, x1, y1 = box
        h = y1 - y0
        quad = np.array([[x0 + 0.3 * (x1 - x0), y0 + 0.2 * h], [x1 - 0.3 * (x1 - x0), y0 + 0.2 * h],
                         [x1 - 0.3 * (x1 - x0), y0 + 0.45 * h], [x0 + 0.3 * (x1 - x0), y0 + 0.45 * h]])
    mask = np.zeros(img_lab.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 1)
    px = img_lab[mask.astype(bool)]
    if len(px) < 20:
        return np.full(3, np.nan, np.float32)
    return np.median(px, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--step", type=int, default=2, help="process every Nth frame (2 → ~30 fps)")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="data/detections.npz")
    a = ap.parse_args()

    cam = Camera.load()
    model = YOLO(a.model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(round(a.start * fps))
    f1 = min(n_total, int(round(a.end * fps)) if a.end else n_total)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    rows = {k: [] for k in ("frame", "box", "conf", "kp", "foot_px", "foot", "colour")}
    frames_done, court_scores = [], []
    batch, idxs = [], []
    t_start = time.time()

    def flush():
        if not batch:
            return
        results = model.predict(batch, imgsz=a.imgsz, conf=0.25, classes=[0], device=device, verbose=False)
        for fi, img, res in zip(idxs, batch, results):
            frames_done.append(fi)
            court_scores.append(line_score(img, cam))
            if res.boxes is None or len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            kps = res.keypoints.data.cpu().numpy()      # (N,17,3) x, y, conf
            feet = np.array([foot_pixel(b, k) for b, k in zip(boxes, kps)])
            floor = cam.to_plane(feet, 0.0)[:, :2]
            keep = (np.abs(floor[:, 0]) < HALF_L + MARGIN_X) & (np.abs(floor[:, 1]) < HALF_W + MARGIN_Y)
            sx0, sy0, sx1, sy1 = SCOREBOARD
            keep &= ~((boxes[:, 2] < sx1) & (boxes[:, 3] < sy1))
            if not keep.any():
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            for j in np.flatnonzero(keep):
                rows["frame"].append(fi); rows["box"].append(boxes[j]); rows["conf"].append(confs[j])
                rows["kp"].append(kps[j]); rows["foot_px"].append(feet[j]); rows["foot"].append(floor[j])
                rows["colour"].append(torso_colour(lab, boxes[j], kps[j]))
        batch.clear(); idxs.clear()

    fi = f0
    while fi < f1:
        ok = cap.grab()
        if not ok:
            break
        if (fi - f0) % a.step == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            batch.append(img); idxs.append(fi)
            if len(batch) == a.batch:
                flush()
                done = fi - f0
                el = time.time() - t_start
                rate = len(frames_done) / el
                eta = (f1 - fi) / a.step / max(rate, 1e-6)
                print(f"\rframe {fi}/{f1}  {rate:5.1f} fps  eta {eta / 60:5.1f} min", end="", flush=True)
        fi += 1
    flush()
    print()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, fps=fps, step=a.step, frames=np.array(frames_done), court_score=np.array(court_scores, np.float32),
        **{k: np.array(v) for k, v in rows.items()},
    )
    per = np.bincount(np.array(rows["frame"]) - f0)[:: a.step] if rows["frame"] else np.array([0])
    print(f"wrote {out}: {len(rows['frame'])} detections over {len(frames_done)} frames "
          f"(people per frame: {np.bincount(per, minlength=7)[:7].tolist()} for 0..6)")


if __name__ == "__main__":
    main()
