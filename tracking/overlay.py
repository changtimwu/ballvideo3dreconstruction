"""QA: draw the tracked players back onto the video.

  uv run python -m tracking.overlay match.mp4 --start 290 --end 320 [--out data/overlay.mp4]

Each player gets a 0.45 m floor circle at the exported position (projected
through the calibrated camera), tagged P0..P3 in tracking_data.json order; raw per-frame estimates are
small dots. Court lines are drawn too, so a camera bump would show. The 2D
ball track (ball_track) is a yellow ring, magenta where interpolated.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, draw_court


def hex_bgr(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def floor_circle(cam, x, y, r=0.45, n=40):
    a = np.linspace(0, 2 * np.pi, n)
    return cam.project(np.c_[x + r * np.cos(a), y + r * np.sin(a), np.zeros(n)]).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--players", default="data/players.npz")
    ap.add_argument("--data", default="tracking_data.json")
    ap.add_argument("--out", default="data/overlay.mp4")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--ball", default="data/ball2d.npz", help="2D ball track to draw, if present")
    a = ap.parse_args()

    cam = Camera.load()
    P = np.load(a.players)
    J = json.load(open(a.data))
    meta = J["players"]
    t_exp = np.array([f["t"] for f in J["frames"]])
    pos = np.array([[p if p else [np.nan, np.nan] for p in f["players"]] for f in J["frames"]], float)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cam.size[0] * a.scale), int(cam.size[1] * a.scale)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out],
                          stdin=subprocess.PIPE)
    raw_t = P["t"]; raw = P["raw"]
    ball = np.load(a.ball) if Path(a.ball).exists() else None
    while True:
        fi = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        t = fi / fps
        ok, img = cap.read()
        if not ok or t > a.end:
            break
        draw_court(img, cam, (255, 255, 255), 1)
        j = int(np.clip(np.searchsorted(raw_t, t), 0, len(raw_t) - 1))
        for k in range(raw.shape[1]):
            if np.isfinite(raw[j, k, 0]):
                p = cam.project(np.array([[*raw[j, k], 0]]))[0].astype(int)
                cv2.circle(img, tuple(p), 5, (0, 0, 0), -1)
        e = int(np.clip(np.searchsorted(t_exp, t), 0, len(t_exp) - 1))
        for k, m in enumerate(meta):
            x, y = pos[e, k]
            if not np.isfinite(x):
                continue
            col = hex_bgr(m["color"])
            ring = floor_circle(cam, x, y)
            cv2.polylines(img, [ring], True, (0, 0, 0), 8, cv2.LINE_AA)   # dark outline keeps light/dark colours visible
            cv2.polylines(img, [ring], True, col, 4, cv2.LINE_AA)
            c = tuple(cam.project(np.array([[x, y, 0]]))[0].astype(int) + [-20, 40])
            cv2.putText(img, f"P{k}", c, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(img, f"P{k}", c, cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)
        if ball is not None:
            k = fi - int(ball["f0"])
            if 0 <= k < len(ball["xy"]) and np.isfinite(ball["xy"][k, 0]):
                bx, by = ball["xy"][k].astype(int)
                cv2.circle(img, (bx, by), 14, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.circle(img, (bx, by), 14, (0, 255, 255) if ball["observed"][k] else (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{t:7.2f}s", (cam.size[0] - 260, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        ff.stdin.write(cv2.resize(img, (W, H)).tobytes())
    ff.stdin.close(); ff.wait()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
