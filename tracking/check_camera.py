"""Stage 1: confirm the camera never moves, and write tracking/camera.json.

  uv run python -m tracking.check_camera match.mp4 [--every 30] [--ref 900]

Solves the camera at the reference time, then re-detects the court corners
every --every seconds and reports how far they sit from where the reference
camera projects them. Sub-few-pixel drift means one camera serves the whole video.
"""
import argparse
import json

import cv2
import numpy as np

from .camera import CAMERA_JSON, CORNERS, solve, draw_court


def read_frame(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    return img if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=30)
    ap.add_argument("--ref", type=float, default=900)
    ap.add_argument("--overlay", default="data/camera_overlay.jpg")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    ref_img = read_frame(cap, a.ref)
    cam, err, _ = solve(ref_img)
    obj = np.array([v[1] for v in CORNERS.values()], np.float64)
    ref_px = cam.project(obj)
    print(f"reference t={a.ref:.0f}s  reprojection {err:.2f}px  vfov {cam.vfov_deg:.2f}°  centre {cam.center.round(3)}")

    drifts = []
    for t in np.arange(5, dur - 1, a.every):
        img = read_frame(cap, t)
        if img is None:
            continue
        try:
            _, e, pts = solve(img)
        except ValueError as ex:
            print(f"t={t:6.0f}s  skipped ({ex})")
            continue
        px = np.array([pts[k] for k in CORNERS])
        d = np.linalg.norm(px - ref_px, axis=1)
        drifts.append(np.median(d))
        flag = "  <-- check" if np.median(d) > 4 else ""
        print(f"t={t:6.0f}s  corner drift median {np.median(d):5.2f}px  max {d.max():5.2f}px  (fit {e:.2f}px){flag}")

    print(f"\nmedian drift over video: {np.median(drifts):.2f}px, worst sample {np.max(drifts):.2f}px")
    CAMERA_JSON.write_text(json.dumps(cam.to_json(), indent=2) + "\n")
    print(f"wrote {CAMERA_JSON}")
    if a.overlay:
        import pathlib
        pathlib.Path(a.overlay).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.overlay, cv2.resize(draw_court(ref_img.copy(), cam), (960, 540)))


if __name__ == "__main__":
    main()
