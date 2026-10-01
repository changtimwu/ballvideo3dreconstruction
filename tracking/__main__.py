"""Analyse a YouTube pickleball video end to end and publish its tracking data.

  uv run python -m tracking analyze <youtube-url> [--no-push] [--no-commit] [--force] [--no-gui]

1. downloads the video with yt-dlp into work/<id>/ (gitignored),
2. runs the pipeline stages, skipping any whose output already exists (so an
   interrupted run resumes; --force reruns everything):
     calibrate → detect → appearance → track → ball_detect → ball_track → ball3d → export
3. writes videos/<id>/tracking_data.json, camera.json and meta.json,
4. commits them and pushes; GitHub CI then rebuilds the site's video list.

Assumes a fixed camera that sees the whole court (doubles, 4 players). The first
stage tries automatic court calibration and opens a click-the-court-points window
if that fails (--no-gui makes it fail instead).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORMAT = "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/b[ext=mp4][height<=1080]/bv*[height<=1080]+ba/b"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def stage(module: str, *args) -> list[str]:
    return [sys.executable, "-m", f"tracking.{module}", *args]


def video_info(url: str) -> dict:
    out = subprocess.run(["yt-dlp", "-J", "--no-playlist", url], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def stats(data: dict) -> dict:
    frames = data["frames"]
    n = len(frames)
    all4 = sum(all(p is not None for p in f["players"]) for f in frames)
    ball = sum(f["ball"] is not None for f in frames)
    ev = data.get("events", [])
    return {
        "frames": n,
        "players_all_visible": round(all4 / n, 3),
        "ball_visible": round(ball / n, 3),
        "hits": sum(e["type"] == "hit" for e in ev),
        "bounces": sum(e["type"] == "bounce" for e in ev),
    }


def analyze(a) -> None:
    t_start = time.time()
    print(f"→ reading video info: {a.url}")
    info = video_info(a.url)
    vid = info["id"]
    work = ROOT / "work" / vid
    out = ROOT / "videos" / vid
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f"  {info.get('title')} ({vid}), {info.get('duration')} s")

    video = work / "video.mp4"
    if not video.exists():
        print("→ downloading")
        run(["yt-dlp", "-f", FORMAT, "--merge-output-format", "mp4", "--no-playlist", "-o", video, a.url])

    cam = work / "camera.json"
    W = lambda name: work / name
    steps = [
        ("calibrate", [cam], stage("calibrate", video, "--out", cam, "--overlay", W("calib.jpg"),
                                   *(["--no-gui"] if a.no_gui else []))),
        ("detect", [W("detections.npz")], stage("detect", video, "--camera", cam, "--out", W("detections.npz"))),
        ("appearance", [W("appearance.npz")], stage("appearance", video, "--det", W("detections.npz"),
                                                    "--out", W("appearance.npz"))),
        ("track", [W("players.npz")], stage("track", "--camera", cam, "--det", W("detections.npz"),
                                            "--app", W("appearance.npz"), "--out", W("players.npz"))),
        ("ball_detect", [W("ball_candidates.npz")], stage("ball_detect", video, "--camera", cam,
                                                          "--det", W("detections.npz"), "--out", W("ball_candidates.npz"))),
        ("ball_track", [W("ball2d.npz")], stage("ball_track", "--camera", cam, "--cand", W("ball_candidates.npz"),
                                                "--out", W("ball2d.npz"))),
        ("ball3d", [W("ball3d.npz")], stage("ball3d", "--camera", cam, "--ball", W("ball2d.npz"),
                                            "--players", W("players.npz"), "--out", W("ball3d.npz"))),
    ]
    for name, outputs, cmd in steps:
        if not a.force and all(o.exists() for o in outputs):
            print(f"→ {name}: done already, skipping")
            continue
        print(f"→ {name}")
        t0 = time.time()
        run(cmd)
        print(f"  {name} took {time.time() - t0:.0f} s")

    print("→ export")
    run(stage("export", "--camera", cam, "--players", W("players.npz"), "--ball3d", W("ball3d.npz"),
              "--out", out / "tracking_data.json"))
    shutil.copy(cam, out / "camera.json")
    data = json.loads((out / "tracking_data.json").read_text())
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {
        "id": vid,
        "url": info.get("webpage_url") or a.url,
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "video_fps": info.get("fps"),
        "analysed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "pipeline": sha,
        "players": [p["label"] for p in data["players"]],
        "stats": stats(data),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"→ wrote {out.relative_to(ROOT)}/ ({json.dumps(meta['stats'])}) in {(time.time() - t_start) / 60:.1f} min")

    if a.no_commit:
        return
    run(["git", "add", out.relative_to(ROOT)])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("→ nothing changed; not committing")
        return
    run(["git", "commit", "-m", f"Add tracking for {meta['title']} ({vid})"])
    if not a.no_push:
        run(["git", "push"])
        print("→ pushed; GitHub CI will rebuild the site")


def main():
    ap = argparse.ArgumentParser(prog="python -m tracking")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze", help="analyse a YouTube video and publish its tracking data")
    p.add_argument("url")
    p.add_argument("--force", action="store_true", help="rerun every stage")
    p.add_argument("--no-gui", action="store_true", help="never open the calibration click window")
    p.add_argument("--no-commit", action="store_true")
    p.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    if a.cmd == "analyze":
        analyze(a)


if __name__ == "__main__":
    main()
