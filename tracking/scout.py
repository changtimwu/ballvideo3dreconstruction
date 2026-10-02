"""Find YouTube pickleball videos and screen them for `tracking analyze`.

  uv run python -m tracking scout --channel <url> [--channel …] [--search "query"] [--url <video>] [--limit 30]
  uv run python -m tracking scout --report                 # ranked list
  uv run python -m tracking scout --review                 # local page: contact sheets + accept/reject
  uv run python -m tracking scout --rescore                # re-apply thresholds to stored measurements
  uv run python -m tracking scout --seed-check             # thresholds vs. hand labels in scout/seed.json
  uv run python -m tracking scout --analyze-top 3 [--include-auto]

A video is suitable when it matches what the pipeline assumes: a fixed camera that
sees the whole court, doubles (4 players), few captions/graphics, little music or
sound effects, and embeddable (the site plays it through the YouTube player).

Stages, cheapest first; a video stops at the first failed gate:
  1. metadata (yt-dlp -J): public, embeddable, not live, 5–60 min, ≥ 720p
  2. storyboard (YouTube's 320×180 preview mosaics, ~10 s apart, ~1 MB): is every
     frame the same fixed view (ORB homography to the median frame)? how often does
     it cut away? how much is covered by graphics or captions?
  3. ~24 frames from a 480p copy (~60 MB per 15 min, deleted afterwards):
     does tracking.calibrate find the court? are there 4 players on it?
  4. audio only (worst-quality stream): share of time with music (sustained
     spectral peaks); the paddle-pop rate is reported but not scored
Every measurement goes to scout/candidates.json (committed) with the score, the
gate results and a decision; videos are never screened twice (--rescreen forces).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "scout" / "candidates.json"
SEED = ROOT / "scout" / "seed.json"
CACHE = ROOT / "work" / "scout"

# ── thresholds (tune with --seed-check, apply with --rescore) ───────────────────
MIN_S, MAX_S, MIN_HEIGHT = 5 * 60, 60 * 60, 720
FIXED_MIN = 0.80          # share of storyboard frames aligned with the median view
CUTS_MAX = 1.0            # cuts per minute
PLAYERS_MIN = 3           # median people standing on the calibrated court
PASS_SCORE = 70
BAD_TITLE = ("highlight", "rules", "how to", "tutorial", "reaction", "drill", "lesson", "tips", "review",
             "unboxing", "#shorts", "top 10", "best shots", "recap")
GOOD_TITLE = ("full game", "full match", "doubles", "rec ", "4.0", "4.5", "5.0", "3.5", "mixed", "men's", "women's")


# ── registry ────────────────────────────────────────────────────────────────────
def load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"videos": {}}


def save_registry(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


# ── stage 0/1: harvest + metadata ───────────────────────────────────────────────
def ytdlp_json(args: list[str]) -> dict:
    out = subprocess.run(["yt-dlp", "-J", *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def harvest(channels, searches, urls, limit) -> list[str]:
    ids = []
    for ch in channels:
        d = ytdlp_json(["--flat-playlist", "--playlist-end", str(limit), ch])
        ids += [e["id"] for e in d.get("entries", []) if e.get("id")]
    for q in searches:
        d = ytdlp_json(["--flat-playlist", f"ytsearch{limit}:{q}"])
        ids += [e["id"] for e in d.get("entries", []) if e.get("id")]
    for u in urls:
        ids.append(ytdlp_json(["--skip-download", "--no-playlist", u])["id"])
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
    return out


def metadata_gate(info: dict) -> tuple[dict, list[str]]:
    title = (info.get("title") or "").lower()
    m = {
        "duration": info.get("duration"), "height": info.get("height"), "fps": info.get("fps"),
        "embeddable": info.get("playable_in_embed"), "availability": info.get("availability"),
        "live": info.get("live_status"),
        "title_prior": sum(k in title for k in GOOD_TITLE) - 2 * sum(k in title for k in BAD_TITLE),
    }
    fails = []
    if m["availability"] not in (None, "public", "unlisted"):
        fails.append(f"not public ({m['availability']})")
    if m["embeddable"] is False:
        fails.append("embedding disabled")
    if m["live"] not in (None, "not_live", "was_live"):
        fails.append(f"live ({m['live']})")
    if not m["duration"] or not MIN_S <= m["duration"] <= MAX_S:
        fails.append(f"duration {m['duration']} s outside {MIN_S}–{MAX_S} s")
    if (m["height"] or 0) < MIN_HEIGHT:
        fails.append(f"{m['height']}p < {MIN_HEIGHT}p")
    return m, fails


# ── stage 2: storyboard probe ───────────────────────────────────────────────────
def storyboard(info: dict, cache: Path):
    """All storyboard tiles as (N, h, w, 3) BGR plus their timestamps."""
    sb = [f for f in info.get("formats", []) if f.get("format_note") == "storyboard" or str(f.get("format_id", "")).startswith("sb")]
    if not sb:
        return None, None
    f = max(sb, key=lambda f: f.get("width") or 0)
    rows, cols, w, h = f["rows"], f["columns"], f["width"], f["height"]
    tiles, times, t0 = [], [], 0.0
    for k, frag in enumerate(f["fragments"]):
        p = cache / f"sb_{k:03d}.jpg"
        if not p.exists():
            with urllib.request.urlopen(frag["url"], timeout=30) as r:
                p.write_bytes(r.read())
        img = cv2.imread(str(p))
        n_here = rows * cols
        dur = frag.get("duration") or n_here / (f.get("fps") or 0.1)
        for i in range(n_here):
            r_, c_ = divmod(i, cols)
            tile = img[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w]
            if tile.shape[:2] != (h, w) or tile.mean() < 3:     # past the end: blank
                continue
            tiles.append(tile); times.append(t0 + (i + 0.5) * dur / n_here)
        t0 += dur
    return np.array(tiles), np.array(times)


def visual_probe(tiles: np.ndarray, times: np.ndarray, cache: Path) -> dict:
    grey = np.array([cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in tiles])
    med = np.median(grey, axis=0).astype(np.uint8)
    h, w = med.shape
    orb = cv2.ORB_create(1000)
    kp0, d0 = orb.detectAndCompute(med, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    aligned = np.zeros(len(grey), bool)
    disp = np.full(len(grey), np.nan)
    # share of each frame that looks like the median view (a cut or camera move changes
    # nearly everything; a player close to the camera only covers part of the frame)
    unchanged = np.array([(cv2.absdiff(cv2.GaussianBlur(g, (5, 5), 0), cv2.GaussianBlur(med, (5, 5), 0)) < 22).mean()
                          for g in grey])
    for i, g in enumerate(grey):
        kp, d = orb.detectAndCompute(g, None)
        ok_h = False
        if d is not None and d0 is not None and len(kp) >= 20:
            mt = bf.match(d0, d)
            if len(mt) >= 20:
                src = np.float32([kp0[m.queryIdx].pt for m in mt]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt for m in mt]).reshape(-1, 1, 2)
                Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if Hm is not None and inl.sum() >= 15:
                    disp[i] = float(np.abs(cv2.perspectiveTransform(corners, Hm) - corners).reshape(-1, 2).max())
                    ok_h = True
        if ok_h:
            # a measured move is a move, unless the frame is still mostly the median view
            aligned[i] = disp[i] <= 0.015 * w or unchanged[i] > 0.75
        else:
            # too few background features (dark venue, a player filling the frame)
            aligned[i] = unchanged[i] > 0.55
    # cuts: big jumps in colour histogram between consecutive frames
    hists = [cv2.normalize(cv2.calcHist([cv2.cvtColor(t, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]), None).ravel()
             for t in tiles]
    jumps = sum(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA) > 0.35 for a, b in zip(hists[:-1], hists[1:]))
    minutes = max((times[-1] - times[0]) / 60, 1e-6)
    # graphics / captions on frames that are the fixed view
    graphic, caption = 0, 0
    for i in np.flatnonzero(aligned):
        diff = cv2.absdiff(grey[i], med) > 45
        diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if diff.mean() > 0.22:
            graphic += 1
        # caption: wide, short, bright component (text line) that isn't in the median frame
        hsv = cv2.cvtColor(tiles[i], cv2.COLOR_BGR2HSV)
        bright = ((hsv[..., 2] > 200) & (hsv[..., 1] < 50) & (diff > 0)).astype(np.uint8)
        bright = cv2.dilate(bright, np.ones((3, 7), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(bright)
        if any(st[j, 2] > 0.18 * w and st[j, 3] < 0.12 * h and st[j, 2] > 3.5 * st[j, 3] for j in range(1, n)):
            caption += 1
    na = max(aligned.sum(), 1)
    # contact sheet for the review page
    pick = np.linspace(0, len(tiles) - 1, min(24, len(tiles))).astype(int)
    sheet = [cv2.resize(tiles[i], (160, 90)) for i in pick]
    for k, i in enumerate(pick):
        cv2.rectangle(sheet[k], (0, 0), (159, 89), (0, 200, 0) if aligned[i] else (0, 0, 255), 2)
    while len(sheet) % 6:
        sheet.append(np.zeros_like(sheet[0]))
    cv2.imwrite(str(cache / "sheet.jpg"), np.vstack([np.hstack(sheet[r:r + 6]) for r in range(0, len(sheet), 6)]))
    cv2.imwrite(str(cache / "median_sb.jpg"), cv2.cvtColor(med, cv2.COLOR_GRAY2BGR))
    return {
        "frames": int(len(tiles)),
        "fixed_share": round(float(aligned.mean()), 3),
        "cuts_per_min": round(float(jumps / minutes), 2),
        "graphic_share": round(graphic / na, 3),
        "caption_share": round(caption / na, 3),
        "aligned_times": [round(float(t), 1) for t in times[aligned]],
    }


# ── stage 3: frame probe (480p) ──────────────────────────────────────────────
def probe_frames(vid: str, times, cache: Path, height: int = 480):
    """Frames at `times` from a low-resolution copy of the video.

    Random access into YouTube's full-resolution streams isn't possible: the signed
    URLs refuse byte ranges beyond the first ~10–30 MB (ffmpeg seeking gets 403), and
    HLS isn't offered to the clients yt-dlp can use. A 480p video-only stream is
    ~60 MB per 15 min, and auto-calibration on it agrees with 1080p to ~10 cm. The
    file is deleted after probing."""
    path = cache / "probe.mp4"
    if not path.exists():
        subprocess.run(["yt-dlp", "-q", "--no-playlist", "-f",
                        f"bv*[height<={height}][vcodec^=avc1]/bv*[height<={height}]/b[height<={height}]",
                        "-o", str(path), f"https://www.youtube.com/watch?v={vid}"], check=True, capture_output=True)
    cap = cv2.VideoCapture(str(path))
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    path.unlink(missing_ok=True)
    return frames


_YOLO = None


def fullres_probe(vid: str, aligned_times: list[float], cache: Path, n: int = 24) -> dict:
    from .calibrate import auto_calibrate
    from .camera import HALF_L, HALF_W, draw_court, line_score
    from .detect import foot_pixel
    global _YOLO
    ts = np.array(aligned_times)
    ts = ts[np.linspace(0, len(ts) - 1, min(n, len(ts))).astype(int)] if len(ts) else np.array([])
    frames = probe_frames(vid, ts, cache)
    if len(frames) < 5:
        return {"calibrated": False, "calib_note": f"only {len(frames)} frames could be read"}
    H = max(f.shape[0] for f in frames)
    frames = [f for f in frames if f.shape[0] == H]
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cam, note = auto_calibrate(bg)
    out = {"calibrated": cam is not None, "calib_note": note, "frames_read": len(frames)}
    if cam is None:
        cv2.imwrite(str(cache / "median.jpg"), bg)
        return out
    cv2.imwrite(str(cache / "calib.jpg"), draw_court(bg.copy(), cam, thickness=1))
    if _YOLO is None:
        import torch
        from ultralytics import YOLO
        _YOLO = (YOLO("yolo11m-pose.pt"), "mps" if torch.backends.mps.is_available() else "cpu")
    model, device = _YOLO
    counts, visible = [], []
    for f in frames:
        visible.append(line_score(f, cam))
        r = model.predict(f, imgsz=960, conf=0.3, classes=[0], device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            counts.append(0); continue
        feet = np.array([foot_pixel(b, k) for b, k in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy())])
        xy = cam.to_plane(feet, 0.0)[:, :2]
        counts.append(int(((np.abs(xy[:, 0]) < HALF_L + 2) & (np.abs(xy[:, 1]) < HALF_W + 1.5)).sum()))
    out.update({
        "players_median": float(np.median(counts)),
        "court_visible": round(float(np.mean(np.array(visible) > 0.6)), 3),
        "camera_height_m": round(float(cam.center[2]), 2),
        "vfov_deg": round(cam.vfov_deg, 1),
    })
    return out


# ── stage 4: audio probe ────────────────────────────────────────────────────────
def audio_probe(vid: str, cache: Path) -> dict:
    path = cache / "audio"
    if not list(cache.glob("audio.*")):
        subprocess.run(["yt-dlp", "-f", "wa", "-o", str(path) + ".%(ext)s", "--no-playlist",
                        f"https://www.youtube.com/watch?v={vid}"], check=True, capture_output=True)
    src = next(cache.glob("audio.*"))
    out = audio_metrics(src)
    src.unlink(missing_ok=True)
    return out


def audio_metrics(src: Path, sr: int = 16000) -> dict:
    """Music share and paddle-pop rate of an audio (or video) file.

    Music and jingles hold narrow spectral peaks (notes) steady for hundreds of ms;
    rally audio is broadband noise plus pops, and speech pitch keeps moving. So the
    music score of a 1 s window is the share of its 0–4 kHz energy in spectral peaks
    that persist ≥ ~0.35 s. Checked on the reference video: 2.8% of windows > 0.25;
    with a synthetic chord bed mixed in at −12 / −6 / 0 dB: 16% / 47% / 94%.
    (Harmonic–percussive separation was tried first and failed: steady hall noise
    counts as "harmonic".)
    """
    from scipy.ndimage import maximum_filter, median_filter
    from scipy.signal import find_peaks, stft
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    if len(x) < sr * 30:
        return {"audio": False}
    # tonal persistence (hop 32 ms)
    _, _, Z = stft(x, sr, nperseg=2048, noverlap=1536)
    S = np.abs(Z).astype(np.float32)[:512]
    D = 20 * np.log10(S + 1e-7)
    peak = (D - median_filter(D, size=(15, 1)) > 10) & (D == maximum_filter(D, size=(3, 1)))
    peak = maximum_filter(peak, size=(3, 1))                      # ±1 bin of pitch wobble
    sustained = median_filter(peak.astype(np.uint8), size=(1, 11)) > 0
    E = S ** 2
    win = 31                                                       # ~1 s
    n = S.shape[1] // win
    tonal = (E * sustained)[:, :n * win].reshape(S.shape[0], n, win).sum((0, 2)) / (
        E[:, :n * win].reshape(S.shape[0], n, win).sum((0, 2)) + 1e-12)
    # paddle pops (informational only): 1.5–6 kHz band jumping ≥ 7 dB above its 1 s
    # running median and falling back within ~50 ms. On the reference video these
    # line up with only ~20% of the tracked hits, so they don't enter the score.
    _, _, Z2 = stft(x, sr, nperseg=512, noverlap=256)
    Bd = 20 * np.log10(np.abs(Z2[int(1500 / (sr / 512)):int(6000 / (sr / 512))]).sum(0) + 1e-7)
    rise = Bd - median_filter(Bd, size=int(sr / 256))
    peaks, _ = find_peaks(rise, height=7.0, distance=int(0.15 * sr / 256))
    peaks = [p for p in peaks if p + 3 < len(Bd) and Bd[p] - Bd[p + 3] > 3.5]
    minutes = len(x) / sr / 60
    return {"audio": True, "music_share": round(float(np.mean(tonal > 0.25)), 3),
            "pops_per_min": round(len(peaks) / minutes, 1)}


# ── scoring ─────────────────────────────────────────────────────────────────────
def score(rec: dict) -> dict:
    m = rec.get("metrics", {})
    gates, reasons = {}, list(rec.get("meta_fails", []))
    gates["metadata"] = not rec.get("meta_fails")
    v = m.get("visual")
    if v:
        gates["fixed"] = v["fixed_share"] >= FIXED_MIN and v["cuts_per_min"] <= CUTS_MAX
        if not gates["fixed"]:
            reasons.append(f"not a fixed camera (fixed {v['fixed_share']:.0%}, {v['cuts_per_min']} cuts/min)")
    f = m.get("fullres")
    if f:
        gates["calibrated"] = bool(f.get("calibrated"))
        if not gates["calibrated"]:
            reasons.append(f"court not calibratable ({f.get('calib_note')})")
        if f.get("calibrated"):
            gates["players"] = f.get("players_median", 0) >= PLAYERS_MIN
            if not gates["players"]:
                reasons.append(f"{f.get('players_median')} players on court (want 4)")
    a = m.get("audio") or {}
    s = None
    if v and f and f.get("calibrated"):
        s = 100 * (0.25 * v["fixed_share"]
                   + 0.15 * (1 - min(v["graphic_share"] * 3, 1))
                   + 0.15 * (1 - min(v["caption_share"] * 3, 1))
                   + 0.15 * f.get("court_visible", 0)
                   + 0.20 * (1 - min(a.get("music_share", 0.25) * 4, 1))
                   + 0.10 * min((m.get("meta", {}).get("height") or 0) / 1080, 1))
        s = round(s, 1)
    complete = all(k in gates for k in ("metadata", "fixed", "calibrated", "players"))
    if not all(gates.values()):
        decision = "auto-reject"
    elif complete and s is not None and s >= PASS_SCORE:
        decision = "auto-pass"
    elif complete:
        decision = "review"
        reasons.append(f"score {s} < {PASS_SCORE}")
    else:
        decision = "incomplete"
        if rec.get("blocked"):
            reasons.append(rec["blocked"])
    return {"gates": gates, "score": s, "decision": decision, "reasons": reasons}


# ── screening ───────────────────────────────────────────────────────────────────
def screen(vid: str, reg: dict, skip_audio: bool = False) -> dict:
    cache = CACHE / vid
    cache.mkdir(parents=True, exist_ok=True)
    info = ytdlp_json(["--skip-download", "--no-playlist", f"https://www.youtube.com/watch?v={vid}"])
    rec = {"id": vid, "url": info.get("webpage_url"), "title": info.get("title"),
           "channel": info.get("channel") or info.get("uploader"), "channel_url": info.get("channel_url"),
           "screened_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "metrics": {}}
    meta, fails = metadata_gate(info)
    rec["metrics"]["meta"] = meta
    rec["meta_fails"] = fails

    def done():
        rec.update(score(rec))
        prev = reg["videos"].get(vid, {})
        if prev.get("human"):
            rec["human"] = prev["human"]
        reg["videos"][vid] = rec
        save_registry(reg)
        return rec

    if fails:
        return done()
    tiles, times = storyboard(info, cache)
    if tiles is None or len(tiles) < 10:
        rec["meta_fails"] = fails + ["no storyboard"]
        return done()
    rec["metrics"]["visual"] = visual_probe(tiles, times, cache)
    if not score(rec)["gates"].get("fixed"):
        return done()
    try:
        rec["metrics"]["fullres"] = fullres_probe(vid, rec["metrics"]["visual"]["aligned_times"], cache)
    except subprocess.CalledProcessError as e:
        # media downloads can be refused (YouTube rate-limits bursts); retry later
        rec["blocked"] = f"frame probe download failed: {(e.stderr or b'')[-200:].decode(errors='ignore').strip() or e}"
        return done()
    g = score(rec)["gates"]
    if not (g.get("calibrated") and g.get("players")):
        return done()
    if not skip_audio:
        try:
            rec["metrics"]["audio"] = audio_probe(vid, cache)
        except subprocess.CalledProcessError as e:
            rec["blocked"] = f"audio download failed: {(e.stderr or b'')[-200:].decode(errors='ignore').strip() or e}"
    return done()


# ── reporting / review ──────────────────────────────────────────────────────────
ORDER = {"accepted": 0, "auto-pass": 1, "review": 2, "incomplete": 3, "auto-reject": 4, "rejected": 5}


def effective(rec):
    return rec.get("human") or rec.get("decision")


def ranked(reg):
    return sorted(reg["videos"].values(), key=lambda r: (ORDER.get(effective(r), 9), -(r.get("score") or 0)))


def report(reg) -> None:
    for r in ranked(reg):
        v = r["metrics"].get("visual", {}); f = r["metrics"].get("fullres", {}); a = r["metrics"].get("audio", {})
        print(f"{effective(r):11s} {str(r.get('score') or '–'):>5}  {r['id']}  {(r.get('title') or '')[:60]:60s}  "
              f"fixed {v.get('fixed_share', '–')}  cuts {v.get('cuts_per_min', '–')}  "
              f"players {f.get('players_median', '–')}  music {a.get('music_share', '–')}  "
              f"{'; '.join(r.get('reasons', []))}")


def review_page(reg) -> str:
    rows = []
    for r in ranked(reg):
        vid = html.escape(r["id"])
        v = r["metrics"].get("visual", {}); f = r["metrics"].get("fullres", {}); a = r["metrics"].get("audio", {})
        img = lambda name: f'<img src="/img/{vid}/{name}" loading="lazy">' if (CACHE / r["id"] / name).exists() else ""
        rows.append(f"""
<section class="c {html.escape(effective(r))}"><h2><a href="{html.escape(r.get('url') or '')}" target="_blank">{html.escape(r.get('title') or vid)}</a></h2>
<p class="m">{html.escape(r.get('channel') or '')} · <b>{html.escape(effective(r))}</b> · score {r.get('score') or '–'} ·
fixed {v.get('fixed_share', '–')} · cuts/min {v.get('cuts_per_min', '–')} · graphics {v.get('graphic_share', '–')} ·
captions {v.get('caption_share', '–')} · players {f.get('players_median', '–')} · court visible {f.get('court_visible', '–')} ·
music {a.get('music_share', '–')} · pops/min {a.get('pops_per_min', '–')}</p>
<p class="r">{html.escape('; '.join(r.get('reasons', [])))}</p>
<div class="i">{img('sheet.jpg')}{img('calib.jpg') or img('median.jpg')}</div>
<p><button onclick="d('{vid}','accepted')">accept</button> <button onclick="d('{vid}','rejected')">reject</button>
<button onclick="d('{vid}','')">clear</button></p></section>""")
    return f"""<!doctype html><meta charset="utf-8"><title>scout review</title>
<style>body{{font:14px system-ui;background:#0b0f0e;color:#e7efe9;max-width:1500px;margin:20px auto;padding:0 16px}}
a{{color:#d9f24a}}.c{{border:1px solid #1f2d29;border-radius:10px;padding:12px;margin:14px 0}}
.c.auto-reject,.c.rejected{{opacity:.55}}h2{{font-size:16px;margin:0 0 4px}}.m,.r{{color:#8aa39a;margin:4px 0}}
.r{{color:#f59e0b}}.i img{{max-width:49%;margin-right:1%;vertical-align:top}}button{{padding:4px 10px}}</style>
<h1>Scout review — {len(reg['videos'])} videos</h1><p>Green frames = the fixed view, red = cut away / moved.</p>
{''.join(rows)}
<script>function d(id,v){{fetch('/decide',{{method:'POST',body:JSON.stringify({{id,decision:v}})}}).then(()=>location.reload())}}</script>"""


def serve_review(port: int = 8790) -> None:
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            if self.path.startswith("/img/"):
                _, _, vid, name = self.path.split("/", 3)
                p = CACHE / vid / name
                if p.exists() and p.suffix == ".jpg":
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers()
                    self.wfile.write(p.read_bytes()); return
                self.send_response(404); self.end_headers(); return
            body = review_page(load_registry()).encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                reg = load_registry()
                rec = reg["videos"].get(req["id"])
                if rec is not None:
                    if req["decision"]:
                        rec["human"] = req["decision"]
                    else:
                        rec.pop("human", None)
                    save_registry(reg)
            self.send_response(204); self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print(f"review page: {url}  (Ctrl-C to stop)")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def seed_check(reg) -> None:
    if not SEED.exists():
        print(f"no {SEED.relative_to(ROOT)}"); return
    seeds = json.loads(SEED.read_text())["videos"]
    ok = pending = 0
    for vid, lab in seeds.items():
        rec = reg["videos"].get(vid)
        if not rec:
            print(f"  {vid}: not screened yet (run scout --url)"); continue
        got = score(rec)["decision"]
        if got == "incomplete" and lab["label"] == "pass":
            print(f"  …   {vid} label={lab['label']:6s} got={got:11s} (probes pending; passed every stage so far)")
            pending += 1
            continue
        good = (lab["label"] == "pass") == (got in ("auto-pass", "review"))
        ok += good
        print(f"  {'ok ' if good else 'BAD'} {vid} label={lab['label']:6s} got={got:11s} {lab.get('note', '')}")
    print(f"{ok}/{len(seeds) - pending} seed videos classified as labelled ({pending} pending)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tracking scout")
    ap.add_argument("--channel", action="append", default=[], help="channel or playlist URL")
    ap.add_argument("--search", action="append", default=[], help="YouTube search query")
    ap.add_argument("--url", action="append", default=[], help="a single video")
    ap.add_argument("--limit", type=int, default=30, help="videos per channel/search")
    ap.add_argument("--rescreen", action="store_true", help="screen videos already in the registry again")
    ap.add_argument("--retry-incomplete", action="store_true", help="re-screen videos whose probes didn't finish")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--seed-check", action="store_true")
    ap.add_argument("--analyze-top", type=int, default=0)
    ap.add_argument("--include-auto", action="store_true", help="--analyze-top may take auto-pass videos")
    a = ap.parse_args(argv)
    reg = load_registry()

    if a.channel or a.search or a.url or a.retry_incomplete:
        ids = harvest(a.channel, a.search, a.url, a.limit) if (a.channel or a.search or a.url) else []
        if a.retry_incomplete:
            ids += [v for v, r in reg["videos"].items() if r.get("decision") in ("incomplete", "error")]
        analysed = {p.name for p in (ROOT / "videos").glob("*/")}
        retry = {v for v, r in reg["videos"].items() if a.retry_incomplete and r.get("decision") in ("incomplete", "error")}
        todo = [i for i in dict.fromkeys(ids) if a.rescreen or i in retry or (i not in reg["videos"] and i not in analysed)]
        print(f"{len(ids)} found, {len(todo)} to screen")
        for k, vid in enumerate(todo, 1):
            try:
                r = screen(vid, reg, skip_audio=a.no_audio)
                print(f"[{k}/{len(todo)}] {r['decision']:11s} {r.get('score') or '–':>5} {vid} {(r.get('title') or '')[:55]}"
                      f"{'  — ' + '; '.join(r['reasons']) if r['reasons'] else ''}", flush=True)
            except Exception as e:      # one bad video shouldn't stop a batch
                print(f"[{k}/{len(todo)}] error      {vid}: {e}", flush=True)
                reg["videos"][vid] = {"id": vid, "decision": "error", "reasons": [str(e)[:200]], "metrics": {}}
                save_registry(reg)
    if a.rescore:
        for rec in reg["videos"].values():
            if rec.get("metrics"):
                rec.update(score(rec))
        save_registry(reg)
        print(f"rescored {len(reg['videos'])} videos")
    if a.seed_check:
        seed_check(reg)
    if a.report:
        report(reg)
    if a.review:
        serve_review()
    if a.analyze_top:
        allowed = ("accepted", "auto-pass") if a.include_auto else ("accepted",)
        analysed = {p.name for p in (ROOT / "videos").glob("*/")}
        picks = [r for r in ranked(reg) if effective(r) in allowed and r["id"] not in analysed][: a.analyze_top]
        if not picks:
            print("nothing to analyse: accept candidates in --review first" + ("" if a.include_auto else " (or pass --include-auto)"))
        for r in picks:
            print(f"→ analyse {r['id']} {r.get('title')}")
            subprocess.run([sys.executable, "-m", "tracking", "analyze", r["url"]], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
