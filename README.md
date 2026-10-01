# 匹克球 3D 重建 · Pickleball 3D Reconstruction

An interactive 3D playback viewer for a pickleball doubles match. It plays a Three.js
reconstruction of the court, the players and the ball next to the source video, locked to
the video's clock.

- `index.html`: the whole viewer in one file (Three.js and Tailwind come from CDNs).
- `tracking/`: the Python pipeline that extracts tracking from the video (§4).
- `tracking_data.json`: its output, which the viewer loads.
- `prompts.md`: the original spec.

> **Status (milestone 1 of [#1](https://github.com/changtimwu/ballvideo3dreconstruction/issues/1)):**
> `tracking_data.json` holds **real player floor positions, the 3D ball, and hit/bounce
> events** for the whole match (milestones 1–3). Body poses are still procedural
> (milestone 4). Without
> `tracking_data.json`, the viewer falls back to a generated demo dataset. The badge in the
> top-right corner shows which data is loaded.

## 1. Get the match video (`match.mp4`)

The video is ~365 MB and isn't in git (`*.mp4` is in `.gitignore`), so download it again
from YouTube:

- **Source:** "Advanced Senior Pickleball in Orlando": https://www.youtube.com/watch?v=cRRLvhKbPXM
- **Length:** 935 s (15:35)

**Requirements:** [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) and `ffmpeg` (needed to merge
the separate video and audio streams).

```bash
brew install yt-dlp ffmpeg        # macOS; or: pipx install yt-dlp
```

**Download.** Run this from the repository root:

```bash
yt-dlp -f "299+140" --merge-output-format mp4 -o match.mp4 \
  "https://www.youtube.com/watch?v=cRRLvhKbPXM"
```

- `299` = 1920×1080, 60 fps, H.264 video only
- `140` = AAC 129 kbps audio only
- The output must be named `match.mp4` and sit next to `index.html`.

**Check.** You should see `h264 1920x1080 60000/1001`, `aac`, and a duration of about `935.16`:

```bash
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,r_frame_rate \
  -of compact match.mp4
```

**Why these exact formats:**
- **Use H.264 (`299`).** Don't swap in the smaller VP9/AV1 streams (`303`/`398`). They play in
  Chrome, but the camera calibration and frame stepping assume this exact 1080p60 encode.
- **Use `-f "299+140"`, not `-f best`.** `best` picks a ≤720p file that already contains
  audio, and the calibration is in 1080p pixel space.
- **If YouTube renumbers its formats,** list them with `yt-dlp -F <url>` and pick the 1080p60
  `avc1` video plus the `m4a` audio.

## 2. Run the viewer

Open `index.html` in Chrome. That's enough for the video and the demo data.

**Hosted version:** https://changtimwu.github.io/ballvideo3dreconstruction/

The 365 MB video can't go on GitHub Pages, which caps files at 100 MB. After downloading
`match.mp4` (§1), click **載入影片** in the video panel, or drag the file onto the page. The
file stays on your machine; nothing is uploaded.

To have `tracking_data.json` picked up automatically, serve the folder over HTTP. Chrome
blocks `fetch()` from `file://`. The server must support HTTP Range requests, or video
seeking won't work, so don't use `python3 -m http.server`.

```bash
npx http-server -p 8080 .         # then open http://localhost:8080
```

You can also click **載入 JSON**, or drag a `.json` file onto the page.

### Controls

| Control | Action |
| --- | --- |
| Space | Play / pause |
| ← / → | Step one video frame |
| `1/4x` `1/2x` `1x` | Playback speed |
| `1`–`5` | Camera: 環繞 / 原機位疊合 / 俯視 / 底線後方 / 側面 |
| `T` | Toggle player trails (軌跡) |
| `V` | Toggle the video panel (對照影片) |
| Drag / scroll | Orbit / zoom (dragging from any preset switches to 環繞) |

**Deep links** set the starting view, e.g.
`index.html#cam=broadcast&t=900&panel=0&trails=0`.

## 3. Tracking data format (`tracking_data.json`)

The full schema is in the comment at the top of `index.html`. In short:

- **Units:** metres.
- **Court coordinates:**
  - The origin is the centre of the net, on the floor.
  - `+x` points toward the far baseline. The near baseline is at `x = -6.705`.
  - `+y` is the left side, as seen from behind the near baseline.
  - `z` is up.
- **`frames[]`:** each frame holds `t`, plus per-player `[x, y]` floor positions and a ball
  `[x, y, z]`. Either can be `null` when missing.
- **Optional `poses`:** COCO-17 joints per player. Without them, the figures are posed
  procedurally from footwork and hit timing.
- **Optional `events[]`:** `hit` and `bounce` events. If they're absent, they're
  auto-detected from the ball path.
- **Optional `camera`:** overrides the built-in calibration.
- **Player labels and colours** come from `players[]`. The pipeline names players by
  measured shirt colour (灰衣 / 白衣 / 紅衣 / 藍衣), not 近場 / 遠場, because the teams
  switch ends mid-match.
- **`video_offset`:** maps data time to video time (`video_time = t + video_offset`).

## 4. Tracking pipeline (`tracking/`)

Python 3.12 via `uv`; it uses PyTorch on Apple-silicon GPUs (MPS) when available. The
first `uv sync` downloads ~1 GB.

```bash
uv sync
uv run python -m tracking.check_camera match.mp4      # stage 1: camera check → tracking/camera.json
uv run python -m tracking.detect match.mp4            # stage 2: YOLO pose, ~40 min for the full video
uv run python -m tracking.appearance match.mp4        # stage 2b: hair/torso/shorts/elbow colours, ~2 min
uv run python -m tracking.track                       # stage 3: identities + smoothed floor tracks
uv run python -m tracking.export                      # → tracking_data.json
uv run python -m tracking.ball_detect match.mp4       # ball B1: colour+motion candidates, 60 fps, ~5 min
uv run python -m tracking.ball_track                  # ball B2: link into 2D flight tracks → data/ball2d.npz
uv run python -m tracking.ball3d                      # ball B3: 3D flight fits + hits/bounces, ~2 min
uv run python -m tracking.export                      # re-run to include the ball and events
uv run python -m tracking.ball_check                  # physical sanity numbers for the 3D ball
uv run python -m tracking.overlay match.mp4 --start 290 --end 320   # QA video → data/overlay.mp4
```

`detect` accepts `--start` / `--end` (seconds) to process a test window. Intermediate
files go to `data/` (gitignored).

| Stage | What it does |
| --- | --- |
| `camera.py` / `check_camera` | Fits the near-half court lines, intersects them into corners, and runs `solvePnP` over a focal-length sweep. Re-checks every 30 s. The camera holds still to ~2 px for the whole video. |
| `detect` | Runs YOLO11-pose on every 2nd frame (~30 fps). Keeps people whose feet project onto this court (±2 m), samples each one's shirt colour, and scores whether the court is visible in the frame. |
| `appearance` | Samples the median colour of keypoint-anchored regions per detection: hair, torso, shorts and each elbow. Needed because the near pair wear near-identical light shirts. |
| `track` | See below. |
| `export` | Pairs players into teams (the pair that shares a half), names them by shirt colour, and writes the viewer JSON. Long gaps stay `null`, and the viewer hides the player there. |
| `overlay` | Draws the exported positions back onto the video, to check alignment and identities. |
| `ball_detect` | Every frame: neon-green colour mask, minus a learned static mask (net-post sticker, logos), keeping only moving blobs. Records size, shape, hue/saturation and distance to the nearest ankle. |
| `ball_track` | Links candidates into flight tracks (seed by consistent velocity, extend with a quadratic prediction, allow short occlusion gaps). Keeps tracks that are ball-coloured (hue ≥ 40, sat ≥ 165; this rejects shoe stripes and a paddle grip), moving, and whose size-implied depth puts them over this court (rejects neighbouring-court balls). Then picks one track per frame. |
| `ball3d` | Lifts the 2D track to 3D. Between contacts the ball is ballistic, so each flight segment is 6 unknowns (start position, velocity) fitted to tens of observations through the calibrated camera, initialised from the ball's pixel size. Tracks are split at kinks (hits/bounces) wherever one parabola doesn't fit. Then each rally's segments are refit jointly with air drag (a = g − k\|v\|v, k = 0.04 /m) and soft constraints: consecutive segments meet, bounces lie on the floor. Boundaries become bounce events (on the floor, vertical velocity flips) or hit events (credited to the nearest player). |
| `ball_check` | Physical sanity checks: how well segments join, net-crossing heights, bounce positions, contact heights. |
| `ball_qa` | Crops around random tracked ball positions, to check precision. |
| `identity_frames` | Tiles frames with per-tracker boxes at chosen timestamps. This is the check that catches identity swaps. |

**How `track` assigns identities:**
1. Drop duplicate boxes on one person, and set aside people standing within 0.7 m of
   someone else.
2. Link the rest into tracklets by floor-position continuity, and merge unambiguous
   continuations.
3. **Teams by court side:** the red player's shirt is unmistakable, so his half over time
   tells every tracklet which team it belongs to. Partners share a half, and the ends
   switch only once.
4. **Split tracklets** where the person's appearance flips. The motion linker sometimes
   swaps two players during a crossing.
5. **Who's who within a team** comes from the multi-region appearance. Tracklets that
   overlap in time must be different people, which makes it a two-colouring problem; each
   connected group is oriented by its length-weighted evidence.
6. Fill the set-aside crowded detections back in, place feet (ankles, or hips at each
   player's measured hip height when feet are cut off), reject outliers, fill short gaps,
   and smooth at 2.5 Hz.

Checked by eye at 12 timestamps across the match: identities are correct in all of them,
including after the end switch. 61% of frames have all four players. The rest are mostly
players outside the camera's view, which stay `null`.

**2D ball result:**
- **Coverage:** the ball is tracked in 58% of all frames, 54% directly observed and the
  rest interpolated across short gaps. The median flight track is 26 observations.
- **Precision:** checked by eye on 42 random crops, 41 are the ball; the other is an
  interpolated point behind a player's head.
- **What the untracked frames are:** in sampled frames without a track, the ball was out
  of play (between points, in a hand, or lying still).

**3D ball result (`ball_check`):**
- 1,091 flight segments, 277 bounces and 440 hits.
- The ball is in 50% of exported frames.
- Reprojection error: median 1.3 px.
- Consecutive segments meet: median gap 10 cm.
- Contact height: median 0.91 m.
- Bounces: 90% land within 0.5 m of the court.
- Net crossings: 92% clear the net. Most of the rest are real: balls rolled under the net
  between points, or balls into the net.

**Video quirks the pipeline handles:**
- **A full-screen ad at ~312.5–316.4 s** ("Wear Eye Protection!"). It is detected by
  checking whether the court lines are where the camera says they should be. Players are
  `null` there.
- **The teams switch ends between 7:30 and 10:20.** Identity comes from shirt colour, not
  court side.
- **Near players often step out of the bottom/left edge of the frame.** These become gaps.
- **An animated graphic ball flies at the camera just before the ad** (~310.9–311.5 s).
  It's correctly not tracked.

### Camera

Current solution, used as `CALIBRATED_CAMERA` in `index.html` and stored in
`tracking/camera.json`:

- position `(-8.81, -4.78, 2.19)` m
- vertical FOV `41.4°`
- about 2 px reprojection error; the far court and net posts land within ~12 px.
