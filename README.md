# 匹克球 3D 重建 · Pickleball 3D Reconstruction

An interactive 3D playback viewer for a pickleball doubles match. It plays a Three.js
reconstruction of the court, the players and the ball next to the source video, locked to
the video's clock.

- `index.html`: the whole viewer in one file (Three.js and Tailwind come from CDNs).
- `tools/calibrate_camera.py`: solves the real camera from the court lines in the video.
  This is what drives the `原機位疊合` view.
- `prompts.md`: the original spec.

> **Status:** `tracking_data.json` doesn't exist yet. Until it does, the viewer runs on a
> **generated demo dataset**. The court, the camera calibration and the video sync are
> real; the player and ball motion are synthetic. The badge in the top-right corner shows
> which data is loaded.

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
- **`video_offset`:** maps data time to video time (`video_time = t + video_offset`).

## 4. Camera calibration

The camera in `match.mp4` never moves. `tools/calibrate_camera.py` reads one frame and
works out the camera's pose and focal length:

1. It fits the near-half court lines to white pixels.
2. It intersects those lines to get the court corners.
3. It runs `solvePnP` while sweeping the focal length.

The result reprojects to about 2 px, and the far court and net posts land within about
12 px.

```bash
uv run --with opencv-python-headless --with numpy tools/calibrate_camera.py match.mp4 900
```

It prints the camera in court coordinates and writes `calib_overlay.jpg` for a visual check.
Current solution:

- position `(-8.81, -4.78, 2.20)` m
- vertical FOV `41.4°`

These numbers are hard-coded as `CALIBRATED_CAMERA` in `index.html`.
