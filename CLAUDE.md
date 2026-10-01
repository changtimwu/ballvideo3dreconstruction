# CLAUDE.md

Pickleball doubles 3D reconstruction viewer. See README.md for setup, controls and the data
format. The original spec is in `prompts.md`.

## Layout

- `index.html`: the entire app: one ES-module script, three@0.170.0 and OrbitControls via a
  jsdelivr import map, and Tailwind via CDN. There's no build step and no package.json. Keep
  it single-file.
- `tools/calibrate_camera.py`: an OpenCV camera solve from the court lines. Run it with
  `uv run --with opencv-python-headless --with numpy ...`.
- `match.mp4` is gitignored (~365 MB). README §1 has the exact yt-dlp command to fetch it.
- `tracking_data.json` doesn't exist yet. Without it the viewer uses `generateDemo()`, which
  makes synthetic rallies in the page.

## Conventions that matter

- **Court coordinates everywhere in data and logic:**
  - Metres; the origin is the net centre on the floor.
  - `+x` points to the far baseline, `+y` is left as seen from the near baseline, `z` is up.
- **Convert to three.js only at the render boundary,** with `toV(x, y, z)`, which gives
  `(x, z, -y)`. Don't mix the two spaces.
- **Player index order** comes from `players[]` in the JSON. Indices 0–1 are the near team
  and 2–3 the far team (`team` overrides this).
- **The video is the master clock.** `clock.now()` uses `video.currentTime`, refined with
  `requestVideoFrameCallback`. A free-running clock takes over only when the video fails to
  load. Data time = video time − `video_offset`.
- **All sampling goes through `sampleAt` / `playerXY` / `ballXYZ`.** These lerp over typed
  arrays built by `normalize()`. New data fields should be added to `normalize()`, not read
  from raw JSON in the frame loop.
- **`CALIBRATED_CAMERA` in `index.html` comes from `tools/calibrate_camera.py`.** Re-run the
  tool rather than hand-editing it. It is tied to the 1080p60 H.264 encode (yt-dlp formats
  `299+140`).
- **UI copy is Traditional Chinese** (zh-Hant). Keep labels as written in `prompts.md`.

## Testing

- **Open `file://.../index.html` in Chrome.** Don't serve it with `python3 -m http.server`:
  it has no Range support, so video seeking breaks. Use `npx http-server` if you need
  `fetch` of `tracking_data.json`.
- **Automated tabs are often `document.visibilityState === 'hidden'`.** Chrome then pauses
  requestAnimationFrame, so screenshots show stale frames and camera tweens don't advance.
  For visual checks, use headless Chrome with a deep link:
  ```bash
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \
    --window-size=1600,1000 --timeout=7000 --screenshot=out.png \
    "file://$PWD/index.html#cam=broadcast&t=900&panel=0"
  ```
- **To check `原機位疊合` alignment,** blend that screenshot with the matching video frame
  (`ffmpeg -ss 900 -i match.mp4 -frames:v 1 f.png`). The court lines should coincide.
