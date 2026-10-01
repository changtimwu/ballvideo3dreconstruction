# CLAUDE.md

Pickleball doubles 3D reconstruction viewer. See README.md for setup, controls and the data
format. The original spec is in `prompts.md`.

## Layout

- `index.html`: the entire app: one ES-module script, three@0.170.0 and OrbitControls via a
  jsdelivr import map, and Tailwind via CDN. There's no build step and no package.json. Keep
  it single-file.
- `tracking/`: the Python pipeline that writes `tracking_data.json` (README §4). It uses
  `uv` (`pyproject.toml`, Python 3.12) and stages `check_camera → detect → appearance → track → export`,
  plus `overlay` for QA. Run stages as modules (`uv run python -m tracking.<stage>`).
  Intermediate artefacts go to `data/` (gitignored). `tracking_data.json` itself is
  committed, so GitHub Pages serves real data.
- `tracking/camera.py`: the court model and camera (projection, pixel → floor-plane rays,
  `line_score` court-visibility test). `tracking/camera.json` is the solved camera.
- `match.mp4` is gitignored (~365 MB). README §1 has the exact yt-dlp command to fetch it.
- If `tracking_data.json` can't be fetched (e.g. under `file://`), the viewer uses
  `generateDemo()`, which makes synthetic rallies in the page.

## Conventions that matter

- **Court coordinates everywhere in data and logic:**
  - Metres; the origin is the net centre on the floor.
  - `+x` points to the far baseline, `+y` is left as seen from the near baseline, `z` is up.
- **Convert to three.js only at the render boundary,** with `toV(x, y, z)`, which gives
  `(x, z, -y)`. Don't mix the two spaces.
- **Player index order** comes from `players[]` in the JSON. `team` groups partners; in the
  real data team 0 is whoever starts on the near side.
- **The video is the master clock.** `clock.now()` uses `video.currentTime`, refined with
  `requestVideoFrameCallback`. A free-running clock takes over only when the video fails to
  load. Data time = video time − `video_offset`.
- **All sampling goes through `sampleAt` / `playerXY` / `ballXYZ`.** These lerp over typed
  arrays built by `normalize()`. New data fields should be added to `normalize()`, not read
  from raw JSON in the frame loop.
- **`CALIBRATED_CAMERA` in `index.html` mirrors `tracking/camera.json`** (written by
  `tracking.check_camera`). Re-run that rather than hand-editing. It is tied to the 1080p60
  H.264 encode (yt-dlp formats `299+140`).
- **UI copy is Traditional Chinese** (zh-Hant). Keep the UI chrome as written in
  `prompts.md`. Player legend labels come from the data.

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
- **Pipeline QA:** after changing `track`/`export`, render
  `uv run python -m tracking.overlay match.mp4 --start 290 --end 320` and look at frames
  (ffmpeg `select` + `tile`). Identity swaps and foot-placement errors show up immediately;
  the printed stats don't reveal them.

## Pipeline gotchas

- **The footage has edited-in graphics** (an ad at ~312.5–316.4 s). Anything per-frame must
  respect `court_score` / `court_ok`.
- **Teams switch ends mid-match.** Never infer identity or facing direction from the team or
  side. The viewer faces each figure toward the net from its current half.
- **Per-frame shirt colour is noisy** (white measured L 123–214, grey 57–148 in one
  rally). Only use colour aggregated over tracklets. Even then, the near pair (grey vs
  white) only separates with the multi-region descriptor (`appearance.py`). "Which shirt
  is brighter" is not ground truth either.
- **YOLO emits duplicate boxes on one person.** Without `dedupe_and_split`, they shatter
  tracklets into 1-frame pieces.
- **The motion linker can swap people mid-tracklet** during crossings, so
  `split_on_appearance` must run before identity assignment.
- **Verify identities by eye** with `tracking.identity_frames` (per-tracker boxes) at ~12 timestamps
  (including after the end switch). The tracker's own statistics looked fine while
  identities were wrong.
- **The ball in 2D** (`ball_detect` / `ball_track`): colour separates it best. Ball hue
  43–50 / sat > 190; yellow shoe stripes hue 34–37; the red player's paddle grip sat < 140.
  Don't reject candidates by distance to ankles, because low balls bounce right by
  players' feet. Check precision with `tracking.ball_qa` crops and recall by looking at
  frames with no track.
- **The 3D ball (`ball3d.py`).** Drag matters for a pickleball: about 40% of g at
  10 m/s. Fit rallies jointly (continuity + floor constraints); independent segment fits
  drift in depth by up to a metre at short segments. Use `simulate()` / `simulate_batch()`
  everywhere the ball is evaluated, so export and checks use the same physics as the fit.
  The batched Jacobian in `refit_chain` is why the full match takes 2 min rather than
  hours; don't swap in scipy's default finite differences.
- **Headless Chrome screenshots of WebGL can come out blank at random** (captured before
  the first draw). Retry before debugging the viewer. To render arbitrary times, serve a
  copy of `index.html` + `tracking_data.json` without `match.mp4`: the internal clock can
  seek, while a server without HTTP Range support can't seek the video.
