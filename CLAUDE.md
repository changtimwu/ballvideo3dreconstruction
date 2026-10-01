# CLAUDE.md

Pickleball doubles 3D reconstruction from YouTube videos. See README.md for setup, the
`analyze` CLI, controls and the data format. The original viewer spec is in `prompts.md`.

## Layout and flow

- **Analysis is local; publishing is CI.**
  - `uv run python -m tracking analyze <url>` downloads the video to `work/<id>/`
    (gitignored), runs the stages, writes `videos/<id>/{tracking_data,camera,meta}.json`,
    then commits and pushes.
  - `.github/workflows/pages.yml` runs `tools/build_site.py`, which builds `_site/` from
    `site/*` + `videos/*/` and generates `videos.json`. Pages is deployed by Actions, not
    from a branch.
  - Never commit videos or `work/`.
- **`site/viewer.html`** is the whole viewer: one ES-module script, three@0.170.0 and
  OrbitControls via a jsdelivr import map, Tailwind via CDN, and the YouTube IFrame API.
  There's no build step and no package.json; keep it a single file. It's opened as
  `viewer.html?v=<id>` and fetches `data/<id>.json` and `videos.json`.
- **`site/index.html`** is the video list, rendered from `videos.json`.
- **`tracking/`** is the pipeline, using `uv` (`pyproject.toml`, Python 3.12).
  - `__main__.py`: the `analyze` CLI. Stages are skipped when their output exists, so
    reruns resume.
  - Stage order: `calibrate → detect → appearance → track → ball_detect → ball_track →
    ball3d → export`.
  - QA tools: `overlay`, `identity_frames`, `ball_qa`, `ball_check`.
  - Every camera-dependent stage takes a required `--camera`, so there is no global
    camera.
- **`tracking/camera.py`** holds the court model (`MODEL_LINES`, `MODEL_POINTS`; sidelines
  split at the net), the `Camera` class, `solve_pnp`, `refine` (snap the model to painted
  lines) and `line_score` (is the court visible).

## Conventions that matter

- **Court coordinates everywhere in data and logic:**
  - Metres; the origin is the net centre on the floor.
  - `+x` points to the far baseline, `+y` is left as seen from the near baseline, `z` is up.
  - "Near" means the half closer to the camera; `calibrate.orient_near` enforces it.
- **Convert to three.js only at the render boundary,** with `toV(x, y, z)`, which gives
  `(x, z, -y)`. Don't mix the two spaces.
- **Player index order** comes from `players[]` in the JSON. `team` groups partners;
  team 0 is whoever starts on the near side.
- **The YouTube player is the master clock.** `clock.now()` advances an estimate with wall
  time and pulls it toward `player.getCurrentTime()`. Every transport action goes through
  the player API (`playVideo`, `seekTo`, `setPlaybackRate`, `mute`). Data time = video
  time − `video_offset`.
- **YouTube players must stay ≥ 200×200 px** to keep playing. That's why "collapsing" the
  video panel turns it into a corner mini player instead of hiding it.
- **All sampling goes through `sampleAt` / `playerXY` / `ballXYZ`.** These lerp over typed
  arrays built by `normalize()`. New data fields should be added to `normalize()`, not read
  from raw JSON in the frame loop.
- **Each video's camera lives in its data** (`camera` in `tracking_data.json`, from
  `work/<id>/camera.json`). `CALIBRATED_CAMERA` in the viewer is only a fallback.
- **UI copy is Traditional Chinese** (zh-Hant). Keep the UI chrome as written in
  `prompts.md`. Player legend labels come from the data.

## Testing

- **Local site:** `python3 tools/build_site.py && python3 -m http.server -d _site 8080`.
  Video comes from YouTube, so no Range-capable server is needed.
- **Automated Chrome tabs are often hidden** (`document.visibilityState === 'hidden'`).
  Chrome then pauses requestAnimationFrame, so tweens don't advance and screenshots are
  stale. Drive a separate headless Chrome instead:
  ```bash
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \
    --remote-debugging-port=9333 --user-data-dir=/tmp/cdp --window-size=1600,1000 \
    --autoplay-policy=no-user-gesture-required about:blank &
  BU_NAME=hl BU_CDP_URL=http://localhost:9333 browser-harness <<'PY'
  new_tab("http://localhost:8080/viewer.html?v=cRRLvhKbPXM"); wait_for_load()
  PY
  ```
  In that tab, the readout (`#readout`) is the easiest check that the 3D clock follows
  YouTube.
- **Headless WebGL screenshots can come out blank at random** (captured before the first
  draw). Retry before debugging the viewer.
- **Calibration:** check `work/<id>/calib.jpg` (the court model drawn on the median frame)
  and the printed drift-check numbers.
- **Pipeline QA:** after changing `track`/`export`, look at
  `tracking.identity_frames` / `tracking.overlay` output, not just the printed stats.
  Identity swaps and foot-placement errors only show up in the frames.

## Pipeline gotchas

- **Auto-calibration.**
  - Score homography hypotheses by "white on the line, floor either side". Walls and
    other clutter are white everywhere and otherwise win.
  - Reject collapsed or tiny projections.
  - Solve from near-half intersections first. Far ones are tiny and snap to the
    neighbouring court's lines.
  - Fitting one line through the net mesh biases the sidelines, hence the split.
- **The reference video has edited-in graphics** (an ad at ~312.5–316.4 s, captions). Anything
  per-frame must respect `court_score` / `court_ok`.
- **Teams switch ends mid-match.** Never infer identity or facing direction from the team or
  side. The viewer faces each figure toward the net from its current half.
- **Per-frame shirt colour is noisy** (white measured L 123–214, grey 57–148 in one
  rally). Only use colour aggregated over tracklets. Even then, similar teammates (grey vs
  white) only separate with the multi-region descriptor (`appearance.py`). "Which shirt is
  brighter" is not ground truth either.
- **Team membership comes from court side.** It's anchored on the highest-chroma shirt,
  not colour, because colour alone moved tracklets across teams.
- **YOLO emits duplicate boxes on one person.** Without `dedupe_and_split`, they shatter
  tracklets into 1-frame pieces.
- **The motion linker can swap people mid-tracklet** during crossings, so
  `split_on_appearance` must run before identity assignment.
- **Verify identities by eye** with `tracking.identity_frames` at ~12 timestamps
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
