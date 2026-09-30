I have a video of a pickleball doubles match (`match.mp4`) and a tracking dataset (`tracking_data.json`) containing 3D court coordinates over time for 4 players and the ball.

Build a single-file, production-grade interactive 3D playback viewer using HTML5, Three.js, and Tailwind CSS. The app should load locally and run smoothly in Chrome.

### UI & Layout Requirements

1. **Split-Screen Layout:**
   - Left side: 3D Three.js canvas.
   - Right side: Synchronized `` player displaying the source match footage.
   - Include a toggle button `對照影片` to collapse or expand the video panel so the 3D court can take up full width.

2. **Top Overlay Over 3D Canvas:**
   - Title: "匹克球 3D 重建 · v2" with subtitle "環繞視角 · 拖曳旋轉 · 滾輪縮放".
   - Legend bar showing colored dots and player labels:
     - 近場・黃衣 (Near court, Yellow shirt)
     - 近場・黑衣 (Near court, Black shirt)
     - 遠場・白衣 (Far court, White shirt)
     - 遠場・黑衣 (Far court, Black shirt)

3. **Bottom Transport Bar:**
   - Media controls: Play/Pause button, Step Back (``), time scrubber slider, and time/frame readout (`XX.XXs · #frame`).
   - Playback speed selectors: `1/4x`, `1/2x`, `1x`.
   - Camera angle presets:
     - `環繞` (Default OrbitControls: drag to rotate, scroll to zoom)
     - `原機位疊合` (Fixed camera matching the exact focal length, elevation, and angle of the real video camera)
     - `俯視` (Direct 90-degree top-down 2D court view)
     - `底線後方` (Behind the baseline perspective)
     - `側面` (Side-line spectator angle)
   - Feature toggle: `軌跡` (Toggle player movement trail paths on/off on the court surface).

### 3D Scene Specifications

1. **Court Model:**
   - Standard metric pickleball court dimensions (44 ft × 20 ft / 13.41 m × 6.10 m).
   - Accurate court markings: baseline, sidelines, centerline, and non-volley zone (kitchen line at 7 ft / 2.13 m).
   - Semi-transparent 3D mesh net with top tape (height: 36 inches at sidelines, 34 inches at center).
   - High-contrast dark green surround and blue court surface.

2. **Players & Ball:**
   - Players: Stylized 3D mannequin / simplified humanoid stick figures with articulated joints. Color each player according to their team/jersey color (Yellow, Dark/Black, Cyan/White, Orange/Pink).
   - Position rings: Add an illuminated circular ring under each player's feet projected directly on the court floor.
   - Motion Trails: When `軌跡` is active, render smooth ribbon/line trails behind each player showing their recent footwork path.
   - Ball: High-visibility yellow sphere with an active trajectory arc line connecting bounce points.

3. **Synchronization Logic:**
   - Hook the Three.js update loop directly to `video.currentTime` (or `requestVideoFrameCallback`).
   - Interpolate positions (lerp) smoothly between JSON keyframes so 0.25x/0.5x slow-motion looks fluid.
   - Support smooth camera transitions (tweening position and target) when switching between preset views.

Provide the complete, self-contained HTML/JS code ready to open in the browser.
