"""Write tracking_data.json for the viewer from data/players.npz.

  uv run python -m tracking.export [--players data/players.npz] [--out tracking_data.json]

Teams are the two pairs that share a half most often. Team 0 is whichever pair
starts on the near side. Labels come from the measured shirt colours.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .camera import Camera
from .track import colour_name, lab_to_hex


def display_colour(lab) -> str:
    """Measured shirt colour, nudged so it reads on the blue court."""
    name = colour_name(lab)
    return {"白衣": "#e5e7eb", "灰衣": "#9ca3af", "黑衣": "#3b4252"}.get(name) or lab_to_hex(
        np.array([max(lab[0], 150), 128 + (lab[1] - 128) * 1.6, 128 + (lab[2] - 128) * 1.6]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", default="data/players.npz")
    ap.add_argument("--ball3d", default="data/ball3d.npz")
    ap.add_argument("--out", default="tracking_data.json")
    a = ap.parse_args()

    P = np.load(a.players)
    t, pos, protos = P["t"], P["pos"], P["protos"]
    n, k = pos.shape[:2]

    side = np.sign(pos[..., 0])
    best = max(itertools.combinations(range(k), 2),
               key=lambda pr: np.nanmean(side[:, pr[0]] == side[:, pr[1]]))
    other = tuple(i for i in range(k) if i not in best)
    first = np.flatnonzero(np.isfinite(pos[:, best[0], 0]))[0]
    teams = [best, other] if pos[first, best[0], 0] < 0 else [other, best]
    order = [*teams[0], *teams[1]]
    print("teams:", [[colour_name(protos[i]) for i in tm] for tm in teams],
          f"(same-half agreement {np.nanmean(side[:, best[0]] == side[:, best[1]]) * 100:.0f}%)")

    names = [colour_name(protos[i]) for i in order]
    # Teammates in similar neutral shirts: call the brighter one white, the other grey.
    for t0 in (0, 2):
        j0, j1 = t0, t0 + 1
        if names[j0] == names[j1] and names[j0] in ("白衣", "灰衣", "黑衣"):
            hi, lo = (j0, j1) if protos[order[j0]][0] > protos[order[j1]][0] else (j1, j0)
            names[hi], names[lo] = "白衣", "灰衣"
    for nm in set(names):                     # disambiguate duplicate colour names
        dup = [j for j, x in enumerate(names) if x == nm]
        if len(dup) > 1:
            for c, j in enumerate(dup):
                names[j] = f"{nm} {'AB'[c]}"
    neutral = {"白衣": "#e5e7eb", "灰衣": "#9ca3af", "黑衣": "#3b4252"}
    players = [{"id": f"p{j}", "label": names[j], "color": neutral.get(names[j].split()[0]) or display_colour(protos[i]),
                "team": j // 2} for j, i in enumerate(order)]

    ball = np.full((n, 3), np.nan)
    events = []
    if Path(a.ball3d).exists():
        from .ball3d import LINK_GAP, simulate
        R = np.load(a.ball3d)
        S, E = R["segs"], R["events"]
        for i, sg in enumerate(S):
            # a segment covers its own span plus the gap up to a consecutive successor
            end = sg[1]
            if i + 1 < len(S) and S[i + 1][0] - sg[1] <= LINK_GAP:
                end = S[i + 1][0]
            m = (t >= sg[0]) & (t < end + 1e-6)
            if m.any():
                bp, _ = simulate(sg[2:5], sg[5:8], t[m] - sg[0])
                bp[:, 2] = np.maximum(bp[:, 2], 0.0)
                ball[m] = bp
        export_index = {k_: j for j, k_ in enumerate(order)}
        for te, kind, x, y, z, pl in E:
            if kind == 0:
                events.append({"t": round(float(te), 3), "type": "bounce", "pos": [round(float(x), 3), round(float(y), 3)]})
            elif int(pl) in export_index:
                events.append({"t": round(float(te), 3), "type": "hit", "player": export_index[int(pl)]})
        print(f"ball in {np.isfinite(ball[:, 0]).mean() * 100:.1f}% of frames; {len(events)} events")

    frames = []
    for fi in range(n):
        row = []
        for i in order:
            x, y = pos[fi, i]
            row.append(None if not np.isfinite(x) else [round(float(x), 3), round(float(y), 3)])
        b = None if not np.isfinite(ball[fi, 0]) else [round(float(v), 3) for v in ball[fi]]
        frames.append({"t": round(float(t[fi]), 4), "players": row, "ball": b})

    cam = Camera.load().to_json()
    data = {
        "fps": round(float(P["fps"]) / int(P["step"]), 3),
        "video_offset": 0.0,
        "source": "tracking pipeline (players: milestone 1; ball 3D + events: milestone 3)",
        "camera": {k_: cam[k_] for k_ in ("position", "look_at", "vfov_deg", "aspect")},
        "players": players,
        "frames": frames,
        "events": events,
    }
    with open(a.out, "w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {a.out}: {n} frames, {t[0]:.2f}–{t[-1]:.2f}s, players {[p['label'] for p in players]}")


if __name__ == "__main__":
    main()
