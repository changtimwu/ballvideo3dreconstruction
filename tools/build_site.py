"""Build the static site into _site/ (run by GitHub CI; standard library only).

  python tools/build_site.py [--out _site]

Copies site/* and, for every analysed video in videos/<id>/, its tracking data to
data/<id>.json. Writes videos.json, the list the gallery page reads, from each
video's meta.json (newest analysis first).
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "_site"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out)
    (out / "data").mkdir()

    videos = []
    for d in sorted((ROOT / "videos").glob("*/")):
        meta_p, data_p = d / "meta.json", d / "tracking_data.json"
        if not (meta_p.exists() and data_p.exists()):
            print(f"skipping {d.name}: needs meta.json and tracking_data.json")
            continue
        meta = json.loads(meta_p.read_text())
        shutil.copy(data_p, out / "data" / f"{meta['id']}.json")
        meta["data_bytes"] = data_p.stat().st_size
        videos.append(meta)
    videos.sort(key=lambda m: m.get("analysed_at", ""), reverse=True)
    (out / "videos.json").write_text(json.dumps({"videos": videos}, ensure_ascii=False, indent=1) + "\n")
    (out / ".nojekyll").write_text("")
    print(f"built {out}: {len(videos)} video(s)")


if __name__ == "__main__":
    main()
