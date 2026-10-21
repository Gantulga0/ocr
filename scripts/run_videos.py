"""Qualitative multi-frame test on the three street videos; writes annotated videos + a JSON summary."""
import os, sys, json, glob
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.pipeline import ALPR, draw
from mn_alpr import plate_rules as PR
alpr = ALPR(); os.makedirs("runs/videos", exist_ok=True); summary = {}
for path in sorted(glob.glob("data/videos/*.mp4")):
    alpr.reset_tracks(); cap = cv2.VideoCapture(path); W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    name = os.path.splitext(os.path.basename(path))[0]
    wr = cv2.VideoWriter(f"runs/videos/{name}_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (W, H)); i = 0; times = []; per_frame = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % 2 == 0:
            res, tm = alpr.process_frame(fr, track=True); times.append(tm["t_total_ms"]); wr.write(draw(fr, res))
            per_frame.append([(r["text"], r.get("voted", ""), r.get("track_id")) for r in res])
        i += 1
    cap.release(); wr.release()
    allt = sorted(alpr.all_tracks(), key=lambda t: -t.n)
    tr = []
    for t in allt:
        text, margin, n = t.consensus()
        if n >= 3 and text:
            raw = [h[0] for h in t.history]; agree = sum(r == text for r in raw) / len(raw)
            tr.append({"id": t.id, "voted": PR.pretty(text), "valid": PR.is_valid(text), "frames": n, "margin": round(margin, 3), "single_frame_agreement": round(agree, 3), "region": PR.region_of(text)})
    summary[name] = {"frames_processed": len(times), "res": f"{W}x{H}", "mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95)), "tracks": tr}
    print(name, summary[name])
json.dump(summary, open("runs/videos/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
