"""End-to-end evaluation on the Mongolian hold-out set + latency benchmark + multi-frame simulation.
Outputs runs/eval/results.json and figures used in the report."""
import os, sys, json, time, random, glob, argparse
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.pipeline import ALPR, PlateTrack, iou
from mn_alpr import plate_rules as PR
import editdistance

ap = argparse.ArgumentParser(); ap.add_argument("--det", default="models/plate_det_yolo11n.pt"); ap.add_argument("--rec", default="models/plate_rec_crnn.pt")
ap.add_argument("--out", default="runs/eval"); ap.add_argument("--frames", type=int, default=9)
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); random.seed(0); np.random.seed(0)

alpr = ALPR(args.det, args.rec)
meta = json.load(open("data/rec/real_github/meta.json", encoding="utf-8"))
val = {}
for m in meta:
    if m["split"] == "val": val.setdefault(m["image"], []).append(m)

# ---- 1. single-frame end-to-end ----
res_rows = []; n_ok = n_tot = 0; cer_n = cer_d = 0; det_hit = 0; times = []
for stem, plates in sorted(val.items()):
    frame = cv2.imread(f"data/raw/github_alpr/{stem}.jpg")
    results, tm = alpr.process_frame(frame); times.append(tm)
    for m in plates:
        n_tot += 1; gt = m["text"]; gb = m["box"]
        match = max(results, key=lambda r: iou(r["box"], gb), default=None)
        if match is None or iou(match["box"], gb) < 0.5:
            res_rows.append({"image": stem, "gt": gt, "pred": "", "detected": False, "layout": m["two_line"]}); cer_n += len(gt); cer_d += len(gt); continue
        det_hit += 1; pr = match["text"]; ok = pr == gt; n_ok += ok
        cer_n += editdistance.eval(pr, gt); cer_d += len(gt)
        res_rows.append({"image": stem, "gt": gt, "pred": pr, "ok": ok, "detected": True, "layout": match["layout"], "rec_conf": match["rec_conf"], "det_conf": match["det_conf"]})
single = {"plates": n_tot, "detected": det_hit, "det_recall@0.5": det_hit / n_tot, "plate_acc": n_ok / n_tot, "plate_acc_given_det": n_ok / max(1, det_hit), "cer": cer_n / cer_d}
print("SINGLE-FRAME:", single)

# ---- 2. multi-frame simulation (K jittered views per image, confidence-weighted voting) ----
def jitter(frame):
    h, w = frame.shape[:2]; a = frame.copy()
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-3, 3), random.uniform(0.9, 1.1)); M[:, 2] += np.random.uniform(-0.03, 0.03, 2) * [w, h]
    a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    a = np.clip(a.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-25, 25), 0, 255).astype(np.uint8)
    if random.random() < 0.5: a = cv2.GaussianBlur(a, (random.choice([3, 5]), ) * 2, 0)
    if random.random() < 0.5: a = np.clip(a + np.random.normal(0, 6, a.shape), 0, 255).astype(np.uint8)
    return a

multi_ok = {k: 0 for k in (3, 5, args.frames)}; multi_rows = []
for stem, plates in sorted(val.items()):
    frame = cv2.imread(f"data/raw/github_alpr/{stem}.jpg")
    tracks = {i: PlateTrack(i, m["box"]) for i, m in enumerate(plates)}
    for k in range(1, args.frames + 1):
        results, _ = alpr.process_frame(jitter(frame) if k > 1 else frame)
        for i, m in enumerate(plates):
            match = max(results, key=lambda r: iou(r["box"], m["box"]), default=None)
            if match is not None and iou(match["box"], m["box"]) >= 0.5: tracks[i].add(match["text"], match["rec_conf"], match["box"])
        if k in multi_ok:
            for i, m in enumerate(plates): multi_ok[k] += tracks[i].consensus()[0] == m["text"]
    for i, m in enumerate(plates):
        text, margin, n = tracks[i].consensus(); ok = text == m["text"]
        multi_rows.append({"image": stem, "gt": m["text"], "voted": text, "ok": ok, "n": n, "margin": margin})
multi = {"frames": args.frames, "plate_acc": multi_ok[args.frames] / n_tot, "by_k": {str(k): v / n_tot for k, v in multi_ok.items()}}
print("MULTI-FRAME:", multi)


# ---- 2b. MLUB real crops: recognition + multi-frame simulation (large-sample numbers) ----
mlub_meta = json.load(open("data/rec/real_mlub/meta.json", encoding="utf-8")); mv = [m for m in mlub_meta if m["split"] == "val"]
def crop_jitter(im):
    h, w = im.shape[:2]; a = im.copy()
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-3, 3), random.uniform(0.92, 1.08)); M[:, 2] += np.random.uniform(-0.03, 0.03, 2) * [w, h]
    a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    a = np.clip(a.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-25, 25), 0, 255).astype(np.uint8)
    if random.random() < 0.5: a = cv2.GaussianBlur(a, (3, 3), 0)
    if random.random() < 0.5: a = np.clip(a + np.random.normal(0, 6, a.shape), 0, 255).astype(np.uint8)
    return a
def read_mlub(im, two):
    if two:
        H = im.shape[0]; t, c = alpr.rec.read_lines([im[: int(H * 0.55)], im[int(H * 0.45):]]); return PR.normalize(t[0] + t[1]), float(min(c))
    t, c = alpr.rec.read_lines([im]); return PR.normalize(t[0]), float(c[0])
m_single = 0; m_cer_n = m_cer_d = 0; m_multi = {k: 0 for k in (3, 5, args.frames)}; m_rows = []
for m in mv:
    im = cv2.imread(m["src"]); pr, c = read_mlub(im, m["two_line"]); ok = pr == m["text"]; m_single += ok
    m_cer_n += editdistance.eval(pr, m["text"]); m_cer_d += len(m["text"])
    tr = PlateTrack(0, [0, 0, 1, 1]); tr.add(pr, c, [0, 0, 1, 1]); hist = {1: pr}
    for k in range(2, args.frames + 1):
        p2, c2 = read_mlub(crop_jitter(im), m["two_line"]); tr.add(p2, c2, [0, 0, 1, 1])
        if k in m_multi: m_multi[k] += tr.consensus()[0] == m["text"]
    m_rows.append({"image": m["image"], "gt": m["text"], "pred": pr, "ok": ok, "two_line": m["two_line"], "voted": tr.consensus()[0]})
mlub = {"plates": len(mv), "plate_acc": m_single / len(mv), "cer": m_cer_n / m_cer_d,
        "plate_acc_single_line": sum(r["ok"] for r in m_rows if not r["two_line"]) / max(1, sum(not r["two_line"] for r in m_rows)),
        "plate_acc_two_line": sum(r["ok"] for r in m_rows if r["two_line"]) / max(1, sum(r["two_line"] for r in m_rows)),
        "multi_frame": {str(k): v / len(mv) for k, v in m_multi.items()}}
print("MLUB:", mlub)

# ---- 3. latency benchmark (1080p frames, 1 plate each) ----
bench_imgs = [cv2.imread(f) for f in sorted(glob.glob("data/raw/github_alpr/*.jpg"))[:30]]
for f in bench_imgs[:5]: alpr.process_frame(f)
lat = []; comp = []
for _ in range(3):
    for f in bench_imgs:
        torch.cuda.synchronize(); t0 = time.perf_counter(); _, tm = alpr.process_frame(f); torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1e3); comp.append(tm)
latency = {"n": len(lat), "mean_ms": float(np.mean(lat)), "median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)), "max_ms": float(np.max(lat)),
           "det_mean_ms": float(np.mean([t["t_det_ms"] for t in comp])), "rec_mean_ms": float(np.mean([t["t_rec_ms"] for t in comp])),
           "first_frame_ms": float(times[0]["t_total_ms"])}
print("LATENCY:", latency)

# ---- 4. detector mAP on Mongolian val + generic test ----
from ultralytics import YOLO
det = YOLO(args.det); mp = {}
for name, data in (("mongolian_val", "data/det/github/data.yaml"), ("generic_val", "data/det/hf_plates/data.yaml")):
    r = det.val(data=data, imgsz=640, batch=16, device=0 if __import__("torch").cuda.is_available() else "cpu", plots=False, verbose=False, project=f"{args.out}/det", name=name, exist_ok=True)
    mp[name] = {"mAP50": float(r.box.map50), "mAP50-95": float(r.box.map), "precision": float(r.box.mp), "recall": float(r.box.mr)}
print("DET:", mp)

json.dump({"single_frame": single, "multi_frame": multi, "mlub": mlub, "mlub_rows": m_rows, "latency": latency, "detector": mp, "rows": res_rows, "multi_rows": multi_rows},
          open(f"{args.out}/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", f"{args.out}/results.json")
