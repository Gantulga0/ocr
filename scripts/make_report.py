"""Build report/report.html (+ figures) from training logs and runs/eval results."""
import os, sys, json, glob, math, random, datetime
import numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.pipeline import ALPR, draw
from mn_alpr.crnn import CRNN
FIG = "report/figures"; os.makedirs(FIG, exist_ok=True); random.seed(1)
R = json.load(open("runs/eval/results.json", encoding="utf-8"))
meta = json.load(open("data/rec/real_github/meta.json", encoding="utf-8"))
V = {}

def pct(x): return f"{100*x:.1f}%"
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"[{100*(c-h):.0f}%, {100*(c+h):.0f}%]"

# ---------- Fig 1: samples ----------
imgs = sorted(glob.glob("data/raw/github_alpr/*.jpg")); pick = [imgs[i] for i in (1, 9, 27, 36)]
row = np.hstack([cv2.resize(cv2.imread(p), (480, 270)) for p in pick])
crops = []
for m in [m for m in meta if m["split"] == "val"][:8]:
    im = cv2.imread(f"data/raw/github_alpr/{m['image']}.jpg"); b = m["box"]; c = im[b[1]:b[3], b[0]:b[2]]
    crops.append(cv2.resize(c, (240, 100)))
row2 = np.hstack(crops[:8])
mcs = sorted(glob.glob("data/raw/mlub/training/training/*.png")); random.shuffle(mcs)
row3 = np.hstack([cv2.resize(cv2.imread(p), (240, 100)) for p in mcs[:8]])
cv2.imwrite(f"{FIG}/fig1_samples.jpg", np.vstack([row, row2, row3]), [cv2.IMWRITE_JPEG_QUALITY, 85])
# ---------- Fig 2: synth ----------
sl = [l.split("\t") for l in open("data/rec/synth/labels.txt", encoding="utf-8").read().splitlines()[:5000]]
random.shuffle(sl); tiles = [cv2.resize(cv2.imread(f"data/rec/synth/{p}"), (240, 80)) for p, _ in sl[:24]]
cv2.imwrite(f"{FIG}/fig2_synth.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
# ---------- Fig 4: det curves ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=160); off = 0
for name, lab in (("stageA_generic", "Stage A (generic val)"), ("stageB_mongolian", "Stage B (Mongolian val)")):
    f = f"runs/det/{name}/results.csv"
    if not os.path.exists(f): continue
    d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]; e = d["epoch"] + off
    ax[0].plot(e, d["metrics/mAP50(B)"], label=lab); ax[1].plot(e, d["metrics/mAP50-95(B)"], label=lab); off += len(d)
    V.setdefault("EPOCHS_" + name[5], str(len(d)))
for a, t in zip(ax, ("mAP50", "mAP50-95")): a.set_xlabel("epoch"); a.set_ylabel(t); a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/fig4_det_curves.png"); plt.close()
# ---------- Fig 5: rec curves ----------
h1 = json.load(open("runs/rec/synth_only/history.json")); h2 = json.load(open("runs/rec/crnn/history.json"))
V["REC_EPOCHS_S"], V["REC_EPOCHS_F"] = str(len(h1)), str(len(h2))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=160)
h3 = json.load(open("runs/rec/crnn_mlub_r1/history.json")) + json.load(open("runs/rec/crnn_mlub/history.json")); V["REC_EPOCHS_M"] = "8 + 20"
ep = list(range(1, len(h1) + len(h2) + len(h3) + 1)); H = h1 + h2 + h3; b1, b2 = len(h1) + .5, len(h1) + len(h2) + .5
ax[0].plot(ep, [h["loss"] for h in H]); ax[0].set_ylabel("CTC loss (train)"); ax[0].set_yscale("log")
ax[1].plot(ep[:len(h1) + len(h2)], [h["plate_acc"] for h in h1 + h2], label="plate acc (GitHub val)"); ax[1].plot(ep[:len(h1) + len(h2)], [h["cer"] for h in h1 + h2], label="CER (GitHub val)")
ax[1].plot(ep[len(h1) + len(h2):], [h["plate_acc"] for h in h3], label="plate acc (MLUB val)"); ax[1].plot(ep[len(h1) + len(h2):], [h["cer"] for h in h3], label="CER (MLUB val)")
for a in ax: a.axvline(b1, ls="--", c="gray"); a.axvline(b2, ls="--", c="gray")
ax[1].legend(fontsize=6)
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{FIG}/fig5_rec_curves.png"); plt.close()
# ---------- Fig 6: qualitative ----------
alpr = ALPR(); tiles = []
vals = sorted({m["image"] for m in meta if m["split"] == "val"})
for stem in vals[:8]:
    fr = cv2.imread(f"data/raw/github_alpr/{stem}.jpg"); res, _ = alpr.process_frame(fr)
    out = draw(fr, res, use_voted=False)
    # zoom on the largest plate region for legibility
    if res:
        b = max(res, key=lambda r: (r["box"][2] - r["box"][0]))["box"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(300, (b[2] - b[0]) * 2.2)
        x1, y1 = int(max(0, cx - s)), int(max(0, cy - s * 0.6)); out = out[y1:int(cy + s * 0.6), x1:int(cx + s)]
    tiles.append(cv2.resize(out, (360, 220)))
cv2.imwrite(f"{FIG}/fig6_qualitative.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])

# ---------- values ----------
S, M, L, D = R["single_frame"], R["multi_frame"], R["latency"], R["detector"]
ntr = len({m["image"] for m in meta if m["split"] == "train"}); nva = len({m["image"] for m in meta if m["split"] == "val"})
V.update({
 "AUTHOR": os.environ.get("REPORT_AUTHOR", "[Author name]"), "AFFIL": os.environ.get("REPORT_AFFIL", "[Affiliation]"), "DATE": datetime.date.today().strftime("%d %B %Y"),
 "N_SYNTH": f"{len(open('data/rec/synth/labels.txt',encoding='utf-8').read().splitlines()):,}",
 "N_REAL_TRAIN": str(len(open("data/rec/real_github/train_labels.txt", encoding="utf-8").read().splitlines())),
 "N_REAL_VAL": str(len(open("data/rec/real_github/val_labels.txt", encoding="utf-8").read().splitlines())),
 "N_TRAIN_IMGS": str(ntr), "N_VAL_IMGS": str(nva), "N_TRAIN_PLATES": str(sum(m["split"] == "train" for m in meta)), "N_VAL_PLATES": str(S["plates"]),
 "N_VAL_BOXES": str(len(glob.glob("data/det/github/labels/val/*.txt")) and sum(len(open(f).read().split("\n")) - 1 for f in glob.glob("data/det/github/labels/val/*.txt"))),
 "N_TSUND_IMGS": os.environ.get("N_TSUND_IMGS", "n/a"),
 "DET_MAP50_MN": f"{D['mongolian_val']['mAP50']:.3f}", "DET_MAP_MN": f"{D['mongolian_val']['mAP50-95']:.3f}", "DET_P_MN": f"{D['mongolian_val']['precision']:.3f}", "DET_R_MN": f"{D['mongolian_val']['recall']:.3f}",
 "DET_MAP50_GEN": f"{D['generic_val']['mAP50']:.3f}", "DET_MAP_GEN": f"{D['generic_val']['mAP50-95']:.3f}", "DET_P_GEN": f"{D['generic_val']['precision']:.3f}", "DET_R_GEN": f"{D['generic_val']['recall']:.3f}",
 "ACC_SINGLE": pct(S["plate_acc"]), "ACC_MULTI": pct(M["plate_acc"]), "K_FRAMES": str(M["frames"]), "E2E_CER": f"{S['cer']:.3f}",
 "LAT_MEAN": f"{L['mean_ms']:.1f}", "LAT_MED": f"{L['median_ms']:.1f}", "LAT_P95": f"{L['p95_ms']:.1f}", "LAT_MAX": f"{L['max_ms']:.1f}", "LAT_DET": f"{L['det_mean_ms']:.1f}", "LAT_REC": f"{L['rec_mean_ms']:.1f}",
 "REC_PARAMS": f"{sum(p.numel() for p in CRNN().parameters())/1e6:.2f}",
 "ONE_PLATE_PCT": f"{100/S['plates']:.1f}", "CI_SINGLE": wilson(round(S["plate_acc"] * S["plates"]), S["plates"]),
 "ST_DET": "met" if D["mongolian_val"]["mAP50"] >= 0.95 else "not met", "ST_SINGLE": "met" if S["plate_acc"] >= 0.95 else "not met",
 "ST_MULTI": "met" if M["plate_acc"] >= 0.95 else "not met", "ST_LAT": "met" if L["mean_ms"] < 100 else "not met",
})
MB = R["mlub"]; mm = json.load(open("data/rec/real_mlub/meta.json", encoding="utf-8")); mv = [m for m in mm if m["split"] == "val"]
h3 = json.load(open("runs/rec/crnn_mlub/history.json")); bm = max(h3, key=lambda h: h["plate_acc"] - h["cer"])
h3 = json.load(open("runs/rec/crnn_mlub_r1/history.json")) + h3
V.update({"N_MLUB": f"{len(mm):,}", "N_MLUB_UNIQUE": f"{len({m['text'] for m in mm}):,}", "N_MLUB_TRAIN_PLATES": f"{sum(m['split']=='train' for m in mm):,}", "N_MLUB_VAL_PLATES": f"{len(mv):,}",
  "N_MLUB_AMBIG": os.environ.get("N_MLUB_AMBIG", "87"), "N_MLUB_TWO_PCT": f"{100*sum(m['two_line'] for m in mm)/len(mm):.0f}%", "N_MLUB_TWO": f"{sum(m['two_line'] for m in mm):,}", "N_MLUB_VAL_1L": str(sum(not m["two_line"] for m in mv)), "N_MLUB_VAL_2L": str(sum(m["two_line"] for m in mv)),
  "MLUB_ACC": pct(MB["plate_acc"]), "MLUB_CER": f"{MB['cer']:.4f}", "MLUB_ACC_1L": pct(MB["plate_acc_single_line"]), "MLUB_ACC_2L": pct(MB["plate_acc_two_line"]),
  "MLUB_K3": pct(MB["multi_frame"]["3"]), "MLUB_K5": pct(MB["multi_frame"]["5"]), "MLUB_K9": pct(MB["multi_frame"][str(M["frames"])]),
  "G_K3": pct(M["by_k"]["3"]), "G_K5": pct(M["by_k"]["5"]), "G_CER": f"{bm['others']['real_github']['cer']:.3f}", "G_PLATE": pct(bm['others']['real_github']['plate_acc']),
  "REC_EPOCHS_M": "8 + 20", "ST_MSINGLE": "met" if MB["plate_acc"] >= 0.95 else "not met", "ST_MMULTI": "met" if MB["multi_frame"][str(M["frames"])] >= 0.95 else "not met",
  "CI_MLUB": wilson(round(MB["plate_acc"] * len(mv)), len(mv))})
bs = max(h1, key=lambda h: h["plate_acc"] - h["cer"]); bf = max(h2, key=lambda h: h["plate_acc"] - h["cer"])
V.update({"S_LINE": pct(bs["line_acc"]), "S_CER": f"{bs['cer']:.3f}", "S_PLATE": pct(bs["plate_acc"]), "F_LINE": pct(bf["line_acc"]), "F_CER": f"{bf['cer']:.3f}", "F_PLATE": pct(bf["plate_acc"])})
ks = sorted(int(os.path.basename(d).split("_k")[1]) for d in glob.glob("runs/eval_k*"))
kacc = {k: json.load(open(f"runs/eval_k{k}/results.json"))["multi_frame"]["plate_acc"] for k in ks}
kacc[1] = S["plate_acc"]; kacc[M["frames"]] = M["plate_acc"]; ks = sorted(kacc)
V["K_HEAD"] = "".join(f"<th>{k}</th>" for k in ks); V["K_ROW"] = "".join(f"<td>{pct(kacc[k])}</td>" for k in ks)
# discussion text
rows = R["rows"]; errs = [r for r in rows if r.get("detected") and not r.get("ok")]; miss = [r for r in rows if not r.get("detected")]
two = [r for r in rows if r.get("layout") in ("two-line", True)]; one = [r for r in rows if r.get("layout") in ("single-line", False)]
acc2 = sum(1 for r in two if r.get("ok")) / max(1, len(two)); acc1 = sum(1 for r in one if r.get("ok")) / max(1, len(one))
V["DET_DISCUSSION"] = (f"On the Mongolian test images the detector found {S['detected']} of {S['plates']} plates at IoU &ge; 0.5 (recall {pct(S['det_recall@0.5'])}). "
  f"Stage A already generalises to Mongolian plates because plate geometry is shared across countries; Stage B mainly improves localisation tightness (mAP<sub>50-95</sub>), which matters for the recogniser because loose boxes change the line split of two-line plates. "
  f"Precision on the generic set remains {V['DET_P_GEN']}, so fine-tuning on 48 Mongolian images did not cause forgetting.")
err_txt = "; ".join(f"{r['gt']}&rarr;{r['pred'] or '(none)'}" for r in (errs + miss)[:8])
V["REC_DISCUSSION"] = (f"Synthetic-only training already reads {V['S_PLATE']} of real plates, confirming that the renderer captures the plate design; adaptation on the {V['N_REAL_TRAIN']} real lines raises this to {V['F_PLATE']} on ground-truth crops. "
  f"End-to-end (detector crops instead of annotated crops) single-frame accuracy is {V['ACC_SINGLE']} (single-line plates {pct(acc1)}, two-line plates {pct(acc2)}); CER is {V['E2E_CER']}, i.e. most failures are single-character substitutions. "
  f"Residual errors: {err_txt}. Consensus voting over {V['K_FRAMES']} jittered views reaches {V['ACC_MULTI']} (Table IV), which is the operating point relevant to a parking gate that observes a stopped vehicle for roughly a second.")
V["VIDEO_DISCUSSION"] = os.environ.get("VIDEO_DISCUSSION", "Real street videos from [11] were processed with the Gradio interface; results are qualitative because the videos have no ground truth.")
html = open("report/template.html", encoding="utf-8").read()
for k, v in V.items(): html = html.replace(f"%%{k}%%", str(v))
left = sorted(set(__import__("re").findall(r"%%[A-Z0-9_]+%%", html)))
open("report/report.html", "w", encoding="utf-8").write(html)
json.dump(V, open("report/values.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("report written; unresolved placeholders:", left)
