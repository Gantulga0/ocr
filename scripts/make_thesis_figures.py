"""Generate all figures (Mongolian labels) for the LaTeX progress report -> report/latex/figures/."""
import os, sys, json, glob, random, collections, shutil, re
import numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.charset import CHARS
from mn_alpr.plate_rules import REGION_CODES
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 170, "savefig.dpi": 170, "savefig.bbox": "tight"})
OUT = "report/latex/figures"; os.makedirs(OUT, exist_ok=True); random.seed(0); np.random.seed(0)
R = json.load(open("runs/eval/results.json", encoding="utf-8"))
gh_meta = json.load(open("data/rec/real_github/meta.json", encoding="utf-8"))
ml_meta = json.load(open("data/rec/real_mlub/meta.json", encoding="utf-8"))
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

def save(name): plt.savefig(f"{OUT}/{name}.png"); plt.close(); print("fig", name)

# ---------------- 1. dataset overview ----------------
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
src = ["GitHub\n(бүтэн дүрс)", "MLUB\n(тайралт)", "HuggingFace\n(ерөнхий)", "Roboflow\n(татаагүй)", "Синтетик\n(мөр)"]
n = [68, 7392, 8823, 984, 90021]
b = ax[0].bar(src, n, color=C[:5]); ax[0].set_yscale("log"); ax[0].set_ylabel("зургийн тоо (log)"); ax[0].set_title("Эх үүсвэр тус бүрийн хэмжээ")
for r, v in zip(b, n): ax[0].text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v:,}", ha="center", fontsize=7)
ax[0].tick_params(axis="x", labelsize=7)
# region codes (MLUB)
reg = collections.Counter(m["text"][4:6] for m in ml_meta); top = reg.most_common(12)
ax[1].bar([k for k, _ in top], [v for _, v in top], color=C[0]); ax[1].set_title("Аймгийн код (MLUB, 7,392)"); ax[1].set_ylabel("тоо"); ax[1].tick_params(axis="x", labelsize=7)
# letter frequency (MLUB) all 3 letters
let = collections.Counter(c for m in ml_meta for c in m["text"][4:])
ks = [c for c in CHARS[10:] if let[c] > 0]; ax[2].bar(ks, [let[c] for c in ks], color=C[2]); ax[2].set_yscale("log"); ax[2].set_title("Үсгийн давтамж (MLUB)"); ax[2].tick_params(axis="x", labelsize=7)
save("dataset_overview")

# ---------------- 2. aspect ratio & layout ----------------
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
ars = [m["aspect"] for m in gh_meta]
ax[0].hist(ars, bins=20, color=C[0]); ax[0].axvline(1.8, color="r", ls="--", label="босго 1.8"); ax[0].set_xlabel("өргөн / өндөр"); ax[0].set_ylabel("дугаарын тоо"); ax[0].set_title("GitHub: дугаарын хайрцгийн харьцаа"); ax[0].legend()
two = sum(m["two_line"] for m in ml_meta); one = len(ml_meta) - two
ax[1].bar(["нэг мөрт", "хоёр мөрт"], [one, two], color=[C[0], C[1]]); ax[1].set_title("MLUB: автоматаар шошголсон хэлбэр")
for i, v in enumerate([one, two]): ax[1].text(i, v + 50, f"{v:,} ({100*v/len(ml_meta):.0f}%)", ha="center")
save("aspect_layout")

# ---------------- 3. splits ----------------
fig, ax = plt.subplots(figsize=(7, 2.6))
rows = [("GitHub зураг", 48, 18), ("GitHub дугаар", 53, 19), ("MLUB дугаар", 5901, 1491), ("MLUB мөр", 8094, 2055)]
y = np.arange(len(rows)); tr = [r[1] for r in rows]; va = [r[2] for r in rows]
ax.barh(y, tr, color=C[0], label="сургалт"); ax.barh(y, va, left=tr, color=C[1], label="шалгалт/тест")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xscale("log"); ax.legend(); ax.set_title("Хуваалт (дугаарын текстээр, давхцалгүй)")
for i, (a, b2) in enumerate(zip(tr, va)): ax.text(a + b2, i, f"  {a} / {b2}", va="center", fontsize=8)
save("splits")

# ---------------- 4. real & synthetic samples ----------------
imgs = sorted(glob.glob("data/raw/github_alpr/*.jpg")); pick = [imgs[i] for i in (1, 9, 27, 36)]
row = np.hstack([cv2.resize(cv2.imread(p), (480, 270)) for p in pick])
crops = []
for m in [m for m in gh_meta if m["split"] == "val"][:8]:
    im = cv2.imread(f"data/raw/github_alpr/{m['image']}.jpg"); b = m["box"]; crops.append(cv2.resize(im[b[1]:b[3], b[0]:b[2]], (240, 100)))
mcs = sorted(glob.glob("data/raw/mlub/training/training/*.png")); random.shuffle(mcs)
row3 = np.hstack([cv2.resize(cv2.imread(p), (240, 100)) for p in mcs[:8]])
cv2.imwrite(f"{OUT}/samples_real.jpg", np.vstack([row, np.hstack(crops), row3]), [cv2.IMWRITE_JPEG_QUALITY, 88])
sl = [l.split("\t") for l in open("data/rec/synth/labels.txt", encoding="utf-8").read().splitlines()[:5000]]; random.shuffle(sl)
tiles = [cv2.resize(cv2.imread(f"data/rec/synth/{p}"), (240, 80)) for p, _ in sl[:30]]
cv2.imwrite(f"{OUT}/samples_synth.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, 30, 6)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
# synthetic clean (pre-augmentation) examples: re-render a few
sys.path.insert(0, "scripts"); import synth_plates as SP
SP.random.seed(3); clean = []
for i in range(6):
    d, l = SP.rand_text(); fp = random.choice(SP.FONT_FILES)
    im = SP.render_single(d, l, fp) if i % 2 == 0 else SP.render_two_line(d, l, fp)[0]
    a = np.array(im)[:, :, ::-1]; h, w = a.shape[:2]; clean.append(cv2.resize(a, (int(w * 120 / h), 120)))
canvas = np.full((2 * 130, 3 * 600, 3), 255, np.uint8)
for i, c in enumerate(clean):
    r_, c_ = divmod(i, 3); canvas[r_ * 130 + 5:r_ * 130 + 125, c_ * 600 + 5:c_ * 600 + 5 + c.shape[1]] = c[:, :595]
cv2.imwrite(f"{OUT}/samples_synth_clean.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("fig samples")

# ---------------- 5. detector curves ----------------
dA = pd.read_csv("runs/det/stageA_generic/results.csv"); dB = pd.read_csv("runs/det/stageB_mongolian/results.csv")
for d in (dA, dB): d.columns = [c.strip() for c in d.columns]
fig, ax = plt.subplots(2, 3, figsize=(11, 5.5))
def plot_stage(d, off, lab, col):
    e = d["epoch"] + off
    ax[0, 0].plot(e, d["train/box_loss"], color=col, label=f"{lab} сургалт"); ax[0, 0].plot(e, d["val/box_loss"], color=col, ls="--", label=f"{lab} шалгалт")
    ax[0, 1].plot(e, d["train/cls_loss"], color=col, label=f"{lab} сургалт"); ax[0, 1].plot(e, d["val/cls_loss"], color=col, ls="--", label=f"{lab} шалгалт")
    ax[0, 2].plot(e, d["train/dfl_loss"], color=col, label=f"{lab} сургалт"); ax[0, 2].plot(e, d["val/dfl_loss"], color=col, ls="--", label=f"{lab} шалгалт")
    ax[1, 0].plot(e, d["metrics/precision(B)"], color=col, label=f"{lab} P"); ax[1, 0].plot(e, d["metrics/recall(B)"], color=col, ls="--", label=f"{lab} R")
    ax[1, 1].plot(e, d["metrics/mAP50(B)"], color=col, label=lab); ax[1, 2].plot(e, d["metrics/mAP50-95(B)"], color=col, label=lab)
plot_stage(dA, 0, "A үе (ерөнхий)", C[0]); plot_stage(dB, len(dA), "B үе (монгол)", C[1])
titles = ["Box loss", "Cls loss", "DFL loss", "Precision / Recall (шалгалт)", "mAP50 (шалгалт)", "mAP50-95 (шалгалт)"]
for a, t in zip(ax.ravel(), titles): a.set_title(t); a.set_xlabel("эпох"); a.axvline(len(dA) + 0.5, color="gray", ls=":"); a.legend(fontsize=6)
ax[1, 1].set_ylim(0.6, 1.0); ax[1, 2].set_ylim(0.4, 0.9)
plt.tight_layout(); save("det_curves")
for f in ("BoxPR_curve", "BoxF1_curve", "confusion_matrix_normalized", "val_batch0_pred", "labels"):
    shutil.copy(f"runs/det/stageB_mongolian/{f}.png" if f != "labels" and f != "val_batch0_pred" else f"runs/det/stageB_mongolian/{f}.jpg", f"{OUT}/det_{f}.{'png' if f not in ('labels','val_batch0_pred') else 'jpg'}")

# ---------------- 6. recogniser curves (train loss, val MLUB, test GitHub) ----------------
h1 = json.load(open("runs/rec/synth_only/history.json")); h2 = json.load(open("runs/rec/crnn/history.json"))
h3 = json.load(open("runs/rec/crnn_mlub_r1/history.json")); h4 = json.load(open("runs/rec/crnn_mlub/history.json"))
H = h1 + h2 + h3 + h4; ep = np.arange(1, len(H) + 1); b1, b2, b3 = len(h1), len(h1) + len(h2), len(h1) + len(h2) + len(h3)
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
ax[0].plot(ep, [h["loss"] for h in H], color=C[0]); ax[0].set_yscale("log"); ax[0].set_title("Сургалтын CTC loss"); ax[0].set_ylabel("loss")
gh_acc = [h["plate_acc"] for h in h1 + h2] + [h["others"]["real_github"]["plate_acc"] for h in h3 + h4]
gh_cer = [h["cer"] for h in h1 + h2] + [h["others"]["real_github"]["cer"] for h in h3 + h4]
ml_acc = [None] * b2 + [h["plate_acc"] for h in h3 + h4]; ml_cer = [None] * b2 + [h["cer"] for h in h3 + h4]
ax[1].plot(ep, [100 * v for v in gh_acc], color=C[1], label="GitHub (19 дугаар)"); ax[1].plot(ep[b2:], [100 * v for v in ml_acc[b2:]], color=C[2], label="MLUB (1,491 дугаар)"); ax[1].set_title("Бүтэн дугаарын нарийвчлал, %"); ax[1].legend(fontsize=7)
ax[2].plot(ep, gh_cer, color=C[1], label="GitHub"); ax[2].plot(ep[b2:], ml_cer[b2:], color=C[2], label="MLUB"); ax[2].set_yscale("log"); ax[2].set_title("CER"); ax[2].legend(fontsize=7)
for a in ax:
    a.set_xlabel("эпох")
    for b_ in (b1, b2, b3): a.axvline(b_ + 0.5, color="gray", ls=":")
plt.tight_layout(); save("rec_curves")

# ---------------- 7. character confusion matrix (MLUB test, final model) ----------------
import difflib
def align(gt, pr):
    pairs = []; sm = difflib.SequenceMatcher(None, gt, pr, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs += [(gt[k], gt[k]) for k in range(i1, i2)]
        elif tag == "replace":
            g, p = gt[i1:i2], pr[j1:j2]
            for k in range(max(len(g), len(p))): pairs.append((g[k] if k < len(g) else "∅", p[k] if k < len(p) else "∅"))
        elif tag == "delete": pairs += [(gt[k], "∅") for k in range(i1, i2)]
        elif tag == "insert": pairs += [("∅", pr[k]) for k in range(j1, j2)]
    return pairs
def confusion(rows, name, title):
    pairs = [p for r in rows for p in align(r["gt"], r["pred"])]
    labels = [c for c in CHARS if any(p[0] == c for p in pairs)] + ["∅"]
    idx = {c: i for i, c in enumerate(labels)}; M = np.zeros((len(labels), len(labels)))
    for g, p in pairs:
        if g in idx and p in idx: M[idx[g], idx[p]] += 1
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8)); im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("таамагласан"); ax.set_ylabel("үнэн"); ax.set_title(title); ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if M[i, j] > 0 and i != j: ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=5, color="red")
            elif i == j: ax.text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center", fontsize=5, color="white" if Mn[i, j] > 0.5 else "black")
    plt.colorbar(im, fraction=0.03); save(name)
    # per-class recall
    rec = {c: (M[idx[c], idx[c]] / M[idx[c]].sum(), int(M[idx[c]].sum())) for c in labels[:-1] if M[idx[c]].sum() > 0}
    return rec, pairs
rec_ml, pairs_ml = confusion(R["mlub_rows"], "confusion_mlub", "Тэмдэгтийн андуурлын матриц, MLUB тест (1,491 дугаар), мөрөөр нормчилсон")
rec_gh, _ = confusion([r for r in R["rows"] if r.get("detected")], "confusion_github", "Тэмдэгтийн андуурлын матриц, GitHub тест (төгсгөл хүртэлх, 18 дугаар)")
# per-char accuracy bar
fig, ax = plt.subplots(figsize=(10, 3)); ks = list(rec_ml); vals = [100 * rec_ml[k][0] for k in ks]; sup = [rec_ml[k][1] for k in ks]
bars = ax.bar(ks, vals, color=[C[0] if k.isdigit() else C[2] for k in ks]); ax.set_ylim(80, 101); ax.set_ylabel("recall, %"); ax.set_title("Тэмдэгт тус бүрийн recall, MLUB тест (дээр нь дээжийн тоо)")
for b_, s_ in zip(bars, sup): ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.3, str(s_), ha="center", fontsize=6, rotation=90)
save("perchar_mlub")
# top confusions table (json for LaTeX)
conf = collections.Counter((g, p) for g, p in pairs_ml if g != p)
json.dump({"top_confusions": conf.most_common(15), "per_char": {k: [v[0], v[1]] for k, v in rec_ml.items()}}, open(f"{OUT}/confusions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------- 8. ablation bars ----------------
fig, ax = plt.subplots(figsize=(6, 3))
names = ["Зөвхөн\nсинтетик", "+ GitHub\n74 мөр", "+ MLUB\n(1-р үе)", "+ MLUB\n(2-р үе, эцсийн)"]
gh = [42.1, 84.2, 84.2, 89.5]; ml = [None, None, 91.9, 96.8]
x = np.arange(4); ax.bar(x - 0.2, gh, 0.4, color=C[1], label="GitHub тест (19)"); ax.bar(x[2:] + 0.2, ml[2:], 0.4, color=C[2], label="MLUB тест (1,491)")
for i, v in enumerate(gh): ax.text(i - 0.2, v + 1, f"{v}", ha="center", fontsize=7)
for i, v in enumerate(ml):
    if v: ax.text(i + 0.2, v + 1, f"{v}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel("бүтэн дугаар зөв, %"); ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.set_title("Сургалтын өгөгдлийн нөлөө (ablation)")
save("ablation")

# ---------------- 9. multi-frame K ----------------
fig, ax = plt.subplots(figsize=(5, 3)); K = [1, 3, 5, 9]
ml_k = [100 * R["mlub"]["plate_acc"]] + [100 * R["mlub"]["multi_frame"][str(k)] for k in (3, 5, 9)]
gh_k = [100 * R["single_frame"]["plate_acc"]] + [100 * R["multi_frame"]["by_k"][str(k)] for k in (3, 5, 9)]
ax.plot(K, ml_k, "o-", color=C[2], label="MLUB тайралт (1,491)"); ax.plot(K, gh_k, "s-", color=C[1], label="GitHub төгсгөл хүртэлх (19)")
ax.set_xlabel("санал авсан кадрын тоо K"); ax.set_ylabel("бүтэн дугаар зөв, %"); ax.set_title("Олон кадрын саналын нөлөө (симуляци)"); ax.legend(fontsize=7); ax.set_xticks(K)
save("multiframe_k")

# ---------------- 10. latency ----------------
L = R["latency"]; fig, ax = plt.subplots(figsize=(5, 2.6))
ax.bar(["илрүүлэгч", "уншигч+дүрэм", "нийт (дундаж)", "нийт p95", "нийт max"], [L["det_mean_ms"], L["rec_mean_ms"], L["mean_ms"], L["p95_ms"], L["max_ms"]], color=[C[0], C[2], C[1], C[1], C[1]])
ax.axhline(100, color="r", ls="--", label="зорилт 100 мс"); ax.set_ylabel("мс / кадр (1080p)"); ax.set_yscale("log"); ax.legend(fontsize=7); ax.set_title("Хугацаа, RTX 5080, FP16"); ax.tick_params(axis="x", labelsize=7)
for i, v in enumerate([L["det_mean_ms"], L["rec_mean_ms"], L["mean_ms"], L["p95_ms"], L["max_ms"]]): ax.text(i, v * 1.1, f"{v:.1f}", ha="center", fontsize=7)
save("latency")

# ---------------- 11. video tracks ----------------
S = json.load(open("runs/videos/summary.json", encoding="utf-8")); fig, ax = plt.subplots(figsize=(5.5, 3.2))
for (name, v), col, mk in zip(S.items(), C, "os^"):
    ax.scatter([t["single_frame_agreement"] for t in v["tracks"]], [t["margin"] for t in v["tracks"]], c=[col if t["valid"] else "red" for t in v["tracks"]], marker=mk, label=name, s=[20 + t["frames"] for t in v["tracks"]], alpha=0.8)
ax.axhline(0.8, color="gray", ls="--"); ax.set_xlabel("нэг кадрын санал нийцэл"); ax.set_ylabel("санал нэгдлийн margin"); ax.set_title("Бичлэгийн мөрүүд (улаан = дүрэмд нийцээгүй; хэмжээ = кадрын тоо)"); ax.legend(fontsize=7)
save("video_tracks")

# ---------------- 12. error examples (MLUB) ----------------
errs = [r for r in R["mlub_rows"] if not r["ok"]]; random.shuffle(errs); src = {m["image"]: m["src"] for m in ml_meta}
tiles = []
for r in errs[:12]:
    im = cv2.resize(cv2.imread(src[r["image"]]), (470, 110)); pad = np.full((40, 470, 3), 255, np.uint8); tiles.append(np.vstack([im, pad]))
sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 12, 3)])
from PIL import Image, ImageDraw, ImageFont
pil = Image.fromarray(sheet[:, :, ::-1]); d = ImageDraw.Draw(pil); f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
for i, r in enumerate(errs[:12]):
    rr, cc = divmod(i, 3); d.text((cc * 470 + 8, rr * 150 + 115), f"үнэн {r['gt']}   таамаг {r['pred']}", fill=(200, 0, 0), font=f)
pil.save(f"{OUT}/errors_mlub.jpg", quality=88); print("fig errors")
shutil.copy("report/figures/fig6_qualitative.jpg", f"{OUT}/qualitative_github.jpg"); shutil.copy("report/figures/fig7_video.jpg", f"{OUT}/video_frames.jpg")
print("ALL FIGURES DONE")
