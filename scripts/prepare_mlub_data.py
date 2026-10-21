"""Prepare the MLUB 'mongolian-car-plate-prediction' set (7,394 real plate crops, 470x110,
ground-truth text; found inside github.com/Tsundgerhen/mongolian_licence_plate).

All crops are stretched to 470x110, so two-line (square) plates are stretched too.  We label the
layout automatically: a plate is 'two-line' if reading its upper/lower halves separately reproduces
the ground truth while reading it as one line does not (and vice-versa); unresolved cases fall back
to a horizontal ink-profile test.  Lines are then written in the same format as real_github:
   single-line -> <stem>_0_full.jpg  "1234УБА"
   two-line    -> <stem>_0_top.jpg "1234", <stem>_0_bot.jpg "УБА"
Split 80/20 by plate identity (no plate string appears in both splits)."""
import os, sys, csv, json, random, re, glob
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.crnn import CRNN, preprocess
from mn_alpr.plate_rules import PLATE_RE

RAW = "data/raw/mlub"; OUT = "data/rec/real_mlub"; random.seed(0)
rows = {}
for f in ("training.csv", "test.csv"):
    for r in csv.DictReader(open(f"{RAW}/{f}", encoding="utf-8")):
        rows[r["file_name"]] = r["plate_number"].strip().upper()
items = []
for fn, txt in sorted(rows.items()):
    p = f"{RAW}/training/training/{fn}" if os.path.exists(f"{RAW}/training/training/{fn}") else f"{RAW}/test/test/{fn}"
    if not os.path.exists(p) or not PLATE_RE.match(txt): print("skip", fn, txt); continue
    items.append((fn, p, txt))
print("valid items", len(items))

dev = "cuda"; model = CRNN().to(dev).eval(); model.load_state_dict(torch.load("models/plate_rec_crnn.pt", map_location=dev))

def read(imgs):
    with torch.no_grad():
        x = torch.stack([preprocess(i) for i in imgs]).to(dev); return model.recognize(x)[0]

def ink_profile_two_line(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); g = cv2.resize(g, (235, 110))
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    prof = b[:, 20:-20].mean(1); prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 9), 0).ravel()
    mid = prof[45:65].mean(); top = prof[15:45].mean(); bot = prof[65:95].mean()
    return mid < 0.6 * min(top, bot) and min(top, bot) > 8

layout = {}; stats = {"single": 0, "two": 0, "ink_single": 0, "ink_two": 0}
B = 64
for i in range(0, len(items), B):
    batch = items[i:i + B]; imgs = [cv2.imread(p) for _, p, _ in batch]
    full = read(imgs); h = imgs[0].shape[0]
    tops = read([im[: int(im.shape[0] * 0.55)] for im in imgs]); bots = read([im[int(im.shape[0] * 0.45):] for im in imgs])
    for (fn, p, txt), im, f, t, b in zip(batch, imgs, full, tops, bots):
        s1 = f == txt; s2 = (t + b) == txt
        # Reading the two halves separately can only reproduce the digits/letters if the crop really
        # has two text rows; a one-line reading may also succeed on stretched two-line crops, so the
        # split test takes precedence.
        if s2: layout[fn] = True; stats["two"] += 1
        elif s1: layout[fn] = False; stats["single"] += 1
        else:
            # partial agreement: compare edit distances, else ink profile
            import editdistance
            d1 = editdistance.eval(f, txt); d2 = editdistance.eval(t + b, txt)
            if d1 < d2: layout[fn] = False; stats["single"] += 1
            elif d2 < d1: layout[fn] = True; stats["two"] += 1
            else:
                tl = ink_profile_two_line(im); layout[fn] = tl; stats["ink_two" if tl else "ink_single"] += 1
print("layout stats", stats)

# split by plate identity
plates = sorted({t for _, _, t in items}); random.shuffle(plates)
val_plates = set(plates[: int(0.2 * len(plates))])
for s in ("train", "val"): os.makedirs(f"{OUT}/{s}", exist_ok=True)
labels = {"train": [], "val": []}; meta = []
for fn, p, txt in items:
    s = "val" if txt in val_plates else "train"; stem = os.path.splitext(fn)[0]; im = cv2.imread(p); H = im.shape[0]
    if layout[fn]:
        cv2.imwrite(f"{OUT}/{s}/{stem}_0_top.jpg", im[: int(H * 0.55)], [cv2.IMWRITE_JPEG_QUALITY, 95]); labels[s].append(f"{s}/{stem}_0_top.jpg\t{txt[:4]}")
        cv2.imwrite(f"{OUT}/{s}/{stem}_0_bot.jpg", im[int(H * 0.45):], [cv2.IMWRITE_JPEG_QUALITY, 95]); labels[s].append(f"{s}/{stem}_0_bot.jpg\t{txt[4:]}")
    else:
        cv2.imwrite(f"{OUT}/{s}/{stem}_0_full.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95]); labels[s].append(f"{s}/{stem}_0_full.jpg\t{txt}")
    meta.append({"image": stem, "plate": 0, "text": txt, "two_line": bool(layout[fn]), "split": s, "src": p})
for s in ("train", "val"):
    open(f"{OUT}/{s}_labels.txt", "w", encoding="utf-8").write("\n".join(labels[s]) + "\n")
json.dump(meta, open(f"{OUT}/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
print("train lines", len(labels["train"]), "val lines", len(labels["val"]), "| plates train", sum(m["split"] == "train" for m in meta), "val", sum(m["split"] == "val" for m in meta), "| two-line", sum(m["two_line"] for m in meta))
