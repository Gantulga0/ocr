"""Convert gumbernator/Mongolian-ALPR Pascal-VOC labels (plate + per-character boxes)
into (a) YOLO detection labels and (b) recognition crops with Cyrillic text labels.

Two-line (square) plates are additionally split into a digit line and a letter line,
because the CRNN reads one text line at a time.
"""
import glob, os, random, json, sys, xml.etree.ElementTree as ET
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.charset import TRANSLIT

RAW = "data/raw/github_alpr"
DET = "data/det/github"
REC = "data/rec/real_github"
random.seed(0)
TWO_LINE_AR = 1.8  # plate w/h below this -> two-line plate


def union(boxes, pad=0.08):
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    w, h = x2 - x1, y2 - y1
    return [x1 - pad * w, y1 - pad * h * 2, x2 + pad * w, y2 + pad * h * 2]


def main():
    xmls = sorted(glob.glob(f"{RAW}/*.xml"))
    # Split by *plate identity* (same car photographed several times must not leak
    # from train into val).  Images sharing any plate text go to the same group.
    def plate_texts(x):
        r = ET.parse(x).getroot(); out = []
        for o in r.findall("object"):
            if o.find("name").text == "plate":
                b = o.find("bndbox"); pb = [int(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
                ch = []
                for o2 in r.findall("object"):
                    nm = o2.find("name").text
                    if nm == "plate": continue
                    b2 = o2.find("bndbox"); cb = [int(b2.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
                    if cb[0] >= pb[0] - 5 and cb[2] <= pb[2] + 5 and cb[1] >= pb[1] - 5 and cb[3] <= pb[3] + 5: ch.append((nm, cb))
                d = "".join(nm for nm, _ in sorted([c for c in ch if c[0].isdigit()], key=lambda t: t[1][0]))
                l = "".join(TRANSLIT[nm] for nm, _ in sorted([c for c in ch if not c[0].isdigit()], key=lambda t: t[1][0]))
                out.append(d + l)
        return out
    parent = {}
    def find(a):
        while parent.setdefault(a, a) != a: a = parent[a]
        return a
    for x in xmls:
        for t in plate_texts(x): parent[find(t)] = find(x)
    groups = {}
    for x in xmls: groups.setdefault(find(x), []).append(x)
    glist = list(groups.values()); random.shuffle(glist)
    split = {}; n_val = 0
    for g in glist:
        s = "val" if n_val < int(0.3 * len(xmls)) else "train"
        for x in g: split[x] = s
        if s == "val": n_val += len(g)
    for s in ("train", "val"):
        os.makedirs(f"{DET}/images/{s}", exist_ok=True); os.makedirs(f"{DET}/labels/{s}", exist_ok=True)
        os.makedirs(f"{REC}/{s}", exist_ok=True)
    rec_labels = {"train": [], "val": []}
    meta = []
    for x in sorted(xmls):
        s = split[x]; stem = os.path.basename(x)[:-4]
        r = ET.parse(x).getroot()
        W = int(r.find("size/width").text); H = int(r.find("size/height").text)
        img = Image.open(x[:-4] + ".jpg").convert("RGB")
        plates, chars = [], []
        for o in r.findall("object"):
            b = o.find("bndbox"); box = [int(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
            nm = o.find("name").text
            (plates if nm == "plate" else chars).append((nm, box))
        # YOLO det labels
        img.save(f"{DET}/images/{s}/{stem}.jpg", quality=95)
        with open(f"{DET}/labels/{s}/{stem}.txt", "w") as f:
            for _, (x1, y1, x2, y2) in plates:
                f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        # recognition crops: assign chars to plates
        for pi, (_, pb) in enumerate(plates):
            inside = [(nm, cb) for nm, cb in chars if cb[0] >= pb[0] - 5 and cb[2] <= pb[2] + 5 and cb[1] >= pb[1] - 5 and cb[3] <= pb[3] + 5]
            digits = sorted([(nm, cb) for nm, cb in inside if nm.isdigit()], key=lambda t: t[1][0])
            letters = sorted([(nm, cb) for nm, cb in inside if not nm.isdigit()], key=lambda t: t[1][0])
            text = "".join(nm for nm, _ in digits) + "".join(TRANSLIT[nm] for nm, _ in letters)
            if len(digits) != 4 or len(letters) != 3:
                print("skip", stem, text); continue
            ar = (pb[2] - pb[0]) / max(1, pb[3] - pb[1])
            two_line = ar < TWO_LINE_AR
            # whole-plate crop (with small margin)
            mw, mh = 0.04 * (pb[2] - pb[0]), 0.06 * (pb[3] - pb[1])
            crop = img.crop((pb[0] - mw, pb[1] - mh, pb[2] + mw, pb[3] + mh))
            name = f"{stem}_{pi}_full.jpg"; crop.save(f"{REC}/{s}/{name}", quality=95)
            if two_line:
                for tag, grp, txt in (("top", digits, text[:4]), ("bot", letters, text[4:])):
                    u = union([cb for _, cb in grp])
                    lc = img.crop(tuple(u)); ln = f"{stem}_{pi}_{tag}.jpg"; lc.save(f"{REC}/{s}/{ln}", quality=95)
                    rec_labels[s].append(f"{s}/{ln}\t{txt}")
            else:
                rec_labels[s].append(f"{s}/{name}\t{text}")
            meta.append({"image": stem, "plate": pi, "text": text, "aspect": round(ar, 2), "two_line": two_line, "split": s, "box": pb})
    for s in ("train", "val"):
        with open(f"{REC}/{s}_labels.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(rec_labels[s]) + "\n")
    with open(f"{REC}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    with open(f"{DET}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(DET)}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n")
    n2 = sum(m["two_line"] for m in meta)
    print(f"images: {len(xmls)} (train {len(xmls)-n_val}, val {n_val}); plates with full text: {len(meta)}; two-line: {n2}; single-line: {len(meta)-n2}")
    print("rec lines:", {s: len(v) for s, v in rec_labels.items()})


if __name__ == "__main__":
    main()
