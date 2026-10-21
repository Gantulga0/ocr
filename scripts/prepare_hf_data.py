"""Unzip keremberke/license-plate-object-detection (Roboflow 'Vehicle Registration Plates',
COCO format, 8,823 images) and convert to YOLO format for detector pre-training."""
import json, os, zipfile, glob

RAW = "data/raw/hf_keremberke"; OUT = "data/det/hf_plates"
for split, zname in (("train", "train.zip"), ("val", "valid.zip"), ("test", "test.zip")):
    d = f"{RAW}/{split}"
    if not os.path.isdir(d):
        with zipfile.ZipFile(f"{RAW}/{zname}") as z: z.extractall(d)
    ann = glob.glob(f"{d}/**/*.json", recursive=True)
    coco = json.load(open(ann[0]))
    imgs = {im["id"]: im for im in coco["images"]}
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True); os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
    per = {}
    for a in coco["annotations"]:
        im = imgs[a["image_id"]]; x, y, w, h = a["bbox"]; W, H = im["width"], im["height"]
        per.setdefault(im["file_name"], []).append(f"0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
    n = 0
    for fn, im in ((im["file_name"], im) for im in coco["images"]):
        src = os.path.join(os.path.dirname(ann[0]), fn)
        if not os.path.exists(src): continue
        dst = f"{OUT}/images/{split}/{fn}"
        if not os.path.exists(dst): os.link(src, dst)
        with open(f"{OUT}/labels/{split}/{os.path.splitext(fn)[0]}.txt", "w") as f:
            f.write("\n".join(per.get(fn, [])) + ("\n" if per.get(fn) else ""))
        n += 1
    print(split, n, "images", len(coco["annotations"]), "boxes", "categories", [c["name"] for c in coco["categories"]])
with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n")
