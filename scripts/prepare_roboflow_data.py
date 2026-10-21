"""Turn the Roboflow export (train only, 984 imgs) into data/det/roboflow with an 85/15 split."""
import glob, os, random, shutil
src = "data/raw/roboflow"; out = "data/det/roboflow"; random.seed(0)
imgs = sorted(glob.glob(f"{src}/**/images/*.*", recursive=True)); random.shuffle(imgs)
for i, im in enumerate(imgs):
    s = "val" if i < int(0.15 * len(imgs)) else "train"
    os.makedirs(f"{out}/images/{s}", exist_ok=True); os.makedirs(f"{out}/labels/{s}", exist_ok=True)
    shutil.copy(im, f"{out}/images/{s}/"); lb = im.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if os.path.exists(lb): shutil.copy(lb, f"{out}/labels/{s}/")
open(f"{out}/data.yaml", "w").write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n")
print(len(imgs), "images prepared ->", out)
