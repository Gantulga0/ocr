"""Train YOLO11n plate detector.
Stage A: pre-train on the generic Roboflow/HF 'Vehicle Registration Plates' set (8.8k imgs).
Stage B: fine-tune on Mongolian images (GitHub set, oversampled) + generic set, validate on Mongolian val.
"""
import argparse, os, shutil, glob
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all", choices=["A", "B", "all"])
ap.add_argument("--epochsA", type=int, default=30); ap.add_argument("--epochsB", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--rep", type=int, default=15, help="oversampling factor for Mongolian train images")
args = ap.parse_args()

ROOT = os.path.abspath(".")
GH = f"{ROOT}/data/det/github"; HF = f"{ROOT}/data/det/hf_plates"
# oversample Mongolian train set via hard links
REP = f"{ROOT}/data/det/github_rep"
if not os.path.isdir(REP):
    for sub in ("images/train", "labels/train"):
        os.makedirs(f"{REP}/{sub}", exist_ok=True)
    for img in glob.glob(f"{GH}/images/train/*.jpg"):
        stem = os.path.splitext(os.path.basename(img))[0]
        for k in range(args.rep):
            os.link(img, f"{REP}/images/train/{stem}_r{k}.jpg")
            os.link(f"{GH}/labels/train/{stem}.txt", f"{REP}/labels/train/{stem}_r{k}.txt")
with open(f"{ROOT}/configs/det_stageB.yaml", "w") as f:
    f.write(f"path: {ROOT}/data/det\ntrain: [hf_plates/images/train, github_rep/images/train]\nval: github/images/val\nnames:\n  0: plate\n")

common = dict(imgsz=args.imgsz, batch=args.batch, device=0, workers=8, project=f"{ROOT}/runs/det", exist_ok=True,
              cos_lr=True, close_mosaic=10, degrees=5, shear=3, perspective=0.0005, fliplr=0.0, mixup=0.1, plots=True)
if args.stage in ("A", "all"):
    m = YOLO("yolo11n.pt")
    m.train(data=f"{HF}/data.yaml", epochs=args.epochsA, name="stageA_generic", **common)
if args.stage in ("B", "all"):
    w = f"{ROOT}/runs/det/stageA_generic/weights/best.pt"
    m = YOLO(w if os.path.exists(w) else "yolo11n.pt")
    m.train(data=f"{ROOT}/configs/det_stageB.yaml", epochs=args.epochsB, name="stageB_mongolian", lr0=0.005, **common)
    best = f"{ROOT}/runs/det/stageB_mongolian/weights/best.pt"
    os.makedirs(f"{ROOT}/models", exist_ok=True); shutil.copy(best, f"{ROOT}/models/plate_det_yolo11n.pt")
    print("saved", f"{ROOT}/models/plate_det_yolo11n.pt")
