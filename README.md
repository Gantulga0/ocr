# Монгол улсын дугаар таних систем (Mongolian ALPR)

YOLO11n илрүүлэгч → CRNN + CTC (PaddleOCR-маягийн) уншигч → олон кадрын саналын баталгаажуулалт.
Зогсоолын системд зориулсан суурь загвар. Тайлан: `report/report.html` (IEEE бүтэцтэй).

## Лаптоп дээр суулгаж турших (clone → run)
```bash
git clone <repo-url> mn-alpr && cd mn-alpr
conda create -n alpr python=3.12 -y && conda activate alpr
# PyTorch: GPU байвал cu128, байхгүй бол cpu wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # зөвхөн CPU
pip install -r requirements.txt
python app/gradio_app.py            # http://127.0.0.1:7860 → samples/ доторх зургуудаар турш
```
Сургасан жингүүд (`models/*.pt`) репод орсон тул дахин сургах шаардлагагүй. GPU байхгүй бол автоматаар CPU дээр ажиллана (нэг кадр ~60–200 мс).
Командын мөрөөс шууд турших:
```bash
python -c "from mn_alpr.pipeline import ALPR; import cv2; a=ALPR(); r,t=a.process_frame(cv2.imread('samples/sample_two_line.jpg')); print([x['pretty'] for x in r], t)"
```

## Орчин (энэ машин дээр)
```bash
conda activate alpr          # Python 3.12, torch 2.11 (cu128), ultralytics, gradio
```

## Өгөгдөл бэлтгэх (дахин сургах үед л хэрэгтэй; `data/` репод ороогүй)
```bash
python scripts/prepare_github_data.py     # gumbernator/Mongolian-ALPR (68 зураг, тэмдэгтийн хайрцагтай) -> det + rec
python scripts/prepare_hf_data.py         # keremberke/license-plate-object-detection (8,823 ерөнхий зураг) -> det
python scripts/synth_plates.py --n 60000  # синтетик монгол дугаар (нэг мөр + хоёр мөр)
# MLUB 7,392 бодит дугаарын зураг: github.com/Tsundgerhen/mongolian_licence_plate (scripts/mlub-mongolian-car-plate-prediction) -> data/raw/mlub, дараа нь:
python scripts/prepare_mlub_data.py
export ROBOFLOW_API_KEY=...; python scripts/download_roboflow.py && python scripts/prepare_roboflow_data.py   # сонголтоор (984 зураг)
```

## Сургалт
```bash
python scripts/train_det.py               # YOLO11n: A үе (ерөнхий) -> B үе (монгол) -> models/plate_det_yolo11n.pt
python scripts/train_rec.py               # CRNN+CTC -> runs/rec/crnn/best.pt -> models/plate_rec_crnn.pt
python scripts/eval_pipeline.py           # mAP, дугаарын нарийвчлал, CER, хугацаа, олон кадрын симуляци
python scripts/run_videos.py              # data/videos/*.mp4 дээр олон кадрын туршилт
python scripts/make_report.py             # report/report.html (IEEE бүтэц) үүсгэх
```

## Интерфейс
```bash
python app/gradio_app.py                  # http://127.0.0.1:7860  (зураг / бичлэг оруулах)
```

## Үр дүн (2026-10-14)
| Хэмжүүр | Утга |
|---|---|
| Илрүүлэлт mAP50 / mAP50-95 (монгол шалгалт, 18 зураг) | 0.992 / 0.834 |
| Бүтэн зөв уншилт, MLUB 1,491 бодит дугаар | 96.8% (CER 0.65%) |
| Төгсгөл хүртэлх, GitHub 19 бүтэн дүрстэй зураг | 73.7% (9 кадраар 78.9%) |
| Нэг кадр, 1080p, RTX 5080 (FP16) | 7.3 мс (p95 9.8 мс) |

## Бүтэц
- `mn_alpr/charset.py` — тэмдэгтийн олонлог (0-9 + 35 кирилл үсэг), CTC кодлол
- `mn_alpr/plate_rules.py` — `^\d{4}[А-ЯӨҮ]{3}$` дүрэм, аймгийн кодууд, зөв байх магадлал
- `mn_alpr/crnn.py` — CRNN загвар (ResNet-маягийн CNN + 2x BiLSTM + CTC), 48x192 оролт
- `mn_alpr/pipeline.py` — илрүүлэлт, нэг/хоёр мөрт хуваалт, таних, IoU мөрдөлт, итгэлээр жинлэсэн тэмдэгт тус бүрийн санал
