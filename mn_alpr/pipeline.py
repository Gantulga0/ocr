"""End-to-end Mongolian ALPR pipeline:
  YOLO11n plate detection -> layout split (single / two-line) -> CRNN+CTC recognition
  -> plate-rule normalisation -> (video) IoU tracking + confidence-weighted per-character voting.
"""
import time, os
import numpy as np, cv2, torch
from .crnn import CRNN, preprocess
from . import plate_rules as PR

TWO_LINE_AR = 1.8


def auto_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Recognizer:
    def __init__(self, weights, device=None, fp16=True):
        device = device or auto_device()
        self.device = device; self.model = CRNN().to(device).eval()
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.fp16 = fp16 and device == "cuda"
        if self.fp16: self.model.half()

    @torch.no_grad()
    def read_lines(self, crops):
        if not crops: return [], []
        x = torch.stack([preprocess(c) for c in crops]).to(self.device)
        if self.fp16: x = x.half()
        return self.model.recognize(x)

    def read_plate(self, crop):
        """Returns (text, conf, layout). Two-line plates are split into digit row / letter row."""
        h, w = crop.shape[:2]
        if w / max(1, h) < TWO_LINE_AR:
            top = crop[: int(h * 0.55)]; bot = crop[int(h * 0.45):]
            t, c = self.read_lines([top, bot])
            return PR.normalize(t[0] + t[1]), float(min(c)), "two-line"
        t, c = self.read_lines([crop])
        return PR.normalize(t[0]), float(c[0]), "single-line"


class PlateTrack:
    """Accumulates per-position character votes for one physical plate across frames."""
    def __init__(self, tid, box):
        self.id = tid; self.box = box; self.votes = {}; self.n = 0; self.missed = 0; self.history = []

    def add(self, text, conf, box):
        self.box = box; self.n += 1; self.missed = 0; self.history.append((text, conf))
        if not text: return
        w = conf * (0.5 + PR.plausibility(text))  # confidence x syntactic prior
        # vote per (length, position) so that 7-char hypotheses dominate
        self.votes.setdefault(len(text), [dict() for _ in range(len(text))])
        for i, ch in enumerate(text):
            self.votes[len(text)][i][ch] = self.votes[len(text)][i].get(ch, 0.0) + w

    def consensus(self):
        if not self.votes: return "", 0.0, 0
        L = max(self.votes, key=lambda l: sum(sum(d.values()) for d in self.votes[l]) * (1.3 if l == 7 else 1.0))
        chars, margins = [], []
        for d in self.votes[L]:
            items = sorted(d.items(), key=lambda kv: -kv[1]); tot = sum(d.values())
            chars.append(items[0][0]); margins.append(items[0][1] / tot if tot else 0)
        return "".join(chars), float(np.mean(margins)), self.n


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


class ALPR:
    def __init__(self, det_weights="models/plate_det_yolo11n.pt", rec_weights="models/plate_rec_crnn.pt", device=None, imgsz=640, conf=0.4):
        from ultralytics import YOLO
        device = device or auto_device()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(det_weights): det_weights = os.path.join(root, det_weights)
        if not os.path.isabs(rec_weights): rec_weights = os.path.join(root, rec_weights)
        self.det = YOLO(det_weights); self.rec = Recognizer(rec_weights, device)
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.half = device == "cuda"   # FP16 only on GPU; CPU laptops run in FP32
        self.tracks, self.finished, self.next_id = [], [], 1
        self.det.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, device=device, half=self.half, verbose=False)  # warm-up

    def detect(self, frame):
        r = self.det.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, half=self.half, verbose=False)[0]
        out = []
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append(([float(v) for v in b], float(c)))
        return out

    def process_frame(self, frame, track=False):
        """Returns list of dicts: box, det_conf, text, rec_conf, layout, valid, region, (track info)."""
        t0 = time.perf_counter(); dets = self.detect(frame); t1 = time.perf_counter()
        H, W = frame.shape[:2]; results = []
        for box, dc in dets:
            x1, y1, x2, y2 = box; mw, mh = 0.03 * (x2 - x1), 0.06 * (y2 - y1)
            crop = frame[max(0, int(y1 - mh)):min(H, int(y2 + mh)), max(0, int(x1 - mw)):min(W, int(x2 + mw))]
            if crop.size == 0: continue
            text, rc, layout = self.rec.read_plate(crop)
            results.append({"box": box, "det_conf": dc, "text": text, "rec_conf": rc, "layout": layout,
                            "valid": PR.is_valid(text), "region": PR.region_of(text), "pretty": PR.pretty(text)})
        t2 = time.perf_counter()
        if track: self._update_tracks(results)
        for r in results: r["t_det_ms"] = (t1 - t0) * 1e3; r["t_rec_ms"] = (t2 - t1) * 1e3
        return results, {"t_det_ms": (t1 - t0) * 1e3, "t_rec_ms": (t2 - t1) * 1e3, "t_total_ms": (time.perf_counter() - t0) * 1e3}

    def _update_tracks(self, results, iou_thr=0.3, max_missed=15):
        for t in self.tracks: t.missed += 1
        for r in results:
            best, bi = None, iou_thr
            for t in self.tracks:
                v = iou(t.box, r["box"])
                if v > bi: best, bi = t, v
            if best is None:
                best = PlateTrack(self.next_id, r["box"]); self.next_id += 1; self.tracks.append(best)
            best.add(r["text"], r["rec_conf"], r["box"])
            text, margin, n = best.consensus()
            r.update({"track_id": best.id, "voted": text, "voted_pretty": PR.pretty(text), "vote_margin": margin, "n_frames": n, "voted_valid": PR.is_valid(text)})
        self.finished += [t for t in self.tracks if t.missed > max_missed]
        self.tracks = [t for t in self.tracks if t.missed <= max_missed]

    def reset_tracks(self): self.tracks = []; self.finished = []; self.next_id = 1

    def all_tracks(self): return self.finished + self.tracks


def draw(frame, results, use_voted=True):
    out = frame.copy()
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        txt = r.get("voted_pretty") if use_voted and r.get("voted") else r["pretty"]
        ok = r.get("voted_valid") if use_voted and r.get("voted") else r["valid"]
        col = (0, 200, 0) if ok else (0, 140, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"{txt}  {r['rec_conf']:.2f}" + (f"  n={r['n_frames']}" if "n_frames" in r else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out
