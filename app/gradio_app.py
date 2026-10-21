"""Minimal test UI: upload an image or a video -> detected plates, recognized text,
multi-frame voted result (video), per-frame latency.
Run:  python app/gradio_app.py   (http://127.0.0.1:7860)
"""
import os, sys, time, tempfile
import cv2, numpy as np, gradio as gr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.pipeline import ALPR, draw
from mn_alpr import plate_rules as PR

alpr = ALPR()


def run_image(img):
    if img is None: return None, "", ""
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res, tm = alpr.process_frame(frame, track=False)
    rows = [[r["pretty"], r["layout"], f"{r['det_conf']:.2f}", f"{r['rec_conf']:.2f}", "✓" if r["valid"] else "✗", r["region"] or ""] for r in res]
    txt = "\n".join(f"{r['pretty']}  ({r['region'] or 'тодорхойгүй бүс'})" for r in res) or "Дугаар олдсонгүй"
    return cv2.cvtColor(draw(frame, res), cv2.COLOR_BGR2RGB), txt, f"илрүүлэлт {tm['t_det_ms']:.1f} ms + таних {tm['t_rec_ms']:.1f} ms = {tm['t_total_ms']:.1f} ms"


def run_video(path, every_n=2, max_frames=600, progress=gr.Progress()):
    if path is None: return None, "", ""
    alpr.reset_tracks(); cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25; W = int(cap.get(3)); H = int(cap.get(4)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = os.path.join(tempfile.mkdtemp(), "out.mp4")
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / every_n, (W, H))
    i = 0; times = []; last = []
    while True:
        ok, frame = cap.read()
        if not ok or i >= max_frames * every_n: break
        if i % every_n == 0:
            res, tm = alpr.process_frame(frame, track=True); times.append(tm["t_total_ms"]); last = res
            wr.write(draw(frame, res, use_voted=True))
            progress(min(1.0, i / max(1, min(n, max_frames * every_n))))
        i += 1
    cap.release(); wr.release()
    lines = []
    for t in sorted(alpr.all_tracks(), key=lambda t: -t.n):
        text, margin, nf = t.consensus()
        if nf < 2 or not text: continue
        lines.append(f"Track {t.id}: {PR.pretty(text)}  | кадр {nf} | санал нэгдэл {margin:.2f} | {'хүчинтэй' if PR.is_valid(text) else 'формат буруу'} | {PR.region_of(text) or ''}")
    stats = f"боловсруулсан кадр {len(times)}, дундаж {np.mean(times):.1f} ms/кадр, p95 {np.percentile(times,95):.1f} ms" if times else ""
    return out_path, "\n".join(lines) or "Дугаар олдсонгүй", stats


with gr.Blocks(title="Монгол улсын дугаар таних (YOLO11n + CRNN-CTC)") as demo:
    gr.Markdown("## Монгол улсын дугаар таних — туршилтын интерфейс\nYOLO11n илрүүлэгч → CRNN+CTC уншигч → олон кадрын саналын баталгаажуулалт")
    with gr.Tab("Зураг"):
        with gr.Row():
            inp = gr.Image(type="numpy", label="Зураг"); out = gr.Image(label="Үр дүн")
        txt = gr.Textbox(label="Уншсан дугаар"); lat = gr.Textbox(label="Хугацаа")
        gr.Button("Таних").click(run_image, inp, [out, txt, lat])
    with gr.Tab("Бичлэг"):
        with gr.Row():
            vin = gr.Video(label="Бичлэг"); vout = gr.Video(label="Үр дүн")
        every = gr.Slider(1, 5, 2, step=1, label="Кадр алгасалт (every N)")
        vtxt = gr.Textbox(label="Олон кадрын баталгаажсан дугаарууд", lines=6); vstat = gr.Textbox(label="Статистик")
        gr.Button("Боловсруулах").click(run_video, [vin, every], [vout, vtxt, vstat])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
