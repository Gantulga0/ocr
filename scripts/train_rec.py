"""Train the CRNN+CTC recognizer on synthetic + real Mongolian plate line crops.
Validation = real GitHub crops (val split): line accuracy, CER, and full-plate accuracy
(two-line plates are scored after joining their top+bottom lines)."""
import os, sys, random, argparse, json, time, math
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.charset import encode, CHARS, NUM_CLASSES
from mn_alpr.crnn import CRNN, preprocess, IMG_H, IMG_W
import editdistance


def read_labels(path, root):
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln: continue
        p, t = ln.split("\t"); out.append((os.path.join(root, p), t))
    return out


class LineDS(Dataset):
    def __init__(self, items, train=True):
        self.items, self.train = items, train

    def __len__(self): return len(self.items)

    def aug(self, a):
        h, w = a.shape[:2]
        if random.random() < 0.5:  # crop jitter
            dx0, dx1 = int(random.uniform(-0.05, 0.08) * w), int(random.uniform(-0.05, 0.08) * w)
            dy0, dy1 = int(random.uniform(-0.08, 0.12) * h), int(random.uniform(-0.08, 0.12) * h)
            a = cv2.copyMakeBorder(a, max(0, dy0), max(0, dy1), max(0, dx0), max(0, dx1), cv2.BORDER_REPLICATE)
            a = a[max(0, -dy0):a.shape[0] - max(0, -dy1), max(0, -dx0):a.shape[1] - max(0, -dx1)]
        if random.random() < 0.5:
            M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), random.uniform(-6, 6), random.uniform(0.9, 1.1))
            a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        a = a.astype(np.float32) * random.uniform(0.6, 1.3) + random.uniform(-30, 30)
        a = np.clip(a + np.random.normal(0, random.uniform(0, 8), a.shape), 0, 255).astype(np.uint8)
        if random.random() < 0.3: a = cv2.GaussianBlur(a, (3, 3), 0)
        return a

    def __getitem__(self, i):
        p, t = self.items[i]
        a = cv2.imread(p)
        if a is None: a = np.full((IMG_H, IMG_W, 3), 127, np.uint8)
        if self.train: a = self.aug(a)
        return preprocess(a), t


def collate(b):
    x = torch.stack([i[0] for i in b]); texts = [i[1] for i in b]
    tg = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
    return x, torch.cat(tg), torch.tensor([len(t) for t in tg]), texts


@torch.no_grad()
def evaluate(model, items, device, meta=None):
    model.eval(); dl = DataLoader(LineDS(items, False), batch_size=128, collate_fn=collate, num_workers=4)
    pred = {}; n_ok = 0; cer_n = 0; cer_d = 0
    for x, _, _, texts in dl:
        ts, cs = model.recognize(x.to(device))
        for (p, gt), pr, c in zip(items[len(pred):len(pred) + len(texts)], ts, cs):
            pred[p] = (pr, c); n_ok += pr == gt; cer_n += editdistance.eval(pr, gt); cer_d += len(gt)
    res = {"line_acc": n_ok / len(items), "cer": cer_n / max(1, cer_d)}
    if meta is not None:  # full-plate accuracy
        root = os.path.dirname(os.path.dirname(items[0][0]))
        ok = tot = 0
        for m in meta:
            if m["split"] != "val": continue
            s = m["split"]; stem = f"{m['image']}_{m['plate']}"
            if m["two_line"]:
                pr = pred.get(f"{root}/{s}/{stem}_top.jpg", ("",))[0] + pred.get(f"{root}/{s}/{stem}_bot.jpg", ("",))[0]
            else:
                pr = pred.get(f"{root}/{s}/{stem}_full.jpg", ("",))[0]
            ok += pr == m["text"]; tot += 1
        res["plate_acc"] = ok / max(1, tot); res["n_plates"] = tot
    model.train(); return res, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/rec/synth"); ap.add_argument("--real", nargs="+", default=["data/rec/real_github"], help="real sets (first = primary for model selection)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra label files (root/labels.txt) to add to train")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--real_rep", type=int, default=40)
    ap.add_argument("--out", default="runs/rec/crnn"); ap.add_argument("--resume", default=None)
    ap.add_argument("--no_synth", action="store_true"); ap.add_argument("--min_rep", type=int, default=1, help="minimum repetition of each real set")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = "cuda"
    synth = [] if args.no_synth else read_labels(f"{args.synth}/labels.txt", args.synth)
    reals = []  # (name, train_items, val_items, meta, rep)
    for k, root in enumerate(args.real):
        tr = read_labels(f"{root}/train_labels.txt", root); va = read_labels(f"{root}/val_labels.txt", root)
        meta = json.load(open(f"{root}/meta.json", encoding="utf-8"))
        # repeat small sets more: target ~ real_rep x 74 lines each, capped
        rep = max(args.min_rep, min(args.real_rep, int(args.real_rep * 74 / max(1, len(tr))))) if args.real_rep > 0 else 0
        reals.append((os.path.basename(root), tr, va, meta, rep))
    extra = []
    for e in args.extra: extra += read_labels(e, os.path.dirname(e))
    real_va, meta = reals[0][2], reals[0][3]
    train_items = synth + sum([tr * rep for _, tr, _, _, rep in reals], []) + extra * max(1, args.real_rep // 4)
    print("train lines: synth", len(synth), "+", " + ".join(f"{n} {len(tr)}x{rep}" for n, tr, _, _, rep in reals), "+ extra", len(extra), "=", len(train_items), "; val lines", {n: len(va) for n, _, va, _, _ in reals})
    dl = DataLoader(LineDS(train_items, True), batch_size=args.bs, shuffle=True, num_workers=10, collate_fn=collate, drop_last=True, persistent_workers=True)
    model = CRNN().to(device)
    if args.resume: model.load_state_dict(torch.load(args.resume, map_location=device))
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(dl); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True); scaler = torch.amp.GradScaler()
    best = -1; hist = []
    for ep in range(args.epochs):
        t0 = time.time(); tot = 0; n = 0
        for x, tg, tl, _ in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            lp = logits.float().log_softmax(-1).permute(1, 0, 2)  # T,B,C
            il = torch.full((x.size(0),), lp.size(0), dtype=torch.long)
            loss = ctc(lp, tg, il, tl)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            tot += loss.item(); n += 1
        res, _ = evaluate(model, real_va, device, meta)
        others = {n: evaluate(model, va, device, m)[0] for n, _, va, m, _ in reals[1:]}
        hist.append({"epoch": ep + 1, "loss": tot / n, **res, "others": others, "time": time.time() - t0})
        print(f"ep {ep+1}/{args.epochs} loss {tot/n:.4f} | {reals[0][0]} val line_acc {res['line_acc']:.3f} cer {res['cer']:.4f} plate_acc {res['plate_acc']:.3f} ({res['n_plates']})"
              + "".join(f" | {n}: plate_acc {o['plate_acc']:.3f} cer {o['cer']:.4f} ({o['n_plates']})" for n, o in others.items()) + f" | {time.time()-t0:.0f}s", flush=True)
        score = res["plate_acc"] - res["cer"]
        if score >= best:
            best = score; torch.save(model.state_dict(), f"{args.out}/best.pt")
        torch.save(model.state_dict(), f"{args.out}/last.pt")
        json.dump(hist, open(f"{args.out}/history.json", "w"), indent=1)
    model.load_state_dict(torch.load(f"{args.out}/best.pt")); res, pred = evaluate(model, real_va, device, meta)
    print("BEST", reals[0][0], res, {n: evaluate(model, va, device, m)[0] for n, _, va, m, _ in reals[1:]})
    with open(f"{args.out}/val_predictions.txt", "w", encoding="utf-8") as f:
        for p, gt in real_va: f.write(f"{os.path.basename(p)}\t{gt}\t{pred[p][0]}\t{pred[p][1]:.3f}\n")


if __name__ == "__main__":
    main()
