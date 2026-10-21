"""Synthetic Mongolian plate generator for CRNN pre-training.

Renders single-line (520x110) and two-line (square) plates with the red Soyombo
emblem, the MNG oval, several DIN/Roboto-like fonts, then applies heavy photometric
and geometric augmentation.  Emits text-line crops + labels:
  full single-line plate  -> "1234УБА"
  two-line plate top line -> "1234"
  two-line plate bottom   -> "УБА"
"""
import os, sys, random, glob, json, argparse
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.charset import LETTERS
from mn_alpr.plate_rules import REGION_CODES, SERIAL_LETTERS

FONT_DIR = "assets/fonts"
FONT_FILES = [f for f in glob.glob(f"{FONT_DIR}/*.ttf")]
REGION_W = {"УБ": 40, "УН": 25, "УА": 6, "УЕ": 4, "УК": 3}
for k in REGION_CODES:
    REGION_W.setdefault(k, 1.2)
REG_KEYS, REG_P = list(REGION_W), np.array(list(REGION_W.values()), float)
REG_P /= REG_P.sum()


def load_font(path, size):
    f = ImageFont.truetype(path, size)
    try:  # variable fonts: pick a bold-ish instance
        axes = f.get_variation_axes()
        vals = []
        for a in axes:
            if a["name"] in (b"Weight", "Weight", b"wght"):
                vals.append(random.choice([500, 600, 700]))
            elif a["name"] in (b"Width", "Width", b"wdth"):
                vals.append(random.choice([75, 87, 100]))
            else:
                vals.append(a["default"])
        f.set_variation_by_axes(vals)
    except Exception:
        pass
    return f


def rand_text():
    digits = "".join(random.choice("0123456789") for _ in range(4))
    if random.random() < 0.9:
        reg = np.random.choice(REG_KEYS, p=REG_P)
    else:
        reg = random.choice(LETTERS) + random.choice(LETTERS)
    third = random.choice(SERIAL_LETTERS) if random.random() < 0.9 else random.choice(LETTERS)
    return digits, reg + third


def draw_soyombo(d, x, y, h, color=(200, 20, 30)):
    """Stylised red emblem (a column of flame/sun/moon/bars) standing in for the Soyombo."""
    w = h * 0.42
    d.polygon([(x + w / 2, y), (x + w * 0.85, y + h * 0.18), (x + w / 2, y + h * 0.3), (x + w * 0.15, y + h * 0.18)], fill=color)
    d.ellipse([x + w * 0.2, y + h * 0.3, x + w * 0.8, y + h * 0.45], fill=color)
    d.rectangle([x + w * 0.1, y + h * 0.5, x + w * 0.9, y + h * 0.56], fill=color)
    d.polygon([(x + w * 0.5, y + h * 0.58), (x + w * 0.9, y + h * 0.75), (x + w * 0.1, y + h * 0.75)], fill=color)
    d.rectangle([x + w * 0.1, y + h * 0.78, x + w * 0.9, y + h * 0.84], fill=color)
    d.rectangle([x, y + h * 0.87, x + w, y + h], fill=color)
    return w


def render_single(digits, letters, font_path):
    S = 4; W, H = 520 * S // 2, 110 * S // 2  # 1040 x 220
    bg = tuple(random.randint(225, 255) for _ in range(3))
    im = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(im)
    bw = random.randint(4, 10)
    d.rounded_rectangle([bw // 2, bw // 2, W - bw // 2, H - bw // 2], radius=12, outline=(10, 10, 10), width=bw)
    fs = int(H * random.uniform(0.62, 0.75)); font = load_font(font_path, fs)
    style = random.random()
    x = int(W * 0.05)
    if style < 0.55:  # 2001+ design: soyombo, digits, MNG oval, letters
        sw = draw_soyombo(d, x, int(H * 0.15), int(H * 0.7)); x += sw + int(W * 0.03)
        dtxt = digits
    elif style < 0.8:  # older: digits-hyphen-digits
        dtxt = digits[:2] + "-" + digits[2:]
    else:
        dtxt = digits
    ink = (random.randint(0, 30),) * 3
    d.text((x, H // 2), dtxt, font=font, fill=ink, anchor="lm")
    x += d.textlength(dtxt, font=font) + int(W * random.uniform(0.03, 0.06))
    if style < 0.55 and random.random() < 0.8:
        ow, oh = int(W * 0.055), int(H * 0.28)
        d.ellipse([x, H // 2 - oh // 2, x + ow, H // 2 + oh // 2], outline=(0, 0, 0), width=3)
        sf = load_font(font_path, int(oh * 0.55)); d.text((x + ow / 2, H / 2), "MNG", font=sf, fill=ink, anchor="mm")
        x += ow + int(W * 0.03)
    d.text((x, H // 2), letters, font=font, fill=ink, anchor="lm")
    return im


def render_two_line(digits, letters, font_path):
    W = random.randint(300, 360); H = random.randint(220, 280)
    bg = tuple(random.randint(225, 255) for _ in range(3))
    im = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(im)
    bw = random.randint(3, 8)
    d.rounded_rectangle([bw // 2, bw // 2, W - bw // 2, H - bw // 2], radius=10, outline=(10, 10, 10), width=bw)
    fs = int(H * random.uniform(0.33, 0.4)); font = load_font(font_path, fs)
    ink = (random.randint(0, 30),) * 3
    y1, y2 = int(H * 0.28), int(H * 0.73)
    x = int(W * 0.07)
    if random.random() < 0.7:
        sw = draw_soyombo(d, x, int(H * 0.08), int(H * 0.4)); x += sw + int(W * 0.04)
    d.text((x, y1), digits, font=font, fill=ink, anchor="lm")
    lx = (W - d.textlength(letters, font=font)) / 2 + random.randint(-15, 15)
    d.text((lx, y2), letters, font=font, fill=ink, anchor="lm")
    top = (0, 0, W, int(H * 0.52)); bot = (0, int(H * 0.48), W, H)
    return im, top, bot


def augment(img):
    a = np.array(img)[:, :, ::-1].copy()  # BGR
    h, w = a.shape[:2]
    # random margin / crop jitter (detector box imprecision)
    pad = [int(random.uniform(-0.04, 0.12) * s) for s in (h, h, w, w)]
    border = [int(x) for x in np.random.randint(60, 200, 3)]
    a = cv2.copyMakeBorder(a, max(0, pad[0]), max(0, pad[1]), max(0, pad[2]), max(0, pad[3]), cv2.BORDER_CONSTANT, value=border)
    h, w = a.shape[:2]
    cy0, cy1 = max(0, -pad[0]), h - max(0, -pad[1]); cx0, cx1 = max(0, -pad[2]), w - max(0, -pad[3])
    a = a[cy0:cy1, cx0:cx1]; h, w = a.shape[:2]
    # perspective + rotation
    if random.random() < 0.8:
        dx, dy = 0.08 * w, 0.12 * h
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[random.uniform(-dx, dx), random.uniform(-dy, dy)] for _ in range(4)])
        M = cv2.getPerspectiveTransform(src, dst)
        a = cv2.warpPerspective(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.7:
        ang = random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # resolution loss
    th = random.choice([14, 16, 20, 24, 24, 28, 32, 32, 40, 48, 64])
    a = cv2.resize(a, (max(8, int(w * th / h)), th), interpolation=random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR]))
    # photometric
    a = a.astype(np.float32)
    a = a * random.uniform(0.55, 1.25) + random.uniform(-35, 35)
    a += np.random.uniform(-15, 15, 3)[None, None, :]  # colour cast
    if random.random() < 0.5:  # gamma
        a = 255 * (np.clip(a, 0, 255) / 255) ** random.uniform(0.6, 1.6)
    if random.random() < 0.4:  # shadow band
        hh, ww = a.shape[:2]; x0 = random.randint(0, ww); x1 = random.randint(0, ww)
        mask = np.zeros((hh, ww), np.float32); cv2.fillPoly(mask, [np.int32([[x0, 0], [x1, 0], [x1 + 30, hh], [x0 - 30, hh]])], 1)
        a = a * (1 - 0.5 * mask[:, :, None])
    if random.random() < 0.3:  # dirt spots
        for _ in range(random.randint(1, 5)):
            hh, ww = a.shape[:2]; cv2.circle(a, (random.randint(0, ww), random.randint(0, hh)), random.randint(1, 4), [random.uniform(20, 120)] * 3, -1)
    a += np.random.normal(0, random.uniform(0, 12), a.shape)
    a = np.clip(a, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        k = random.choice([3, 5]); a = cv2.GaussianBlur(a, (k, k), 0)
    if random.random() < 0.3:  # motion blur
        k = random.choice([3, 5, 7]); ker = np.zeros((k, k)); ker[k // 2, :] = 1 / k
        a = cv2.filter2D(a, -1, ker)
    if random.random() < 0.6:
        q = random.randint(30, 90); _, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, q]); a = cv2.imdecode(enc, 1)
    if random.random() < 0.15:
        a = cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60000); ap.add_argument("--out", default="data/rec/synth"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(f"{args.out}/images", exist_ok=True)
    labels = []
    for i in range(args.n):
        digits, letters = rand_text(); fp = random.choice(FONT_FILES)
        r = random.random()
        try:
            if r < 0.5:
                im = render_single(digits, letters, fp); samples = [(im, digits + letters, "s")]
            else:
                im, top, bot = render_two_line(digits, letters, fp)
                samples = [(im.crop(top), digits, "t"), (im.crop(bot), letters, "b")]
        except Exception as e:
            print("render fail", fp, e); continue
        for im, txt, tag in samples:
            a = augment(im); name = f"{i:06d}{tag}.jpg"
            cv2.imwrite(f"{args.out}/images/{name}", a, [cv2.IMWRITE_JPEG_QUALITY, 92])
            labels.append(f"images/{name}\t{txt}")
        if i % 5000 == 0: print(i, len(labels), flush=True)
    with open(f"{args.out}/labels.txt", "w", encoding="utf-8") as f: f.write("\n".join(labels) + "\n")
    print("done", len(labels))


if __name__ == "__main__":
    main()
