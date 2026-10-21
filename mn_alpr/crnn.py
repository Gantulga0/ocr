"""CRNN (CNN backbone + BiLSTM + CTC head) text-line recognizer, PaddleOCR-style.

Input: RGB line image resized to H=48, W=192 (aspect-preserving, right-padded).
Output: T=48 time steps x NUM_CLASSES logits for CTC decoding.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from .charset import NUM_CLASSES, decode_greedy

IMG_H, IMG_W = 48, 192


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = None if (cin == cout and stride == (1, 1)) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden=192):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ConvBlock(32, 64, (2, 2)),    # 24 x 96
            ConvBlock(64, 128, (2, 2)),   # 12 x 48
            ConvBlock(128, 128),
            ConvBlock(128, 256, (2, 1)),  # 6 x 48
            ConvBlock(256, 256),
            ConvBlock(256, 384, (2, 1)),  # 3 x 48
            ConvBlock(384, 384),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(384, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x):            # x: B,3,48,192
        f = self.layers(self.stem(x))  # B,384,3,48
        f = self.pool(f).squeeze(2).permute(0, 2, 1)  # B,T,C
        f, _ = self.rnn(f)
        return self.fc(f)             # B,T,num_classes

    @torch.no_grad()
    def recognize(self, x):
        """Returns (texts, confidences). Confidence = mean max-prob over non-blank emitted steps."""
        logits = self.forward(x); prob = logits.softmax(-1)
        pmax, idx = prob.max(-1)
        texts, confs = [], []
        for p, i in zip(pmax.cpu().numpy(), idx.cpu().numpy()):
            texts.append(decode_greedy(i))
            keep = [p[t] for t in range(len(i)) if i[t] != 0 and (t == 0 or i[t] != i[t - 1])]
            confs.append(float(sum(keep) / len(keep)) if keep else 0.0)
        return texts, confs


def preprocess(img_bgr, out_w=IMG_W, out_h=IMG_H):
    """Aspect-preserving resize + right pad; returns float tensor 3xHxW in [-1,1]."""
    import cv2, numpy as np
    h, w = img_bgr.shape[:2]
    nw = min(out_w, max(8, int(round(w * out_h / max(1, h)))))
    r = cv2.resize(img_bgr, (nw, out_h), interpolation=cv2.INTER_AREA if h > out_h else cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, 3), 127, np.uint8); canvas[:, :nw] = r
    t = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float()
    return (t / 127.5) - 1.0
