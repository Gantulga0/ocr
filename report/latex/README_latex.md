# LaTeX явцын тайлан (МУИС-ийн бакалаврын явцын тайлангийн формат)

Хөрвүүлэх (Tectonic, conda `tex` орчин):
```bash
/home/bolly/miniconda3/envs/tex/bin/tectonic main.tex      # -> main.pdf
```
Overleaf дээр: бүх файлыг (main.tex, chapters/, figures/) оруулаад компиляторыг **XeLaTeX** болгоно.

Засах газрууд:
- `main.tex` эхэнд: `\studentname`, `\studentid`, `\supervisor`, `\cosupervisor`, `\program`, `\reportdate`.
- Үечилсэн төлөвлөгөөний хуудас: скан файлаа `figures/plan.pdf` болгож `\includepdf` хийх (pdfpages багц).
- Зургуудыг дахин үүсгэх: төслийн үндсэн хавтаснаас `python scripts/make_thesis_figures.py` (→ `figures/`).

Бүтэц: нүүр хуудас ×2, төлөвлөгөө, зохиогчийн баталгаа, гарчиг, зураг/хүснэгтийн жагсаалт, Удиртгал,
1 Судалгааны тойм, 2 Өгөгдөл, 3 Арга зүй, 4 Туршилт ба үр дүн, 5 Хэлэлцүүлэг, Дүгнэлт, Ном зүй, Хавсралт A–D.
