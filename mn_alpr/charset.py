"""Character set for Mongolian vehicle registration plates.

Plates carry 4 digits + 3 uppercase Mongolian-Cyrillic letters (e.g. 1234 УБА).
We keep the full 35-letter Mongolian Cyrillic alphabet so that rare third letters
are representable; the plate-rule post-filter handles region-code plausibility.
Index 0 is reserved for the CTC blank symbol.
"""
DIGITS = "0123456789"
# Mongolian Cyrillic uppercase (35 letters) incl. Ө and Ү
LETTERS = "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ"
CHARS = DIGITS + LETTERS
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # + blank

# Latin transliteration used by the gumbernator/Mongolian-ALPR VOC labels
TRANSLIT = {
    "a": "А", "b": "Б", "v": "В", "g": "Г", "d": "Д", "ye": "Е", "yo": "Ё", "j": "Ж",
    "z": "З", "i": "И", "y": "Й", "k": "К", "l": "Л", "m": "М", "n": "Н", "o": "О",
    "q": "Ө", "p": "П", "r": "Р", "s": "С", "t": "Т", "u": "У", "uu": "Ү", "f": "Ф",
    "h": "Х", "ts": "Ц", "ch": "Ч", "sh": "Ш", "sch": "Щ", "e": "Э", "yu": "Ю", "ya": "Я",
}
# Visually similar Latin/Cyrillic pairs that OCR engines or labelers confuse
LATIN2CYR = str.maketrans("ABEKMHOPCTYX", "АВЕКМНОРСТУХ")


def encode(text: str):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def decode_greedy(indices):
    """CTC greedy decode: collapse repeats, drop blanks."""
    out, prev = [], None
    for i in indices:
        if i != prev and i != BLANK:
            out.append(IDX2CHAR[int(i)])
        prev = i
    return "".join(out)
