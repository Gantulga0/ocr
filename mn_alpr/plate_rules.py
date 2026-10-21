"""Syntactic rules for Mongolian plates (used for validation, voting and scoring)."""
import re
from .charset import LETTERS, LATIN2CYR

# Standard private series: 4 digits + 3 letters.  The first two letters are the
# region (aimag / city) code, the third is a serial letter.
PLATE_RE = re.compile(r"^(\d{4})([%s]{3})$" % LETTERS)

# Region codes (Wikipedia: Vehicle registration plates of Mongolia; MongoliaFocus)
REGION_CODES = {
    "АР": "Архангай", "БН": "Баянхонгор", "БР": "Багануур (УБ)", "БӨ": "Баян-Өлгий",
    "БУ": "Булган", "ГА": "Говь-Алтай", "ГС": "Говьсүмбэр", "ДА": "Дархан-Уул",
    "ДГ": "Дорноговь", "ДО": "Дорнод", "ДУ": "Дундговь", "ОР": "Орхон",
    "ХЭ": "Хэнтий", "ХО": "Ховд", "ХӨ": "Хөвсгөл", "НА": "Налайх (УБ)",
    "ӨМ": "Өмнөговь", "ӨВ": "Өвөрхангай", "СЭ": "Сэлэнгэ", "СҮ": "Сүхбаатар",
    "СБ": "Сүхбаатар хот", "ТӨ": "Төв", "УБ": "Улаанбаатар", "УН": "Улаанбаатар",
    "УА": "Улаанбаатар", "УЕ": "Улаанбаатар", "УК": "Улаанбаатар", "УВ": "Увс",
    "ЗА": "Завхан",
    # special series
    "ДК": "Дипломат", "ЦА": "Цэрэг", "ХЦ": "Хилийн цэрэг",
}
SERIAL_LETTERS = "АБВГДЕЗИКЛМНОӨПРСТУҮХЦЧЭЯ"  # letters observed on plates (3rd position)


def normalize(text: str) -> str:
    """Upper-case, map look-alike Latin letters to Cyrillic, strip separators."""
    t = text.upper().translate(LATIN2CYR)
    return re.sub(r"[^0-9%s]" % LETTERS, "", t)


def is_valid(text: str) -> bool:
    return PLATE_RE.match(normalize(text)) is not None


def region_of(text: str):
    m = PLATE_RE.match(normalize(text))
    return REGION_CODES.get(m.group(2)[:2]) if m else None


def plausibility(text: str) -> float:
    """0..1 score used as a prior during multi-frame voting."""
    t = normalize(text)
    m = PLATE_RE.match(t)
    if not m:
        return 0.0
    s = 0.6
    if m.group(2)[:2] in REGION_CODES:
        s += 0.3
    if m.group(2)[2] in SERIAL_LETTERS:
        s += 0.1
    return s


def pretty(text: str) -> str:
    t = normalize(text)
    m = PLATE_RE.match(t)
    return f"{m.group(1)} {m.group(2)}" if m else t
