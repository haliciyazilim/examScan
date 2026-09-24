"""
Taranmış cevap formlarını bir görsel-dil modeline okutup cevap anahtarıyla puanlar.

Kurulum:
  pip install opencv-python numpy pdf2image requests fpdf2   (+ sistemde poppler-utils)
  export ANTHROPIC_API_KEY="sk-ant-..."
  (VISION_PROVIDER "gemini"/"openai" ise GEMINI_API_KEY / OPENAI_API_KEY)

Notlar:
  * Model yanıtları vision_cache/ içinde tutulur; tekrar çalıştırınca API'ye gidilmez.
  * Model bir sayfayı okuyamazsa o sayfa puanlanmaz ve sonda listelenir; tekrar çalıştırın.
  * Koordinatları kontrol etmek için: debug_rois(cv_image)
"""

import os
import re
import csv
import json
import time
import base64
import hashlib
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import cv2
import numpy as np
import requests
from fpdf import FPDF
try:
    from fpdf.enums import XPos, YPos
    _NL = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}
except ImportError:  # eski fpdf paketi
    _NL = {"ln": True}
from pdf2image import convert_from_path

PDF_PATH = "SCAN0000-3.PDF"
ACTIVE_ANSWER_KEY = None               # None = takım ID'sinden otomatik seç

VISION_PROVIDER = "anthropic"          # anthropic | gemini | openai
VISION_MODEL = ""                      # boş = aşağıdaki varsayılan
DEFAULT_VISION_MODELS = {
    # daha ucuzu: "claude-haiku-4-5-20251001" (el yazısı netse yeter)
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
}
VISION_TIMEOUT = 90
VISION_MAX_RETRY = 3
VISION_WORKERS = 4                     # kaç sayfa paralel okunsun
VISION_CACHE_DIR = "vision_cache"
VISION_RECHECK = True                  # şüpheli alanlar için büyütülmüş ikinci tur

REPORT_DIR = "reports"
REVIEW_DIR = "debug_crops"             # kontrol gereken sayfaların görüntüsü
SAVE_REVIEW_PAGES = True

TARGET_W, TARGET_H = 900, 1200

# Karar eşikleri
FUZZY_THRESHOLD = 0.0      # 0 = sadece birebir eşleşme puan alır
NEAR_MISS_RATIO = 0.70     # bu kadar benzeyen yanlışlar "az-fark" -> elle kontrol
LOW_CONF = 55              # bu güvenin altı -> elle kontrol
EMPTY_INK_RATIO = 0.004    # kutudaki mürekkep oranı bunun altındaysa kutu boş

# Cevap anahtarları: TÜRKİYE-CEVAPLAR2.xlsx
ANSWER_KEY_1_2 = {
    1:  "534", 2: "C", 3: "ÜRÜN", 4: "FECD", 5: "C", 6: "ABBA", 7: "13", 8: "1223", 9: "KARANFİL",
    10: ["767876", "6789876"], 11: "CAN", 12: "YOL", 13: "YAN", 14: "SEN", 15: "KENT",
    16: "OLAY", 17: "ASIK", 18: "YAPI", 19: "21", 20: "6", 21: "4", 22: "8", 23: "8", 24: "90", 25: "38"
}

ANSWER_KEY_3_4 = {
    1:  "3832", 2: "A", 3: "ROZET", 4: "FFCDED", 5: "D", 6: "BABAAB", 7: "21", 8: "11433", 9: "KARANFİL",
    # Excel'de 10. soru 1-4. sınıflar için 767876 / 6789876 (5-9 satırı değil)
    10: ["767876", "6789876"], 11: "CAN", 12: "YOL", 13: "YAN", 14: "SEN", 15: "KENT",
    16: "OLAY", 17: "ASIK", 18: "YAPI", 19: "21", 20: "6", 21: "4", 22: "8", 23: "8", 24: "90", 25: "38"
}

ANSWER_KEY_5_8 = {
    1:  "23831", 2: "B", 3: "DÜRBÜN", 4: "FADCEADD", 5: "22", 6: "AABABBAB", 7: "31", 8: "514533", 9: "MERİDYEN",
    10: ["767696", "767898587696"], 11: "ORAN", 12: "KREM", 13: "FEDA", 14: "OZAN", 15: "SANI",
    16: "DERS", 17: "MERA", 18: "PUAN", 19: "15", 20: "17/2", 21: "153", 22: "20", 23: "94", 24: "38", 25: "4/3"
}

ANSWER_KEY_9_12 = {
    1:  "161385", 2: "D", 3: "ESTETİK", 4: "FAAAADCBED", 5: "20", 6: "BBABBAABAA", 7: "51", 8: "4144727", 9: "KATEGORİ",
    10: ["747476", "74789898765476"], 11: "TANI", 12: "KANO", 13: "KAPI", 14: "TANE", 15: "KAYIT",
    16: "YANIK", 17: "SOLUK", 18: "TORUN", 19: "9", 20: "17/2", 21: "-8", 22: "6", 23: "7", 24: "10/3", 25: "23"
}

# Koordinatlar 900x1200'e düzeltilmiş form üzerinde (kontrol: debug_rois)
ANSWER_ROIS = {
    1:  (60, 228, 370, 72),
    3:  (60, 462, 370, 72),
    4:  (60, 578, 370, 72),
    5:  (120, 705, 270, 55),           # 5-12. sınıf: iki gol kutusunu birlikte kapsar
    6:  (495, 228, 370, 72),
    7:  (495, 345, 370, 72),
    8:  (495, 462, 370, 72),
    9:  (495, 578, 370, 72),
    10: (495, 695, 370, 72),
    11: (68, 884, 145, 36), 12: (278, 884, 147, 36), 13: (488, 884, 146, 36), 14: (700, 884, 144, 36),
    15: (68, 930, 145, 36), 16: (278, 930, 147, 36), 17: (488, 930, 146, 36), 18: (700, 930, 144, 36),
    19: (68, 1046, 145, 36), 20: (278, 1046, 147, 36), 21: (488, 1046, 146, 36), 22: (700, 1046, 144, 36),
    23: (68, 1094, 145, 36), 24: (278, 1094, 147, 36), 25: (488, 1094, 146, 36),
}

Q5_BOX_INTERIORS = [(139, 720, 56, 26), (316, 720, 56, 26)]

BUBBLE_BAND = (140, 345, 350, 62)      # Soru 2
BUBBLE_BAND_Q5 = (140, 700, 350, 70)   # 1-4. sınıf Soru 5
BUBBLE_FALLBACK = {"y": 383, "r": 11, "x": {"A": 182, "B": 212, "C": 243, "D": 274, "E": 305}}
BUBBLE_FILL_MIN = 0.45
BUBBLE_MARGIN = 0.25

TEAM_ID_ROI = (450, 1148, 445, 40)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def find_form_and_warp(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    bright = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY)[1]
    low_sat = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.bitwise_and(bright, low_sat)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("[WARN] Beyaz form alanı bulunamadı — tüm görüntü kullanılıyor.")
        return cv2.resize(image, (TARGET_W, TARGET_H))

    largest = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
    if len(approx) == 4:
        rect = order_points(approx.reshape(4, 2).astype("float32"))
    else:
        x, y, w, h = cv2.boundingRect(largest)
        rect = order_points(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32"))

    dst = np.array([[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]],
                   dtype="float32")
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (TARGET_W, TARGET_H))


def remove_grader_ink(form: np.ndarray) -> np.ndarray:
    s = cv2.cvtColor(form, cv2.COLOR_BGR2HSV)[:, :, 1]
    colored = cv2.dilate(cv2.inRange(s, 60, 255), np.ones((3, 3), np.uint8), iterations=1)
    cleaned = form.copy()
    cleaned[colored > 0] = (255, 255, 255)
    return cleaned


def roi_has_ink(form: np.ndarray, roi) -> bool:
    if not roi:
        return False
    x, y, w, h = roi
    crop = form[y:y + h, x:x + w]
    if crop.size == 0:
        return False
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None, fx=3, fy=3,
                      interpolation=cv2.INTER_CUBIC)
    th = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15)
    H, W = th.shape
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 3, 25), 1)))
    vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(H // 2, 25))))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8), 1)
    th = cv2.bitwise_and(th, cv2.bitwise_not(lines))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return float(np.count_nonzero(th)) / th.size >= EMPTY_INK_RATIO


VISION_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "openai": "https://api.openai.com/v1/chat/completions",
}
VISION_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}

PAGE_PROMPT = """Bu bir matematik yarışması cevap formunun taranmış hâli.
Öğrencinin EL YAZISIYLA yazdığı cevapları oku ve aynen aktar.

Kurallar:
- Yalnızca öğrencinin yazdığını yaz; DOĞRU cevabı tahmin etme, düzeltme yapma.
- Puanlayıcının FARKLI RENKTEKİ kalemiyle (mor/mavi/kırmızı/pembe/yeşil) sonradan
  eklediği işaretleri (1, 0, çarpı, yuvarlak, boydan boya çizgi) YOK SAY.
  Onlar cevap değil, puanlama işaretidir. Öğrencinin yazısı kurşun kalem/siyahtır.
- Soru 2 (ve varsa Soru 5) optik işaretlemedir: işaretlenmiş şıkkın harfini yaz.
  İşaret dolu daire, tik, çarpı ya da daireyi kesen bir çizgi olabilir.
- Soru 5 "Puan Tablosu" iki kutuluysa iki sayıyı bitişik yaz (ör. A=2, B=3 -> "23").
- Kutu boşsa "" (boş dize) döndür.
- Okunamıyorsa en iyi tahminini yaz ve o soruyu "supheli" listesine ekle.
- Sayfanın sağ altındaki "Takım ID: 5-88028-1 - EFD15 ..." satırından takım ID'sini
  (yalnızca rakam-tire kısmını) "takim_id" olarak ver.

Çıktı SADECE şu JSON olsun, başka hiçbir metin olmasın:
{"takim_id": "5-88028-1",
 "cevaplar": {"1": "...", "2": "...", ..., "25": "..."},
 "supheli": [3, 17]}
"""

RECHECK_PROMPTS = {
    "recheck": """Bu, bir sınav cevap formundaki TEK bir cevap kutusunun büyütülmüş hâli.
Öğrencinin el yazısıyla yazdığını aynen oku.
- Puanlayıcının farklı renkteki kalemiyle üste çizdiği işaretleri yok say; onların
  ALTINDA kalan karakterler de cevaba dâhildir (ilk karakter çoğu kez böyle örtülür).
- İki ayrı sayı kutusu varsa (A ve B gol sayısı) iki sayıyı bitişik yaz, ör. 2 ve 3 -> "23".
- Düzeltme/tahmin yapma, kutu boşsa "" döndür.
Çıktı sadece: {"cevap": "..."}""",

    "bubble": """Bu, bir sınav formundaki A-E şıklarının bulunduğu satırın büyütülmüş hâli.
Öğrenci hangi şıkkı işaretlemiş? İşaret dolu daire olabileceği gibi tik, çarpı,
üzeri çizik ya da daireyi kesen bir çizgi de olabilir; çizginin ucunun değil,
üzerinden geçtiği dairenin harfi geçerlidir.
Farklı renkteki puanlayıcı kalemini yok say. Hiçbiri işaretli değilse "" döndür.
Çıktı sadece: {"cevap": "A"}""",

    "team": """Bu, bir sınav formunun alt satırı: "Takım ID: X-XXXXX-X - ... Oturum - Masa No: ..."
Takım ID'sinin sadece rakam-tire kısmını aynen oku (ör. "5-88028-1").
Farklı renkteki kalemle üstüne çizilmiş işaretleri yok say.
Çıktı sadece: {"cevap": "5-88028-1"}""",
}


def _png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Görüntü PNG'ye çevrilemedi.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _vision_request(b64: str, prompt: str) -> str:
    provider = VISION_PROVIDER
    model = VISION_MODEL or DEFAULT_VISION_MODELS[provider]
    api_key = os.getenv(VISION_ENV[provider], "")
    if not api_key:
        raise RuntimeError(f"{VISION_ENV[provider]} tanımlı değil")

    url = VISION_ENDPOINTS[provider].format(model=model)
    if provider == "anthropic":
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        payload = {
            "model": model, "max_tokens": 1500, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        }
    elif provider == "gemini":
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/png", "data": b64}},
                {"text": prompt},
            ]}],
            "generationConfig": {"temperature": 0, "response_mime_type": "application/json"},
        }
    else:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        }

    last_err = None
    for attempt in range(VISION_MAX_RETRY):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=VISION_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            if provider == "anthropic":
                return data["content"][0]["text"]
            if provider == "gemini":
                return data["candidates"][0]["content"]["parts"][0]["text"]
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"model isteği başarısız: {last_err}")


def _cached_request(cache_key: str, b64: str, prompt: str) -> str:
    path = os.path.join(VISION_CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    raw = _vision_request(b64, prompt)
    os.makedirs(VISION_CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw)
    return raw


def _json_from(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0) if m else text)


def _clean_team_id(value) -> str:
    m = re.search(r"\d{1,2}-\d{3,6}(?:-\d{1,2})?", str(value or ""))
    return m.group(0) if m else ""


def vision_read_page(form: np.ndarray) -> tuple[dict, set, str]:
    b64 = _png_b64(form)
    key = hashlib.sha1(b64.encode("ascii")).hexdigest()[:16]
    data = _json_from(_cached_request(key, b64, PAGE_PROMPT))

    answers = {}
    for k, v in (data.get("cevaplar") or {}).items():
        try:
            answers[int(str(k).strip())] = "" if v is None else str(v).strip()
        except ValueError:
            continue
    unsure = set()
    for q in data.get("supheli") or []:
        try:
            unsure.add(int(q))
        except (TypeError, ValueError):
            pass
    return answers, unsure, _clean_team_id(data.get("takim_id"))


def vision_recheck(form: np.ndarray, roi, kind: str = "recheck") -> str:
    if not roi:
        return ""
    x, y, w, h = roi
    pad, pad_left = 8, 26
    crop = form[max(0, y - pad):y + h + pad, max(0, x - pad_left):x + w + pad]
    if crop.size == 0:
        return ""
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    b64 = _png_b64(crop)
    key = hashlib.sha1((kind + b64).encode("ascii")).hexdigest()[:16]
    try:
        return str(_json_from(_cached_request(key, b64, RECHECK_PROMPTS[kind])).get("cevap", "")).strip()
    except (ValueError, AttributeError):
        return ""


def _find_bubbles(gray: np.ndarray, band) -> list:
    x0, y0, w, h = band
    area = gray[y0:y0 + h, x0:x0 + w]
    cands = []

    circles = cv2.HoughCircles(cv2.GaussianBlur(area, (3, 3), 0), cv2.HOUGH_GRADIENT, dp=1,
                               minDist=18, param1=100, param2=16, minRadius=7, maxRadius=15)
    if circles is not None:
        cands += [(int(c[0]), int(c[1]), int(c[2])) for c in np.round(circles[0]).astype(int)]

    _, th = cv2.threshold(area, 120, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, cent = cv2.connectedComponentsWithStats(th, 8)
    for i in range(1, n):
        _, _, bw, bh, a = stats[i]
        if 14 <= bw <= 32 and 14 <= bh <= 32 and a > 120 and 0.6 <= bw / bh <= 1.7:
            cands.append((int(cent[i][0]), int(cent[i][1]), int(max(bw, bh) // 2)))
    if not cands:
        return []

    med_y = float(np.median([c[1] for c in cands]))
    cands = sorted((c for c in cands if abs(c[1] - med_y) <= 10), key=lambda c: c[0])
    uniq = []
    for c in cands:
        if not uniq or c[0] - uniq[-1][0] >= 16:
            uniq.append(c)

    if len(uniq) > 5:
        uniq = min((uniq[i:i + 5] for i in range(len(uniq) - 4)),
                   key=lambda seq: float(np.std(np.diff([c[0] for c in seq]))))
    return [(c[0] + x0, c[1] + y0, c[2]) for c in uniq]


def _fill_ratio(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    r = max(r, 7)
    patch = gray[cy - r:cy + r, cx - r:cx + r]
    if patch.size == 0:
        return 0.0
    mask = np.zeros(patch.shape, np.uint8)
    cv2.circle(mask, (r, r), max(int(r * 0.7), 3), 255, -1)
    dark = (patch < 140).astype(np.uint8) * 255
    return float(np.count_nonzero(cv2.bitwise_and(dark, dark, mask=mask))) / max(np.count_nonzero(mask), 1)


def read_bubbles(clean: np.ndarray, band) -> tuple[str, float]:
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    bubbles = _find_bubbles(gray, band)
    if len(bubbles) != 5 and band == BUBBLE_BAND:
        bubbles = [(cx, BUBBLE_FALLBACK["y"], BUBBLE_FALLBACK["r"]) for cx in BUBBLE_FALLBACK["x"].values()]
    if len(bubbles) != 5:
        return "", 0.0

    fills = [_fill_ratio(gray, *b) for b in bubbles]
    order = np.argsort(fills)[::-1]
    if fills[order[0]] < BUBBLE_FILL_MIN or fills[order[0]] - fills[order[1]] < BUBBLE_MARGIN:
        return "", 0.0
    return "ABCDE"[order[0]], min(100.0, fills[order[0]] * 100)


TR_UPPER = str.maketrans({"İ": "I", "ı": "I", "i": "I", "Ö": "O", "ö": "O", "Ü": "U", "ü": "U",
                          "Ş": "S", "ş": "S", "Ç": "C", "ç": "C", "Ğ": "G", "ğ": "G"})


def question_kind(key_value) -> str:
    text = "".join(str(v) for v in (key_value if isinstance(key_value, list) else [key_value]))
    has_digit = any(c.isdigit() for c in text)
    has_alpha = any(c.isalpha() for c in text)
    if has_digit and not has_alpha:
        return "digit"
    if has_alpha and not has_digit:
        return "letter"
    return "mixed"


def normalize(value, kind: str) -> str:
    text = unicodedata.normalize("NFC", str(value)).translate(TR_UPPER).upper()
    text = re.sub(r"[\s.,_·•]", "", text)
    if kind == "digit":
        return re.sub(r"[^0-9/+\-]", "", text)
    if kind == "letter":
        return re.sub(r"[^A-Z]", "", text)
    return re.sub(r"[^A-Z0-9/+\-]", "", text)


def compare(read_value: str, key_value, kind: str) -> tuple[bool, str, float]:
    candidates = key_value if isinstance(key_value, list) else [key_value]
    student = normalize(read_value, kind)
    if not student:
        return False, "bos", 0.0
    if any(student == normalize(c, kind) for c in candidates):
        return True, "tam", 1.0

    ratio = max((SequenceMatcher(None, student, normalize(c, kind)).ratio()
                 for c in candidates if normalize(c, kind)), default=0.0)
    if FUZZY_THRESHOLD > 0 and ratio >= FUZZY_THRESHOLD:
        return True, f"yakin({ratio:.2f})", ratio
    return False, "eslesmedi", ratio


def pick_answer_key(team_id: str) -> tuple[dict, str]:
    m = re.match(r"(\d{1,2})-", team_id or "")
    grade = int(m.group(1)) if m else None
    for grades, key, name in (((1, 2), ANSWER_KEY_1_2, "1-2"), ((3, 4), ANSWER_KEY_3_4, "3-4"),
                              ((5, 6, 7, 8), ANSWER_KEY_5_8, "5-8"), ((9, 10, 11, 12), ANSWER_KEY_9_12, "9-12")):
        if grade in grades:
            return key, name
    raise RuntimeError(f"takım ID'sinden sınıf çıkarılamadı ({team_id!r}); "
                       f"ACTIVE_ANSWER_KEY ile anahtarı elle verin")


def process_form(image: np.ndarray, answer_key: dict | None = None, page_num: int = 0) -> tuple[str, dict]:
    if image is None or image.size == 0:
        raise ValueError("boş görüntü")

    form = find_form_and_warp(image)
    clean = remove_grader_ink(form)

    answers, unsure, team_id = vision_read_page(form)
    if not team_id:
        team_id = _clean_team_id(vision_recheck(form, TEAM_ID_ROI, "team"))
    if not team_id:
        team_id = f"UNKNOWN_p{page_num:02d}"

    if answer_key is None:
        answer_key, key_name = pick_answer_key(team_id)
    else:
        key_name = "elle"
    print(f"[INFO] Sayfa {page_num} — Takım ID: {team_id}  (anahtar: {key_name})")

    details, correct_count, review_count = [], 0, 0
    for q in sorted(answer_key):
        key_value = answer_key[q]
        kind = question_kind(key_value)
        model_value = answers.get(q, "")
        is_bubble = q == 2 or (q == 5 and kind == "letter")

        if is_bubble:
            band = BUBBLE_BAND if q == 2 else BUBBLE_BAND_Q5
            read_value, conf = read_bubbles(clean, band)
            if not read_value:
                second = re.sub(r"[^A-E]", "", vision_recheck(form, band, "bubble").upper())[:1]
                if second:
                    print(f"[INFO] Sayfa {page_num} S{q}: baloncuk ikinci tur -> '{second}'")
                    read_value, conf = second, 70.0
                elif model_value:
                    read_value, conf = model_value, 50.0
                else:
                    conf = 95.0
        else:
            read_value = model_value
            if read_value:
                conf = 40.0 if q in unsure else 92.0
            else:
                if q == 5:
                    has_ink = any(roi_has_ink(clean, b) for b in Q5_BOX_INTERIORS)
                else:
                    has_ink = roi_has_ink(clean, ANSWER_ROIS.get(q))
                conf = 40.0 if has_ink else 95.0

        is_correct, match_type, ratio = compare(read_value, key_value, kind)
        near_miss = not is_correct and ratio >= NEAR_MISS_RATIO
        clash = bool(is_bubble and model_value and normalize(model_value, kind) != normalize(read_value, kind))
        needs_review = bool(near_miss or clash or conf < LOW_CONF
                            or match_type.startswith("yakin") or (not read_value and conf < 90))
        if near_miss:
            match_type = f"az-fark({ratio:.2f})"

        if VISION_RECHECK and needs_review and read_value and not is_bubble and q in ANSWER_ROIS:
            second = vision_recheck(form, ANSWER_ROIS[q], "recheck")
            if second and normalize(second, kind) != normalize(read_value, kind):
                print(f"[INFO] Sayfa {page_num} S{q}: ikinci tur '{read_value}' -> '{second}'")
                ok2, mt2, _ = compare(second, key_value, kind)
                read_value, is_correct, match_type, conf = second, ok2, mt2 + "+2.tur", 75.0

        correct_count += int(is_correct)
        review_count += int(needs_review)
        details.append({
            "question": q,
            "read_answer": read_value,
            "correct_answer": " / ".join(map(str, key_value)) if isinstance(key_value, list) else str(key_value),
            "match_type": match_type,
            "confidence": round(float(conf), 1),
            "is_correct": bool(is_correct),
            "needs_review": needs_review,
        })

    if SAVE_REVIEW_PAGES and review_count:
        os.makedirs(REVIEW_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(REVIEW_DIR, f"p{page_num:02d}_sayfa.png"), form)

    result = {
        "page": page_num,
        "team_id": team_id,
        "answer_key": key_name,
        "score_percentage": round(correct_count / len(answer_key) * 100, 2),
        "total_correct": correct_count,
        "total_questions": len(answer_key),
        "review_count": review_count,
        "details": details,
    }
    return generate_pdf_report(team_id, result), result


TR_ASCII = str.maketrans({"İ": "I", "ı": "i", "Ö": "O", "ö": "o", "Ü": "U", "ü": "u",
                          "Ş": "S", "ş": "s", "Ç": "C", "ç": "c", "Ğ": "G", "ğ": "g"})


def _l1(text) -> str:
    return str(text).translate(TR_ASCII).encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(team_id: str, result: dict) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Degerlendirme Raporu", align="C", **_NL)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, f"Takim ID: {_l1(team_id)}   (anahtar: {result['answer_key']})", align="C", **_NL)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(0, 9, f"Puan: {result['total_correct']} / {result['total_questions']}"
                   f"   ({result['score_percentage']:.1f}%)   |  Gozden gecir: {result['review_count']}",
             fill=True, align="C", **_NL)
    pdf.ln(6)

    col_w = [12, 50, 50, 30, 20, 28]
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(col_w, ["S#", "Okunan Cevap", "Dogru Cevap", "Eslesme", "Guven", "Sonuc"]):
        pdf.cell(w, 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for row in result["details"]:
        if row["needs_review"]:
            pdf.set_fill_color(255, 240, 190)
        elif row["is_correct"]:
            pdf.set_fill_color(200, 240, 200)
        else:
            pdf.set_fill_color(255, 200, 200)
        cells = [
            str(row["question"]),
            _l1(row["read_answer"] or "-"),
            _l1(row["correct_answer"]),
            _l1(row["match_type"]),
            f"{row['confidence']:.0f}",
            ("DOGRU" if row["is_correct"] else "YANLIS") + ("*" if row["needs_review"] else ""),
        ]
        for w, val in zip(col_w, cells):
            pdf.cell(w, 7, val[:28], border=1, align="C", fill=True)
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 5, _l1("* = model emin degil, cevap bos ya da anahtara cok yakin (az-fark); "
                             "elle kontrol onerilir. Sayfa goruntusu debug_crops/ klasorunde."))

    safe_id = re.sub(r"[^\w\-]", "_", team_id) or "Unknown_Team"
    path = os.path.join(REPORT_DIR, f"{safe_id}.pdf")
    pdf.output(path)
    return path


def write_summary_csv(results: list, path: str = "sonuclar.csv") -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Sayfa", "Takim ID", "Anahtar", "Soru", "Okunan", "Dogru cevap",
                    "Eslesme", "Guven", "Sonuc", "Gozden gecir"])
        for r in results:
            for d in r["details"]:
                w.writerow([r["page"], r["team_id"], r["answer_key"], d["question"], d["read_answer"],
                            d["correct_answer"], d["match_type"], d["confidence"],
                            "DOGRU" if d["is_correct"] else "YANLIS", "EVET" if d["needs_review"] else ""])
    print(f"[OK] Ayrıntılı tablo -> {path}")


def write_scores_csv(results: list, path: str = "puanlar.csv") -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Sayfa", "Takim ID", "Anahtar", "Dogru", "Soru sayisi", "Yuzde", "Gozden gecir"])
        for r in results:
            w.writerow([r["page"], r["team_id"], r["answer_key"], r["total_correct"],
                        r["total_questions"], r["score_percentage"], r["review_count"]])
    print(f"[OK] Puan tablosu -> {path}")


def debug_rois(image: np.ndarray, output_path: str = "debug_rois.png") -> None:
    """ROI'leri ve bulunan baloncukları form üzerine çizer."""
    form = find_form_and_warp(image)
    dbg = form.copy()
    for q, (x, y, w, h) in ANSWER_ROIS.items():
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(dbg, f"Q{q}", (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
    gray = cv2.cvtColor(remove_grader_ink(form), cv2.COLOR_BGR2GRAY)
    for band in (BUBBLE_BAND, BUBBLE_BAND_Q5):
        for cx, cy, r in _find_bubbles(gray, band):
            cv2.circle(dbg, (cx, cy), r, (255, 0, 0), 2)
    x, y, w, h = TEAM_ID_ROI
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 200, 200), 2)
    cv2.imwrite(output_path, dbg)
    print(f"[DEBUG] ROI önizleme -> {output_path}")


if __name__ == "__main__":
    print(f"[INFO] PDF yükleniyor: {PDF_PATH} ...")
    pages = convert_from_path(PDF_PATH, dpi=200)
    failed = []

    def handle(job):
        i, page = job
        try:
            pdf_file, result = process_form(cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR),
                                            ACTIVE_ANSWER_KEY, page_num=i)
            print(f"[OK] Sayfa {i} — {result['team_id']}: {result['total_correct']}/"
                  f"{result['total_questions']} — gözden geçir: {result['review_count']} -> {pdf_file}")
            return result
        except Exception as exc:
            print(f"[ERROR] Sayfa {i} PUANLANMADI: {exc}")
            failed.append(i)
            return None

    with ThreadPoolExecutor(max_workers=VISION_WORKERS) as pool:
        all_results = sorted((r for r in pool.map(handle, enumerate(pages, 1)) if r),
                             key=lambda r: r["page"])

    if all_results:
        write_summary_csv(all_results)
        write_scores_csv(all_results)
        with open("sonuclar.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print("[OK] sonuclar.json yazıldı")

    if failed:
        print(f"\n[UYARI] {len(failed)} sayfa puanlanmadı: {sorted(failed)} — "
              f"tekrar çalıştırın, okunan sayfalar önbellekten gelir.")
