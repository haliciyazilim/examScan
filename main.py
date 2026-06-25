import cv2
import numpy as np
import pytesseract
import json
import re
import os
from fpdf import FPDF
from pdf2image import convert_from_path

TARGET_W = 900
TARGET_H = 1200

# ════════════════════════════════════════════════════════════════════════════
# ANSWER KEYS
# ════════════════════════════════════════════════════════════════════════════

ANSWER_KEY_1_2 = {
    1:  "534", 2: "C", 3: "ÜRÜN", 4: "FECD", 5: "C", 6: "ABBA", 7: "13", 8: "1223", 9: "KARANFİL",
    10: ["767876", "6789876"], 11: "CAN", 12: "YOL", 13: "YAN", 14: "SEN", 15: "KENT",
    16: "OLAY", 17: "ASIK", 18: "YAPI", 19: "21", 20: "6", 21: "4", 22: "8", 23: "8", 24: "90", 25: "38"
}

ANSWER_KEY_3_4 = {
    1:  "3832", 2: "A", 3: "ROZET", 4: "FFCDED", 5: "D", 6: "BABAAB", 7: "21", 8: "11433", 9: "KARANFİL",
    10: ["767696", "767898587696"], 11: "CAN", 12: "YOL", 13: "YAN", 14: "SEN", 15: "KENT",
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

MARK_STRIP_W = 45       # Width of strip containing grader mark
AR_THRESHOLD = 0.40     # Aspect ratio threshold (width/height)
MIN_AREA = 80           # Ignore tiny blobs
MIN_HEIGHT = 25         # Ignore very short blobs

def read_mark_section(form, x, y, rw, rh):
    # Crop only the left strip where the grader writes 1 or 0
    strip = form[y:y+rh, x:x+MARK_STRIP_W]

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Remove tiny specks
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for c in cnts:
        area = cv2.contourArea(c)

        if area < MIN_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)

        if bh < MIN_HEIGHT:
            continue

        if area > best_area:
            best = (bx, by, bw, bh)
            best_area = area

    if best is None:
        return False

    bx, by, bw, bh = best

    aspect_ratio = bw / float(bh)

    if aspect_ratio < AR_THRESHOLD:
        return True
    return False

# ════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE CORRECTION
# ════════════════════════════════════════════════════════════════════════════

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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("[WARN] White region not found — using full image.")
        return cv2.resize(image, (TARGET_W, TARGET_H))

    largest = max(cnts, key=cv2.contourArea)
    peri    = cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        pts  = approx.reshape(4, 2).astype("float32")
        rect = order_points(pts)
    else:
        x, y, w, h = cv2.boundingRect(largest)
        rect = order_points(np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h],
        ], dtype="float32"))

    dst = np.array(
        [[0, 0], [TARGET_W-1, 0], [TARGET_W-1, TARGET_H-1], [0, TARGET_H-1]],
        dtype="float32"
    )
    M      = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (TARGET_W, TARGET_H))
    return warped

# ════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING (For Tesseract / Bubble Detection)
# ════════════════════════════════════════════════════════════════════════════

def preprocess_for_tesseract(roi: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled  = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    thresh  = cv2.adaptiveThreshold(
        scaled, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    cleaned = cv2.medianBlur(thresh, 3)
    return cleaned

# ════════════════════════════════════════════════════════════════════════════
# TEAM ID EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

TEAM_ID_ROI = (450, 1148, 445, 40)

def extract_team_id(form: np.ndarray, page_num: int = 0) -> str:
    x, y, rw, rh = TEAM_ID_ROI
    roi = form[y : y + rh, x : x + rw]
    processed = preprocess_for_tesseract(roi)

    raw = pytesseract.image_to_string(processed, config="--psm 7").strip()
    
    normalized = re.sub(r'\b[Il1][O0Oo]\b', 'ID', raw)
    normalized = normalized.lstrip('|( ').strip()

    truncated = re.split(r'[-–]\s*[A-Z]{2,}|EFD|\bOturum\b', normalized, flags=re.IGNORECASE)[0]

    m = re.search(r"(\d{1,2})[-–](\d{3,6})(?:[-–](\d{1,2}))?", truncated)
    if m:
        p1, p2, p3 = str(int(m.group(1))), m.group(2), m.group(3)
        return f"{p1}-{p2}-{p3}" if p3 else f"{p1}-{p2}"

    h, w = roi.shape[:2]
    id_slice = roi[:, w//3 : (w * 2)//3]
    raw2 = pytesseract.image_to_string(
        preprocess_for_tesseract(id_slice),
        config="--psm 7 -c tessedit_char_whitelist=0123456789-"
    ).strip()

    m = re.search(r"(\d{1,2})-(\d{3,6})-(\d{1,2})", raw2)
    if m:
        return f"{int(m.group(1))}-{m.group(2)}-{m.group(3)}"

    cv2.imwrite(f"manual_review_page_{page_num:02d}.png", form)
    print(f"[WARN] Page {page_num} Team ID unreadable — saved for manual review")
    return "UNKNOWN"

# ════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(team_id: str, result: dict) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Grading Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 7, f"Team ID: {team_id}", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(220, 220, 220)
    score_line = (
        f"Score: {result['total_correct']} / {result['total_questions']}"
        f"   ({result['score_percentage']:.1f}%)"
    )
    pdf.cell(0, 9, score_line, ln=True, fill=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", "B", 10)
    col_w = [25, 65, 65, 25]
    headers = ["Q #", "Student Answer", "Correct Answer", "Result"]
    for i, h_txt in enumerate(headers):
        pdf.cell(col_w[i], 8, h_txt, border=1, align="C")
    pdf.ln()

    pdf.set_font("Arial", "", 10)

    for row in result["details"]:
        correct = row["is_correct"]

        if correct:
            pdf.set_fill_color(200, 240, 200)
        else:
            pdf.set_fill_color(255, 200, 200)

        mark = "OK" if correct else "X"

        s_ans = str(row["read_answer"]).encode('latin-1', 'replace').decode('latin-1')
        c_ans = str(row["correct_answer"]).encode('latin-1', 'replace').decode('latin-1')

        cells = [str(row["question"]), s_ans, c_ans, mark]

        for i, val in enumerate(cells):
            pdf.cell(col_w[i], 7, val, border=1, align="C", fill=True)

        pdf.ln()

    pdf.ln(8)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(0, 6, "Raw JSON", ln=True)

    pdf.set_font("Courier", "", 7)
    pdf.set_fill_color(245, 245, 245)

    json_str = json.dumps(result, indent=4, ensure_ascii=False)

    for line in json_str.split("\n"):
        pdf.cell(
            0,
            4.5,
            line[:130].encode("latin-1", "replace").decode("latin-1"),
            ln=True,
            fill=True,
        )

    safe_id = re.sub(r"[^\w\-]", "_", team_id)
    filename = f"{safe_id}.pdf" if safe_id else "Unknown_Team.pdf"

    pdf.output(filename)
    return filename

# ════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING PIPELINE
# ════════════════════════════════════════════════════════════════════════════

ROIS = {
    "Q1":         (10, 225, 420, 75),
    "Q2":         (10, 340, 420, 75),
    "Q3":         (10, 460, 420, 75),
    "Q4":         (10, 575, 420, 75),
    "Q5":         (10, 695, 420, 75),

    "Q6":         (445, 225, 420, 75),
    "Q7":         (445, 340, 420, 75),
    "Q8":         (445, 460, 420, 75),
    "Q9":         (445, 575, 420, 75),
    "Q10":        (445, 695, 420, 75),

    "Q11":        (5, 860, 225, 65),
    "Q12":        (220, 860, 225, 65),
    "Q13":        (430, 860, 225, 65),
    "Q14":        (640, 860, 225, 65),

    "Q15":        (5, 920, 225, 65),
    "Q16":        (220, 920, 225, 65),
    "Q17":        (430, 920, 225, 65),
    "Q18":        (640, 920, 225, 65),

    "Q19":        (5, 1040, 225, 65),
    "Q20":        (220, 1040, 225, 65),
    "Q21":        (430, 1040, 225, 65),
    "Q22":        (640, 1040, 225, 65),

    "Q23":        (5, 1085, 225, 65),
    "Q24":        (220, 1085, 225, 65),
    "Q25":        (430, 1085, 225, 65),
}

def process_form(image: np.ndarray, answer_key: dict, page_num: int = 0) -> tuple[str, dict]:
    if image is None or image.size == 0:
        raise ValueError("Provided image array is empty.")

    form = find_form_and_warp(image)
    team_id = extract_team_id(form, page_num=page_num)
    print(f"[INFO] Team ID detected: {team_id}")

    results_list  = []
    correct_count = 0

    for field, (x, y, rw, rh) in ROIS.items():
        roi_img = form[y : y + rh, x : x + rw]

        q_num = int(field.replace("Q", ""))

        display_correct_ans = answer_key.get(q_num, "")

        is_correct = read_mark_section(form, x, y, rw, rh)

        answer = "1 (Marked Correct)" if is_correct else "0 (Marked Incorrect)"

        if is_correct:
            correct_count += 1

        results_list.append({
            "question":       q_num,
            "read_answer":    answer,
            "correct_answer": display_correct_ans,
            "is_correct":     is_correct
        })

    results_list.sort(key=lambda d: d["question"])

    score_pct = (correct_count / len(answer_key)) * 100 if answer_key else 0
    result = {
        "team_id":          team_id,
        "score_percentage": round(score_pct, 2),
        "total_correct":    correct_count,
        "total_questions":  len(answer_key),
    }

    result["details"] = results_list
    pdf_path = generate_pdf_report(team_id, result)
    return pdf_path, result

def debug_rois(image: np.ndarray, output_path: str = "debug_rois.png") -> None:
    if image is None or image.size == 0:
        raise ValueError("Provided image array is empty.")

    form  = find_form_and_warp(image)
    debug = form.copy()

    for name, (x, y, rw, rh) in ROIS.items():
        cv2.rectangle(debug, (x, y), (x + rw, y + rh), (0, 200, 0), 2)
        cv2.putText(debug, name, (x + 3, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    tx, ty, tw, th = TEAM_ID_ROI
    cv2.rectangle(debug, (tx, ty), (tx + tw, ty + th), (0, 200, 0), 2)
    cv2.putText(debug, "Team_ID", (tx + 3, ty + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    cv2.imwrite(output_path, debug)
    print(f"[DEBUG] ROI overlay saved -> {output_path}")

# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    #change based on document
    ACTIVE_ANSWER_KEY = ANSWER_KEY_5_8
    PDF_PATH = "SCAN0000-3.PDF"

    print(f"[INFO] Loading PDF: {PDF_PATH}...")
    try:
        pages = convert_from_path(PDF_PATH)

        for i, page in enumerate(pages):
            print(f"\n--- Processing Page {i + 1} of {len(pages)} ---")
            img_array = np.array(page)
            cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            pdf_file, result = process_form(cv_image, ACTIVE_ANSWER_KEY, page_num=i + 1)
            print(f"[OK] Report saved for Team: {result['team_id']} -> {pdf_file}")

    except Exception as exc:
        print(f"[ERROR] {exc}")

# ════════════════════════════════════════════════════════════════════════════
# UI DEBUGGER
# ════════════════════════════════════════════════════════════════════════════
try:
    debug_rois(cv_image, "debug_rois.png")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            img = param.copy()

            cv2.putText(
                img,
                f"({x}, {y})",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            cv2.imshow("ROI Picker", img)

    form = find_form_and_warp(cv_image)

    cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Picker", 600, 800)
    cv2.imshow("ROI Picker", form)
    cv2.setMouseCallback("ROI Picker", mouse_callback, form)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

except NameError:
    print("[WARN] UI Debugger skipped because cv_image was not defined (Check PDF path).")
