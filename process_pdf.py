import os
import re
import json
import time
import io
from pathlib import Path

import fitz
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
# -------------------------------
# CONFIG
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------------------
# LOAD JSON
# -------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    if not text:
        return ""
    text = text.replace("\t", " ")
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# -------------------------------
# OCR WITH LAYOUT
# -------------------------------
def ocr_image_with_layout(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        enhanced = Image.fromarray(thresh)
        config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(enhanced, config=config)
        return clean_text(text)
    except Exception as e:
        print(f"⚠ OCR error: {e}")
        return ""


# -------------------------------
# TABLE OCR (SCANNED)
# -------------------------------
def extract_table_from_image_ocr(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        enhanced = Image.fromarray(thresh)

        tsv = pytesseract.image_to_data(
            enhanced, output_type=pytesseract.Output.DICT
        )

        lines = {}
        for i, txt in enumerate(tsv["text"]):
            if txt.strip():
                y = tsv["top"][i] // 10
                x = tsv["left"][i]
                lines.setdefault(y, []).append((x, txt))

        output = []
        for y in sorted(lines):
            row = " | ".join(t[1] for t in sorted(lines[y]))
            output.append(row)

        return "\n".join(output)
    except Exception as e:
        print(f"⚠ Table OCR error: {e}")
        return ""


# -------------------------------
# TABLE DETECTION
# -------------------------------
def image_contains_table(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
        )
        if lines is None:
            return False

        h, v = 0, 0
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 or angle > 170:
                h += 1
            elif 80 < angle < 100:
                v += 1

        return h >= 3 and v >= 2
    except:
        return False


# ------------------------------
# NATIVE TABLE EXTRACTION
# ------------------------------
def extract_tables_native(pdf_path, page_number):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number]
            tables = page.extract_tables()
            if not tables:
                return None

            lines = []
            for table in tables:
                for row in table:
                    lines.append(" | ".join(str(c or "") for c in row))
                lines.append("")
            return "\n".join(lines).strip()
    except:
        return None


# -------------------------------
# MAIN PDF EXTRACTION (REPLACEMENT)
# -------------------------------
def extract_pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages_out = []

    for i in range(len(doc)):
        page = doc[i]
        parts = [f"\n{'='*60}\nPAGE {i+1}\n{'='*60}\n"]

        has_text = False
        for block in page.get_text("blocks"):
            txt = clean_text(block[4])
            if txt:
                parts.append(txt)
                has_text = True

        table_text = extract_tables_native(pdf_path, i)
        if table_text:
            parts.append("\n[TABLE - NATIVE]\n")
            parts.append(table_text)

        images = page.get_images(full=True)
        if images and not has_text:
            for img in images:
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base["image"]

                if image_contains_table(img_bytes):
                    table = extract_table_from_image_ocr(img_bytes)
                    if table:
                        parts.append("\n[TABLE - OCR]\n")
                        parts.append(table)
                else:
                    ocr = ocr_image_with_layout(img_bytes)
                    if ocr:
                        parts.append("\n[TEXT - OCR]\n")
                        parts.append(ocr)

        pages_out.append("\n".join(parts))

    doc.close()
    return "\n".join(pages_out)


# -------------------------------
# SAFE REPLACE
# -------------------------------
def normalize_for_match(text):
    """
    Normalize text for fuzzy matching:
    - lowercase
    - remove extra spaces
    - normalize commas
    """
    text = text.lower()
    text = re.sub(r"\s*,\s*", ",", text)   # normalize commas
    text = re.sub(r"\s+", " ", text)       # normalize spaces
    return text.strip()


def safe_replace(text, original, dummy):
    """
    Replace original value in text ignoring:
    - case
    - spacing
    - comma formatting
    """
    norm_original = normalize_for_match(original)

    # Build regex pattern
    escaped = re.escape(norm_original)
    escaped = escaped.replace(",", r"\s*,\s*")
    escaped = escaped.replace(r"\ ", r"\s+")

    pattern = re.compile(escaped, flags=re.IGNORECASE)

    return pattern.sub(dummy, text)


def normalize_name(text):
    """
    Normalize a name for fuzzy matching:
    - lowercase
    - remove punctuation
    - normalize spaces
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_name_regex(original_name):
    """
    Build OCR-tolerant regex for free-floating names.
    Matches:
    - Ronald Handrop
    - Handrop Ronald
    - Ronaid Handrop (OCR error)
    """
    norm = normalize_name(original_name)
    parts = norm.split()

    if len(parts) < 2:
        return None

    last = parts[0]               # handrop
    first = parts[-1]             # ronald

    # Use first 3 chars to tolerate OCR errors
    f = re.escape(first[:3])
    l = re.escape(last[:3])

    pattern = (
        rf"\b{f}\w*\s+{l}\w*\b"   # Ronald Handrop
        r"|"
        rf"\b{l}\w*\s+{f}\w*\b"   # Handrop Ronald
    )

    return re.compile(pattern, flags=re.IGNORECASE)


def replace_free_floating_name(text, original_name, dummy_name):
    regex = build_name_regex(original_name)
    if not regex:
        return text, False

    new_text, count = regex.subn(dummy_name, text)
    return new_text, count > 0


# -------------------------------
# PII REPLACEMENT
# -------------------------------
def replace_pii(text, pii_data, dummy_data):
    replace_map = {}

    for key, original in pii_data.items():
        dummy = dummy_data.get(key)
        if not dummy:
            continue

        # 1️⃣ Labeled / normalized replacement
        new_text = safe_replace(text, original, dummy)

        replaced = new_text != text
        text = new_text

        # 2️⃣ Free-floating name replacement (ONLY for patient name)
        if key.lower() == "patient name":
            text, free_replaced = replace_free_floating_name(
                text, original, dummy
            )
            replaced = replaced or free_replaced

        if replaced:
            replace_map[key] = {
                "original": original,
                "dummy": dummy,
                "type": "labeled+free" if key.lower() == "patient name" else "labeled"
            }

    return text, replace_map



# -------------------------------
# RUN PIPELINE
# -------------------------------
def run_pipeline():
    os.makedirs("output", exist_ok=True)

    pdf_path = "input/2024.06.30 Bear Valley Community.pdf"

    # 1️⃣ ADVANCED EXTRACTION
    extracted_text = extract_pdf_to_text(pdf_path)
    with open("output/extracted.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    # 2️⃣ LOAD PII
    pii_data = load_json("pii.json")
    dummy_data = load_json("dummy_val.json")

    # 3️⃣ SANITIZE
    sanitized_text, replace_map = replace_pii(
        extracted_text, pii_data, dummy_data
    )

    # 4️⃣ SAVE OUTPUTS
    with open("output/sanitized.txt", "w", encoding="utf-8") as f:
        f.write(sanitized_text)

    with open("output/replace.json", "w", encoding="utf-8") as f:
        json.dump(replace_map, f, indent=4)

    print("✔ PDF extracted + sanitized successfully")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run_pipeline()
