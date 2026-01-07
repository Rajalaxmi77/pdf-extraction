import os
import re
import json
import time
import io
from pathlib import Path
from collections import defaultdict

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


def build_honorific_name_regex(original_name):
    """
    Match honorific + last name.
    Examples:
    - Mr. Handrop
    - Mr Handrop
    - Dr. Handrop
    """
    norm = normalize_name(original_name)
    parts = norm.split()

    if not parts:
        return None

    # Last name is usually first in normalized "Handrop Ronald"
    last = parts[0][:4]  # first 4 chars for OCR tolerance

    honorifics = r"(mr|mrs|ms|dr)"
    pattern = rf"\b{honorifics}\.?\s+{re.escape(last)}\w*\b"

    return re.compile(pattern, flags=re.IGNORECASE)


def replace_honorific_name(text, original_name, dummy_last_name):
    """
    Replace 'Mr. Handrop' → 'Mr Doe'
    """
    regex = build_honorific_name_regex(original_name)
    if not regex:
        return text, False

    def repl(match):
        honorific = match.group(1)
        return f"{honorific.capitalize()} {dummy_last_name}"

    new_text, count = regex.subn(repl, text)
    return new_text, count > 0


# -------------------------------
# PII DETECTION AND REPORTING
# -------------------------------
def escape_for_regex(text):
    """Escape text for regex, handling spaces properly."""
    escaped = re.escape(text)
    # Replace escaped spaces with \s+ pattern
    escaped = escaped.replace(r'\ ', r'\s+')
    return escaped


def detect_remaining_pii(text, pii_data, page_number=None):
    """
    Detect which PII values are still present in the text.
    Returns a dictionary of detected PII with context.
    """
    detected = defaultdict(list)
    
    for key, original in pii_data.items():
        # Skip if original is None or empty
        if not original or not isinstance(original, str):
            continue
        
        # Create multiple search patterns for each PII value
        patterns = []
        
        # 1. Exact match (case-insensitive)
        escaped = re.escape(original)
        patterns.append(re.compile(rf'\b{escaped}\b', re.IGNORECASE))
        
        # 2. Match with spaces normalized
        # Replace spaces in original with \s+ pattern
        space_pattern = original.replace(' ', r'\s+')
        escaped_space = re.escape(space_pattern).replace(r'\\s\+', r'\s+')
        patterns.append(re.compile(rf'\b{escaped_space}\b', re.IGNORECASE))
        
        # 3. For names, try partial matches
        if key.lower() in ["patient name", "patient", "person", "name"]:
            name_parts = original.split()
            if len(name_parts) >= 2:
                # Match first name alone
                patterns.append(re.compile(rf'\b{re.escape(name_parts[0])}\b', re.IGNORECASE))
                # Match last name alone
                patterns.append(re.compile(rf'\b{re.escape(name_parts[-1])}\b', re.IGNORECASE))
        
        # 4. For dates in various formats
        if any(date_key in key.lower() for date_key in ["date", "dob", "birth"]):
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            ]
            for date_pattern in date_patterns:
                patterns.append(re.compile(date_pattern, re.IGNORECASE))
        
        # 5. For phone numbers
        if any(phone_key in key.lower() for phone_key in ["phone", "mobile", "contact"]):
            phone_pattern = r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
            patterns.append(re.compile(phone_pattern))
        
        # 6. For emails
        if "email" in key.lower():
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            patterns.append(re.compile(email_pattern, re.IGNORECASE))
        
        # Search for matches
        for pattern in patterns:
            try:
                matches = pattern.finditer(text)
                for match in matches:
                    matched_text = match.group()
                    
                    # Skip if match is too short (likely false positive)
                    if len(matched_text.strip()) < 2:
                        continue
                    
                    # Get context (50 chars before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Clean up context
                    if start > 0:
                        context = "..." + context
                    if end < len(text):
                        context = context + "..."
                    
                    # Check if this is similar to original (fuzzy match)
                    # Simple check: if original is in matched text or vice versa
                    original_lower = original.lower()
                    matched_lower = matched_text.lower()
                    
                    if (original_lower in matched_lower or 
                        matched_lower in original_lower or
                        len(original_lower) > 3 and any(
                            part in matched_lower for part in original_lower.split()
                        )):
                        
                        detected[key].append({
                            "original_value": original,
                            "matched_text": matched_text,
                            "context": context,
                            "page": page_number,
                            "position": match.start()
                        })
            except re.error as e:
                # Skip patterns that cause regex errors
                print(f"⚠ Regex error for pattern {pattern.pattern if hasattr(pattern, 'pattern') else pattern}: {e}")
                continue
    
    return dict(detected)


def analyze_pii_detection(pages_text, pii_data):
    """
    Analyze PII detection across all pages.
    """
    all_detected = defaultdict(list)
    
    # Split text by pages
    page_pattern = re.compile(r'\n={60}\nPAGE (\d+)\n={60}\n')
    page_matches = list(page_pattern.finditer(pages_text))
    
    if not page_matches:
        # If no page markers found, treat entire text as page 1
        detected = detect_remaining_pii(pages_text, pii_data, 1)
        for key, items in detected.items():
            all_detected[key].extend(items)
    else:
        for i, match in enumerate(page_matches):
            page_num = int(match.group(1))
            
            # Extract text for this page
            start_pos = match.end()
            end_pos = page_matches[i+1].start() if i+1 < len(page_matches) else len(pages_text)
            page_text = pages_text[start_pos:end_pos]
            
            # Detect PII on this page
            detected = detect_remaining_pii(page_text, pii_data, page_num)
            
            # Merge results
            for key, items in detected.items():
                all_detected[key].extend(items)
    
    # Remove duplicates (same PII value in same position)
    for key in all_detected:
        unique_items = []
        seen_positions = set()
        for item in all_detected[key]:
            pos_key = (item['page'], item['position'])
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_items.append(item)
        all_detected[key] = unique_items
    
    return dict(all_detected)


def generate_pii_report(detected_pii, sanitized_text, pii_data, dummy_data):
    """
    Generate a comprehensive report of remaining PII.
    """
    report = {
        "summary": {
            "total_pii_items": len(pii_data),
            "items_with_remaining_pii": len(detected_pii),
            "total_remaining_instances": sum(len(items) for items in detected_pii.values())
        },
        "details": {},
        "recommendations": []
    }
    
    # Add details for each PII type
    for key, items in detected_pii.items():
        report["details"][key] = {
            "original_value": pii_data.get(key, "N/A"),
            "dummy_value": dummy_data.get(key, "N/A"),
            "remaining_instances": len(items),
            "locations": items
        }
    
    # Generate recommendations
    if detected_pii:
        report["recommendations"].append(
            "Review the remaining PII instances and consider:"
        )
        report["recommendations"].append(
            "1. Adding more robust regex patterns for detected values"
        )
        report["recommendations"].append(
            "2. Checking if PII values have OCR errors (e.g., '0' vs 'O')"
        )
        report["recommendations"].append(
            "3. Adding manual replacement rules for specific patterns"
        )
    else:
        report["recommendations"].append(
            "All PII appears to have been successfully sanitized!"
        )
    
    # Find pages with remaining PII
    pages_with_pii = set()
    for items in detected_pii.values():
        for item in items:
            pages_with_pii.add(item['page'])
    
    report["summary"]["pages_with_remaining_pii"] = sorted(list(pages_with_pii))
    
    return report


# -------------------------------
# PII REPLACEMENT
# -------------------------------
def replace_pii(text, pii_data, dummy_data):
    replace_map = {}
    
    for key, original in pii_data.items():
        dummy = dummy_data.get(key)
        if not dummy:
            continue

        replaced = False

        # 1️⃣ Labeled replacement
        new_text = safe_replace(text, original, dummy)
        if new_text != text:
            replaced = True
            text = new_text

        # 2️⃣ Patient name logic (full + honorific)
        if key.lower() in ["patient name", "patient", "person"]:
            # Free-floating full name
            text, free_replaced = replace_free_floating_name(
                text, original, dummy
            )

            # Honorific + last name
            dummy_last = dummy.split()[-1]
            text, honorific_replaced = replace_honorific_name(
                text, original, dummy_last
            )

            replaced = replaced or free_replaced or honorific_replaced

        if replaced:
            replace_map[key] = {
                "original": original,
                "dummy": dummy,
                "type": (
                    "labeled+free+honorific"
                    if key.lower() in ["patient name", "patient", "person"]
                    else "labeled"
                )
            }

    return text, replace_map


# -------------------------------
# RUN PIPELINE
# -------------------------------
def run_pipeline():
    os.makedirs("output", exist_ok=True)

    pdf_path = "input/2024.03.07 Imaging Healthcare Specialists.pdf"

    # 1️⃣ ADVANCED EXTRACTION
    print("🔍 Extracting PDF text...")
    extracted_text = extract_pdf_to_text(pdf_path)
    with open("output/extracted.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)
    print("✅ Extraction complete")

    # 2️⃣ LOAD PII
    print("📋 Loading PII data...")
    pii_data = load_json("pii.json")
    dummy_data = load_json("dummy_val.json")
    print(f"✅ Loaded {len(pii_data)} PII items")

    # 3️⃣ SANITIZE
    print("🛡️ Sanitizing PII...")
    sanitized_text, replace_map = replace_pii(
        extracted_text, pii_data, dummy_data
    )

    # 4️⃣ DETECT REMAINING PII
    print("🔎 Detecting remaining PII...")
    detected_pii = analyze_pii_detection(sanitized_text, pii_data)
    
    # 5️⃣ GENERATE REPORT
    pii_report = generate_pii_report(detected_pii, sanitized_text, pii_data, dummy_data)
    
    # 6️⃣ SAVE OUTPUTS
    print("💾 Saving outputs...")
    with open("output/sanitized.txt", "w", encoding="utf-8") as f:
        f.write(sanitized_text)

    with open("output/replace.json", "w", encoding="utf-8") as f:
        json.dump(replace_map, f, indent=4)
    
    with open("output/pii_report.json", "w", encoding="utf-8") as f:
        json.dump(pii_report, f, indent=4)
    
    # 7️⃣ PRINT SUMMARY
    print("\n" + "="*60)
    print("PII SANITIZATION REPORT")
    print("="*60)
    print(f"Total PII items: {pii_report['summary']['total_pii_items']}")
    print(f"Items with remaining PII: {pii_report['summary']['items_with_remaining_pii']}")
    print(f"Total remaining instances: {pii_report['summary']['total_remaining_instances']}")
    
    if pii_report['summary']['pages_with_remaining_pii']:
        print(f"Pages with remaining PII: {pii_report['summary']['pages_with_remaining_pii']}")
    
    if detected_pii:
        print("\n⚠️ REMINING PII DETECTED:")
        for key, items in detected_pii.items():
            print(f"\n  {key}:")
            print(f"    Original: {pii_data.get(key, 'N/A')}")
            print(f"    Dummy: {dummy_data.get(key, 'N/A')}")
            print(f"    Remaining instances: {len(items)}")
            for i, item in enumerate(items[:3], 1):  # Show first 3 instances
                print(f"    Instance {i}: Page {item['page']}")
                print(f"      Matched: '{item['matched_text']}'")
                print(f"      Context: {item['context']}")
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more instances")
    else:
        print("\n✅ All PII successfully sanitized!")
    
    print("\n📁 Output files created:")
    print("  - output/extracted.txt (raw extracted text)")
    print("  - output/sanitized.txt (PII-replaced text)")
    print("  - output/replace.json (replacement mapping)")
    print("  - output/pii_report.json (detailed PII detection report)")
    
    # 8️⃣ Save a simple text summary
    with open("output/pii_summary.txt", "w", encoding="utf-8") as f:
        f.write("PII SANITIZATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total PII items: {pii_report['summary']['total_pii_items']}\n")
        f.write(f"Items with remaining PII: {pii_report['summary']['items_with_remaining_pii']}\n")
        f.write(f"Total remaining instances: {pii_report['summary']['total_remaining_instances']}\n")
        
        if pii_report['summary']['pages_with_remaining_pii']:
            f.write(f"Pages with remaining PII: {pii_report['summary']['pages_with_remaining_pii']}\n")
        
        if detected_pii:
            f.write("\nREMAINING PII DETAILS:\n")
            for key, items in detected_pii.items():
                f.write(f"\n{key}:\n")
                f.write(f"  Original: {pii_data.get(key, 'N/A')}\n")
                f.write(f"  Dummy: {dummy_data.get(key, 'N/A')}\n")
                f.write(f"  Instances: {len(items)}\n")
                for item in items:
                    f.write(f"  - Page {item['page']}: '{item['matched_text']}'\n")
        else:
            f.write("\n✅ All PII successfully sanitized!\n")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    start_time = time.time()
    run_pipeline()
    end_time = time.time()
    print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")