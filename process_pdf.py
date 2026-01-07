import os
import re
import json
import time
import io
from pathlib import Path
from collections import defaultdict
import random

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
# NORMALIZATION AND MERGE FUNCTIONS
# -------------------------------
def normalize(value: str) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - remove spaces and common separators
    """
    return re.sub(r"[\s\-\(\)\.,#]", "", value.lower())


def ensure_list(value) -> list:
    """
    Convert:
    - string -> [string]
    - list -> list
    - None -> []
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def normalize_pii_registry(raw_pii: dict) -> dict:
    """
    Force every PII field to be an array
    """
    normalized = {}
    for field, value in raw_pii.items():
        normalized[field] = ensure_list(value)
    return normalized


def merge_pii_registry(
    pii_path="pii.json",
    detected_path="detected-leaks.json",
    output_dir="output"
):
    """
    Merge pii.json and detected-leaks.json into merged-pii.json
    """
    print("🔄 Merging PII files...")
    
    # Load inputs
    raw_pii = load_json(pii_path)
    detected = load_json(detected_path)

    # Normalize pii.json (strings -> arrays)
    pii_registry = normalize_pii_registry(raw_pii)

    # Build normalized lookup for fast comparison
    normalized_lookup = {
        field: {normalize(v) for v in values}
        for field, values in pii_registry.items()
    }

    # Merge detected leaks
    for item in detected.get("leaked_fields", []):
        field = item.get("pii_type")
        value = item.get("matched_text", "").strip()

        if not field or not value:
            continue

        # Initialize unknown PII fields automatically
        pii_registry.setdefault(field, [])
        normalized_lookup.setdefault(field, set())

        norm_value = normalize(value)

        # Add only if truly new
        if norm_value not in normalized_lookup[field]:
            pii_registry[field].append(value)
            normalized_lookup[field].add(norm_value)
            print(f"  + Added new PII: {field} = {value}")

    # Write output
    output_path = os.path.join(output_dir, "merged-pii.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pii_registry, f, indent=4, ensure_ascii=False)

    print(f"✅ Merged PII registry generated at: {output_path}")
    print(f"  Total PII categories: {len(pii_registry)}")
    print(f"  Total PII values: {sum(len(v) for v in pii_registry.values())}")
    
    return pii_registry


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
# HELPER: GET DUMMY VALUE FROM ARRAY
# -------------------------------
def get_dummy_value(dummy_array, index=None):
    """
    Get a dummy value from an array.
    If index is provided, use that index, otherwise use first value.
    Returns a string.
    """
    if not dummy_array:
        return ""
    
    if isinstance(dummy_array, list):
        if index is not None and 0 <= index < len(dummy_array):
            return str(dummy_array[index])
        elif dummy_array:
            # Use first value by default
            return str(dummy_array[0])
    elif isinstance(dummy_array, str):
        return dummy_array
    
    return ""


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
    Returns: (new_text, replacement_count)
    """
    # Ensure dummy is a string
    if isinstance(dummy, list):
        dummy = get_dummy_value(dummy)
    
    norm_original = normalize_for_match(original)

    # Build regex pattern
    escaped = re.escape(norm_original)
    escaped = escaped.replace(",", r"\s*,\s*")
    escaped = escaped.replace(r"\ ", r"\s+")

    pattern = re.compile(escaped, flags=re.IGNORECASE)
    
    # Count replacements
    replacement_count = len(pattern.findall(text))
    
    # Perform replacement
    new_text = pattern.sub(dummy, text)
    
    return new_text, replacement_count


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
    """
    Replace free-floating names with dummy name.
    dummy_name can be a string or array.
    Returns: (new_text, replacement_count)
    """
    # Get dummy name as string
    if isinstance(dummy_name, list):
        dummy_name = get_dummy_value(dummy_name)
    
    regex = build_name_regex(original_name)
    if not regex:
        return text, 0

    new_text = text
    replacement_count = 0
    
    # Find all matches and replace
    for match in regex.finditer(text):
        matched_text = match.group()
        # Check if this is actually the name we're looking for (not a partial match)
        if original_name.lower() in matched_text.lower() or matched_text.lower() in original_name.lower():
            new_text = new_text[:match.start()] + dummy_name + new_text[match.end():]
            replacement_count += 1
    
    return new_text, replacement_count


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
    dummy_last_name can be a string or array.
    Returns: (new_text, replacement_count)
    """
    # Get dummy last name as string
    if isinstance(dummy_last_name, list):
        dummy_last_name = get_dummy_value(dummy_last_name)
    
    # Extract last name from dummy if it's a full name
    if dummy_last_name and " " in dummy_last_name:
        dummy_last_name = dummy_last_name.split()[-1]
    
    regex = build_honorific_name_regex(original_name)
    if not regex:
        return text, 0

    def repl(match):
        honorific = match.group(1)
        return f"{honorific.capitalize()} {dummy_last_name}"

    # Count replacements
    replacement_count = len(regex.findall(text))
    
    # Perform replacement
    new_text = regex.sub(repl, text)
    
    return new_text, replacement_count


# -------------------------------
# PII REPLACEMENT FOR ARRAY VALUES
# -------------------------------
def replace_pii_with_arrays(text, pii_data, dummy_data):
    """
    Replace PII values with dummy values.
    Both pii_data and dummy_data are dictionaries with array values.
    Returns: (new_text, replacement_map)
    """
    replacement_map = {}
    current_text = text
    
    print(f"\n🔄 Processing {len(pii_data)} PII categories...")
    
    for key, original_values in pii_data.items():
        if not original_values:
            continue
        
        dummy_array = dummy_data.get(key)
        if not dummy_array:
            print(f"⚠ No dummy data for category: {key}")
            continue
        
        # Track replacements for this category
        category_replacements = {
            "Original_values": original_values,
            "DUMMY_values": dummy_array,
            "replacements": []
        }
        
        total_category_replacements = 0
        
        # Process each original value in this category
        for original_idx, original in enumerate(original_values):
            # Get corresponding dummy value (cycle through dummy array)
            dummy_idx = original_idx % len(dummy_array)
            dummy = get_dummy_value(dummy_array, dummy_idx)
            
            if not dummy:
                continue
            
            # 1️⃣ Labeled replacement
            new_text, direct_count = safe_replace(current_text, original, dummy)
            if direct_count > 0:
                current_text = new_text
                total_category_replacements += direct_count
                category_replacements["replacements"].append({
                    "original": original,
                    "dummy": dummy,
                    "method": "labeled",
                    "count": direct_count
                })
            
            # 2️⃣ Special handling for names
            name_categories = ["patient name", "patient", "person", "name", "first name", "last name", "re"]
            if key.lower() in name_categories:
                # Free-floating full name
                new_text, free_count = replace_free_floating_name(
                    current_text, original, dummy
                )
                if free_count > 0:
                    current_text = new_text
                    total_category_replacements += free_count
                    category_replacements["replacements"].append({
                        "original": original,
                        "dummy": dummy,
                        "method": "free_floating",
                        "count": free_count
                    })
                
                # Honorific + last name
                dummy_last = dummy.split()[-1] if " " in dummy else dummy
                new_text, honorific_count = replace_honorific_name(
                    current_text, original, dummy_last
                )
                if honorific_count > 0:
                    current_text = new_text
                    total_category_replacements += honorific_count
                    category_replacements["replacements"].append({
                        "original": original,
                        "dummy": dummy_last,
                        "method": "honorific",
                        "count": honorific_count
                    })
        
        # Store category results
        replacement_map[key] = {
            "Original_values_count": len(original_values),
            "DUMMY_values_count": len(dummy_array),
            "total_replacements": total_category_replacements,
            "status": "replaced" if total_category_replacements > 0 else "not_replaced",
            "details": category_replacements["replacements"]
        }
        
        if total_category_replacements > 0:
            print(f"  ✅ {key}: {total_category_replacements} replacements")
        else:
            print(f"  ⚠ {key}: No replacements made")
    
    return current_text, replacement_map


# -------------------------------
# CREATE FINAL REPLACED JSON
# -------------------------------
def create_final_replaced_json(pii_data, dummy_data, replacement_map, extracted_text, sanitized_text):
    """
    Create a replaced.json file that tracks which original values
    were replaced with which dummy values.
    """
    final_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pii_categories": len(pii_data),
            "total_pii_values": sum(len(v) for v in pii_data.values()),
            "replaced_categories": sum(1 for item in replacement_map.values() if item.get("status") == "replaced"),
            "not_replaced_categories": sum(1 for item in replacement_map.values() if item.get("status") == "not_replaced"),
            "text_statistics": {
                "original_characters": len(extracted_text),
                "sanitized_characters": len(sanitized_text),
                "total_replacements": sum(item.get("total_replacements", 0) for item in replacement_map.values())
            }
        },
        "mappings": [],
        "summary_by_category": {}
    }
    
    # Create mappings for each PII category
    for key, original_values in pii_data.items():
        dummy_array = dummy_data.get(key, [])
        replacement_info = replacement_map.get(key, {})
        
        # Get all replacements for this category
        category_replacements = []
        for orig_idx, original in enumerate(original_values):
            dummy_idx = orig_idx % len(dummy_array) if dummy_array else 0
            dummy = get_dummy_value(dummy_array, dummy_idx)
            
            # Check if this specific original was replaced
            was_replaced = False
            replacement_count = 0
            if "details" in replacement_info:
                for detail in replacement_info["details"]:
                    if detail["original"] == original:
                        was_replaced = True
                        replacement_count += detail["count"]
            
            category_replacements.append({
                "Original": original,
                "DUMMY": dummy,
                "replaced": was_replaced,
                "replacement_count": replacement_count,
                "dummy_index": dummy_idx if dummy_array else None
            })
        
        # Create category mapping entry
        mapping_entry = {
            "category": key,
            "Original_values": original_values,
            "DUMMY_values": dummy_array,
            "status": replacement_info.get("status", "not_replaced"),
            "total_replacements": replacement_info.get("total_replacements", 0),
            "replacements": category_replacements
        }
        
        final_data["mappings"].append(mapping_entry)
        
        # Add to summary
        final_data["summary_by_category"][key] = {
            "Original_count": len(original_values),
            "DUMMY_count": len(dummy_array),
            "status": mapping_entry["status"],
            "total_replacements": mapping_entry["total_replacements"]
        }
    
    return final_data


# -------------------------------
# RUN PIPELINE
# -------------------------------
def run_pipeline():
    os.makedirs("output", exist_ok=True)

    pdf_path = "input/2024.03.04 Senta Neurosurgery.pdf"

    # 1️⃣ MERGE PII FILES
    print("🔄 Merging PII files...")
    pii_data = merge_pii_registry(
        pii_path="pii.json",
        detected_path="detected-leaks.json",
        output_dir="output"
    )
    
    # Save the merged file
    with open("output/merged-pii.json", "w", encoding="utf-8") as f:
        json.dump(pii_data, f, indent=4, ensure_ascii=False)

    # 2️⃣ LOAD DUMMY DATA
    print("\n📋 Loading dummy data...")
    dummy_data = load_json("dummy_val.json")
    print(f"✅ Loaded dummy data with {len(dummy_data)} categories")

    # 3️⃣ ADVANCED EXTRACTION
    print("\n🔍 Extracting PDF text...")
    extracted_text = extract_pdf_to_text(pdf_path)
    with open("output/extracted.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)
    print("✅ Extraction complete")
    print(f"  Extracted {len(extracted_text):,} characters")

    # 4️⃣ SANITIZE WITH MERGED PII
    print("\n🛡️ Sanitizing PII with merged data...")
    sanitized_text, replacement_map = replace_pii_with_arrays(
        extracted_text, pii_data, dummy_data
    )

    # 5️⃣ CREATE FINAL REPLACED JSON
    print("\n📝 Creating replaced.json...")
    final_replaced_data = create_final_replaced_json(
        pii_data, dummy_data, replacement_map, extracted_text, sanitized_text
    )
    
    # 6️⃣ SAVE OUTPUTS
    print("\n💾 Saving outputs...")
    with open("output/sanitized.txt", "w", encoding="utf-8") as f:
        f.write(sanitized_text)
    
    with open("output/replaced.json", "w", encoding="utf-8") as f:
        json.dump(final_replaced_data, f, indent=4)
    
    # 7️⃣ PRINT SUMMARY
    print("\n" + "="*60)
    print("PII SANITIZATION REPORT")
    print("="*60)
    
    metadata = final_replaced_data["metadata"]
    print(f"\n📊 SUMMARY:")
    print(f"  Total PII categories: {metadata['total_pii_categories']}")
    print(f"  Total PII values: {metadata['total_pii_values']}")
    print(f"  Categories replaced: {metadata['replaced_categories']}")
    print(f"  Categories not replaced: {metadata['not_replaced_categories']}")
    print(f"  Total replacements made: {metadata['text_statistics']['total_replacements']}")
    print(f"  Original text size: {metadata['text_statistics']['original_characters']:,} chars")
    print(f"  Sanitized text size: {metadata['text_statistics']['sanitized_characters']:,} chars")
    
    # Show detailed replacement status
    print(f"\n📋 REPLACEMENT DETAILS BY CATEGORY:")
    print("-" * 80)
    print(f"{'Category':<25} {'Status':<12} {'PII Values':<10} {'Replacements':<12}")
    print("-" * 80)
    
    for mapping in final_replaced_data["mappings"]:
        status_icon = "✅" if mapping["status"] == "replaced" else "❌"
        category = mapping["category"]
        status = mapping["status"]
        pii_count = len(mapping["Original_values"])
        replacements = mapping["total_replacements"]
        
        print(f"{category:<25} {status_icon} {status:<10} {pii_count:<10} {replacements:<12}")
    
    # Show sample of actual replacements
    print(f"\n🔍 SAMPLE REPLACEMENTS (first 3 categories):")
    replaced_categories = [m for m in final_replaced_data["mappings"] if m["status"] == "replaced" and m["total_replacements"] > 0]
    
    for i, category in enumerate(replaced_categories[:3]):
        print(f"\n  {i+1}. {category['category']}:")
        print(f"     Total replacements: {category['total_replacements']}")
        
        # Show up to 3 sample replacements
        sample_count = 0
        for replacement in category["replacements"]:
            if replacement["replaced"] and sample_count < 3:
                print(f"     - {replacement['Original']} → {replacement['DUMMY']} (count: {replacement['replacement_count']})")
                sample_count += 1
    
    # Show categories that weren't replaced
    not_replaced = [m for m in final_replaced_data["mappings"] if m["status"] == "not_replaced"]
    if not_replaced:
        print(f"\n⚠️ CATEGORIES NOT REPLACED ({len(not_replaced)}):")
        for i, category in enumerate(not_replaced[:5]):
            pii_values = category["Original_values"]
            sample_values = pii_values[:2] if len(pii_values) > 2 else pii_values
            print(f"  {i+1}. {category['category']}: {len(pii_values)} values")
            if sample_values:
                print(f"     Sample: {', '.join(sample_values)}")
        if len(not_replaced) > 5:
            print(f"  ... and {len(not_replaced) - 5} more categories")
    
    print(f"\n📁 Output files created:")
    print("  - output/merged-pii.json (merged PII data)")
    print("  - output/extracted.txt (raw extracted text)")
    print("  - output/sanitized.txt (PII-replaced text)")
    print("  - output/replaced.json (Original->DUMMY mapping)")
    
    # Save a simple text summary
    with open("output/pii_summary.txt", "w", encoding="utf-8") as f:
        f.write("PII SANITIZATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {metadata['timestamp']}\n")
        f.write(f"Total PII categories: {metadata['total_pii_categories']}\n")
        f.write(f"Total PII values: {metadata['total_pii_values']}\n")
        f.write(f"Categories replaced: {metadata['replaced_categories']}\n")
        f.write(f"Categories not replaced: {metadata['not_replaced_categories']}\n")
        f.write(f"Total replacements made: {metadata['text_statistics']['total_replacements']}\n\n")
        
        f.write("REPLACEMENT DETAILS BY CATEGORY:\n")
        f.write("-" * 60 + "\n")
        for mapping in final_replaced_data["mappings"]:
            status = "✓" if mapping["status"] == "replaced" else "✗"
            f.write(f"{status} {mapping['category']}: {len(mapping['Original_values'])} values, {mapping['total_replacements']} replacements\n")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    start_time = time.time()
    run_pipeline()
    end_time = time.time()
    print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")