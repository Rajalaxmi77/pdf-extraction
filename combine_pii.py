import json
import re
import os
from typing import Dict, List


# -------------------------------
# NORMALIZATION
# -------------------------------
def normalize(value: str) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - remove spaces and common separators
    """
    return re.sub(r"[\s\-\(\)\.,#]", "", value.lower())


# -------------------------------
# JSON HELPERS
# -------------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# -------------------------------
# PII REGISTRY NORMALIZATION
# -------------------------------
def ensure_list(value) -> List[str]:
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


def normalize_pii_registry(raw_pii: dict) -> Dict[str, List[str]]:
    """
    Force every PII field to be an array
    """
    normalized = {}
    for field, value in raw_pii.items():
        normalized[field] = ensure_list(value)
    return normalized


# -------------------------------
# CORE MERGE LOGIC
# -------------------------------
def merge_pii_registry(
    pii_path="pii.json",
    detected_path="detected-leaks.json",
    output_dir="output"
):
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

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "merged-pii.json")
    save_json(output_path, pii_registry)

    return output_path


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    output = merge_pii_registry()
    print(f"✅ Merged PII registry generated at: {output}")