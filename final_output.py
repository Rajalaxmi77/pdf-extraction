import json
import re
import os
from typing import Dict, List, Set, Tuple, Any
import argparse

def load_json_file(file_path: str) -> Dict[str, List[str]]:
    """Load JSON file containing key-value pairs."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_string(s: str) -> str:
    """Normalize string by removing extra spaces, punctuation, and standardizing case."""
    if not s:
        return s
    
    # Remove extra whitespace
    s = ' '.join(s.split())
    
    # Remove common punctuation but keep important ones
    # Keep: @ . - _ / (for emails, dates, etc.)
    # Remove: , ; : ! ? " ' ( ) [ ] { }
    punctuation_to_remove = ',;:!?"\'()[]{}'
    for char in punctuation_to_remove:
        s = s.replace(char, ' ')
    
    # Clean up extra spaces again
    s = ' '.join(s.split())
    
    return s.strip()

def create_value_mapping(pii_data: Dict[str, List[str]], dummy_data: Dict[str, List[str]]) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    """
    Create mapping from original PII values to dummy values.
    Returns:
        - value_mapping: Dict[original_value] = dummy_value
        - replacement_log: Structured log with Original->DUMMY mappings
    """
    value_mapping = {}
    replacement_log = {
        "summary": {
            "total_original_values": 0,
            "total_dummy_values": 0,
            "mapped_pairs": 0
        },
        "mappings_by_category": {},
        "all_mappings": []
    }
    
    # Get all keys that exist in both dictionaries
    common_keys = set(pii_data.keys()) & set(dummy_data.keys())
    
    print(f"Processing {len(common_keys)} common keys...")
    
    for key in common_keys:
        pii_values = pii_data[key]
        dummy_values = dummy_data[key]
        
        # Use the shorter list length
        min_length = min(len(pii_values), len(dummy_values))
        
        if min_length == 0:
            continue
        
        print(f"  Processing key: {key} ({min_length} values)")
        
        # Initialize category in log
        replacement_log["mappings_by_category"][key] = {
            "original_values": [],
            "dummy_values": [],
            "mappings": []
        }
        
        for i in range(min_length):
            original = str(pii_values[i]).strip()
            dummy = str(dummy_values[i]).strip()
            
            if not original or not dummy:
                continue
            
            # Store the direct mapping
            value_mapping[original] = dummy
            
            # Generate variations for better matching
            variations = generate_variations(original)
            for variation in variations:
                if variation and variation not in value_mapping:
                    value_mapping[variation] = dummy
            
            # Also map the normalized version
            normalized = normalize_string(original)
            if normalized and normalized != original and normalized not in value_mapping:
                value_mapping[normalized] = dummy
            
            # Add to log
            mapping_entry = {
                "Original": original,
                "DUMMY": dummy,
                "variations": list(variations) if variations else [],
                "normalized": normalized if normalized != original else None
            }
            
            replacement_log["mappings_by_category"][key]["mappings"].append(mapping_entry)
            replacement_log["mappings_by_category"][key]["original_values"].append(original)
            replacement_log["mappings_by_category"][key]["dummy_values"].append(dummy)
            
            # Add to all mappings list
            replacement_log["all_mappings"].append({
                "category": key,
                "Original": original,
                "DUMMY": dummy
            })
            
            # Update summary
            replacement_log["summary"]["mapped_pairs"] += 1
        
        replacement_log["summary"]["total_original_values"] += len(pii_values)
        replacement_log["summary"]["total_dummy_values"] += len(dummy_values)
    
    return value_mapping, replacement_log

def generate_variations(value: str) -> Set[str]:
    """Generate common variations of a value for better matching."""
    variations = set()
    
    if not value:
        return variations
    
    # Original value
    variations.add(value)
    
    # Case variations
    variations.add(value.lower())
    variations.add(value.upper())
    variations.add(value.title())
    
    # Common formatting variations
    # Handle spaces
    if ' ' in value:
        variations.add(value.replace(' ', ''))  # No spaces
        variations.add(value.replace(' ', '-'))  # Hyphens instead of spaces
        variations.add(value.replace(' ', '_'))  # Underscores instead of spaces
        
        # Try different spacing
        parts = value.split()
        if len(parts) == 2:
            # For names: "John Doe" -> "Doe, John"
            variations.add(f"{parts[1]}, {parts[0]}")
            variations.add(f"{parts[1]}, {parts[0][0]}.")  # Last, First Initial
            variations.add(f"{parts[0]} {parts[1][0]}.")   # First Last Initial
    
    # Handle hyphens and underscores
    if '-' in value:
        variations.add(value.replace('-', ' '))
        variations.add(value.replace('-', ''))
    
    if '_' in value:
        variations.add(value.replace('_', ' '))
        variations.add(value.replace('_', ''))
    
    # Handle periods
    if '.' in value:
        variations.add(value.replace('.', ''))
        variations.add(value.replace('.', ' '))
    
    # Handle parentheses
    if '(' in value or ')' in value:
        no_paren = re.sub(r'[()]', '', value)
        variations.add(no_paren)
        variations.add(no_paren.strip())
    
    # Handle common prefixes/suffixes
    for prefix in ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Mr', 'Mrs', 'Ms', 'Dr']:
        if value.startswith(prefix):
            without_prefix = value[len(prefix):].strip()
            variations.add(without_prefix)
            variations.add(f"{prefix} {without_prefix}")
    
    # Normalized version
    normalized = normalize_string(value)
    if normalized:
        variations.add(normalized)
    
    # Remove empty strings
    variations.discard('')
    
    return variations

def sanitize_content(content: str, value_mapping: Dict[str, str], replacement_log: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Replace PII values in content with dummy values.
    Returns:
        - sanitized_content: The sanitized text
        - updated_log: Log with replacement statistics
    """
    print(f"Starting sanitization with {len(value_mapping)} mappings...")
    
    sanitized = content
    replacement_stats = {
        "total_replacements": 0,
        "unique_originals_replaced": 0,
        "replacements_by_category": {},
        "detailed_replacements": []
    }
    
    # Initialize stats by category
    for category in replacement_log["mappings_by_category"]:
        replacement_stats["replacements_by_category"][category] = {
            "total_originals": len(replacement_log["mappings_by_category"][category]["original_values"]),
            "replaced_count": 0,
            "unreplaced_originals": []
        }
    
    # Track which originals were replaced
    replaced_originals = set()
    
    # Process mappings in order (longest originals first)
    sorted_mappings = sorted(value_mapping.items(), key=lambda x: len(x[0]), reverse=True)
    
    for original, dummy in sorted_mappings:
        # Try exact match first
        if original in sanitized:
            count = sanitized.count(original)
            sanitized = sanitized.replace(original, dummy)
            
            # Log this replacement
            replacement_stats["detailed_replacements"].append({
                "Original": original,
                "DUMMY": dummy,
                "method": "exact_match",
                "count": count,
                "replaced_text": original,
                "replacement_text": dummy
            })
            
            replacement_stats["total_replacements"] += count
            replaced_originals.add(original)
            print(f"  Replaced '{original}' with '{dummy}' ({count} times)")
        
        # Try case-insensitive replacement
        elif original.lower() != original:
            # Create case-insensitive pattern
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            matches = list(pattern.finditer(sanitized))
            
            if matches:
                # Replace all matches
                for match in reversed(matches):  # Reverse to maintain positions
                    start, end = match.span()
                    matched_text = sanitized[start:end]
                    sanitized = sanitized[:start] + dummy + sanitized[end:]
                
                replacement_stats["detailed_replacements"].append({
                    "Original": original,
                    "DUMMY": dummy,
                    "method": "case_insensitive",
                    "count": len(matches),
                    "replaced_text": matches[0].group() if matches else original,
                    "replacement_text": dummy
                })
                
                replacement_stats["total_replacements"] += len(matches)
                replaced_originals.add(original)
                print(f"  Replaced case-insensitive '{original}' with '{dummy}' ({len(matches)} times)")
    
    # Update category statistics
    for category in replacement_log["mappings_by_category"]:
        category_data = replacement_log["mappings_by_category"][category]
        category_stats = replacement_stats["replacements_by_category"][category]
        
        for mapping in category_data["mappings"]:
            original = mapping["Original"]
            if original in replaced_originals:
                category_stats["replaced_count"] += 1
            else:
                category_stats["unreplaced_originals"].append(original)
    
    replacement_stats["unique_originals_replaced"] = len(replaced_originals)
    
    print(f"\nSanitization complete.")
    print(f"Total replacements made: {replacement_stats['total_replacements']}")
    print(f"Unique originals replaced: {replacement_stats['unique_originals_replaced']}")
    
    # Add replacement stats to the main log
    replacement_log["replacement_stats"] = replacement_stats
    
    return sanitized, replacement_log

def save_replacement_log(replacement_log: Dict[str, Any], output_dir: str, filename: str = "final-replaced.json") -> str:
    """Save the replacement log to a JSON file."""
    output_path = os.path.join(output_dir, filename)
    
    print(f"\nSaving replacement log to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(replacement_log, f, indent=2, ensure_ascii=False)
    
    return output_path

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Sanitize text file by replacing PII values with dummy data.'
    )
    parser.add_argument(
        '--root',
        default='.',
        help='Root directory path (default: current directory)'
    )
    parser.add_argument(
        '--pii', 
        default='merged-pii.json',
        help='PII JSON filename (default: merged-pii.json)'
    )
    parser.add_argument(
        '--dummy',
        default='dummy_val.json',
        help='Dummy values JSON filename (default: dummy_val.json)'
    )
    parser.add_argument(
        '--input',
        default='sanitized.txt',
        help='Input text filename (default: sanitized.txt)'
    )
    parser.add_argument(
        '--output',
        default='final-sanitized.txt',
        help='Output text filename (default: final-sanitized.txt)'
    )
    parser.add_argument(
        '--replacement-log',
        default='final-replaced.json',
        help='Replacement log filename (default: final-replaced.json)'
    )
    
    args = parser.parse_args()
    
    # Construct file paths
    output_dir = os.path.join(args.root, 'output')
    
    pii_path = os.path.join(output_dir, args.pii)
    dummy_path = os.path.join(args.root, args.dummy)
    input_path = os.path.join(output_dir, args.input)
    output_path = os.path.join(output_dir, args.output)
    log_path = os.path.join(output_dir, args.replacement_log)
    
    print(f"Root directory: {args.root}")
    print(f"Output directory: {output_dir}")
    
    # Check if files exist
    files_to_check = [
        (pii_path, "PII JSON"),
        (dummy_path, "Dummy JSON"),
        (input_path, "Input text")
    ]
    
    missing_files = []
    for path, name in files_to_check:
        if not os.path.exists(path):
            missing_files.append((name, path))
    
    if missing_files:
        print("\nError: The following files were not found:")
        for name, path in missing_files:
            print(f"  {name}: {path}")
        print("\nPlease ensure all files exist in the output directory.")
        return
    
    # Load data files
    print(f"\nLoading PII data from: {pii_path}")
    pii_data = load_json_file(pii_path)
    
    print(f"Loading dummy data from: {dummy_path}")
    dummy_data = load_json_file(dummy_path)
    
    print(f"\nOriginal file stats:")
    print(f"  PII keys: {len(pii_data)}")
    print(f"  Dummy keys: {len(dummy_data)}")
    
    # Check if keys match
    pii_keys = set(pii_data.keys())
    dummy_keys = set(dummy_data.keys())
    common_keys = pii_keys & dummy_keys
    
    if len(common_keys) == 0:
        print("\nError: No common keys found between PII and dummy data!")
        return
    
    print(f"\nCreating value mapping...")
    value_mapping, replacement_log = create_value_mapping(pii_data, dummy_data)
    
    print(f"Reading input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    print(f"\nInput file stats:")
    print(f"  Size: {len(original_content):,} characters")
    print(f"  Lines: {original_content.count(chr(10)) + 1}")
    
    print(f"\nSanitizing content...")
    sanitized_content, replacement_log = sanitize_content(original_content, value_mapping, replacement_log)
    
    # Add file information to log
    replacement_log["file_info"] = {
        "input_file": os.path.basename(input_path),
        "output_file": os.path.basename(output_path),
        "log_file": os.path.basename(log_path),
        "pii_file": os.path.basename(pii_path),
        "dummy_file": os.path.basename(dummy_path),
        "original_size": len(original_content),
        "sanitized_size": len(sanitized_content)
    }
    
    # Add processing summary
    replacement_log["processing_summary"] = {
        "total_mappings_created": len(value_mapping),
        "input_characters": len(original_content),
        "output_characters": len(sanitized_content),
        "replacement_ratio": f"{replacement_log['replacement_stats']['total_replacements'] / len(original_content) * 100:.2f}%" if len(original_content) > 0 else "0%"
    }
    
    print(f"\nWriting sanitized output to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sanitized_content)
    
    # Save replacement log
    save_replacement_log(replacement_log, output_dir, args.replacement_log)
    
    # Print final summary
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    stats = replacement_log["replacement_stats"]
    print(f"\nREPLACEMENT SUMMARY:")
    print(f"  Total Original-DUMMY pairs created: {replacement_log['summary']['mapped_pairs']}")
    print(f"  Total replacements made: {stats['total_replacements']}")
    print(f"  Unique originals replaced: {stats['unique_originals_replaced']}")
    
    print(f"\nREPLACEMENTS BY CATEGORY:")
    print("-" * 50)
    for category, cat_stats in stats["replacements_by_category"].items():
        total = cat_stats["total_originals"]
        replaced = cat_stats["replaced_count"]
        percentage = (replaced / total * 100) if total > 0 else 0
        print(f"  {category.ljust(20)}: {replaced}/{total} ({percentage:.1f}%)")
        if cat_stats["unreplaced_originals"]:
            print(f"    Unreplaced: {', '.join(cat_stats['unreplaced_originals'][:3])}")
            if len(cat_stats["unreplaced_originals"]) > 3:
                print(f"    ... and {len(cat_stats['unreplaced_originals']) - 3} more")
    
    print(f"\nOUTPUT FILES:")
    print(f"  Sanitized text: {output_path}")
    print(f"  Replacement log: {log_path}")
    
    # Show sample mappings
    print(f"\nSAMPLE MAPPINGS (first 5):")
    print("-" * 50)
    for i, mapping in enumerate(replacement_log["all_mappings"][:5]):
        print(f"  {i+1}. {mapping['category']}:")
        print(f"       Original: {mapping['Original']}")
        print(f"       DUMMY:    {mapping['DUMMY']}")
        print()

if __name__ == "__main__":
    main()