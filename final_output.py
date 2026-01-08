import json
import re
import os
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime

def load_json_file(file_path: str) -> Dict[str, List[str]]:
    """Load JSON file containing key-value pairs."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_string(s: str) -> str:
    """Normalize string for comparison."""
    if not s:
        return s
    # Convert to lowercase and remove common punctuation
    s = s.lower().strip()
    # Remove punctuation except basic separators
    s = re.sub(r'[,\'"!?;:()\[\]{}]', ' ', s)
    # Remove extra spaces
    s = ' '.join(s.split())
    return s

def create_mappings(pii_data: Dict[str, List[str]], dummy_data: Dict[str, List[str]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Create mappings from PII to dummy values."""
    mappings = {}
    log = {
        "summary": {"total_pairs": 0},
        "mappings": []
    }
    
    for category in pii_data:
        if category in dummy_data:
            pii_list = pii_data[category]
            dummy_list = dummy_data[category]
            
            min_len = min(len(pii_list), len(dummy_list))
            for i in range(min_len):
                original = str(pii_list[i]).strip()
                dummy = str(dummy_list[i]).strip()
                
                if original and dummy:
                    mappings[original] = dummy
                    log["mappings"].append({
                        "category": category,
                        "original": original,
                        "replaced": dummy,
                        # "status": "not_replaced"
                    })
                    log["summary"]["total_pairs"] += 1
    
    return mappings, log

def find_and_replace(content: str, mappings: Dict[str, str], log: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Find and replace PII values in content."""
    sanitized = content
    replaced_count = 0
    replaced_originals = set()
    
    # Sort by longest first to avoid partial replacements
    sorted_items = sorted(mappings.items(), key=lambda x: len(x[0]), reverse=True)
    
    for original, dummy in sorted_items:
        if original in sanitized:
            # Count occurrences
            count = sanitized.count(original)
            sanitized = sanitized.replace(original, dummy)
            replaced_count += count
            replaced_originals.add(original)
            
            # Update log
            for mapping in log["mappings"]:
                if mapping["original"] == original:
                    mapping["status"] = "replaced"
                    mapping["count"] = count
                    break
    
    log["summary"]["replaced_count"] = replaced_count
    log["summary"]["unique_replaced"] = len(replaced_originals)
    
    return sanitized, log

def check_pii_in_text(pii_data: Dict[str, List[str]], text: str) -> Dict[str, Any]:
    """Check which PII values exist in the text."""
    results = {
        "found": {},
        "not_found": {},
        "summary": {
            "total_checked": 0,
            "found_count": 0,
            "not_found_count": 0
        }
    }
    
    for category, values in pii_data.items():
        found_in_category = []
        not_found_in_category = []
        
        for value in values:
            if not isinstance(value, str):
                continue
                
            original = value.strip()
            if not original:
                continue
            
            results["summary"]["total_checked"] += 1
            
            # Check if value exists in text
            if original in text:
                found_in_category.append({
                    "value": original,
                    "count": text.count(original)
                })
                results["summary"]["found_count"] += 1
            else:
                # Try case-insensitive search
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                matches = list(pattern.finditer(text))
                if matches:
                    found_in_category.append({
                        "value": original,
                        "count": len(matches),
                        "note": "Case-insensitive match"
                    })
                    results["summary"]["found_count"] += 1
                else:
                    not_found_in_category.append(original)
                    results["summary"]["not_found_count"] += 1
        
        if found_in_category:
            results["found"][category] = found_in_category
        if not_found_in_category:
            results["not_found"][category] = not_found_in_category
    
    return results

def check_pii_in_final_text(pii_data: Dict[str, List[str]], final_text: str) -> Dict[str, Any]:
    """Check if any original PII values remain in the final sanitized text."""
    print("\n5. Checking for PII in final-sanitized.txt...")
    
    results = {
        "found_pii": {},
        "summary": {
            "total_pii_found": 0,
            "total_occurrences": 0,
            "categories_with_pii": []
        }
    }
    
    for category, values in pii_data.items():
        found_in_category = []
        
        for value in values:
            if not isinstance(value, str):
                continue
                
            original = value.strip()
            if not original:
                continue
            
            # Check if original PII exists in final text
            if original in final_text:
                count = final_text.count(original)
                found_in_category.append({
                    "value": original,
                    "count": count
                })
                results["summary"]["total_pii_found"] += 1
                results["summary"]["total_occurrences"] += count
            else:
                # Try case-insensitive search
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                matches = list(pattern.finditer(final_text))
                if matches:
                    found_in_category.append({
                        "value": original,
                        "count": len(matches),
                        "note": "Case-insensitive match"
                    })
                    results["summary"]["total_pii_found"] += 1
                    results["summary"]["total_occurrences"] += len(matches)
        
        if found_in_category:
            results["found_pii"][category] = found_in_category
            results["summary"]["categories_with_pii"].append(category)
    
    return results

def generate_summary(pii_check: Dict[str, Any], replacement_log: Dict[str, Any], 
                    original_content: str, sanitized_content: str,
                    final_pii_check: Dict[str, Any], output_dir: str) -> str:
    """Generate final summary report."""
    summary_path = os.path.join(output_dir, "final_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PII SANITIZATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Section 1: PII Detection in Original Text
        f.write("1. PII DETECTION IN ORIGINAL TEXT:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total PII values in merged-pii.json: {pii_check['summary']['total_checked']}\n")
        f.write(f"Found in sanitized.txt: {pii_check['summary']['found_count']}\n")
        f.write(f"Not found in sanitized.txt: {pii_check['summary']['not_found_count']}\n\n")
        
        # Show categories with found PII
        if pii_check["found"]:
            f.write("Categories with PII found:\n")
            for category, items in pii_check["found"].items():
                f.write(f"  {category}: {len(items)} values\n")
                # Show first 3 values
                for i, item in enumerate(items[:3]):
                    count_text = f" ({item['count']}x)" if item.get('count', 0) > 1 else ""
                    f.write(f"    • {item['value']}{count_text}")
                    if item.get('note'):
                        f.write(f" [{item['note']}]")
                    f.write("\n")
                if len(items) > 3:
                    f.write(f"    ... and {len(items) - 3} more\n")
        f.write("\n")
        
        # Section 2: Replacement Results
        f.write("2. REPLACEMENT RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total PII-Dummy mappings: {replacement_log['summary']['total_pairs']}\n")
        f.write(f"Total replacements made: {replacement_log['summary'].get('replaced_count', 0)}\n")
        f.write(f"Unique PII values replaced: {replacement_log['summary'].get('unique_replaced', 0)}\n\n")
        
        # Count replaced vs not replaced
        replaced = len([m for m in replacement_log["mappings"] if m.get("status") == "replaced"])
        not_replaced = len([m for m in replacement_log["mappings"] if m.get("status") == "not_replaced"])
        
        f.write(f"Successfully replaced: {replaced}/{replacement_log['summary']['total_pairs']}\n")
        if not_replaced > 0:
            f.write(f"Not replaced: {not_replaced} (PII not found in text)\n")
        
        
        
        # Section 3: Final Sanitized Text Analysis
        f.write("3. FINAL SANITIZED TEXT ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        if final_pii_check["summary"]["total_pii_found"] == 0:
            f.write("✅ No original PII values found in final-sanitized.txt\n")
            f.write("All PII has been successfully replaced with dummy values.\n")
        else:
            f.write(f"⚠️  {final_pii_check['summary']['total_pii_found']} original PII values found ")
            f.write(f"({final_pii_check['summary']['total_occurrences']} total occurrences) ")
            f.write("in final-sanitized.txt\n\n")
            
            f.write("Categories with remaining PII:\n")
            for category in final_pii_check["summary"]["categories_with_pii"]:
                items = final_pii_check["found_pii"][category]
                f.write(f"  {category}: {len(items)} values\n")
                for i, item in enumerate(items[:5]):  # Show first 5
                    count_text = f" ({item['count']}x)" if item.get('count', 0) > 1 else ""
                    f.write(f"    • {item['value']}{count_text}")
                    if item.get('note'):
                        f.write(f" [{item['note']}]")
                    f.write("\n")
                if len(items) > 5:
                    f.write(f"    ... and {len(items) - 5} more\n")
        f.write("\n")
        
        # Section 4: File Information
        f.write("4. FILE INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Input file size: {len(original_content):,} characters\n")
        f.write(f"Output file size: {len(sanitized_content):,} characters\n")
        f.write(f"Size change: {len(sanitized_content) - len(original_content):+,} characters\n")
        
        # Section 5: Final Status
        f.write("\n" + "=" * 60 + "\n")
        f.write("FINAL STATUS\n")
        f.write("=" * 60 + "\n")
        
        # Determine overall success
        all_replaced_in_final = (final_pii_check["summary"]["total_pii_found"] == 0)
        
        if all_replaced_in_final:
            if replaced == replacement_log['summary']['total_pairs']:
                f.write("✅ PERFECT SUCCESS: All PII values have been replaced!\n")
                f.write("   - All PII found in original text was replaced\n")
                f.write("   - No original PII remains in final text\n")
            else:
                f.write("✅ SUCCESS: All PII that was found has been replaced!\n")
                f.write("   - Some PII values were not in the original text\n")
                f.write("   - No original PII remains in final text\n")
        else:
            f.write("⚠️  ISSUES DETECTED: Original PII remains in final text\n")
            f.write("   Recommendations:\n")
            f.write("   1. Check the mapping logic\n")
            f.write("   2. Verify PII values have proper dummy mappings\n")
            f.write("   3. Check for case sensitivity or formatting issues\n")
        
        # Add timestamp
        f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return summary_path

def main():
    parser = argparse.ArgumentParser(description='Sanitize PII from text file')
    parser.add_argument('--root', default='.', help='Root directory')
    parser.add_argument('--pii', default='merged-pii.json', help='PII JSON file')
    parser.add_argument('--dummy', default='dummy_val.json', help='Dummy values JSON file')
    parser.add_argument('--input', default='sanitized.txt', help='Input text file')
    parser.add_argument('--output', default='final-sanitized.txt', help='Output text file')
    parser.add_argument('--log', default='final-replaced.json', help='Replacement log file')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = os.path.join(args.root, 'output')
    pii_path = os.path.join(output_dir, args.pii)
    dummy_path = os.path.join(args.root, args.dummy)
    input_path = os.path.join(output_dir, args.input)
    output_path = os.path.join(output_dir, args.output)
    log_path = os.path.join(output_dir, args.log)
    
    # Check files exist
    for path, name in [(pii_path, "PII"), (dummy_path, "Dummy"), (input_path, "Input")]:
        if not os.path.exists(path):
            print(f"Error: {name} file not found: {path}")
            return
    
    # Load data
    pii_data = load_json_file(pii_path)
    dummy_data = load_json_file(dummy_path)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    print(f"Loaded {len(original_content)} characters from {args.input}")
    
    # Step 1: Check which PII values exist in the original text
    print("\n1. Checking which PII values exist in the text...")
    pii_check = check_pii_in_text(pii_data, original_content)
    print(f"   Found: {pii_check['summary']['found_count']}")
    print(f"   Not found: {pii_check['summary']['not_found_count']}")
    
    # Step 2: Create mappings and replace
    print("\n2. Creating mappings and replacing PII...")
    mappings, replacement_log = create_mappings(pii_data, dummy_data)
    print(f"   Created {len(mappings)} mappings")
    
    sanitized_content, replacement_log = find_and_replace(original_content, mappings, replacement_log)
    print(f"   Made {replacement_log['summary'].get('replaced_count', 0)} replacements")
    print(f"   Unique PII replaced: {replacement_log['summary'].get('unique_replaced', 0)}")
    
    # Step 3: Save output files
    print("\n3. Saving output files...")
    
    # Save sanitized text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sanitized_content)
    print(f"   Saved sanitized text: {output_path}")
    
    # Step 4: Check for PII in final sanitized text
    print("\n4. Checking for PII in final-sanitized.txt...")
    final_pii_check = check_pii_in_final_text(pii_data, sanitized_content)
    
    if final_pii_check["summary"]["total_pii_found"] == 0:
        print("   ✅ No original PII found in final-sanitized.txt")
    else:
        print(f"   ⚠️  Found {final_pii_check['summary']['total_pii_found']} original PII values")
        print(f"      ({final_pii_check['summary']['total_occurrences']} total occurrences)")
        print(f"      Categories affected: {', '.join(final_pii_check['summary']['categories_with_pii'])}")
    
    # Step 5: Save replacement log with timestamp
    print("\n5. Saving replacement log...")
    replacement_log["timestamp"] = datetime.now().isoformat()
    replacement_log["final_pii_check"] = final_pii_check["summary"]
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(replacement_log, f, indent=2, ensure_ascii=False)
    print(f"   Saved replacement log: {log_path}")
    
    # Step 6: Generate summary
    print("\n6. Generating summary report...")
    summary_path = generate_summary(
        pii_check, replacement_log, 
        original_content, sanitized_content,
        final_pii_check, output_dir
    )
    print(f"   Saved summary: {summary_path}")
    
    # Final message
    print("\n" + "="*60)
    print("PROCESS COMPLETE")
    print("="*60)
    
    # Show quick stats
    total_mappings = replacement_log['summary']['total_pairs']
    replaced = len([m for m in replacement_log["mappings"] if m.get("status") == "replaced"])
    pii_in_final = final_pii_check["summary"]["total_pii_found"]
    
    if pii_in_final == 0:
        print("✅ SUCCESS: No original PII found in final-sanitized.txt")
    else:
        print(f"⚠️  WARNING: {pii_in_final} original PII values found in final-sanitized.txt")
        print(f"   ({final_pii_check['summary']['total_occurrences']} total occurrences)")
    
    print(f"\nReplacement: {replaced}/{total_mappings} PII values replaced")
    print(f"   ({total_mappings - replaced} were not found in original text)")
    
    print(f"\nOutput files in {output_dir}:")
    print(f"  • {args.output} - Sanitized text")
    print(f"  • {args.log} - Replacement details")
    print(f"  • final_summary.txt - Summary report")

if __name__ == "__main__":
    main()