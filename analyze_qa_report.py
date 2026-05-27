import json
import sys
from collections import Counter

def analyze_report(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        sys.exit(1)

    total_elements = 0
    flagged_elements = 0
    
    tag_errors = Counter()
    page_errors = Counter()
    reason_counts = Counter()

    # The json structure: list of page objects, each with "results"
    for page_data in data.get("results", []):
        page_num = page_data.get("page_number", "Unknown")
        results = page_data.get("data", [])
        
        for item in results:
            total_elements += 1
            if item.get("is_correct") is False:
                flagged_elements += 1
                
                tag = item.get("current_tag", "Unknown")
                tag_errors[tag] += 1
                page_errors[page_num] += 1
                
                reason = item.get("corrective_reasoning", "")
                if reason:
                    # Just grab a truncated version or the full string for analysis
                    reason_counts[reason] += 1

    print("=== QA REPORT ANALYSIS ===")
    print(f"Total Elements: {total_elements}")
    print(f"Flagged Elements: {flagged_elements}")
    if total_elements > 0:
        print(f"Error Rate: {(flagged_elements / total_elements) * 100:.2f}%\n")
    
    print("--- Errors by Tag Type ---")
    for tag, count in tag_errors.most_common():
        print(f"{tag}: {count}")
        
    print("\n--- Errors by Page ---")
    for page, count in sorted(page_errors.items()):
        print(f"Page {page}: {count}")
        
    print("\n--- Top Corrective Reasons ---")
    for reason, count in reason_counts.most_common(5):
        print(f"({count}x) {reason}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_report(sys.argv[1])
    else:
        print("Usage: python analyze_qa_report.py <json_file>")
