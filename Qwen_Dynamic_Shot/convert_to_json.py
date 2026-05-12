import os
import json
import string
import glob
import csv

def qwen_dynamic_sanitization():
    # Dedicated path for dynamic few-shot research data
    base_path = os.path.expanduser("~/Research/ZAQ/evals/evals/registry/data/qwen_dynamic")
    os.makedirs(base_path, exist_ok=True)
    jsonl_filename = os.path.join(base_path, "qwen_dynamic_eval_data.jsonl")
    
    # Target your specific result CSVs in the current directory
    csv_files = glob.glob("*_Thumbnails_Results.csv")
    
    # Preserve formatting (newlines/tabs) for the judge's Chain-of-Thought analysis
    safe_chars = set(string.ascii_letters + string.digits + string.punctuation + " \n\t")
    total = 0

    print(f"🚀 Starting Dynamic Few-Shot sanitization of {len(csv_files)} files...")

    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for file in csv_files:
            # Detect Ground Truth based on naming convention
            ground_truth = "Not Misleading" if "NON_MTV" in file else "Misleading"
            
            with open(file, 'r', encoding='utf-8') as f:
                # csv.reader handles multi-line reasoning blocks correctly
                reader = csv.reader(f)
                try:
                    next(reader)  # Skip the CSV header
                except StopIteration:
                    continue

                for row in reader:
                    if len(row) >= 2:
                        video_id = row[0].strip()
                        
                        # Standard 11-char YouTube ID validation
                        if len(video_id) == 11:
                            raw_text = row[1]
                            
                            # Clean text while preserving reasoning structure
                            clean_text = "".join(c for c in raw_text if c in safe_chars)
                            
                            # Truncate to 2000 chars to fit judge context windows
                            clean_text = clean_text[:2000].strip()
                            
                            if clean_text:
                                record = {
                                    "input": f"Ground Truth: {ground_truth}",
                                    "completion": clean_text,
                                    "video_id": video_id
                                }
                                jsonl_file.write(json.dumps(record) + '\n')
                                total += 1

    print(f"✨ Dynamic Sanitization Complete: {total} rows saved to {jsonl_filename}")

if __name__ == "__main__":
    qwen_dynamic_sanitization()