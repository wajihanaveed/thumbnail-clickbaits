import os
import json
import string
import glob
import csv

def llava_zero_shot_sanitization():
    # Updated path to a specific llava-zero-shot directory
    base_path = os.path.expanduser("~/Research/ZAQ/evals/evals/registry/data/llava_zero_shot")
    os.makedirs(base_path, exist_ok=True)
    jsonl_filename = os.path.join(base_path, "llava_zero_shot_eval_data.jsonl")
    
    # Target standard results files in the current directory
    csv_files = glob.glob("*_Thumbnails_Results.csv")
    
    # Preserving formatting (newlines/tabs) is vital for the judge to read the CoT
    safe_chars = set(string.ascii_letters + string.digits + string.punctuation + " \n\t")
    total = 0

    print(f"🚀 Starting LLaVA Zero-Shot sanitization of {len(csv_files)} files...")

    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for file in csv_files:
            # Filename-based Ground Truth detection logic (Consistent with Qwen run)
            ground_truth = "Not Misleading" if "NON_MTV" in file else "Misleading"
            
            with open(file, 'r', encoding='utf-8') as f:
                # Using csv.reader to safely handle multi-line LLaVA reasonings
                reader = csv.reader(f)
                try:
                    next(reader)  # Skip the header
                except StopIteration:
                    continue

                for row in reader:
                    if len(row) >= 2:
                        video_id = row[0].strip()
                        
                        # Validate the 11-character YouTube Video ID format
                        if len(video_id) == 11:
                            raw_text = row[1]
                            
                            # Clean text while keeping the reasoning structure intact
                            clean_text = "".join(c for c in raw_text if c in safe_chars)
                            
                            # Truncate to 2000 chars to respect judge context windows
                            clean_text = clean_text[:2000].strip()
                            
                            if clean_text:
                                # Input: The context for the judge
                                # Completion: The actual LLaVA output
                                record = {
                                    "input": f"Ground Truth: {ground_truth}",
                                    "completion": clean_text,
                                    "video_id": video_id
                                }
                                jsonl_file.write(json.dumps(record) + '\n')
                                total += 1

    print(f"✨ LLaVA Zero-Shot Sanitization Complete: {total} rows saved to {jsonl_filename}")

if __name__ == "__main__":
    llava_zero_shot_sanitization()