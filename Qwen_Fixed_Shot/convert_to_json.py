import os
import json
import string
import glob
import csv

def total_sanitization():
    # Central Registry Path for your Qwen research data
    base_path = os.path.expanduser("~/Research/ZAQ/evals/evals/registry/data/qwen")
    os.makedirs(base_path, exist_ok=True)
    jsonl_filename = os.path.join(base_path, "qwen_eval_data.jsonl")
    
    csv_files = glob.glob("*_Thumbnails_Results.csv")
    
    # Adding \n and \t to safe_chars keeps the AI's formatting intact for the judge
    safe_chars = set(string.ascii_letters + string.digits + string.punctuation + " \n\t")
    total = 0

    print(f"🚀 Starting sanitization of {len(csv_files)} files...")

    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for file in csv_files:
            # Determine ground truth based on the filename convention
            ground_truth = "Not Misleading" if "NON_MTV" in file else "Misleading"
            
            with open(file, 'r', encoding='utf-8') as f:
                # csv.reader is essential here: it treats multi-line quoted text as a single field
                reader = csv.reader(f)
                try:
                    next(reader)  # Skip the CSV header
                except StopIteration:
                    continue

                for row in reader:
                    # Expecting at least [video_id, qwen_output]
                    if len(row) >= 2:
                        video_id = row[0].strip()
                        
                        # Validate the 11-character YouTube Video ID format
                        if len(video_id) == 11:
                            raw_text = row[1]
                            
                            # Clean the text but preserve the structure of the reasoning
                            clean_text = "".join(c for c in raw_text if c in safe_chars)
                            
                            # We truncate to 2000 to keep the judge's context window manageable
                            clean_text = clean_text[:2000].strip()
                            
                            if clean_text:
                                # 'input' provides the ground truth context
                                # 'completion' is the specific Qwen response the judge will grade
                                record = {
                                    "input": f"Ground Truth: {ground_truth}",
                                    "completion": clean_text,
                                    "video_id": video_id
                                }
                                jsonl_file.write(json.dumps(record) + '\n')
                                total += 1

    print(f"✨ Total Sanitization Complete: {total} rows saved to {jsonl_filename}")

if __name__ == "__main__":
    total_sanitization()