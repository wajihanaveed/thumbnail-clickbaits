import os
import json
import string
import glob
import csv

def llava_fixed_shot_sanitization():
    # Path for your LLaVA fixed-shot research data
    base_path = os.path.expanduser("~/Research/ZAQ/evals/evals/registry/data/llava_fixed_shot")
    os.makedirs(base_path, exist_ok=True)
    jsonl_filename = os.path.join(base_path, "llava_fixed_shot_eval_data.jsonl")
    
    # Target standard results files in the current LLaVA directory
    csv_files = glob.glob("*_Thumbnails_Results.csv")
    
    # Adding \n and \t keeps the AI's Chain-of-Thought formatting intact for the judge
    safe_chars = set(string.ascii_letters + string.digits + string.punctuation + " \n\t")
    total = 0

    print(f"🚀 Starting LLaVA Fixed-Shot sanitization of {len(csv_files)} files...")

    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for file in csv_files:
            # Determine ground truth based on filename (MTV vs NON_MTV)
            ground_truth = "Not Misleading" if "NON_MTV" in file else "Misleading"
            
            with open(file, 'r', encoding='utf-8') as f:
                # csv.reader handles the multi-line quoted reasoning blocks
                reader = csv.reader(f)
                try:
                    next(reader)  # Skip the CSV header
                except StopIteration:
                    continue

                for row in reader:
                    # Expecting [video_id, llava_output]
                    if len(row) >= 2:
                        video_id = row[0].strip()
                        
                        # Validate the 11-character YouTube Video ID
                        if len(video_id) == 11:
                            raw_text = row[1]
                            
                            # Clean the text but preserve formatting/structure
                            clean_text = "".join(c for c in raw_text if c in safe_chars)
                            
                            # Truncate to 2000 chars to respect the judge's context window
                            clean_text = clean_text[:2000].strip()
                            
                            if clean_text:
                                record = {
                                    "input": f"Ground Truth: {ground_truth}",
                                    "completion": clean_text,
                                    "video_id": video_id
                                }
                                jsonl_file.write(json.dumps(record) + '\n')
                                total += 1

    print(f"✨ LLaVA Fixed-Shot Sanitization Complete: {total} rows saved to {jsonl_filename}")

if __name__ == "__main__":
    llava_fixed_shot_sanitization()