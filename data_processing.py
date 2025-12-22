import json
import random
import glob
import os

def process_clean_and_shuffle():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Directories and Files
    input_folder = os.path.join(current_dir, 'data')
    output_file = os.path.join(current_dir, 'train_data_final.jsonl')
    duplicate_file = os.path.join(current_dir, 'removed_duplicates.jsonl')

    print(f"Script location: {current_dir}")
    print(f"Input folder: {input_folder}")
    print("-" * 50)
    print("MODE: STRICT DUPLICATE DETECTION")
    print("   Criteria: exact match of BOTH Question AND Answer.")
    print("   (Variations of the same question with different answers will be KEPT)")
    print("-" * 50)

    all_data = []
    duplicate_data = []
    seen_records = set()    # Stores tuples of (question, answer)
    total_lines = 0

    # Find files
    search_path = os.path.join(input_folder, '*.jsonl')
    list_files = glob.glob(search_path)

    if not list_files:
        print(f"ERROR: No .jsonl files found in 'data' folder.")
        return

    print(f"Found {len(list_files)} files. Processing...")

    for file_path in list_files:
        filename = os.path.basename(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)

                        # Get Question & Answer
                        q = item.get("input") or item.get("cau_hoi") or item.get("question") or item.get("instruction")
                        a = item.get("output") or item.get("dap_an") or item.get("answer") or item.get("response")

                        if q and a:
                            total_lines += 1

                            # Clean whitespace only (standard practice)
                            q_clean = q.strip()
                            a_clean = str(a).strip()

                            # UNIQUE SIGNATURE: (Question + Answer)
                            record_signature = (q_clean, a_clean)

                            # Check if this EXACT pair has been seen before
                            if record_signature in seen_records:
                                duplicate_data.append({
                                    "question": q_clean,
                                    "answer": a_clean,
                                    "source_file": filename,
                                    "note": "Exact Duplicate (Q+A match)"
                                })
                                continue

                            # New unique record found
                            seen_records.add(record_signature)
                            all_data.append({"question": q_clean, "answer": a_clean})

                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Shuffle
    print(f"Shuffling {len(all_data)} unique records...")
    random.shuffle(all_data)

    # Save Clean Data
    print(f"Saving CLEAN data to: train_data_final.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Save Duplicates
    if duplicate_data:
        print(f"Saving {len(duplicate_data)} DUPLICATES to: removed_duplicates.jsonl")
        with open(duplicate_file, 'w', encoding='utf-8') as f:
            for entry in duplicate_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    else:
        print(f"Clean sweep! No exact duplicates found.")

    print("="*50)
    print(f"DONE!")
    print(f"Total inputs: {total_lines}")
    print(f"Unique kept: {len(all_data)}")
    print(f"Removed:     {len(duplicate_data)}")
    print("="*50)

if __name__ == "__main__":
    process_clean_and_shuffle()