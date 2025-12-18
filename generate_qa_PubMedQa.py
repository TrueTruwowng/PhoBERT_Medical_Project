import os
import json
import time
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from google.api_core import exceptions

# --- CẤU HÌNH ---
API_KEY = "AIzaSyDBAuG-dHHuI1M5iImr1NgM4QN5Ky4uaqE"

INPUT_PARQUET_FILE = "train2.parquet" 
OUTPUT_FILE = "train_dataset_pubmedqa.jsonl"

BATCH_SIZE = 50 

genai.configure(api_key=API_KEY)
MODEL_NAME = 'models/gemini-2.5-flash-preview-09-2025'

# --- 1. SYSTEM PROMPT ---
SYSTEM_PROMPT = """
Bạn là chuyên gia dữ liệu y tế. Nhiệm vụ: Chuyển đổi danh sách câu hỏi (Tiếng Anh) sang câu khẳng định (Tiếng Việt).

INPUT: JSON {id, question, label}.
OUTPUT: JSON {id, cau_hoi, dap_an}.

QUY TẮC:
1. Dựa vào 'label' để viết 'cau_hoi' (Tiếng Việt):
   - yes -> Viết câu khẳng định ĐÚNG thực tế. (Đáp án: Đúng)
   - no -> Viết câu khẳng định SAI thực tế. (Đáp án: Sai)
2. 'cau_hoi' là câu kể.
3. Trả về JSON thuần.
"""

# --- 2. HÀM GỌI API (BATCH) ---
def process_batch(batch_items):
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    
    mini_batch_input = []
    for item in batch_items:
        # Trong file parquet, tên cột là 'pubid', 'question', 'final_decision'
        mini_batch_input.append({
            "id": str(item['pubid']),
            "question": item['question'],
            "label": item['final_decision']
        })

    user_prompt = json.dumps(mini_batch_input)

    for attempt in range(3):
        try:
            response = model.generate_content(
                user_prompt,
                generation_config={"response_mime_type": "application/json"},
                request_options={'timeout': 90}
            )
            return json.loads(response.text)

        except exceptions.ResourceExhausted:
            print(f"   ⏳ Hết quota. Đợi 60s...")
            time.sleep(60)
        except Exception as e:
            print(f"   ⚠️ Lỗi batch: {e}")
            time.sleep(5)
    return None

# --- 3. HÀM ĐỌC FILE PARQUET (AUTO-DETECT PATH) ---
def load_parquet_data():
    # Lấy đường dẫn thư mục chứa file code này
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, INPUT_PARQUET_FILE)
    
    print(f"📖 Đang đọc file: {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"❌ LỖI: Không tìm thấy file '{INPUT_PARQUET_FILE}'.")
        print(f"👉 Hãy chắc chắn file '{INPUT_PARQUET_FILE}' nằm cùng thư mục với file code.")
        exit()
        
    try:
        df = pd.read_parquet(file_path)
        # Lọc chỉ lấy yes/no
        df = df[df['final_decision'].isin(['yes', 'no'])]
        print(f"✅ Đã tải {len(df)} dòng dữ liệu hợp lệ.")
        return df.to_dict('records')
    except Exception as e:
        print(f"❌ Lỗi đọc file Parquet: {e}")
        exit()

# --- 4. CHECKPOINT ---
def get_processed_ids(filepath):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, filepath)
    
    processed = set()
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'original_id' in data: processed.add(data['original_id'])
                except: pass
    print(f"✅ Đã tìm thấy {len(processed)} câu đã xong.")
    return processed

# --- 5. MAIN ---
def run():
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, OUTPUT_FILE)
    
    all_items = load_parquet_data()
    processed_ids = get_processed_ids(OUTPUT_FILE)
    
    todo_items = [item for item in all_items if str(item['pubid']) not in processed_ids]
    
    print(f"🔥 Sẵn sàng xử lý {len(todo_items)} mẫu (Batch Size: {BATCH_SIZE})...")
    
    with open(output_path, "a", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(todo_items), BATCH_SIZE), desc="Đang dịch"):
            batch = todo_items[i : i + BATCH_SIZE]
            
            result_list = process_batch(batch)
            
            if result_list and isinstance(result_list, list):
                for res in result_list:
                    final_record = {
                        "cau_hoi": res.get("cau_hoi"),
                        "dap_an": res.get("dap_an"),
                        "nguon": "PubMedQA (Artificial)",
                        "original_id": str(res.get("id"))
                    }
                    json.dump(final_record, f_out, ensure_ascii=False)
                    f_out.write('\n')
                f_out.flush()
            
            time.sleep(5)

    print(f"\n🎉 HOÀN TẤT! File kết quả: {output_path}")

if __name__ == "__main__":
    run()