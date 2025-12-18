import os
import json
import time
from tqdm import tqdm
from collections import defaultdict
import google.generativeai as genai
from google.api_core import exceptions

# --- CẤU HÌNH ---
# API Key của bạn
API_KEY = "AIzaSyDhnjhEHH1Bvl3I4RiyPgfwcYJR1DnnjwU"

INPUT_FILENAME = "vinmec_data_preprocessed.jsonl"
OUTPUT_FILENAME = "training_dataset_final.jsonl" 

# MỤC TIÊU: 50 CÂU TRONG 1 LẦN GỌI
TARGET_QUESTIONS = 50

genai.configure(api_key=API_KEY)

# --- 1. HÀM TÌM MODEL (ƯU TIÊN TUYỆT ĐỐI 2.5 FLASH) ---
def get_flash_2_5_model():
    print("🔍 Đang tìm model Gemini 2.5 Flash...")
    try:
        all_models = list(genai.list_models())
        valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        # Ưu tiên số 1: Tìm đích danh Gemini 2.5 Flash
        for m in valid_models:
            if "gemini-2.5-flash" in m and "pro" not in m and "image" not in m:
                print(f"🎯 Đã chọn (Ưu tiên): {m}")
                return m
        
        print("⚠️ Không thấy tên 'gemini-2.5-flash' trong danh sách trả về.")
        print("👉 Đang thử dùng tên mặc định 'models/gemini-2.5-flash-preview-09-2025'...")
        return 'models/gemini-2.5-flash-preview-09-2025'

    except Exception as e:
        print(f"❌ Lỗi lấy danh sách model: {e}")
        return 'models/gemini-2.5-flash-preview-09-2025'

# Chọn model
VERIFIED_MODEL_NAME = get_flash_2_5_model()

# --- 2. SYSTEM PROMPT ---
SYSTEM_PROMPT = f"""
Bạn là chuyên gia y tế. Nhiệm vụ: Đọc toàn bộ kiến thức về một bệnh (Context) và soạn bộ câu hỏi trắc nghiệm Đúng/Sai.

YÊU CẦU NGHIÊM NGẶT:
1. Số lượng: Phải sinh ra ĐỦ {TARGET_QUESTIONS} câu hỏi.
2. Phân bổ nội dung (BẮT BUỘC):
   - Khoảng 10-15 câu về: Nguyên nhân, Cơ chế bệnh sinh, Đường lây.
   - Khoảng 15-20 câu về: Triệu chứng lâm sàng, Dấu hiệu nhận biết, Chẩn đoán.
   - Khoảng 15-20 câu về: Điều trị, Thuốc, Phòng ngừa và Biến chứng.
3. Chất lượng: 
   - Câu Sai phải có tính đánh lừa cao (ví dụ sai về nhóm thuốc, nhầm triệu chứng sang bệnh khác).
   - KHÔNG đặt câu hỏi quá dễ hoặc ngớ ngẩn.
4. Định dạng: Trả về duy nhất một JSON List chứa các object.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "bo_cau_hoi": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "cau_hoi": {"type": "STRING"},
                    "dap_an": {"type": "STRING", "enum": ["Đúng", "Sai"]}
                },
                "required": ["cau_hoi", "dap_an"]
            }
        }
    }
}

# --- 3. HÀM GỌI API ---
def call_gemini_single_shot(full_context, disease_name):
    model = genai.GenerativeModel(
        model_name=VERIFIED_MODEL_NAME, 
        system_instruction=SYSTEM_PROMPT
    )
    
    user_prompt = f"""
    Tên bệnh: {disease_name}
    
    Dựa vào thông tin chi tiết dưới đây, hãy sinh ra {TARGET_QUESTIONS} câu hỏi Đúng/Sai bao phủ mọi khía cạnh (Nguyên nhân, Triệu chứng, Điều trị).
    
    Thông tin tham khảo:
    ---
    {full_context}
    ---
    """
    
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.4
    )

    for attempt in range(3):
        try:
            # Tăng timeout lên 600s (10 phút) vì sinh 50 câu tốn thời gian
            response = model.generate_content(
                user_prompt,
                generation_config=generation_config,
                request_options={'timeout': 600} 
            )
            return json.loads(response.text)
            
        except exceptions.ResourceExhausted:
            print(f"   ⏳ Hết quota (429). Đợi 120s...") # Gemini 2.5 cần đợi lâu hơn
            time.sleep(120)
        except exceptions.DeadlineExceeded:
            print(f"   🐢 Mạng chậm (504). Đợi 10s thử lại...")
            time.sleep(10)
        except Exception as e:
            print(f"   ⚠️ Lỗi ({attempt+1}): {e}")
            time.sleep(5)
            
    return None

# --- 4. TIỆN ÍCH ---
def group_remaining_data(filepath):
    print(f"🔄 Đang đọc dữ liệu...")
    disease_map = defaultdict(list)
    if not os.path.exists(filepath): return disease_map
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                name = record.get('benh', 'Unknown')
                if name and record.get('noi_dung'):
                    disease_map[name].append(record)
            except: continue
    return disease_map

def get_completed_diseases(filepath):
    if not os.path.exists(filepath): return set()
    
    disease_counts = defaultdict(int)
    print(f"🔍 Đang kiểm tra tiến độ cũ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if 'chu_de' in record:
                    disease_counts[record['chu_de']] += 1
            except: pass
            
    # Nếu bệnh nào đã có trên 40 câu -> BỎ QUA
    completed = {k for k, v in disease_counts.items() if v >= 40}
    print(f"✅ Đã có {len(completed)} bệnh đã hoàn thành (sẽ được bỏ qua).")
    return completed

# --- 5. MAIN ---
def run_single_shot_50():
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, INPUT_FILENAME)
    output_path = os.path.join(base_path, OUTPUT_FILENAME)
    
    if not os.path.exists(input_path):
        print("❌ Không tìm thấy file input.")
        return
    
    grouped_data = group_remaining_data(input_path)
    completed_diseases = get_completed_diseases(output_path)
    
    total_new_q = 0
    
    print(f"🚀 Bắt đầu chiến dịch '1 Phát Ăn Ngay' (50 câu/request) với model {VERIFIED_MODEL_NAME}...")
    
    with open(output_path, 'a', encoding='utf-8') as f_out:
        
        for disease_name, contents in tqdm(grouped_data.items(), desc="Tiến độ"):
            
            if disease_name in completed_diseases:
                continue 
            
            full_text = ""
            source_url = contents[0].get('url', '')
            for item in contents:
                full_text += f"\n[Mục: {item.get('muc','')}]\n{item.get('noi_dung','')}\n"
            
            if len(full_text) < 200: continue

            data = call_gemini_single_shot(full_text, disease_name)
            
            if data and "bo_cau_hoi" in data:
                questions = data["bo_cau_hoi"]
                
                if len(questions) < 10:
                    print(f"   ⚠️ Sinh quá ít ({len(questions)} câu). Bỏ qua.")
                    continue

                for q in questions:
                    final_record = {
                        "cau_hoi": q["cau_hoi"],
                        "dap_an": q["dap_an"],
                        "nguon": source_url,
                        "chu_de": disease_name
                    }
                    json.dump(final_record, f_out, ensure_ascii=False)
                    f_out.write('\n')
                
                total_new_q += len(questions)
                f_out.flush()
                
                # Nghỉ 10s để an toàn cho Quota Gemini 2.5
                time.sleep(10) 

    print(f"\n🎉 HOÀN TẤT! Đã sinh thêm {total_new_q} câu hỏi.")
    print(f"👉 Dữ liệu: {output_path}")

if __name__ == "__main__":
    run_single_shot_50()