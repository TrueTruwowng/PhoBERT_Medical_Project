import os
import json
import re
from tqdm import tqdm

# --- CẤU HÌNH ---
# File bẩn (Input)
INPUT_FILE = "vinmec_data_step2_final.jsonl" 
# File sạch (Output)
OUTPUT_FILE = "vinmec_data_preprocessed.jsonl"

# --- 1. ĐỊNH NGHĨA BỘ LỌC RÁC ---
# Dùng RegEx để tìm và xóa
# | nghĩa là "hoặc"
SPAM_PATTERNS = re.compile(
    r'Xem thêm:|ĐẶT LỊCH KHÁM|TẠI ĐÂY|hotline|Bệnh viện Đa khoa Quốc tế|Vinmec|Bài viết này được viết cho người đọc|Nguồn tham khảo|được bảo vệ bản quyền|Bấm nút theo dõi|SĐT|www\.vinmec\.com',
    flags=re.IGNORECASE # Không phân biệt hoa thường
)

# Các mục không có giá trị học thuật cao
MUC_LOAI_BO = ["Thông tin khác", ""]

# Ngưỡng ký tự
MIN_LENGTH = 50 # Ngắn hơn 50 ký tự -> bỏ
MAX_LENGTH = 3000 # Dài hơn 3000 ký tự -> bỏ

# --- 2. HÀM CHÍNH (MAIN) ---
def run_preprocessing():
    
    # --- PHẦN SỬA LỖI ---
    # Lấy đường dẫn thư mục hiện tại của file code (F:\...)
    base_path = os.path.dirname(os.path.abspath(__file__)) 
    
    # Nối đường dẫn đầy đủ cho file Input và Output
    input_path = os.path.join(base_path, INPUT_FILE)
    output_path = os.path.join(base_path, OUTPUT_FILE)
    # --- KẾT THÚC SỬA LỖI ---

    if not os.path.exists(input_path): # Kiểm tra đường dẫn đầy đủ
        print(f"LỖI: Không tìm thấy file {input_path}!") # In đường dẫn đầy đủ
        return

    # Set để kiểm tra trùng lặp
    seen_content = set() 
    
    # Biến đếm thống kê
    count_read = 0
    count_written = 0
    count_spam = 0
    count_duplicate = 0
    count_short = 0
    count_irrelevant = 0

    print(f"🚀 Bắt đầu tiền xử lý file {input_path}...") # In đường dẫn đầy đủ
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        # Dùng tqdm để xem tiến trình
        for line in tqdm(f_in, desc="Đang làm sạch"):
            count_read += 1
            try:
                data = json.loads(line)
                noi_dung = data.get("noi_dung")
                muc = data.get("muc")

                if not noi_dung or not muc:
                    continue
                
                # --- 1. LỌC MỤC KHÔNG LIÊN QUAN ---
                if muc in MUC_LOAI_BO:
                    count_irrelevant += 1
                    continue
                
                # --- 2. LÀM SẠCH (CLEAN) ---
                # Xóa các từ spam
                clean_text = re.sub(SPAM_PATTERNS, '', noi_dung)
                # Chuẩn hóa khoảng trắng
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                # --- 3. LỌC ĐỘ DÀI ---
                if not (MIN_LENGTH < len(clean_text) < MAX_LENGTH):
                    count_short += 1
                    continue
                
                # --- 4. KHỬ TRÙNG (DEDUPLICATE) ---
                if clean_text in seen_content:
                    count_duplicate += 1
                    continue
                
                # Nếu vượt qua tất cả
                seen_content.add(clean_text)
                
                # Cập nhật lại data với nội dung đã làm sạch
                data["noi_dung"] = clean_text
                
                # Ghi ra file mới
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')
                count_written += 1

            except Exception as e:
                print(f"Lỗi đọc dòng: {e}")

    print("\n" + "="*50)
    print("🎉 TIỀN XỬ LÝ HOÀN TẤT!")
    print(f"   - Đã đọc:      {count_read} dòng")
    print(f"   - Đã ghi:      {count_written} dòng (đã làm sạch)")
    print("-" * 50)
    print(f"   - Đã loại bỏ (Quá ngắn/dài): {count_short}")
    print(f"   - Đã loại bỏ (Trùng lặp):   {count_duplicate}")
    print(f"   - Đã loại bỏ (Mục rác):     {count_irrelevant}")
    print(f"👉 File sạch sẵn sàng tại: {output_path}") # In đường dẫn đầy đủ

if __name__ == "__main__":
    # Cài đặt thư viện nếu cần: pip install tqdm
    run_preprocessing()