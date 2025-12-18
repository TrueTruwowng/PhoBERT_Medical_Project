import requests
from bs4 import BeautifulSoup
import time
import random
import json
import os
import re

# --- CẤU HÌNH ---
# Đọc file link đã cào ở Bước 1
INPUT_FILE = "vinmec_links_az.txt"
# Tên file kết quả mới, sạch sẽ
OUTPUT_FILE = "vinmec_data_step2_final.jsonl" 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# --- HÀM 1: GỬI REQUEST KIÊN TRÌ ---
def get_with_retry(url, max_retries=3):
    """Cố gắng truy cập URL. Nếu lỗi mạng/timeout, đợi và thử lại."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            
            # Kiểm tra soft ban (chặn IP)
            if "Access Denied" in response.text or "Verify" in response.text:
                print(f"   🚨 CẢNH BÁO: Bị chặn IP. Đang đợi 30 giây...")
                time.sleep(30)
                continue # Thử lại sau khi đợi
                
            if response.status_code == 200:
                return response
            elif response.status_code in [403, 429, 503]:
                print(f"   ⚠️ Server bận (Code {response.status_code}). Đợi 10s...")
                time.sleep(10)
            else:
                return None # Lỗi 404...
                
        except requests.exceptions.RequestException as e:
            wait = (attempt + 1) * 5
            print(f"   ⚠️ Lỗi mạng (Lần {attempt+1}): {e}. Đợi {wait}s...")
            time.sleep(wait)
            
    print(f"   ❌ BỎ QUA: Không thể truy cập {url} sau {max_retries} lần.")
    return None

# --- HÀM 2: CHUẨN HÓA TIÊU ĐỀ ---
def normalize_header(text):
    """Gom nhóm các tiêu đề về 8 mục chính"""
    text = text.lower()
    if any(x in text for x in ["tổng quan", "là gì", "là bệnh gì"]): return "Tổng quan"
    if "nguyên nhân" in text: return "Nguyên nhân"
    if any(x in text for x in ["triệu chứng", "dấu hiệu", "biểu hiện"]): return "Triệu chứng"
    if any(x in text for x in ["lây truyền", "lây lan", "đường lây"]): return "Đường lây truyền"
    if any(x in text for x in ["đối tượng", "nguy cơ", "ai mắc"]): return "Đối tượng nguy cơ"
    if any(x in text for x in ["phòng ngừa", "phòng bệnh", "chế độ sinh hoạt"]): return "Phòng ngừa"
    if any(x in text for x in ["chẩn đoán", "xét nghiệm"]): return "Biện pháp chẩn đoán"
    if any(x in text for x in ["điều trị", "chữa trị", "thuốc"]): return "Biện pháp điều trị"
    return "Thông tin khác"

# --- HÀM 3: BÓC TÁCH THÔNG MINH (Bản sửa lỗi) ---
def parse_details_smart(url):
    """Hàm này là trái tim của code, bóc tách nội dung."""
    response = get_with_retry(url)
    if not response: return []
    
    try:
        soup = BeautifulSoup(response.content, 'lxml')
        
        # --- 1. SỬA LỖI UNKNOWN TÊN BỆNH ---
        benh_name = "Unknown"
        h1 = soup.find('h1')
        if h1 and len(h1.text.strip()) > 2:
            benh_name = h1.text.strip()
        else:
            # Lấy dự phòng từ thẻ <title>
            title_tag = soup.find('title')
            if title_tag:
                raw_title = title_tag.text
                parts = re.split(r'[:|\-]', raw_title) # Cắt tên bệnh trước dấu : hoặc -
                if parts:
                    benh_name = parts[0].strip()

        # --- 2. SỬA LỖI 0 BÀI VIẾT (Tìm khung nội dung) ---
        content_div = None
        keywords = ["Nguyên nhân", "Triệu chứng", "Điều trị", "Tổng quan"]
        
        # Chiến thuật 1: Tìm theo từ khóa (Ưu tiên)
        for kw in keywords:
            target_header = soup.find(lambda tag: tag.name in ['h2', 'h3'] and kw in tag.get_text())
            if target_header:
                content_div = target_header.parent
                if content_div and len(content_div.get_text(strip=True)) < 200 and content_div.parent:
                    content_div = content_div.parent # Leo lên 1 cấp nếu khung quá nhỏ
                break
        
        # Chiến thuật 2: Nếu không thấy từ khóa, tìm theo class (Dự phòng)
        if not content_div:
             classes = ['collapsible-content', 'main-content', 'post-content', 'body-content']
             for cls in classes:
                 div = soup.find('div', class_=cls)
                 if div and len(div.get_text(strip=True)) > 200:
                     content_div = div
                     break

        if not content_div:
            print(f"   ❌ Không tìm thấy khung nội dung nào.")
            return [] # Bỏ qua bài này

        # --- 3. BÓC TÁCH DỮ LIỆU ---
        extracted_data = []
        current_category = "Tổng quan"
        current_content = []
        
        for tag in content_div.find_all(['h2', 'h3', 'p', 'ul']): # Chỉ lấy 4 thẻ quan trọng này
            text = tag.get_text(separator=" ").strip()
            if not text: continue
            
            if tag.name in ['h2', 'h3']:
                # Gặp tiêu đề mới -> Lưu đoạn cũ
                if current_content:
                    full_text = " ".join(current_content).strip()
                    if len(full_text) > 20:
                        extracted_data.append({
                            "benh": benh_name,
                            "muc": current_category,
                            "noi_dung": full_text
                        })
                current_category = normalize_header(text)
                current_content = []
            
            elif tag.name in ['p', 'ul']:
                # Lọc rác quảng cáo
                if any(s in text for s in ["Vinmec", "ĐẶT LỊCH", "TẠI ĐÂY", "Xem thêm"]): 
                    continue
                current_content.append(text)

        # Lưu đoạn cuối cùng
        if current_content:
            extracted_data.append({
                "benh": benh_name,
                "muc": current_category,
                "noi_dung": " ".join(current_content).strip()
            })
            
        return extracted_data

    except Exception as e:
        print(f"   ❌ Lỗi xử lý HTML: {e}")
        return []

# --- HÀM CHÍNH (MAIN) ---
def run_crawler():
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, INPUT_FILE)
    output_path = os.path.join(base_path, OUTPUT_FILE)
    
    if not os.path.exists(input_path):
        print(f"⚠️ LỖI: Không tìm thấy file '{input_path}'! Hãy chạy Bước 1 trước.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        # Lọc bỏ các link rác (nếu có)
        urls = [line.strip() for line in f.readlines() if line.strip() and "/benh/" in line]

    total_links = len(urls)
    total_chunks = 0
    print(f"🚀 Bắt đầu cào nội dung từ {total_links} link (Bản sửa lỗi)...")
    print(f"📂 Dữ liệu sẽ lưu vào: {output_path}")
    
    # Mở file chế độ 'a' (append)
    with open(output_path, 'a', encoding='utf-8') as f_out:
        for i, url in enumerate(urls):
            print(f"[{i+1}/{total_links}] Đang cào: {url}")
            
            chunks = parse_details_smart(url)
            
            if chunks:
                # In tên bệnh để kiểm tra
                print(f"   ✅ Lấy được {len(chunks)} đoạn. (Bệnh: {chunks[0]['benh']})")
                
                for chunk in chunks:
                    json.dump(chunk, f_out, ensure_ascii=False)
                    f_out.write('\n')
                total_chunks += len(chunks)
                
                # --- SỬA LỖI FILE RỖNG ---
                # Ép Python xả dữ liệu từ RAM ra ổ cứng ngay lập tức
                f_out.flush() 
            
            # Nghỉ ngơi để không bị chặn
            time.sleep(random.uniform(1.5, 3))

    print("-" * 50)
    print(f"🎉 HOÀN TẤT!")
    print(f"   - Số bài viết đã xử lý thành công: {total_links} (trừ các link lỗi)")
    print(f"   - Tổng số mẩu dữ liệu (chunks) thu được: {total_chunks}")
    print(f"👉 Kiểm tra file kết quả: {output_path}")

if __name__ == "__main__":
    run_crawler()