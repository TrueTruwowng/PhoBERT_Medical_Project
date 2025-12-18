import requests
from bs4 import BeautifulSoup
import time
import random
import os
import string # Thư viện để lấy bảng chữ cái a-z

# --- CẤU HÌNH THEO HÌNH ẢNH ---
# URL gốc chưa có chữ cái. Chúng ta sẽ cộng chuỗi "a", "b", "c" vào sau.
BASE_URL_PREFIX = "https://www.vinmec.com/vie/tra-cuu-benh/"

# File lưu kết quả
current_folder = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(current_folder, "vinmec_links_az.txt")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def crawl_by_alphabet():
    print("🚀 BẮT ĐẦU CHIẾN DỊCH CÀO THEO BẢNG CHỮ CÁI A-Z")
    print(f"📂 Kết quả lưu tại: {OUTPUT_FILE}")
    print("-" * 60)

    total_links = 0
    
    # Tạo danh sách chữ cái: ['a', 'b', 'c', ..., 'z']
    alphabet_list = list(string.ascii_lowercase) 
    
    # Mở file chế độ 'a' (append)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        
        # VÒNG LẶP TỪ A ĐẾN Z
        for char in alphabet_list:
            # Tạo link: https://www.vinmec.com/vie/tra-cuu-benh/a
            url = f"{BASE_URL_PREFIX}{char}"
            print(f"\n📡 Đang quét chữ cái [{char.upper()}]: {url}")
            
            try:
                response = requests.get(url, headers=HEADERS, timeout=15)
                
                # Nếu chữ cái đó không có bệnh (ví dụ chữ X, Y có thể ít), Vinmec vẫn trả về 200
                if response.status_code != 200:
                    print(f"⚠️ Lỗi truy cập chữ {char}. Bỏ qua.")
                    continue

                soup = BeautifulSoup(response.content, 'lxml')
                
                # --- BÓC TÁCH LINK ---
                # Dựa vào ảnh bạn gửi, danh sách bệnh nằm dưới chữ cái A to tướng
                # Thường Vinmec để link bệnh trong thẻ <li> -> <a> hoặc <h2> -> <a>
                
                # Cách an toàn nhất: Lấy tất cả thẻ <a> trong vùng nội dung chính
                # Và lọc những link chứa '/vie/benh/' hoặc '/vi/benh/'
                
                all_links = soup.find_all('a')
                
                count_char = 0
                for tag in all_links:
                    raw_link = tag.get('href')
                    
                    if raw_link:
                        # 1. Chuẩn hóa link
                        if not raw_link.startswith('http'):
                            full_link = "https://www.vinmec.com" + raw_link
                        else:
                            full_link = raw_link
                        
                        # 2. BỘ LỌC (Quan trọng)
                        # Link bệnh trong ảnh 1 bạn gửi có dạng: .../vie/benh/addison...
                        # Vì vậy ta chỉ lấy link chứa '/benh/'
                        # Và loại bỏ chính cái link trang danh mục (/tra-cuu-benh/)
                        if '/benh/' in full_link and '/tra-cuu-benh/' not in full_link:
                            
                            # Lưu vào file
                            f.write(full_link + '\n')
                            
                            # In ra vài cái để kiểm tra (không in hết cho đỡ rối mắt)
                            if count_char < 3: 
                                print(f"   + {full_link}")
                            elif count_char == 3:
                                print("   + ... (và các bài khác)")
                                
                            count_char += 1
                            total_links += 1

                if count_char == 0:
                    print(f"❌ Không tìm thấy bệnh nào bắt đầu bằng chữ {char.upper()}")
                else:
                    print(f"✅ Chữ {char.upper()}: Lấy được {count_char} bệnh.")

                # Ngủ để tránh chặn (quan trọng vì vòng lặp a-z chạy khá nhanh)
                time.sleep(random.uniform(1.5, 3))

            except Exception as e:
                print(f"❌ Lỗi tại chữ {char}: {e}")

    print("-" * 60)
    print(f"🎉 HOÀN TẤT A-Z! Tổng cộng: {total_links} đường link.")

if __name__ == "__main__":
    crawl_by_alphabet()