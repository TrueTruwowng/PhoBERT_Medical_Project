from bs4 import BeautifulSoup
import requests
import csv
import json
import time

BASE = "https://medlatec.vn"
START_URL = BASE + "/tu-dien-benh-ly"

print("Crawling index page...")

html_text = requests.get(START_URL).text
soup = BeautifulSoup(html_text, "lxml")

# Lấy URLs bệnh
lists = soup.find_all("ul", class_="disease-list")
all_urls = []
for ul in lists:
    for a in ul.find_all("a"):
        href = a.get("href")
        if href:
            all_urls.append(BASE + href)

print("Total found:", len(all_urls))

# Ánh xạ heading -> field JSON
SECTION_MAP = {
    "nguyên nhân": "cause",
    "triệu chứng": "symptoms",
    "dấu hiệu": "symptoms",
    "điều trị": "treatment",
    "phòng ngừa": "prevention",
    "dự phòng": "prevention",
    "chẩn đoán": "diagnosis",
    "biến chứng": "complications",
    "tổng quan": "disease_description",
    "là gì": "disease_description"
}

output = []

def identify_section(title):
    t = title.lower()
    for key in SECTION_MAP:
        if key in t:
            return SECTION_MAP[key]
    return None  


for i, url in enumerate(all_urls, 1):
    print(f"\n--- Crawling {i}: {url}")

    html = requests.get(url).text
    soup = BeautifulSoup(html, "lxml")

    # Lấy tên bệnh thuần túy, bỏ phần ": ..."
    disease_title = soup.find("h1", class_="page-title")
    disease_name = disease_title.get_text(strip=True).split(" :")[0] if disease_title else ""

    content_area = soup.find("div", class_="description")
    if not content_area:
        continue

    # Loại phần không cần
    for tag in content_area.find_all(["img", "figure", "figcaption"]):
        tag.decompose()
    for tag in content_area.find_all(style=lambda x: x and "text-align: center" in x):
        tag.decompose()
    for div in content_area.find_all("div", style=lambda x: x and "background:#eee" in x):
        strong = div.find("strong")
        if strong and "tài liệu tham khảo" in strong.get_text(strip=True).lower():
            div.decompose()

    # Extract tất cả H2 làm section header
    sections = content_area.find_all("h2")

    # Chuẩn bị output fields
    data = {
        "url": url,
        "source": "Medlatec",
        "disease": disease_name,
        "cause": "",
        "symptoms": "",
        "treatment": "",
        "prevention": "",
        "diagnosis": "",
        "complications": "",
        "disease_description": ""
    }

    for h2 in sections:
        section_title = h2.get_text(" ", strip=True)
        section_key = identify_section(section_title)
        if not section_key:
            continue  # skip heading không match

        # Lấy nội dung dưới H2 tới H2 tiếp theo
        content = []
        next_nodes = h2.find_all_next()
        for node in next_nodes:
            if node.name == "h2":
                break
            if node.name in ["p", "li", "h3"]:
                text = node.get_text(" ", strip=True)
                if text:
                    content.append(text)

        data[section_key] = "\n".join(content)

    output.append(data)
    time.sleep(0.3)

# Save JSON
with open("medlatec_structured.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Save CSV
with open("medlatec_structured.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=output[0].keys())
    writer.writeheader()
    writer.writerows(output)

print("\nDONE! Saved structured JSON + CSV.")

