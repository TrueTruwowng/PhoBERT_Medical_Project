import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin

class MedlinePlusScraper:
    def __init__(self):
        self.base_url = "https://medlineplus.gov"
        self.encyclopedia_url = "https://medlineplus.gov/encyclopedia.html"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.articles = []

    def get_alphabet_links(self):
        """Lấy tất cả các link A-Z từ trang encyclopedia"""
        print("Đang lấy danh sách các chữ cái A-Z...")
        response = requests.get(self.encyclopedia_url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Tìm các link A-Z - cấu trúc thực tế
        alphabet_links = []

        # Tìm phần "Other encyclopedia articles A-Z"
        # Các link A-Z thường nằm trong div có class hoặc id chứa "index" hoặc trực tiếp là các link
        az_links = soup.find_all('a', href=True)

        for link in az_links:
            href = link.get('href')
            text = link.text.strip()
            # Tìm các link có dạng /encyclopedia_A.htm, /encyclopedia_B.htm, etc
            if href and '/encyclopedia_' in href and len(text) <= 3:
                full_url = urljoin(self.base_url, href)
                alphabet_links.append({
                    'letter': text,
                    'url': full_url
                })

        # Nếu không tìm thấy, thử cách khác
        if not alphabet_links:
            print("Thử phương pháp khác để tìm link A-Z...")
            # Tạo link trực tiếp theo pattern
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for letter in letters:
                alphabet_links.append({
                    'letter': letter,
                    'url': f'{self.base_url}/ency/encyclopedia_{letter}.htm'
                })

        print(f"Tìm thấy {len(alphabet_links)} chữ cái")
        return alphabet_links

    def scrape_article_list(self, letter_url):
        """Cào danh sách bài viết từ một trang chữ cái"""
        try:
            print(f"Đang cào: {letter_url}")
            response = requests.get(letter_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            articles = []

            # Theo cấu trúc HTML thực tế: các bài viết nằm trong <ul id="index">
            article_list = soup.find('ul', {'id': 'index'})

            if article_list:
                links = article_list.find_all('a', href=True)
                print(f"  Tìm thấy {len(links)} links trong ul#index")

                for link in links:
                    href = link.get('href')
                    title = link.text.strip()

                    if href and title:
                        # Tạo URL đầy đủ
                        if href.startswith('http'):
                            article_url = href
                        else:
                            # href có dạng: patientinstructions/000823.htm hoặc article/003640.htm
                            article_url = urljoin(letter_url, href)

                        articles.append({
                            'title': title,
                            'url': article_url
                        })
            else:
                print("  Không tìm thấy ul#index")

            # Loại bỏ duplicate
            seen = set()
            unique_articles = []
            for article in articles:
                if article['url'] not in seen:
                    seen.add(article['url'])
                    unique_articles.append(article)

            print(f"  Tổng cộng: {len(unique_articles)} bài viết unique")
            return unique_articles
        except Exception as e:
            print(f"Lỗi khi cào {letter_url}: {str(e)}")
            return []

    def scrape_article_detail(self, article):
        """Cào chi tiết một bài viết"""
        try:
            print(f"Đang cào chi tiết: {article['title']}")
            response = requests.get(article['url'], headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Lấy tiêu đề từ page-title
            page_title = soup.find('div', class_='page-title')
            if page_title:
                h1 = page_title.find('h1')
                if h1:
                    article['title'] = h1.text.strip()

            # Lấy phần tóm tắt từ ency_summary
            summary = ""
            ency_summary = soup.find('div', {'id': 'ency_summary'})
            if ency_summary:
                paragraphs = ency_summary.find_all('p')
                summary = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

            article['summary'] = summary if summary else None

            # Lấy các sections
            sections = []
            section_divs = soup.find_all('div', class_='section')

            for section_div in section_divs:
                # Lấy tiêu đề section
                section_title_div = section_div.find('div', class_='section-title')
                if section_title_div:
                    h2 = section_title_div.find('h2')
                    if h2:
                        section_title = h2.get_text(strip=True)

                        # Lấy nội dung section
                        section_body = section_div.find('div', class_='section-body')
                        if section_body:
                            section_content = []

                            # Lấy tất cả paragraphs và lists
                            for elem in section_body.find_all(['p', 'ul', 'ol']):
                                if elem.name == 'p':
                                    text = elem.get_text(strip=True)
                                    if text:
                                        section_content.append(text)
                                elif elem.name in ['ul', 'ol']:
                                    # Lấy các list items
                                    items = elem.find_all('li')
                                    for item in items:
                                        text = item.get_text(strip=True)
                                        if text:
                                            section_content.append(f"• {text}")

                            if section_content:
                                sections.append({
                                    'heading': section_title,
                                    'content': '\n'.join(section_content)
                                })

            article['sections'] = sections if sections else None

            # Thêm delay để tránh quá tải server
            time.sleep(1)

            return article
        except Exception as e:
            print(f"Lỗi khi cào chi tiết {article['url']}: {str(e)}")
            article['error'] = str(e)
            return article

    def scrape_all(self, include_details=False, max_letters=None, max_articles_per_letter=None):
        """Cào tất cả dữ liệu"""
        print("Bắt đầu cào dữ liệu MedlinePlus Encyclopedia...")

        # Lấy danh sách chữ cái
        alphabet_links = self.get_alphabet_links()

        if max_letters:
            alphabet_links = alphabet_links[:max_letters]

        # Cào từng chữ cái
        for idx, letter_info in enumerate(alphabet_links, 1):
            print(f"\n--- [{idx}/{len(alphabet_links)}] Đang xử lý chữ cái: {letter_info['letter']} ---")
            articles = self.scrape_article_list(letter_info['url'])

            print(f"Tìm thấy {len(articles)} bài viết")

            # Giới hạn số bài viết nếu cần
            if max_articles_per_letter:
                articles = articles[:max_articles_per_letter]
                print(f"Giới hạn {len(articles)} bài viết")

            # Nếu cần lấy chi tiết từng bài viết
            if include_details and articles:
                print("Đang lấy chi tiết các bài viết...")
                for i, article in enumerate(articles, 1):
                    print(f"  [{i}/{len(articles)}] ", end="")
                    self.scrape_article_detail(article)

            self.articles.extend(articles)

            # Delay giữa các chữ cái
            time.sleep(2)

        print(f"\n\n Hoàn thành! Tổng cộng: {len(self.articles)} bài viết")
        return self.articles

    def save_to_json(self, filename='medlineplus_data.json'):
        """Lưu dữ liệu vào file JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'total_articles': len(self.articles),
                'articles': self.articles,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu dữ liệu vào {filename}")


def main():
    scraper = MedlinePlusScraper()
    articles = scraper.scrape_all(include_details=True, max_letters=None, max_articles_per_letter=None)

    # Lưu vào file JSON
    scraper.save_to_json('medlineplus_encyclopedia_full.json')

    if articles:
        print(json.dumps(articles[:3], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

