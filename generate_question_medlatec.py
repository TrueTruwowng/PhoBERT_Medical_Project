#SINH CAU HOI MEDLATEC
import json
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-AWQ"
INPUT_FILE = 'medlatec_benh.json'  # Your data file name
OUTPUT_FILE = 'train_dataset_medlatec.jsonl'

print(f"Đang khởi động Engine vLLM với {MODEL_ID}...") # Retained print

# Initialize vLLM Engine
# gpu_memory_utilization=0.9: Use 90% of A100 VRAM
llm = LLM(
    model=MODEL_ID,
    quantization="awq",
    dtype="half",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    enforce_eager=True  # Add this line if encountering CUDA graph errors on Colab
)

# Text generation configuration
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=1024,
    stop=["<|im_end|>", "<|endoftext|>"]
)

print("Engine sẵn sàng! Chuẩn bị đua tốc độ...") # Retained print


def parse_sections(text):
    """Parses text into structured sections based on Vietnamese keywords."""
    if not text: return {}

    # Mapping output keys to Vietnamese keywords
    keywords = {
        "Tong_quan": ["Tổng quan", "Định nghĩa", "là gì"],
        "Nguyen_nhan": ["Nguyên nhân", "Căn nguyên"],
        "Trieu_chung": ["Triệu chứng", "Dấu hiệu"],
        "Chan_doan": ["Chẩn đoán", "Xét nghiệm"],
        "Dieu_tri": ["Điều trị", "Chữa trị", "Thuốc"],
        "Phong_ngua": ["Phòng ngừa", "Dự phòng"],
        "Bien_chung": ["Biến chứng", "Hậu quả"]
    }

    sections = {}
    lines = text.split('\n')
    current_key = "Tong_quan"
    current_content = []

    for line in lines:
        line = line.strip()
        if not line: continue

        is_header = False
        for key, variants in keywords.items():
            # Check if line contains a section header keyword (case-insensitive)
            if any(v.lower() in line.lower() for v in variants) and len(line) < 100:
                if current_content:
                    # Store content of the previous section
                    sections[current_key] = "\n".join(current_content)

                # Start new section
                current_key = key
                current_content = []
                is_header = True
                break

        if not is_header:
            current_content.append(line)

    # Store the content of the last section
    if current_content: sections[current_key] = "\n".join(current_content)

    return sections

def make_prompt(disease_name, category, context):
    """Creates the Qwen Chat standard prompt for question generation."""
    return f"""<|im_start|>system
Bạn là chuyên gia y tế. Tạo bộ câu hỏi trắc nghiệm Đúng/Sai.<|im_end|>
<|im_start|>user
Bệnh: {disease_name}
Mục: {category}
Nội dung: "{context[:3000]}"

YÊU CẦU:
1. Sinh 4-6 câu nhận định (Statement).
2. 50% câu ĐÚNG, 50% câu SAI.
3. Input là câu KHẲNG ĐỊNH. KHÔNG viết câu hỏi.
4. Chỉ trả về JSON List: [{{"input": "...", "output": "..."}}]<|im_end|>
<|im_start|>assistant
"""

def main():
    """Main function to load data, generate prompts, run inference, and save results."""
    if not os.path.exists(INPUT_FILE):
        print("Không tìm thấy file input!") # Retained print
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict): data = [data]

    # CREATE LIST OF ALL PROMPTS
    all_prompts = []
    metadata = []  # Stores accompanying info to map results back

    print("Đang chuẩn bị dữ liệu...") # Retained print
    for item in tqdm(data, desc="Parsing"):
        clean_text = item.get('clean_text', '')
        url = item.get('url', '')
        # Extract disease name, handling potential "Tổng quan" prefix
        disease_name = clean_text.split('\n')[0].replace("Tổng quan", "").strip() or "Bệnh lý"

        sections = parse_sections(clean_text)

        for sec_key, sec_content in sections.items():
            if len(sec_content) < 50: continue # Skip very short sections

            prompt = make_prompt(disease_name, sec_key, sec_content)
            all_prompts.append(prompt)
            metadata.append({"url": url, "category": sec_key})

    print(f"Tổng cộng có {len(all_prompts)} tác vụ cần xử lý.") # Retained print

    # RUN BATCH INFERENCE 
    print("BẮT ĐẦU CHẠY BATCH TRÊN A100...") # Retained print
    # This part is much faster than traditional iterative generation
    outputs = llm.generate(all_prompts, sampling_params)

    # STEP 3: SAVE RESULTS
    count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            meta = metadata[i]

            try:
                # Manual JSON parsing/extraction (robust to extra text)
                start = generated_text.find('[')
                end = generated_text.rfind(']') + 1

                if start != -1 and end != -1:
                    # Load the list of Q&A dictionaries
                    qa_list = json.loads(generated_text[start:end])

                    for qa in qa_list:
                        inp = qa.get('input', '')
                        # Basic filtering for quality
                        if len(inp) < 15 or "?" in inp: continue

                        final_obj = {
                            # Instruction generation based on category
                            "instruction": f"Dựa vào kiến thức y học về {meta['category']}, hãy xác định nhận định sau là Đúng hay Sai.",
                            "input": inp,
                            "output": qa.get('output', ''),
                            "source": meta['url'],
                            "category": meta['category']
                        }

                        # Write the final JSON object in jsonl format
                        json.dump(final_obj, f_out, ensure_ascii=False)
                        f_out.write('\n')
                        count += 1
            except:
                # Skip if JSON parsing fails
                continue

    print(f"\n HOÀN TẤT! Đã sinh được {count} câu hỏi.") # Retained print
    print(f"Kết quả lưu tại: {OUTPUT_FILE}") # Retained print

if __name__ == "__main__":

    main()

