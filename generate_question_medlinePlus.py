import json
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Sử dụng model mạnh về đa ngôn ngữ và y tế
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-AWQ"
INPUT_FILE = 'data_medlineplus_vi.json'   # File dữ liệu MedlinePlus
OUTPUT_FILE = 'train_dataset_medlineplus.jsonl'

# Cấu hình VLLM
llm = LLM(
    model=MODEL_ID,
    quantization="awq",
    dtype="half",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    enforce_eager=True
)

# Cấu hình sinh văn bản 
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    max_tokens=1500,
    stop=["<|im_end|>", "<|endoftext|>"]
)

def make_cross_lingual_prompt(title, content):
    """
    Prompt đặc biệt: Input tiếng Anh (hoặc Việt) -> Output bắt buộc Tiếng Việt.
    """
    return f"""<|im_start|>system
Bạn là chuyên gia y tế song ngữ Anh-Việt. Nhiệm vụ của bạn là tạo dữ liệu huấn luyện cho mô hình AI y tế Việt Nam.<|im_end|>
<|im_start|>user
Dựa trên thông tin dưới đây từ MedlinePlus:

**Chủ đề:** {title}
**Nội dung:** "{content[:3500]}"

YÊU CẦU:
1. Tạo 4-6 nhận định (statement) bằng **TIẾNG VIỆT**.
2. Đảm bảo 50% là nhận định ĐÚNG (True), 50% là nhận định SAI (False).
3. Nhận định phải dựa hoàn toàn vào nội dung cung cấp, không bịa đặt.
4. Với câu SAI, hãy sửa đổi một chi tiết quan trọng (ví dụ: nguyên nhân, triệu chứng, tên thuốc) để làm nó sai.
5. Trả về định dạng JSON List chính xác: [{{ "input": "...", "output": "..." }}]
<|im_end|>
<|im_start|>assistant
"""

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Không tìm thấy file {INPUT_FILE}")
        return

    # 1. Đọc dữ liệu đầu vào
    print("Đang đọc dữ liệu MedlinePlus...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Xử lý trường hợp data là dict hoặc list
        if isinstance(data, dict) and 'feed' in data: # Cấu trúc thường thấy của XML Medline convert sang JSON
            data = data['feed'].get('entry', [])
        elif isinstance(data, dict):
            data = [data]

    all_prompts = []
    metadata = []

    # 2. Tạo Prompt
    print("Đang tạo prompt...")
    for item in tqdm(data):
        # MedlinePlus có: title, summary, content, hoặc description
        title = item.get('title', {}).get('#text', '') if isinstance(item.get('title'), dict) else item.get('title', 'Y tế')

        # Lấy nội dung: ưu tiên content dài, nếu không có thì lấy summary
        content = item.get('content', {}).get('#text', '') if isinstance(item.get('content'), dict) else item.get('content', '')
        if not content:
            content = item.get('summary', {}).get('#text', '') if isinstance(item.get('summary'), dict) else item.get('summary', '')

        # Bỏ qua nếu nội dung quá ngắn
        if len(str(content)) < 100: continue

        prompt = make_cross_lingual_prompt(title, content)
        all_prompts.append(prompt)
        metadata.append({"source": "MedlinePlus", "topic": title})

    print(f"Tổng số prompt cần chạy: {len(all_prompts)}")

    # 3. Chạy Inference (Batch)
    print("Bắt đầu sinh dữ liệu...")
    outputs = llm.generate(all_prompts, sampling_params)

    # 4. Lưu kết quả
    valid_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            meta = metadata[i]

            try:
                # Trích xuất JSON từ text sinh ra
                start = generated_text.find('[')
                end = generated_text.rfind(']') + 1

                if start != -1 and end != -1:
                    qa_list = json.loads(generated_text[start:end])

                    for qa in qa_list:
                        inp = qa.get('input', '')
                        out = qa.get('output', '')

                        # Kiểm tra xem output có phải tiếng Việt không (sơ bộ)
                        # Nếu output vẫn là tiếng Anh thì bỏ qua (hoặc lọc sau)
                        if len(inp) < 10: continue

                        final_obj = {
                            "instruction": f"Xác định tính Đúng/Sai của nhận định sau dựa trên kiến thức về {meta['topic']}.",
                            "input": inp,
                            "output": str(out).upper(), # Chuẩn hóa TRUE/FALSE
                            "source_type": "MedlinePlus",
                            "topic": meta['topic']
                        }

                        json.dump(final_obj, f_out, ensure_ascii=False)
                        f_out.write('\n')
                        valid_count += 1
            except Exception as e:
                continue

    print(f"\nHOÀN TẤT! Đã sinh được {valid_count} câu hỏi tiếng Việt từ MedlinePlus.")
    print(f"Lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":

    main()
