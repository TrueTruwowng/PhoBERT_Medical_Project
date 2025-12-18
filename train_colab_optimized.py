
import json
import os
import torch
import time
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset

#1. KẾT NỐI DRIVE
if not os.path.exists('/content/drive'):
    print("Đang kết nối Google Drive...")
    drive.mount('/content/drive')

#2. CẤU HÌNH ĐƯỜNG DẪN
BASE_PATH = "/content/drive/MyDrive/PhoBERT_Medical_Project"

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
    print(f"✅ Thư mục làm việc: {BASE_PATH}")

MODEL_NAME = "vinai/phobert-large" 
DATA_FILE = "merged_train_data.jsonl" 

OUTPUT_DIR = os.path.join(BASE_PATH, "phobert_large_final")
CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")
MAX_LENGTH = 128     

BATCH_SIZE = 32       
GRAD_ACCUMULATION = 1 
LEARNING_RATE = 2e-5  
EPOCHS = 1           

def print_status(msg):
    print(f"\n🚀 {msg}")

def check_environment():
    print("="*50)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} | VRAM: {vram:.2f} GB")
        print("✅ Cấu hình: MAX SPEED (Len 128 - Batch 32).")
    else:
        print("❌ LỖI: Không tìm thấy GPU! Hãy bật GPU trong Runtime settings.")
        exit()
    print("="*50)

try:
    from underthesea import word_tokenize
except ImportError:
    print("⚠️ Đang cài underthesea...")
    os.system("pip install underthesea")
    from underthesea import word_tokenize

def segment_text(text):
    if not text: return ""
    try:
        return word_tokenize(text, format="text")
    except:
        return text

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def find_cache_file(base_path):
    possible_names = ["train_data_cached.pt", "train_data_cached_colab.pt", "cached_data.pt"]
    print(f"📂 Đang tìm Cache trong: {base_path}")
    for name in possible_names:
        full_path = os.path.join(base_path, name)
        if os.path.exists(full_path):
            return full_path
    return None

def main():
    check_environment()
    torch.backends.cudnn.benchmark = True 

    texts = []
    labels = []
    label_map = {"Sai": 0, "Đúng": 1} 

    #3. XỬ LÝ DỮ LIỆU
    found_cache = find_cache_file(BASE_PATH)

    if found_cache:
        print_status(f"✅ TÌM THẤY CACHE: {found_cache}")
        print("   -> Đang load dữ liệu (Siêu nhanh)...")
        cached_data = torch.load(found_cache)
        texts = cached_data['texts']
        labels = cached_data['labels']
    else:
        print_status("⚠️ KHÔNG TÌM THẤY CACHE. BẮT ĐẦU XỬ LÝ GỐC...")
        file_path_to_read = DATA_FILE
        drive_data_path = os.path.join(BASE_PATH, DATA_FILE)
        
        if os.path.exists(drive_data_path):
            file_path_to_read = drive_data_path
        elif not os.path.exists(DATA_FILE):
             print(f"❌ Lỗi: Không tìm thấy file jsonl. Hãy upload lên Drive!")
             return

        try:
            with open(file_path_to_read, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()
            
            print(f"   Đang tách từ {len(raw_lines)} dòng...")
            count = 0
            for line in raw_lines:
                try:
                    record = json.loads(line)
                    q = str(record.get('question', '')).strip()
                    a = str(record.get('answer', '')).strip()
                    if q and a in label_map:
                        texts.append(segment_text(q))
                        labels.append(label_map[a])
                        count += 1
                        if count % 5000 == 0: print(f"   {count}...", end='\r')
                except: continue
            
            new_cache_path = os.path.join(BASE_PATH, "train_data_cached.pt")
            print(f"\n✅ Xử lý xong. Lưu Cache vào: {new_cache_path}")
            torch.save({'texts': texts, 'labels': labels}, new_cache_path)
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return

    if len(texts) == 0: return

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    print_status(f"LOAD MODEL: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "Sai", 1: "Đúng"}, label2id={"Sai": 0, "Đúng": 1}
    )

    def preprocess_function(examples):

        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    print("⚙️ Đang Tokenize (Mã hóa dữ liệu)...")
    encoded_train = train_dataset.map(preprocess_function, batched=True)
    encoded_val = val_dataset.map(preprocess_function, batched=True)

    #4. CẤU HÌNH TRAINER
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR, 
        learning_rate=LEARNING_RATE,
        
        per_device_train_batch_size=BATCH_SIZE, 
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        
        eval_strategy="steps",     
        eval_steps=1000,                
        save_strategy="steps",          
        save_steps=1000,           
        save_total_limit=1,       
        load_best_model_at_end=True,
        
        fp16=True,                 
        group_by_length=True,      
        dataloader_num_workers=4,  
        dataloader_pin_memory=True,
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    #5. TỰ ĐỘNG KHÔI PHỤC (RESUME)
    latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        print_status(f"⚠️ TÌM THẤY CHECKPOINT: {latest_ckpt}")
        print("   -> Đang khôi phục train (Resume)...")
        trainer.train(resume_from_checkpoint=latest_ckpt)
    else:
        print_status(f"BẮT ĐẦU TRAIN TỐC ĐỘ CAO (Dự kiến 2-3 tiếng)...")
        trainer.train()

    #6. LƯU KẾT QUẢ
    print_status(f"LƯU MODEL VÀO DRIVE: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ XONG! Model đã an toàn trên Drive.")

if __name__ == "__main__":
    main()