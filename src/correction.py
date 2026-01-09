import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class VietAdminCorrection:
    def __init__(self, model_path="protonx-models/protonx-legal-tc", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Đang tải mô hình sửa lỗi ProtonX lên {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def correct_chunk(self, text):
        """Sửa lỗi cho một đoạn văn bản ngắn (< 160 tokens)"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=160
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=160,
                num_beams=5,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def process_large_text(self, text, chunk_size=30):
        """
        Chia văn bản thành các nhóm từ (ví dụ 30 từ mỗi cụm) 
        để đảm bảo không vượt quá giới hạn 160 tokens của model.
        """
        words = text.split()
        corrected_text = []
        
        print(f"Đang xử lý sửa lỗi cho {len(words)} từ...")
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                corrected_chunk = self.correct_chunk(chunk)
                corrected_text.append(corrected_chunk)
        
        return " ".join(corrected_text)

if __name__ == "__main__":
    # Test nhanh với một đoạn lỗi từ kết quả OCR của bạn
    sample_error = "SỞ GIÁO DUC VÀ ĐÀO TẠO Đà ngày 26tháng nám 2020 V/v khán truong thực hiện các biện pháp"
    
    corrector = VietAdminCorrection()
    clean_text = corrector.process_large_text(sample_error)
    
    print("\n[GỐC]:", sample_error)
    print("\n[SAU KHI SỬA]:", clean_text)
