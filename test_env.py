# test_env.py (Bản cập nhật dùng EasyOCR - PyTorch based)
import torch
import easyocr
from transformers import AutoTokenizer

def run_test():
    print("="*50)
    print("KIỂM TRA MÔI TRƯỜNG VIETADMIN (EASYOCR + PYTORCH)")
    print("="*50)

    print("\n--- 1. Kiểm tra PyTorch & GPU ---")
    print(f"PyTorch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"Số lượng GPU: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        # Thử tạo một tensor trên GPU
        x = torch.rand(5, 3).cuda()
        print("Test Tensor trên GPU: OK")

    print("\n--- 2. Kiểm tra EasyOCR (OCR trên GPU) ---")
    try:
        # Khởi tạo EasyOCR với ngôn ngữ tiếng Việt
        # Model sẽ được tải về trong lần chạy đầu tiên (~100MB)
        reader = easyocr.Reader(['vi'], gpu=True)
        print("Khởi tạo EasyOCR: THÀNH CÔNG")
    except Exception as e:
        print(f"Lỗi khởi tạo EasyOCR: {e}")

    print("\n--- 3. Kiểm tra Transformers (ProtonX Model) ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained("protonx-models/protonx-legal-tc")
        print("Tải Tokenizer ProtonX: THÀNH CÔNG")
    except Exception as e:
        print(f"Lỗi kết nối HuggingFace: {e}")

    print("\n" + "="*50)

if __name__ == "__main__":
    run_test()
