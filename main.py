import os
from src.ocr_engine import VietAdminOCR
from src.correction import VietAdminCorrection

def pipeline(input_file):
    # 1. Khởi tạo các module
    ocr_engine = VietAdminOCR(use_gpu=True)
    corrector = VietAdminCorrection()

    # 2. Trích xuất OCR thô
    print(f"\n--- BẮT ĐẦU XỬ LÝ: {input_file} ---")
    ocr_results = ocr_engine.extract_text(input_file)
    
    final_output = []

    for page in ocr_results:
        raw_text = page['raw_text']
        print(f"Đang sửa lỗi trang {page['page']}...")
        
        # 3. Sửa lỗi bằng model ProtonX
        corrected_text = corrector.process_large_text(raw_text)
        
        final_output.append({
            'page': page['page'],
            'raw': raw_text,
            'clean': corrected_text
        })

    # 4. Lưu kết quả ra file để làm tài liệu viết Paper
    file_name = os.path.basename(input_file).split('.')[0]
    result_path = f"results/{file_name}_comparison.txt"
    
    with open(result_path, "w", encoding="utf-8") as f:
        for item in final_output:
            f.write(f"=== TRANG {item['page']} ===\n")
            f.write(f"[OCR THÔ]: {item['raw']}\n\n")
            f.write(f"[SỬA LỖI]: {item['clean']}\n")
            f.write("-" * 50 + "\n")
            
    print(f"\n Đã lưu kết quả so sánh tại: {result_path}")

if __name__ == "__main__":
    # Thử nghiệm với file ảnh 4_1.jpg trong data của bạn
    test_path = "data/4_1.jpg"
    if os.path.exists(test_path):
        pipeline(test_path)
    else:
        print("Không tìm thấy file test!")
