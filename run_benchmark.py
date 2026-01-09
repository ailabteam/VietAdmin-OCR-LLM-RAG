import os
import pandas as pd
from jiwer import wer, cer
from src.ocr_engine import VietAdminOCR
from src.correction import VietAdminCorrection

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def run_experiment():
    # 1. Khởi tạo Modules
    ocr_engine = VietAdminOCR(use_gpu=True)
    corrector = VietAdminCorrection()
    
    # Danh sách các ảnh cần test trong thư mục augmented
    augmented_dir = "data/augmented"
    image_files = sorted([f for f in os.listdir(augmented_dir) if f.endswith(".jpg")])
    
    results_data = []

    print(f"\n>>> BẮT ĐẦU BENCHMARK TRÊN {len(image_files)} MẪU THỬ NGHIỆM...")

    for img_name in image_files:
        print(f"\nĐang xử lý: {img_name}")
        img_path = os.path.join(augmented_dir, img_name)
        
        # Xác định file Ground Truth tương ứng (4_1 hoặc 4_2)
        gt_file = "data/4_1_gt.txt" if "4_1" in img_name else "data/4_2_gt.txt"
        if not os.path.exists(gt_file):
            print(f"Bỏ qua {img_name} vì thiếu file Ground Truth.")
            continue
            
        ground_truth = load_text(gt_file)

        # 2. Bước OCR Thô
        ocr_results = ocr_engine.extract_text(img_path)
        raw_text = " ".join([page['raw_text'] for page in ocr_results])
        
        # 3. Bước Sửa lỗi
        clean_text = corrector.process_large_text(raw_text)

        # 4. Tính toán chỉ số lỗi (WER - Word Error Rate)
        wer_raw = wer(ground_truth, raw_text) * 100
        wer_clean = wer(ground_truth, clean_text) * 100
        
        # Tính tỉ lệ cải thiện
        improvement = wer_raw - wer_clean

        results_data.append({
            "Image": img_name,
            "Variant": img_name.split("_v")[1].split(".")[0], # v0, v1, v2...
            "WER_Raw (%)": round(wer_raw, 2),
            "WER_Clean (%)": round(wer_clean, 2),
            "Improvement (%)": round(improvement, 2)
        })
        
        print(f"Result for {img_name}: Raw WER={wer_raw:.2f}% -> Clean WER={wer_clean:.2f}%")

    # 5. Lưu kết quả ra CSV
    df = pd.DataFrame(results_data)
    csv_path = "results/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n>>> TỔNG KẾT THỰC NGHIỆM ĐÃ LƯU TẠI: {csv_path}")
    print("\nBảng kết quả tóm tắt:")
    print(df)

if __name__ == "__main__":
    run_experiment()
