import os
import re
import cv2
import torch
import pandas as pd
import numpy as np
from jiwer import wer
from src.ocr_engine import VietAdminOCR
from src.correction import VietAdminCorrection

# --- UTILS ---
def normalize_text(text):
    if not text: return ""
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

def load_text(file_path):
    if not os.path.exists(file_path): return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# --- TILING LOGIC (Hàm bóc tách nâng cao) ---
def get_tiled_text(img, ocr_reader):
    """
    Chiến lược bóc tách: Chia ảnh thành 4 vùng (2x2), 
    phóng to và làm nét từng vùng để không sót chữ.
    """
    h, w = img.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    overlap = 100 # Chồng lấp để không mất chữ ở đường cắt
    
    # Định nghĩa 4 vùng: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    coords = [
        (0, mid_h + overlap, 0, mid_w + overlap),
        (0, mid_h + overlap, mid_w - overlap, w),
        (mid_h - overlap, h, 0, mid_w + overlap),
        (mid_h - overlap, h, mid_w - overlap, w)
    ]
    
    tiled_raw_text = []
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) # Sharpening kernel

    for i, (y1, y2, x1, x2) in enumerate(coords):
        tile = img[y1:y2, x1:x2]
        # Tiền xử lý từng mảnh: Phóng to 1.5x và làm nét
        tile = cv2.resize(tile, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)
        tile = cv2.filter2D(tile, -1, kernel)
        
        # OCR từng mảnh
        result = ocr_reader.readtext(tile, detail=0)
        tiled_raw_text.extend(result)
    
    return " ".join(tiled_raw_text)

# --- MAIN EXPERIMENT ---
def run_experiment():
    print("\n>>> ĐANG KHỞI TẠO HỆ THỐNG TRÊN RTX 4090...")
    ocr_engine = VietAdminOCR(use_gpu=True)
    corrector = VietAdminCorrection()
    
    augmented_dir = "data/augmented"
    image_files = sorted([f for f in os.listdir(augmented_dir) if f.endswith(".jpg")])
    
    results_data = []

    print(f"\n>>> BẮT ĐẦU SO SÁNH CHIẾN LƯỢC TRÊN {len(image_files)} MẪU...")

    for img_name in image_files:
        print(f"\n--- Processing: {img_name} ---")
        img_path = os.path.join(augmented_dir, img_name)
        img = cv2.imread(img_path)
        
        # Xác định Ground Truth
        gt_file = "data/4_1_gt.txt" if "4_1" in img_name else "data/4_2_gt.txt"
        gt_raw = load_text(gt_file)
        if not gt_raw: continue
        gt_norm = normalize_text(gt_raw)

        # --- CHIẾN LƯỢC 1: GLOBAL OCR (Cũ) ---
        raw_global = ocr_engine.extract_text(img_path)
        raw_global_text = " ".join([p['raw_text'] for p in raw_global])
        clean_global_text = corrector.process_large_text(raw_global_text)
        
        wer_global_raw = wer(gt_norm, normalize_text(raw_global_text))
        wer_global_clean = wer(gt_norm, normalize_text(clean_global_text))

        # --- CHIẾN LƯỢC 2: TILED OCR (Đề xuất mới) ---
        raw_tiled_text = get_tiled_text(img, ocr_engine.reader)
        clean_tiled_text = corrector.process_large_text(raw_tiled_text)
        
        wer_tiled_raw = wer(gt_norm, normalize_text(raw_tiled_text))
        wer_tiled_clean = wer(gt_norm, normalize_text(clean_tiled_text))

        # --- Ghi nhận kết quả ---
        results_data.append({
            "Image": img_name,
            "Variant": img_name.split("_v")[1].split(".")[0],
            "WER_Global_Raw (%)": round(wer_global_raw * 100, 2),
            "WER_Global_Clean (%)": round(wer_global_clean * 100, 2),
            "WER_Tiled_Raw (%)": round(wer_tiled_raw * 100, 2),
            "WER_Tiled_Clean (%)": round(wer_tiled_clean * 100, 2),
            "Best_Improvement (%)": round((wer_global_raw - wer_tiled_clean) * 100, 2)
        })
        
        print(f"Global Clean WER: {round(wer_global_clean*100, 2)}% | Tiled Clean WER: {round(wer_tiled_clean*100, 2)}%")

    # Lưu kết quả
    df = pd.DataFrame(results_data)
    csv_path = "results/strategy_comparison.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n>>> KẾT QUẢ SO SÁNH ĐÃ LƯU TẠI: {csv_path}")
    print("\n" + "="*50)
    print("BẢNG SO SÁNH CHIẾN LƯỢC BÓC TÁCH (WER %)")
    print("="*50)
    print(df[["Image", "WER_Global_Clean (%)", "WER_Tiled_Clean (%)", "Best_Improvement (%)"]])

if __name__ == "__main__":
    run_experiment()
