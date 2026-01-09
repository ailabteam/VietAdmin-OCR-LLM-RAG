import os
import re
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import wer
from difflib import SequenceMatcher
from src.ocr_engine import VietAdminOCR
from src.correction import VietAdminCorrection

# --- 1. CHUẨN HÓA & TIỆN ÍCH ---
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

# --- 2. THUẬT TOÁN GHÉP MẢNH KHỬ TRÙNG (STITCHING) ---
def smart_merge(s1, s2):
    """
    Tìm điểm cắt tối ưu giữa 2 đoạn văn bản chồng lấp để triệt tiêu lặp từ.
    """
    words1 = s1.split()
    words2 = s2.split()
    
    # Lấy 30 từ cuối của s1 và 30 từ đầu của s2 để tìm điểm giao
    tail_len = min(len(words1), 30)
    head_len = min(len(words2), 30)
    
    tail = " ".join(words1[-tail_len:])
    head = " ".join(words2[:head_len])
    
    matcher = SequenceMatcher(None, tail, head)
    match = matcher.find_longest_match(0, len(tail), 0, len(head))
    
    if match.size > 10: # Nếu trùng nhau đủ lớn
        # Cắt bỏ phần trùng ở đầu đoạn sau
        overlap_str = head[match.b + match.size:].strip()
        return s1 + " " + overlap_str + " " + " ".join(words2[head_len:])
    
    # Nếu không tìm thấy điểm cắt, dùng tập hợp để lọc bớt từ lặp cực bộ
    last_10_words = set(words1[-10:])
    filtered_head = [w for w in words2[:head_len] if w not in last_10_words]
    return s1 + " " + " ".join(filtered_head) + " " + " ".join(words2[head_len:])

# --- 3. CHIẾN LƯỢC TILING (STRIPES) ---
def get_tiled_text(img, ocr_reader):
    h, w = img.shape[:2]
    h_step = h // 3
    overlap = 150
    coords = [
        (0, h_step + overlap, 0, w),
        (h_step - overlap, 2*h_step + overlap, 0, w),
        (2*h_step - overlap, h, 0, w)
    ]
    
    segments = []
    for (y1, y2, x1, x2) in coords:
        tile = img[y1:y2, x1:x2]
        tile = cv2.resize(tile, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LANCZOS4)
        res = ocr_reader.readtext(tile, detail=0, paragraph=True)
        segments.append(" ".join(res))
    
    # Ghép thông minh
    full_text = segments[0]
    for i in range(1, len(segments)):
        full_text = smart_merge(full_text, segments[i])
    return full_text

# --- 4. VẼ BIỂU ĐỒ CHUẨN IEEE ---
def save_figures(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: WER Comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df['Variant']))
    width = 0.35
    
    plt.bar(x - width/2, df['G_WER (%)'], width, label='Global-OCR (Baseline)', color='#e74c3c')
    plt.bar(x + width/2, df['T_WER (%)'], width, label='Tiled-OCR (Proposed)', color='#2ecc71')
    
    plt.ylabel('Word Error Rate (%)')
    plt.title('Performance Comparison: Global vs Tiled OCR Strategies')
    plt.xticks(x, df['Variant'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/wer_comparison.pdf')
    print(">>> Đã lưu biểu đồ: results/wer_comparison.pdf")

    # Figure 2: Word Recovery (Recall)
    plt.figure(figsize=(10, 6))
    plt.plot(df['Variant'], df['Words_G'], marker='o', label='Global Word Count', color='#e74c3c', linestyle='--')
    plt.plot(df['Variant'], df['Words_T'], marker='s', label='Tiled Word Count', color='#2ecc71', linewidth=2)
    plt.axhline(y=480, color='blue', linestyle=':', label='Ground Truth (Avg)')
    
    plt.ylabel('Number of Extracted Words')
    plt.title('Word Recovery (Recall) across Different Conditions')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/word_recovery.pdf')
    print(">>> Đã lưu biểu đồ: results/word_recovery.pdf")

# --- 5. MAIN PIPELINE ---
def run_experiment():
    print("\n" + "="*50)
    print("VIETADMIN RESEARCH BENCHMARK - FINAL RUN")
    print("="*50)

    ocr_engine = VietAdminOCR(use_gpu=True)
    corrector = VietAdminCorrection()
    
    augmented_dir = "data/augmented"
    image_files = sorted([f for f in os.listdir(augmented_dir) if f.endswith(".jpg")])
    gt_count_ref = 480 # Giá trị tham chiếu cho biểu đồ

    results_data = []

    for img_name in image_files:
        print(f"\n[Processing]: {img_name}")
        img_path = os.path.join(augmented_dir, img_name)
        img = cv2.imread(img_path)
        
        gt_file = "data/4_1_gt.txt" if "4_1" in img_name else "data/4_2_gt.txt"
        gt_raw = load_text(gt_file)
        if not gt_raw: continue
        gt_norm = normalize_text(gt_raw)

        # Global Strategy
        raw_g_res = ocr_engine.extract_text(img_path)
        raw_g_text = " ".join([p['raw_text'] for p in raw_g_res])
        clean_g_text = corrector.process_large_text(raw_g_text)
        
        # Tiled Strategy
        raw_t_text = get_tiled_text(img, ocr_engine.reader)
        clean_t_text = corrector.process_large_text(raw_t_text)

        # Metrics
        w_g = wer(gt_norm, normalize_text(clean_g_text))
        w_t = wer(gt_norm, normalize_text(clean_t_text))
        
        results_data.append({
            "Variant": img_name.replace(".jpg", "").replace("4_1_", "").replace("4_2_", ""),
            "G_WER (%)": round(w_g * 100, 2),
            "T_WER (%)": round(w_t * 100, 2),
            "Words_G": len(clean_g_text.split()),
            "Words_T": len(clean_t_text.split())
        })

    df = pd.DataFrame(results_data)
    # Gom nhóm theo Variant để vẽ biểu đồ trung bình
    df_avg = df.groupby('Variant').mean().reset_index()
    
    # Xuất dữ liệu
    df.to_csv("results/final_strategy_comparison.csv", index=False)
    save_figures(df_avg)
    
    print("\n" + "="*50)
    print("THỰC NGHIỆM HOÀN TẤT")
    print("="*50)
    print(df_avg)

if __name__ == "__main__":
    run_experiment()
