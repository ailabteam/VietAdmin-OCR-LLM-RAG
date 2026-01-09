import easyocr
import cv2
import numpy as np
import fitz 
import os

class VietAdminOCR:
    def __init__(self, use_gpu=True):
        print("Đang khởi tạo EasyOCR Nâng Cao...")
        self.reader = easyocr.Reader(['vi'], gpu=use_gpu)

    def preprocess_image(self, img):
        """Tiền xử lý ảnh để tăng khả năng nhận diện"""
        # 1. Chuyển sang ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Phóng to ảnh (Upscaling) - Cực kỳ quan trọng nếu chữ nhỏ
        # Phóng to gấp 1.5 hoặc 2 lần
        height, width = gray.shape[:2]
        img_resized = cv2.resize(gray, (int(width * 1.5), int(height * 1.5)), interpolation=cv2.INTER_CUBIC)
        
        # 3. Tăng độ tương phản (Optional - có thể thử nếu ảnh mờ)
        # img_enhanced = cv2.detailEnhance(img_resized, sigma_s=10, sigma_r=0.15)
        
        return img_resized

    def pdf_to_images(self, pdf_path):
        print(f"Đang xử lý PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            # Tăng DPI khi render PDF (zoom=3 ~ 300 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            images.append(img_bgr)
        return images

    def extract_text(self, input_path):
        ext = os.path.splitext(input_path)[1].lower()
        img_list = self.pdf_to_images(input_path) if ext == '.pdf' else [cv2.imread(input_path)]

        full_text_pages = []
        for i, img in enumerate(img_list):
            if img is None: continue
            
            # Tiền xử lý ảnh trước khi đưa vào OCR
            processed_img = self.preprocess_image(img)
            
            # Tinh chỉnh tham số EasyOCR:
            # - paragraph=True: Giúp gom các dòng lại thành đoạn, tránh sót và giữ ngữ cảnh cho ProtonX
            # - width_ths, height_ths: Giảm ngưỡng để bắt được các từ đứng gần nhau hoặc xa nhau
            # - add_margin: Thêm lề xung quanh từ để nhận diện dấu tốt hơn
            ocr_result = self.reader.readtext(
                processed_img, 
                detail=1, 
                paragraph=True, # Gom nhóm văn bản
                width_ths=0.7,  # Tăng khả năng ghép nối từ theo chiều ngang
                add_margin=0.1, # Thêm khoảng trống bao quanh chữ
                min_size=10     # Không bỏ qua chữ nhỏ (tính theo pixel)
            )
            
            page_text = " ".join([res[1] for res in ocr_result])
            full_text_pages.append({
                'page': i + 1,
                'raw_text': page_text,
                'details': ocr_result 
            })
            print(f"--- Đã xong trang {i+1} (Dùng chế độ Paragraph) ---")
            
        return full_text_pages
