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
        # 1. Chuyển sang ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Phóng to ảnh (giữ tỉ lệ để không vỡ chữ)
        height, width = gray.shape[:2]
        img_resized = cv2.resize(gray, (int(width * 2.0), int(height * 2.0)), interpolation=cv2.INTER_CUBIC)
        
        # 3. Áp dụng Adaptive Thresholding (Nhị phân hóa thích nghi)
        # Giúp làm nổi bật chữ đen trên nền trắng, loại bỏ nhiễu nền mờ
        binary_img = cv2.adaptiveThreshold(
            img_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Khử nhiễu nhẹ (Denoising)
        denoised = cv2.fastNlMeansDenoising(binary_img, None, 10, 7, 21)
        
        return denoised
    

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
                paragraph=False,     # Thử tắt paragraph để nó đọc từng dòng nhỏ trước
                contrast_ths=0.1,    # Nhạy hơn với chữ mờ
                adjust_contrast=0.7, # Tự động tăng tương phản
                text_threshold=0.5,  # Giảm ngưỡng nhận diện để bắt chữ nhỏ
                width_ths=0.5,
                mag_ratio=1.5        # Phóng to nội bộ khi nhận diện
            )


            page_text = " ".join([res[1] for res in ocr_result])
            full_text_pages.append({
                'page': i + 1,
                'raw_text': page_text,
                'details': ocr_result
            })
            print(f"--- Đã xong trang {i+1} (Dùng chế độ Paragraph) ---")

        return full_text_pages
