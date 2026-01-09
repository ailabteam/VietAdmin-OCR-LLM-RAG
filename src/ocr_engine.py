import easyocr
import cv2
import numpy as np
import fitz  # PyMuPDF
import os

class VietAdminOCR:
    def __init__(self, use_gpu=True):
        print("Đang khởi tạo EasyOCR với ngôn ngữ Tiếng Việt...")
        # Sử dụng GPU để đạt hiệu năng tốt nhất trên 4090
        self.reader = easyocr.Reader(['vi'], gpu=use_gpu)

    def pdf_to_images(self, pdf_path):
        """Chuyển đổi các trang PDF thành danh sách ảnh sử dụng PyMuPDF"""
        print(f"Đang xử lý PDF bằng PyMuPDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            # Tăng độ phân giải để OCR chính xác hơn (zoom=2 tương đương 150-200 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            # Chuyển từ RGB sang BGR cho OpenCV
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            images.append(img_bgr)
        return images

    def extract_text(self, input_path):
        """Hàm chính để trích xuất text từ file ảnh hoặc PDF"""
        ext = os.path.splitext(input_path)[1].lower()
        
        if ext == '.pdf':
            img_list = self.pdf_to_images(input_path)
        else:
            img = cv2.imread(input_path)
            if img is None:
                print(f"Lỗi: Không thể đọc file ảnh {input_path}")
                return []
            img_list = [img]

        print(f"Bắt đầu OCR cho {len(img_list)} trang...")
        full_text_pages = []
        
        for i, img in enumerate(img_list):
            # detail=1 để lấy tọa độ và confidence phục vụ phân tích sau này
            ocr_result = self.reader.readtext(img, detail=1)
            
            # Ghép các đoạn text lại
            page_text = " ".join([res[1] for res in ocr_result])
            full_text_pages.append({
                'page': i + 1,
                'raw_text': page_text,
                'details': ocr_result 
            })
            print(f"--- Đã xong trang {i+1} ---")
            
        return full_text_pages

if __name__ == "__main__":
    engine = VietAdminOCR()
    
    # Thử nghiệm với file ảnh hoặc pdf trong data/
    test_file = "data/4_1.jpg" # Hoặc "data/1.pdf"
    
    if os.path.exists(test_file):
        output = engine.extract_text(test_file)
        if output:
            print("\n[KẾT QUẢ OCR THÔ]:")
            print(output[0]['raw_text'])
    else:
        print(f"Vui lòng kiểm tra lại đường dẫn file: {test_file}")
