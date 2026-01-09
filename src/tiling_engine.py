import cv2
import numpy as np

class TilingOCR:
    def __init__(self, ocr_engine):
        self.ocr_engine = ocr_engine

    def get_tiles(self, img, rows=2, cols=2, overlap=100):
        """Chia ảnh thành các ô nhỏ có phần chồng lấp"""
        h, w = img.shape[:2]
        tile_h = h // rows
        tile_w = w // cols
        
        tiles = []
        for r in range(rows):
            for c in range(cols):
                y1 = max(0, r * tile_h - overlap)
                y2 = min(h, (r + 1) * tile_h + overlap)
                x1 = max(0, c * tile_w - overlap)
                x2 = min(w, (c + 1) * tile_w + overlap)
                
                tile = img[y1:y2, x1:x2]
                tiles.append(tile)
        return tiles

    def process_with_tiling(self, img_path):
        img = cv2.imread(img_path)
        # Chia làm 4 mảnh (2x2)
        tiles = self.get_tiles(img, rows=2, cols=2)
        
        all_text = []
        print(f"Đang chạy Tiling OCR trên {len(tiles)} vùng...")
        
        for i, tile in enumerate(tiles):
            # Dùng lại engine OCR đã có nhưng trên ảnh mảnh nhỏ (zoom tốt hơn)
            # Chúng ta sẽ làm nét từng mảnh
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            tile_sharp = cv2.filter2D(tile, -1, kernel)
            
            # Giả định extract_text của bạn nhận được ảnh numpy
            result = self.ocr_engine.reader.readtext(tile_sharp, detail=0)
            all_text.extend(result)
            print(f"Xong vùng {i+1}")

        # Loại bỏ các từ trùng lặp do phần overlap (đơn giản hóa bằng set hoặc giữ nguyên vì ProtonX có thể xử lý)
        return " ".join(all_text)
