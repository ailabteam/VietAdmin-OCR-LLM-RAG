import cv2
import numpy as np
import os

def add_gaussian_noise(image):
    """Mô phỏng nhiễu hạt của máy scan cũ"""
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss * 20 # Độ mạnh của nhiễu
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_blur(image):
    """Mô phỏng ảnh bị mất nét (defocus)"""
    return cv2.GaussianBlur(image, (5, 5), 0)

def adjust_brightness_contrast(image, brightness=-50, contrast=30):
    """Mô phỏng văn bản bị thiếu sáng hoặc độ tương phản kém"""
    beta = brightness
    alpha = 1 + (contrast / 127)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def rotate_image(image, angle=1):
    """Mô phỏng văn bản bị đặt lệch trong máy scan (xoay nhẹ)"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

def generate_variants(img_path, output_dir):
    img = cv2.imread(img_path)
    if img is None: return
    
    base_name = os.path.basename(img_path).split('.')[0]
    
    # 1. Bản gốc (để đối soát)
    cv2.imwrite(f"{output_dir}/{base_name}_v0_original.jpg", img)
    
    # 2. Bản nhiễu hạt
    cv2.imwrite(f"{output_dir}/{base_name}_v1_noisy.jpg", add_gaussian_noise(img))
    
    # 3. Bản bị mờ
    cv2.imwrite(f"{output_dir}/{base_name}_v2_blur.jpg", add_blur(img))
    
    # 4. Bản bị tối và tương phản kém
    cv2.imwrite(f"{output_dir}/{base_name}_v3_dark.jpg", adjust_brightness_contrast(img))
    
    # 5. Bản bị xoay lệch
    cv2.imwrite(f"{output_dir}/{base_name}_v4_rotated.jpg", rotate_image(img))

if __name__ == "__main__":
    output_path = "data/augmented"
    os.makedirs(output_path, exist_ok=True)
    
    # Chạy cho 4.1 và 4.2
    generate_variants("data/4_1.jpg", output_path)
    generate_variants("data/4_2.jpg", output_path)
    print(f"Đã tạo xong các biến thể ảnh tại {output_path}")
