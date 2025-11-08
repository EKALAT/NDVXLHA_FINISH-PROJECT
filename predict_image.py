"""
Script để dự đoán loại trái cây từ một ảnh mới
Sử dụng mô hình đã được train (fruit_model.h5)
"""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras

# Tên các lớp trái cây
CLASS_NAMES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon","coconut"]
IMG_SIZE = 224
MODEL_PATH = "fruit_model.h5"

def predict_image(image_path, model_path=MODEL_PATH):
    """
    Dự đoán loại trái cây từ một ảnh mới
    
    Args:
        image_path: Đường dẫn đến ảnh cần dự đoán
        model_path: Đường dẫn đến file mô hình đã train
    """
    # Load mô hình
    print(f"Đang tải mô hình từ {model_path}...")
    try:
        model = keras.models.load_model(model_path)
        print("Đã tải mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        print("Vui lòng đảm bảo mô hình đã được train và lưu tại:", model_path)
        return None
    
    # Load và tiền xử lý ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = img_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    
    # Dự đoán
    print("Đang dự đoán...")
    predictions = model.predict(img_array, verbose=0)
    predicted_class_id = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_id]
    predicted_class_name = CLASS_NAMES[predicted_class_id]
    
    # Hiển thị kết quả
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ DỰ ĐOÁN")
    print(f"{'='*60}")
    print(f"Ảnh: {image_path}")
    print(f"Loại trái cây dự đoán: {predicted_class_name}")
    print(f"Độ tin cậy: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"\nXác suất tất cả các lớp:")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = predictions[0][i]
        bar = "█" * int(prob * 50)  # Thanh bar đơn giản
        print(f"  {class_name:12s}: {prob:.4f} ({prob*100:5.2f}%) {bar}")
    
    # Hiển thị ảnh
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {predicted_class_name} ({confidence*100:.2f}%)", 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return predicted_class_name, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách sử dụng: python predict_image.py <đường_dẫn_ảnh>")
        print("Ví dụ: python predict_image.py test/images/example.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_image(image_path)

