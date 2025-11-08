"""
Hệ thống nhận diện trái cây qua camera thời gian thực
Sử dụng mô hình YOLOv8 đã được train sẵn
"""

import cv2
from ultralytics import YOLO
import os

# Đường dẫn đến file mô hình đã train
MODEL_PATH = "yolov8s.pt"

# Tên các loại trái cây (theo thứ tự trong data.yaml)
CLASS_NAMES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon","coconut"]

def main():
    """
    Hàm chính để chạy hệ thống nhận diện trái cây qua camera
    """
    print("Đang khởi tạo hệ thống nhận diện trái cây...")
    
    # Kiểm tra xem file mô hình có tồn tại không
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}")
        return
    
    # Load mô hình YOLOv8 đã được train sẵn
    print("Đang tải mô hình YOLOv8...")
    model = YOLO(MODEL_PATH)
    print("Đã tải mô hình thành công!")
    
    # Mở camera máy tính (0 là camera mặc định)
    print("Đang mở camera...")
    cap = cv2.VideoCapture(0)
    
    # Kiểm tra xem camera có mở được không
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera!")
        return
    
    print("Camera đã sẵn sàng. Nhấn phím 'q' để thoát.")
    print("-" * 50)
    
    # Vòng lặp chính để xử lý từng khung hình
    while True:
        # Đọc khung hình từ camera
        ret, frame = cap.read()
        
        # Kiểm tra xem có đọc được khung hình không
        if not ret:
            print("Lỗi: Không thể đọc khung hình từ camera!")
            break
        
        # Dự đoán các loại trái cây trong khung hình hiện tại
        results = model(frame, verbose=False)
        
        # Tạo bản sao của khung hình để vẽ lên
        annotated_frame = frame.copy()
        
        # Lấy thông tin chi tiết về các đối tượng được phát hiện
        for result in results:
            boxes = result.boxes
            
            # Duyệt qua từng đối tượng được phát hiện
            for box in boxes:
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Lấy độ tin cậy (confidence)
                confidence = float(box.conf[0].cpu().numpy())
                
                # Lấy class ID và tên loại trái cây
                class_id = int(box.cls[0].cpu().numpy())
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                
                # Vẽ bounding box (màu xanh lá, độ dày 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Tạo nhãn hiển thị (tên trái cây + độ tin cậy)
                label = f"{class_name}: {confidence:.2f}"
                
                # Tính toán kích thước text để vẽ nền
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Vẽ nền cho text (màu đen) để text dễ đọc hơn
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    (0, 0, 0),
                    -1
                )
                
                # Vẽ text (tên trái cây và độ tin cậy) màu xanh lá
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        
        # Hiển thị khung hình đã được xử lý
        cv2.imshow("Nhận diện trái cây - Nhấn 'q' để thoát", annotated_frame)
        
        # Kiểm tra xem người dùng có nhấn phím 'q' không
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nĐang thoát chương trình...")
            break
    
    # Giải phóng camera và đóng tất cả cửa sổ
    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng camera và thoát chương trình.")

if __name__ == "__main__":
    main()

