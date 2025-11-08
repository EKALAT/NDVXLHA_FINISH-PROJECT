# 📖 Giải thích Hệ thống Nhận dạng Trái cây

## 🎯 Tổng quan
Hệ thống sử dụng Deep Learning (CNN/MobileNetV2) để phân loại 6 loại trái cây từ hình ảnh: **Apple, Banana, Grape, Orange, Pineapple, Watermelon**.

---

## 🔄 QUY TRÌNH HOẠT ĐỘNG

### **BƯỚC 1: Load và Tiền xử lý Dữ liệu**
- Đọc ảnh từ thư mục `train/valid/test`
- Đọc file label (định dạng YOLO) để lấy class_id (0-5)
- Resize ảnh về 224x224 pixels
- Chuẩn hóa pixel từ [0-255] → [0-1]
- Chuyển labels sang one-hot encoding (0 → [1,0,0,0,0,0])

**Kết quả:** Dataset sẵn sàng cho training với ~7108 ảnh train, ~914 ảnh validation, ~457 ảnh test

---

### **BƯỚC 2: Xây dựng Mô hình**
- **MobileNetV2 (mặc định):** Sử dụng Transfer Learning từ ImageNet
  - Base model trích xuất đặc trưng
  - Lớp Dense(128) học pattern
  - Lớp Dense(6) output xác suất cho 6 lớp
- **CNN từ đầu:** Xây dựng mạng từ đầu với 4 lớp Conv2D + Dense

**Kết quả:** Mô hình đã được compile với optimizer Adam, loss Categorical Crossentropy

---

### **BƯỚC 3: Huấn luyện Mô hình**
- **Data Augmentation:** Tăng cường dữ liệu bằng cách xoay, dịch chuyển, lật ngang, zoom ảnh
- **Training:** 20 epochs, batch size 32
- **Callbacks:** 
  - ModelCheckpoint: Tự động lưu mô hình tốt nhất
  - EarlyStopping: Dừng sớm nếu không cải thiện
- **Theo dõi:** Vẽ biểu đồ Accuracy và Loss

**Kết quả:** File `fruit_model.h5` chứa mô hình đã train, file `training_history.png` chứa biểu đồ

---

### **BƯỚC 4: Đánh giá Mô hình**
- Tính Test Accuracy trên tập test
- Tạo Confusion Matrix để xem dự đoán đúng/sai cho từng lớp
- Báo cáo Precision, Recall, F1-score

**Kết quả:** File `confusion_matrix.png` và báo cáo đánh giá chi tiết

---

### **BƯỚC 5: Dự đoán Ảnh Mới**
- Load ảnh mới và tiền xử lý (resize, normalize)
- Đưa vào mô hình để dự đoán
- Mô hình trả về xác suất cho 6 lớp
- Lấy lớp có xác suất cao nhất làm kết quả

**Kết quả:** Tên trái cây dự đoán + độ tin cậy (%)

---

## 📚 NHIỆM VỤ CÁC THƯ VIỆN

### **1. TensorFlow**
- **Nhiệm vụ:** Framework chính cho deep learning
- **Chức năng:** Xây dựng, train và chạy mô hình neural network, tối ưu hóa với GPU

### **2. Keras (tensorflow.keras)**
- **Nhiệm vụ:** API đơn giản hóa TensorFlow
- **Chức năng:** Cung cấp các hàm tiện ích (to_categorical, ImageDataGenerator), quản lý callbacks, lưu/tải mô hình

### **3. OpenCV (cv2)**
- **Nhiệm vụ:** Xử lý hình ảnh
- **Chức năng:** Đọc ảnh (imread), resize, chuyển đổi không gian màu (BGR ↔ RGB)

### **4. NumPy (np)**
- **Nhiệm vụ:** Tính toán với mảng đa chiều
- **Chức năng:** Lưu trữ dữ liệu (ảnh, labels), các phép toán (array, argmax, sum)

### **5. Matplotlib (plt)**
- **Nhiệm vụ:** Vẽ biểu đồ và hiển thị ảnh
- **Chức năng:** Vẽ biểu đồ Accuracy/Loss (plot), hiển thị ảnh (imshow), lưu biểu đồ (savefig)

### **6. Scikit-learn (sklearn)**
- **Nhiệm vụ:** Công cụ đánh giá mô hình
- **Chức năng:** Tạo confusion matrix, báo cáo classification (precision, recall, F1-score)

### **7. Seaborn (sns)**
- **Nhiệm vụ:** Visualization đẹp hơn matplotlib
- **Chức năng:** Vẽ heatmap cho confusion matrix

### **8. tqdm**
- **Nhiệm vụ:** Hiển thị thanh tiến trình
- **Chức năng:** Hiển thị % hoàn thành khi xử lý nhiều file

### **9. Pillow (PIL)**
- **Nhiệm vụ:** Xử lý ảnh (hỗ trợ Keras)
- **Chức năng:** Load và chuyển đổi ảnh (load_img, img_to_array)

### **10. Pathlib**
- **Nhiệm vụ:** Xử lý đường dẫn file/folder
- **Chức năng:** Tìm file theo pattern (glob), nối đường dẫn

### **11. os**
- **Nhiệm vụ:** Tương tác với hệ điều hành
- **Chức năng:** Kiểm tra file tồn tại (path.exists)

### **12. sys**
- **Nhiệm vụ:** Tương tác với Python interpreter
- **Chức năng:** Lấy tham số dòng lệnh (argv), thoát chương trình (exit)

---

## 📊 SƠ ĐỒ QUY TRÌNH

```
Dataset (8479 ảnh)
    ↓
Chia: Train | Valid | Test
    ↓
Tiền xử lý: Resize + Normalize
    ↓
Xây dựng Model (MobileNetV2/CNN)
    ↓
Training với Data Augmentation
    ↓
Lưu Model (fruit_model.h5)
    ↓
Đánh giá: Accuracy + Confusion Matrix
    ↓
Dự đoán ảnh mới
```

---

## 💡 CÁC KHÁI NIỆM QUAN TRỌNG

- **Transfer Learning:** Sử dụng mô hình đã train trên dataset lớn (ImageNet), chỉ train lại layers cuối
- **Data Augmentation:** Tăng số lượng dữ liệu bằng cách biến đổi ảnh (xoay, lật, zoom)
- **One-Hot Encoding:** Chuyển label số thành vector binary (0 → [1,0,0,0,0,0])
- **Normalization:** Chia pixel cho 255 để đưa giá trị về [0-1]
- **Epoch:** 1 lần duyệt qua toàn bộ dataset
- **Batch Size:** Số ảnh xử lý cùng lúc (32 ảnh/batch)
- **Confusion Matrix:** Ma trận hiển thị số lượng dự đoán đúng/sai cho mỗi lớp

---

## ✅ TÓM TẮT

**Input:** Ảnh trái cây  
**Process:** MobileNetV2 → Dense layers → Softmax  
**Output:** Tên trái cây + độ tin cậy  
**Accuracy:** ~90-95%  
**Thời gian train:** 5-15 phút (tùy GPU)  
**Thời gian dự đoán:** <100ms/ảnh

