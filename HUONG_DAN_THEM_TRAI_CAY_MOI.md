# 🍎 Hướng dẫn Train thêm Trái cây Mới

## 📋 Tổng quan

Hướng dẫn này sẽ giúp bạn thêm các loại trái cây mới vào hệ thống nhận dạng hiện tại. Hiện tại hệ thống đang nhận diện **6 loại trái cây**: Apple, Banana, Grape, Orange, Pineapple, Watermelon.

---

## 🎯 QUY TRÌNH THÊM TRÁI CÂY MỚI

### **BƯỚC 1: Chuẩn bị Dữ liệu Ảnh**

#### 1.1. Thu thập ảnh
- **Số lượng tối thiểu**: 100-200 ảnh cho mỗi loại trái cây mới
- **Số lượng khuyến nghị**: 300-500 ảnh để có độ chính xác cao
- **Định dạng**: `.jpg` hoặc `.png`
- **Chất lượng**: Ảnh rõ nét, đủ ánh sáng, góc chụp đa dạng

#### 1.2. Chia dataset
Chia ảnh thành 3 tập:
- **Train**: 70% (dùng để train model)
- **Valid**: 15% (dùng để đánh giá trong quá trình train)
- **Test**: 15% (dùng để đánh giá cuối cùng)

**Ví dụ với 200 ảnh Mango:**
- Train: 140 ảnh
- Valid: 30 ảnh
- Test: 30 ảnh

---

### **BƯỚC 2: Tạo File Labels (YOLO Format)**

#### 2.1. Hiểu về YOLO format
Mỗi ảnh cần có 1 file `.txt` cùng tên chứa thông tin bounding box.

**Format:** `class_id x_center y_center width height`

Trong đó:
- `class_id`: Số thứ tự của class (0, 1, 2, ...)
- `x_center, y_center`: Tọa độ trung tâm (normalized 0-1)
- `width, height`: Chiều rộng và cao (normalized 0-1)

#### 2.2. Xác định class_id mới
- **Class hiện tại:**
  - 0 = Apple
  - 1 = Banana
  - 2 = Grape
  - 3 = Orange
  - 4 = Pineapple
  - 5 = Watermelon

- **Class mới:**
  - 6 = Mango (ví dụ)
  - 7 = Kiwi (nếu thêm tiếp)
  - 8 = ... (tiếp tục)

#### 2.3. Tạo file label
**Ví dụ:** Ảnh `mango_001.jpg` → File `mango_001.txt`

Nội dung file `mango_001.txt`:
```
6 0.5 0.5 0.6 0.7
```

**Giải thích:**
- `6`: class_id của Mango
- `0.5 0.5`: Trung tâm ảnh ở giữa
- `0.6 0.7`: Bounding box chiếm 60% chiều rộng, 70% chiều cao

**Lưu ý:** Nếu ảnh có nhiều trái cây, mỗi dòng là 1 object:
```
6 0.3 0.4 0.2 0.3
6 0.7 0.6 0.25 0.35
```

#### 2.4. Công cụ tạo labels
Bạn có thể sử dụng:
- **LabelImg**: Tool GUI để vẽ bounding box và tự động tạo file label
- **Roboflow**: Platform online để annotate ảnh
- **CVAT**: Computer Vision Annotation Tool

---

### **BƯỚC 3: Copy Ảnh và Labels vào Dataset**

#### 3.1. Cấu trúc thư mục
```
Fruits-detection/
├── train/
│   ├── images/
│   │   ├── apple_001.jpg
│   │   ├── banana_001.jpg
│   │   └── mango_001.jpg    ← Ảnh mới
│   └── labels/
│       ├── apple_001.txt
│       ├── banana_001.txt
│       └── mango_001.txt    ← Label mới
├── valid/
│   ├── images/
│   │   └── mango_002.jpg    ← Ảnh validation
│   └── labels/
│       └── mango_002.txt    ← Label validation
└── test/
    ├── images/
    │   └── mango_003.jpg    ← Ảnh test
    └── labels/
        └── mango_003.txt    ← Label test
```

#### 3.2. Copy files
1. Copy ảnh Mango vào:
   - `train/images/` (70% ảnh)
   - `valid/images/` (15% ảnh)
   - `test/images/` (15% ảnh)

2. Copy labels tương ứng vào:
   - `train/labels/` (file .txt cùng tên)
   - `valid/labels/`
   - `test/labels/`

**Lưu ý:** Tên file ảnh và label phải giống nhau (chỉ khác extension):
- `mango_001.jpg` ↔ `mango_001.txt` ✅
- `mango_001.jpg` ↔ `mango_002.txt` ❌ SAI

---

### **BƯỚC 4: Cập nhật Code**

#### 4.1. File `fruit_classification.py`

**Tìm dòng 33:**
```python
# TRƯỚC:
CLASS_NAMES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]
```

**Sửa thành:**
```python
# SAU (thêm Mango):
CLASS_NAMES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon", "Mango"]
```

**Lưu ý:** 
- Thứ tự trong list phải khớp với class_id
- Mango = class_id 6 → đứng thứ 7 trong list (index 6)

#### 4.2. File `data.yaml`

**Tìm và sửa:**
```yaml
# TRƯỚC:
names:
- Apple
- Banana
- Grape
- Orange
- Pineapple
- Watermelon
nc: 6

# SAU:
names:
- Apple
- Banana
- Grape
- Orange
- Pineapple
- Watermelon
- Mango    # Thêm dòng này
nc: 7      # Đổi từ 6 thành 7
```

#### 4.3. File `predict_image.py`

**Tìm dòng 13:**
```python
# TRƯỚC:
CLASS_NAMES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]

# SAU:
CLASS_NAMES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon", "Mango"]
```

#### 4.4. File `fruit_detection_camera.py` (nếu có)

**Tìm dòng 14 và cập nhật tương tự:**
```python
CLASS_NAMES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon", "Mango"]
```

---

### **BƯỚC 5: Kiểm tra Labels**

#### 5.1. Kiểm tra class_id
Đảm bảo tất cả file label của trái cây mới dùng đúng class_id.

**Ví dụ với Mango:**
- Tất cả file label của Mango phải bắt đầu bằng `6`
- Nếu có file bắt đầu bằng số khác, cần sửa lại

#### 5.2. Script kiểm tra (tùy chọn)
Bạn có thể tạo script Python để kiểm tra:
```python
import os
from pathlib import Path

# Kiểm tra labels trong train/labels
label_dir = Path("train/labels")
for label_file in label_dir.glob("*.txt"):
    with open(label_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            class_id = int(first_line.split()[0])
            if class_id >= 6:  # Class mới
                print(f"{label_file.name}: class_id = {class_id}")
```

---

### **BƯỚC 6: Train Model Mới**

#### 6.1. Xóa model cũ (tùy chọn)
Model cũ (`fruit_model.h5`) không thể dùng vì số lớp đã thay đổi. Bạn có thể:
- Xóa file `fruit_model.h5` cũ
- Hoặc đổi tên để backup: `fruit_model_old_6classes.h5`

#### 6.2. Chạy training
```bash
cd Fruits-detection
python fruit_classification.py
```

#### 6.3. Quá trình training
- Model sẽ tự động đọc tất cả classes (bao gồm Mango)
- Tạo output layer mới với 7 neurons (thay vì 6)
- Train lại từ đầu với tất cả dữ liệu

**Thời gian:** 5-15 phút (tùy GPU và số lượng ảnh)

---

## 📝 VÍ DỤ CỤ THỂ: Thêm Mango

### Tóm tắt các bước:

1. **Thu thập 200 ảnh Mango**
   - Train: 140 ảnh
   - Valid: 30 ảnh
   - Test: 30 ảnh

2. **Tạo labels với class_id = 6**
   - Mỗi ảnh có 1 file `.txt` cùng tên
   - Nội dung: `6 0.5 0.5 0.6 0.7` (ví dụ)

3. **Copy vào dataset:**
   ```
   train/images/   → 140 ảnh Mango
   train/labels/   → 140 file .txt
   valid/images/   → 30 ảnh Mango
   valid/labels/   → 30 file .txt
   test/images/    → 30 ảnh Mango
   test/labels/    → 30 file .txt
   ```

4. **Cập nhật code:**
   - `fruit_classification.py`: Thêm "Mango" vào CLASS_NAMES
   - `data.yaml`: Thêm "Mango" và đổi nc: 7
   - `predict_image.py`: Thêm "Mango" vào CLASS_NAMES

5. **Train lại:**
   ```bash
   python fruit_classification.py
   ```

---

## 🔄 VÍ DỤ: Thêm Nhiều Trái cây Cùng Lúc

### Thêm Mango + Kiwi + Strawberry

#### 1. Class mapping:
- 0 = Apple
- 1 = Banana
- 2 = Grape
- 3 = Orange
- 4 = Pineapple
- 5 = Watermelon
- **6 = Mango** (mới)
- **7 = Kiwi** (mới)
- **8 = Strawberry** (mới)

#### 2. Cập nhật `CLASS_NAMES`:
```python
CLASS_NAMES = [
    "Apple", "Banana", "Grape", "Orange", 
    "Pineapple", "Watermelon", 
    "Mango", "Kiwi", "Strawberry"  # 3 class mới
]
```

#### 3. Cập nhật `data.yaml`:
```yaml
names:
- Apple
- Banana
- Grape
- Orange
- Pineapple
- Watermelon
- Mango
- Kiwi
- Strawberry
nc: 9
```

#### 4. Labels:
- Mango: class_id = 6
- Kiwi: class_id = 7
- Strawberry: class_id = 8

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. Model cũ không dùng được
- Model cũ (`fruit_model.h5`) có 6 output neurons
- Model mới cần 7+ output neurons
- **Phải train lại từ đầu**

### 2. Số lượng ảnh
- **Tối thiểu**: 100-200 ảnh/class
- **Khuyến nghị**: 300-500 ảnh/class
- **Càng nhiều càng tốt** để độ chính xác cao

### 3. Cân bằng dữ liệu
- Các class nên có số lượng ảnh tương đương
- Nếu 1 class có quá ít ảnh → model sẽ học kém class đó

### 4. Chất lượng ảnh
- Ảnh rõ nét, đủ ánh sáng
- Góc chụp đa dạng
- Nền đơn giản (dễ nhận diện hơn)

### 5. Kiểm tra labels
- Đảm bảo class_id đúng
- Đảm bảo tên file ảnh và label khớp nhau
- Format YOLO đúng: `class_id x y w h`

---

## 🛠️ TROUBLESHOOTING

### Lỗi: "Index out of range"
**Nguyên nhân:** Class_id trong label lớn hơn số lượng classes
**Giải pháp:** Kiểm tra lại class_id trong labels, đảm bảo < NUM_CLASSES

### Lỗi: "Model không nhận diện được class mới"
**Nguyên nhân:** Chưa cập nhật CLASS_NAMES trong `predict_image.py`
**Giải pháp:** Cập nhật CLASS_NAMES trong tất cả file

### Lỗi: "Accuracy thấp cho class mới"
**Nguyên nhân:** 
- Quá ít ảnh training
- Ảnh chất lượng kém
- Labels không chính xác
**Giải pháp:** 
- Tăng số lượng ảnh
- Kiểm tra lại labels
- Data augmentation sẽ giúp một phần

---

## ✅ CHECKLIST

Trước khi train, đảm bảo:

- [ ] Đã thu thập đủ ảnh (100-200+ ảnh/class)
- [ ] Đã chia train/valid/test (70/15/15)
- [ ] Đã tạo labels YOLO format với class_id đúng
- [ ] Đã copy ảnh vào `train/images/`, `valid/images/`, `test/images/`
- [ ] Đã copy labels vào `train/labels/`, `valid/labels/`, `test/labels/`
- [ ] Đã cập nhật `CLASS_NAMES` trong `fruit_classification.py`
- [ ] Đã cập nhật `CLASS_NAMES` trong `predict_image.py`
- [ ] Đã cập nhật `data.yaml` (names và nc)
- [ ] Đã kiểm tra class_id trong labels
- [ ] Đã backup model cũ (nếu cần)

---

## 📚 TÀI LIỆU THAM KHẢO

- **LabelImg**: https://github.com/tzutalin/labelImg
- **YOLO Format**: https://docs.ultralytics.com/datasets/
- **Data Augmentation**: Đã được tích hợp sẵn trong code

---

## 🎯 KẾT QUẢ SAU KHI TRAIN

Sau khi train xong, bạn sẽ có:
- ✅ Model mới: `fruit_model.h5` (với số lớp mới)
- ✅ Biểu đồ: `training_history.png`
- ✅ Confusion Matrix: `confusion_matrix.png`
- ✅ Model có thể nhận diện thêm trái cây mới

**Test model:**
```bash
python predict_image.py test/images/mango_001.jpg
```

Kết quả sẽ hiển thị: `Predicted: Mango (95.23%)`

---

**Chúc bạn thành công! 🎉**

