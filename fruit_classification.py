"""
Hệ thống nhận dạng loại trái cây qua hình ảnh sử dụng CNN/MobileNet
Dataset: Fruit Detection Dataset (YOLO format)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CẤU HÌNH
# ============================================================================

# Đường dẫn đến dataset
TRAIN_IMAGES_DIR = "train/images"
TRAIN_LABELS_DIR = "train/labels"
VALID_IMAGES_DIR = "valid/images"
VALID_LABELS_DIR = "valid/labels"
TEST_IMAGES_DIR = "test/images"
TEST_LABELS_DIR = "test/labels"

# Tên các lớp trái cây (theo thứ tự trong data.yaml)
CLASS_NAMES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon","coconut"]
NUM_CLASSES = len(CLASS_NAMES)

# Tham số cho mô hình
IMG_SIZE = 224  # Kích thước ảnh đầu vào (224x224 cho MobileNetV2)
BATCH_SIZE = 32
EPOCHS = 20

# Đường dẫn lưu mô hình
MODEL_SAVE_PATH = "fruit_model.h5"

# ============================================================================
# 1. LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU
# ============================================================================

def get_class_from_label_file(label_path):
    """
    Đọc file label YOLO format và trả về class_id đầu tiên (class chính trong ảnh)
    Format YOLO: class_id x_center y_center width height (normalized)
    """
    if not os.path.exists(label_path):
        return None
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                # Lấy class_id từ dòng đầu tiên
                first_line = lines[0].strip().split()
                if len(first_line) > 0:
                    class_id = int(first_line[0])
                    return class_id
    except:
        pass
    return None

def load_dataset(images_dir, labels_dir):
    """
    Load dataset từ thư mục images và labels
    Trả về: (images, labels) - numpy arrays
    """
    images = []
    labels = []
    
    print(f"Đang load dữ liệu từ {images_dir}...")
    
    # Lấy danh sách tất cả file ảnh
    image_files = list(Path(images_dir).glob("*.jpg"))
    
    for img_path in tqdm(image_files, desc="Loading images"):
        # Tìm file label tương ứng
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        
        # Lấy class_id từ label
        class_id = get_class_from_label_file(str(label_path))
        
        if class_id is not None and class_id < NUM_CLASSES:
            # Load và resize ảnh
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0  # Normalize về [0, 1]
                
                images.append(img)
                labels.append(class_id)
    
    return np.array(images), np.array(labels)

def prepare_datasets():
    """
    Load và chuẩn bị datasets cho train, validation và test
    """
    print("=" * 60)
    print("BƯỚC 1: LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    
    # Load dữ liệu
    X_train, y_train = load_dataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    X_valid, y_valid = load_dataset(VALID_IMAGES_DIR, VALID_LABELS_DIR)
    X_test, y_test = load_dataset(TEST_IMAGES_DIR, TEST_LABELS_DIR)
    
    # Chuyển đổi labels sang one-hot encoding
    y_train_categorical = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_valid_categorical = keras.utils.to_categorical(y_valid, NUM_CLASSES)
    y_test_categorical = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    print(f"\nKích thước dữ liệu:")
    print(f"  Train: {X_train.shape} - Labels: {y_train.shape}")
    print(f"  Valid: {X_valid.shape} - Labels: {y_valid.shape}")
    print(f"  Test:  {X_test.shape} - Labels: {y_test.shape}")
    
    # Hiển thị số lượng mẫu mỗi lớp
    print(f"\nPhân bố lớp trong tập train:")
    for i, class_name in enumerate(CLASS_NAMES):
        count = np.sum(y_train == i)
        print(f"  {class_name}: {count} mẫu")
    
    return X_train, y_train_categorical, X_valid, y_valid_categorical, X_test, y_test_categorical, y_test

# ============================================================================
# 2. XÂY DỰNG MÔ HÌNH
# ============================================================================

def build_cnn_model():
    """
    Xây dựng mô hình CNN từ đầu
    """
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Flatten và Dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def build_mobilenet_model():
    """
    Xây dựng mô hình sử dụng MobileNetV2 (pretrained) - Transfer Learning
    """
    # Load MobileNetV2 pretrained (không bao gồm top layer)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Đóng băng các layer của base model (optional - có thể fine-tune)
    base_model.trainable = False
    
    # Xây dựng model mới
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_model(use_mobilenet=True):
    """
    Tạo mô hình (CNN hoặc MobileNetV2)
    """
    print("=" * 60)
    print("BƯỚC 2: XÂY DỰNG MÔ HÌNH")
    print("=" * 60)
    
    if use_mobilenet:
        print("Sử dụng MobileNetV2 (Transfer Learning)...")
        model = build_mobilenet_model()
    else:
        print("Sử dụng CNN từ đầu...")
        model = build_cnn_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTóm tắt mô hình:")
    model.summary()
    
    return model

# ============================================================================
# 3. HUẤN LUYỆN MÔ HÌNH
# ============================================================================

def train_model(model, X_train, y_train, X_valid, y_valid):
    """
    Huấn luyện mô hình và hiển thị biểu đồ
    """
    print("=" * 60)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH")
    print("=" * 60)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Huấn luyện
    print(f"\nBắt đầu huấn luyện với {EPOCHS} epochs...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_valid, y_valid),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Vẽ biểu đồ Accuracy và Loss
    plot_training_history(history)
    
    return history

def plot_training_history(history):
    """
    Vẽ biểu đồ Accuracy và Loss
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Biểu đồ Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Biểu đồ Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nĐã lưu biểu đồ vào 'training_history.png'")

# ============================================================================
# 4. ĐÁNH GIÁ MÔ HÌNH
# ============================================================================

def evaluate_model(model, X_test, y_test, y_test_original):
    """
    Đánh giá mô hình trên tập test
    """
    print("=" * 60)
    print("BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)
    
    # Dự đoán
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Tính accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_original, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test_original, y_pred, target_names=CLASS_NAMES))
    
    return test_accuracy, y_pred

def plot_confusion_matrix(cm, class_names):
    """
    Vẽ Confusion Matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nĐã lưu Confusion Matrix vào 'confusion_matrix.png'")

# ============================================================================
# 5. DỰ ĐOÁN ẢNH MỚI
# ============================================================================

def predict_image(model, image_path):
    """
    Dự đoán loại trái cây từ một ảnh mới
    """
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
        print(f"  {class_name}: {predictions[0][i]:.4f} ({predictions[0][i]*100:.2f}%)")
    
    # Hiển thị ảnh
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {predicted_class_name} ({confidence*100:.2f}%)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return predicted_class_name, confidence

# ============================================================================
# HÀM CHÍNH
# ============================================================================

def main():
    """
    Hàm chính để chạy toàn bộ quy trình
    """
    print("\n" + "="*60)
    print("HỆ THỐNG NHẬN DẠNG LOẠI TRÁI CÂY QUA HÌNH ẢNH")
    print("="*60 + "\n")
    
    # 1. Load và tiền xử lý dữ liệu
    X_train, y_train, X_valid, y_valid, X_test, y_test, y_test_original = prepare_datasets()
    
    # 2. Tạo mô hình (sử dụng MobileNetV2 - có thể đổi thành False để dùng CNN)
    model = create_model(use_mobilenet=True)
    
    # 3. Huấn luyện mô hình
    history = train_model(model, X_train, y_train, X_valid, y_valid)
    
    # 4. Đánh giá mô hình
    test_accuracy, y_pred = evaluate_model(model, X_test, y_test, y_test_original)
    
    # 5. Lưu mô hình (đã được lưu trong callback, nhưng lưu lại để chắc chắn)
    model.save(MODEL_SAVE_PATH)
    print(f"\nĐã lưu mô hình vào: {MODEL_SAVE_PATH}")
    
    print("\n" + "="*60)
    print("HOÀN THÀNH!")
    print("="*60)
    print(f"\nĐể dự đoán ảnh mới, sử dụng:")
    print(f"  model = keras.models.load_model('{MODEL_SAVE_PATH}')")
    print(f"  predict_image(model, 'đường_dẫn_ảnh.jpg')")

if __name__ == "__main__":
    main()

