# Script: split_and_copy_coconut.py
import os
import shutil
from pathlib import Path
import random

# Đường dẫn
coconut_images_dir = Path("coconut_raw/images")
coconut_labels_dir = Path("coconut_raw/labels")

train_images_dir = Path("train/images")
train_labels_dir = Path("train/labels")
valid_images_dir = Path("valid/images")
valid_labels_dir = Path("valid/labels")
test_images_dir = Path("test/images")
test_labels_dir = Path("test/labels")

# Tạo thư mục đích nếu chưa tồn tại
train_images_dir.mkdir(parents=True, exist_ok=True)
train_labels_dir.mkdir(parents=True, exist_ok=True)
valid_images_dir.mkdir(parents=True, exist_ok=True)
valid_labels_dir.mkdir(parents=True, exist_ok=True)
test_images_dir.mkdir(parents=True, exist_ok=True)
test_labels_dir.mkdir(parents=True, exist_ok=True)

# Lấy danh sách ảnh đã có label
all_images = []
for img_path in sorted(coconut_images_dir.glob("*.jpg")) + sorted(coconut_images_dir.glob("*.png")):
    label_path = coconut_labels_dir / (img_path.stem + ".txt")
    if label_path.exists():  # Chỉ lấy ảnh đã có label
        all_images.append(img_path)

random.shuffle(all_images)  # Xáo trộn ngẫu nhiên

# Tính số lượng
total = len(all_images)
train_count = int(total * 0.7)
valid_count = int(total * 0.15)
test_count = total - train_count - valid_count

print(f"Tổng số ảnh Coconut đã có label: {total}")
print(f"Train: {train_count}, Valid: {valid_count}, Test: {test_count}")

# Chia và copy
def copy_files(image_list, img_dest, label_dest):
    copied = 0
    for img_path in image_list:
        try:
            # Copy ảnh
            shutil.copy2(img_path, img_dest / img_path.name)
            
            # Copy label
            label_path = coconut_labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, label_dest / label_path.name)
                copied += 1
            else:
                print(f"⚠️  Warning: Không tìm thấy label cho {img_path.name}")
        except Exception as e:
            print(f"❌ Lỗi khi copy {img_path.name}: {e}")
    return copied

# Copy train
print("\nĐang copy train...")
train_copied = copy_files(all_images[:train_count], train_images_dir, train_labels_dir)
print(f"✅ Đã copy {train_copied}/{train_count} ảnh và labels vào train")

# Copy valid
print("Đang copy valid...")
valid_copied = copy_files(all_images[train_count:train_count+valid_count], valid_images_dir, valid_labels_dir)
print(f"✅ Đã copy {valid_copied}/{valid_count} ảnh và labels vào valid")

# Copy test
print("Đang copy test...")
test_copied = copy_files(all_images[train_count+valid_count:], test_images_dir, test_labels_dir)
print(f"✅ Đã copy {test_copied}/{test_count} ảnh và labels vào test")

print("\n" + "="*60)
print("✅ HOÀN THÀNH!")
print(f"Tổng cộng: {train_copied + valid_copied + test_copied} ảnh đã được copy")
print("="*60)