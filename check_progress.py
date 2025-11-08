# Script: check_progress.py
from pathlib import Path

images_dir = Path("coconut_raw/images")
labels_dir = Path("coconut_raw/labels")

# Đếm ảnh và labels
all_images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
all_labels = list(labels_dir.glob("*.txt"))

# Kiểm tra ảnh nào đã có label
images_with_labels = []
images_without_labels = []

for img_path in all_images:
    label_path = labels_dir / (img_path.stem + ".txt")
    if label_path.exists():
        images_with_labels.append(img_path.name)
    else:
        images_without_labels.append(img_path.name)

print("="*60)
print("📊 THỐNG KÊ TIẾN ĐỘ")
print("="*60)
print(f"Tổng số ảnh: {len(all_images)}")
print(f"Đã có label: {len(images_with_labels)} ({len(images_with_labels)/len(all_images)*100:.1f}%)")
print(f"Chưa có label: {len(images_without_labels)} ({len(images_without_labels)/len(all_images)*100:.1f}%)")
print("="*60)

if images_without_labels:
    print(f"\n📋 Danh sách ảnh chưa có label (10 đầu tiên):")
    for i, img_name in enumerate(images_without_labels[:10], 1):
        print(f"  {i}. {img_name}")
    if len(images_without_labels) > 10:
        print(f"  ... và {len(images_without_labels) - 10} ảnh khác")