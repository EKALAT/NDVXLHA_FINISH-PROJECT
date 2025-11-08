# Script: check_coconut_in_dataset.py
from pathlib import Path

train_labels = Path("train/labels")
valid_labels = Path("valid/labels")
test_labels = Path("test/labels")

# Đếm labels có class_id = 6 (Coconut)
def count_coconut_labels(labels_dir):
    count = 0
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                class_id = int(first_line.split()[0])
                if class_id == 6:
                    count += 1
    return count

train_count = count_coconut_labels(train_labels)
valid_count = count_coconut_labels(valid_labels)
test_count = count_coconut_labels(test_labels)

print("="*60)
print("📊 KIỂM TRA COCONUT TRONG DATASET")
print("="*60)
print(f"Train: {train_count} ảnh Coconut")
print(f"Valid: {valid_count} ảnh Coconut")
print(f"Test: {test_count} ảnh Coconut")
print(f"Tổng: {train_count + valid_count + test_count} ảnh Coconut")
print("="*60)