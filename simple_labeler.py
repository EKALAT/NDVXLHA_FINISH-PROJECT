"""
Script đơn giản để tạo labels YOLO format cho Coconut
Tương thích Python 3.12
Sử dụng OpenCV để vẽ bounding box
"""
import cv2
import os
from pathlib import Path

class SimpleLabeler:
    def __init__(self, images_dir, labels_dir, class_id=6):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_id = class_id
        self.images = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
        
        # Tự động tìm ảnh chưa có label đầu tiên
        self.current_idx = self.find_first_unlabeled_image()
        
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_image = None
        self.current_image_path = None
        self.display_image = None
        
        # Tạo thư mục labels nếu chưa có
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def find_first_unlabeled_image(self):
        """
        Tìm index của ảnh chưa có label đầu tiên
        """
        for idx, img_path in enumerate(self.images):
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"📍 Tìm thấy ảnh chưa có label đầu tiên: {img_path.name} (ảnh {idx + 1}/{len(self.images)})")
                return idx
        # Nếu tất cả đã có label
        print("✅ Tất cả ảnh đã có label!")
        return 0
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self.update_display()
    
    def update_display(self):
        if self.display_image is not None:
            img_copy = self.display_image.copy()
            if self.start_point and self.end_point:
                cv2.rectangle(img_copy, self.start_point, self.end_point, (0, 255, 0), 2)
            cv2.imshow('Labeler - Coconut (Class ID: 6)', img_copy)
    
    def save_label(self):
        if self.start_point and self.end_point:
            h, w = self.current_image.shape[:2]
            
            # Tính toán tọa độ YOLO format
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # Đảm bảo x1 < x2 và y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Kiểm tra hợp lệ
            if x2 - x1 < 10 or y2 - y1 < 10:
                print("⚠️  Bounding box quá nhỏ! Vui lòng vẽ lại.")
                return False
            
            # Normalize về [0, 1]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # Đảm bảo trong khoảng [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Lưu file label
            label_path = self.labels_dir / (self.current_image_path.stem + ".txt")
            with open(label_path, 'w') as f:
                f.write(f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            print(f"✅ Đã lưu label: {label_path.name}")
            return True
        return False
    
    def load_image(self, idx):
        if 0 <= idx < len(self.images):
            self.current_image_path = self.images[idx]
            self.current_image = cv2.imread(str(self.current_image_path))
            if self.current_image is not None:
                # Resize nếu ảnh quá lớn (giữ tỷ lệ)
                h, w = self.current_image.shape[:2]
                max_size = 1200
                if w > max_size or h > max_size:
                    scale = max_size / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    self.current_image = cv2.resize(self.current_image, (new_w, new_h))
                
                self.display_image = self.current_image.copy()
                return True
        return False
    
    def show_info(self):
        label_path = self.labels_dir / (self.current_image_path.stem + ".txt")
        has_label = label_path.exists()
        status = "✅ Đã có label" if has_label else "❌ Chưa có label"
        
        info_text = f"Ảnh {self.current_idx + 1}/{len(self.images)}: {self.current_image_path.name} | {status}"
        print(f"\n{'='*60}")
        print(info_text)
        print(f"{'='*60}")
    
    def run(self):
        if not self.images:
            print("❌ Không tìm thấy ảnh nào trong thư mục!")
            print(f"   Kiểm tra thư mục: {self.images_dir}")
            return
        
        cv2.namedWindow('Labeler - Coconut (Class ID: 6)')
        cv2.setMouseCallback('Labeler - Coconut (Class ID: 6)', self.mouse_callback)
        
        print("\n" + "="*60)
        print("🍥 SIMPLE LABELER - COCONUT (Class ID: 6)")
        print("="*60)
        print("\n📋 HƯỚNG DẪN:")
        print("  1. Click và kéo chuột để vẽ bounding box quanh quả Coconut")
        print("  2. Nhấn 'S' hoặc 'SPACE' để lưu label")
        print("  3. Nhấn 'N' hoặc '→' để ảnh tiếp theo")
        print("  4. Nhấn 'P' hoặc '←' để ảnh trước")
        print("  5. Nhấn 'D' để xóa label hiện tại")
        print("  6. Nhấn 'Q' hoặc 'ESC' để thoát")
        print("="*60 + "\n")
        
        if not self.load_image(self.current_idx):
            print("❌ Không thể load ảnh!")
            return
        
        self.show_info()
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q hoặc ESC
                print("\n👋 Thoát chương trình...")
                break
            elif key == ord('s') or key == ord(' '):  # S hoặc SPACE
                if self.save_label():
                    self.show_info()
                else:
                    print("⚠️  Vui lòng vẽ bounding box trước! (Click và kéo chuột)")
            elif key == ord('n') or key == 83:  # N hoặc mũi tên phải
                if self.save_label():
                    print("💾 Đã tự động lưu label trước khi chuyển ảnh")
                # Tìm ảnh chưa có label tiếp theo
                self.current_idx = self.find_next_unlabeled_image()
                if self.load_image(self.current_idx):
                    self.show_info()
                    self.update_display()
                    self.start_point = None
                    self.end_point = None
            elif key == ord('p') or key == 81:  # P hoặc mũi tên trái
                if self.save_label():
                    print("💾 Đã tự động lưu label trước khi chuyển ảnh")
                # Tìm ảnh chưa có label trước đó
                self.current_idx = self.find_prev_unlabeled_image()
                if self.load_image(self.current_idx):
                    self.show_info()
                    self.update_display()
                    self.start_point = None
                    self.end_point = None
            elif key == ord('d'):  # D để xóa label
                label_path = self.labels_dir / (self.current_image_path.stem + ".txt")
                if label_path.exists():
                    label_path.unlink()
                    print(f"🗑️  Đã xóa label: {label_path.name}")
                    self.show_info()
        
        cv2.destroyAllWindows()
        
        # Thống kê
        total_labels = len(list(self.labels_dir.glob("*.txt")))
        print(f"\n{'='*60}")
        print(f"📊 THỐNG KÊ:")
        print(f"   Tổng số ảnh: {len(self.images)}")
        print(f"   Số ảnh đã có label: {total_labels}")
        print(f"   Số ảnh chưa có label: {len(self.images) - total_labels}")
        print(f"{'='*60}\n")
    
    def find_next_unlabeled_image(self):
        """
        Tìm ảnh chưa có label tiếp theo
        """
        start_idx = (self.current_idx + 1) % len(self.images)
        for i in range(len(self.images)):
            idx = (start_idx + i) % len(self.images)
            label_path = self.labels_dir / (self.images[idx].stem + ".txt")
            if not label_path.exists():
                return idx
        return self.current_idx  # Không tìm thấy, giữ nguyên
    
    def find_prev_unlabeled_image(self):
        """
        Tìm ảnh chưa có label trước đó
        """
        start_idx = (self.current_idx - 1) % len(self.images)
        for i in range(len(self.images)):
            idx = (start_idx - i) % len(self.images)
            label_path = self.labels_dir / (self.images[idx].stem + ".txt")
            if not label_path.exists():
                return idx
        return self.current_idx  # Không tìm thấy, giữ nguyên

if __name__ == "__main__":
    # Cấu hình
    IMAGES_DIR = "coconut_raw/images"
    LABELS_DIR = "coconut_raw/labels"
    CLASS_ID = 6  # Class ID cho Coconut
    
    print("🚀 Đang khởi động Simple Labeler...")
    labeler = SimpleLabeler(IMAGES_DIR, LABELS_DIR, CLASS_ID)
    labeler.run()