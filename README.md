# ĐỀ XUẤT ĐỀ TÀI: HỆ THỐNG PHÁT HIỆN HỎA HOẠN SỚM THỜI GIAN THỰC SỬ DỤNG YOLOv11

**Lĩnh vực:** Computer Vision / Deep Learning

---

## 1. Đặt vấn đề & Lý do chọn đề tài (Problem Statement)

### 1.1. Hạn chế kỹ thuật của giải pháp hiện hành
Các hệ thống báo cháy truyền thống (cảm biến ion hóa, quang điện hoặc nhiệt) tồn tại những nhược điểm cố hữu về mặt vật lý:
* **Độ trễ phát hiện (Detection Latency):** Cần sự tiếp xúc vật lý trực tiếp với nồng độ khói hoặc ngưỡng nhiệt độ cao để kích hoạt. Khi cảm biến báo động, đám cháy thường đã lan rộng.
* **Vùng mù không gian (Spatial Limitation):** Không hiệu quả trong không gian mở (bãi tập kết, nhà xưởng trần cao) do sự phân tán của khói vào không khí.

### 1.2. Giải pháp kỹ thuật đề xuất
Xây dựng hệ thống **Thị giác máy tính (Computer Vision)** sử dụng kiến trúc mạng nơ-ron tích chập (CNN) hiện đại nhất là **YOLOv11** để phân tích luồng video từ camera giám sát (CCTV).
* **Cơ chế:** Chuyển đổi bài toán phát hiện cháy từ "cảm biến tiếp xúc" sang "nhận dạng mẫu hình ảnh" (Pattern Recognition).
* **Ưu điểm kỹ thuật:** Cho phép phát hiện đám cháy ngay từ giai đoạn khởi phát (Early Stage) thông qua đặc trưng hình ảnh của ngọn lửa và khói trước khi nhiệt độ môi trường tăng lên đáng kể.

---

## 2. Mục tiêu Kỹ thuật (Technical Objectives)

Đề tài tập trung giải quyết bài toán Object Detection với các ràng buộc cụ thể:

1.  **Đa lớp đối tượng (Multi-class Detection):** Phân loại và định vị chính xác bounding box cho hai lớp: `Fire` (Lửa) và `Smoke` (Khói).
2.  **Hiệu năng thời gian thực (Real-time Performance):** Đạt tốc độ suy luận (Inference Speed) **≥ 30 FPS** trên thiết bị biên hoặc máy tính cá nhân, đảm bảo độ trễ cảnh báo < 100ms.
3.  **Chiến lược tối ưu hóa (Optimization Strategy):**
    * Ưu tiên chỉ số **Recall (Độ nhạy)** cao để giảm thiểu tối đa tình trạng bỏ sót báo động (False Negatives) - yếu tố sống còn trong bài toán an toàn.
    * Xử lý vấn đề **Negative Mining**: Giảm báo động giả (False Positives) từ các đối tượng nhiễu có đặc tính thị giác tương đồng (ánh đèn led, sương mù, mây, vật thể màu đỏ).

---

## 3. Cơ sở Công nghệ (Technology Stack)

Lựa chọn **YOLOv11 (You Only Look Once - v11)** làm hạt nhân xử lý với các cải tiến vượt trội so với các phiên bản tiền nhiệm:

* **Kiến trúc C3k2 Block:** Cải thiện khả năng trích xuất đặc trưng (Feature Extraction) cho các đối tượng có hình dạng biến thiên (deformable objects) như khói và lửa, khắc phục hạn chế của module C2f trong YOLOv8.
* **Cơ chế Spatial Attention:** Tăng cường khả năng tập trung vào vùng đối tượng kích thước nhỏ (Small Object Detection), giúp phát hiện đốm lửa từ khoảng cách xa.
* **Computational Efficiency:** Giảm lượng tham số (Parameters) và FLOPs, cho phép triển khai nhẹ nhàng trên phần cứng hạn chế mà không cần GPU Server đắt tiền.

---

## 4. Phạm vi & Giới hạn (Scope & Limitations)

### 4.1. Phạm vi thực hiện (In Scope)
* **Data Engineering:** Xây dựng và tiền xử lý bộ dữ liệu dựa trên D-Fire Dataset (chuẩn hóa Label, Resize, Augmentation).
* **Model Training:** Huấn luyện Fine-tuning mô hình YOLOv11s hoặc YOLOv11n.
* **Inference App:** Xây dựng ứng dụng Desktop (Python/Qt hoặc Web/Streamlit) hiển thị cảnh báo trực quan trên luồng video.

### 4.2. Giới hạn (Out of Scope)
* Không bao gồm thiết kế mạch phần cứng tích hợp (Embedded Hardware Design).
* Không xử lý các tác vụ hậu cảnh báo như điều khiển vòi phun nước hay gọi điện tự động (tập trung thuần túy vào thuật toán nhận diện).

---

## 5. Phương pháp Thực hiện (Methodology)

### Giai đoạn 1: Chuẩn bị Dữ liệu (Data Centric AI)
* **Dataset:** Sử dụng bộ D-Fire (~21.000 ảnh).
* **Hard Negative Mining:** Bổ sung vào tập train các ảnh "Negative" (ảnh không có lửa nhưng dễ gây nhầm lẫn như hoàng hôn, đèn đường, sương mù) để mô hình học được ranh giới quyết định (decision boundary) tốt hơn.

### Giai đoạn 2: Huấn luyện & Tinh chỉnh (Training & Fine-tuning)
* **Môi trường:** PyTorch, Ultralytics Framework trên Google Colab (T4 GPU).
* **Hyperparameters:** Tinh chỉnh Learning rate, Momentum và sử dụng kỹ thuật Mosaic Augmentation để tăng cường khả năng nhận diện đối tượng nhỏ.
* **Loss Function:** Sử dụng CIoU hoặc DFL (Distribution Focal Loss) để tối ưu hóa độ chính xác của khung bao (Bounding Box).

### Giai đoạn 3: Đánh giá & Demo
* Đánh giá mô hình dựa trên Confusion Matrix và đường cong PR (Precision-Recall Curve).
* Xây dựng kịch bản Demo: Test trực tiếp với video giả lập và Webcam thời gian thực.

---

## 6. Tính Khả thi & Kết quả Dự kiến

* **Tính khả thi kỹ thuật:**
    * Công nghệ YOLO là tiêu chuẩn công nghiệp (Industry Standard) đã được kiểm chứng.
    * Dữ liệu huấn luyện (D-Fire) là bộ dữ liệu mở, chất lượng cao và phù hợp với năng lực xử lý của sinh viên.
* **Sản phẩm bàn giao (Deliverables):**
    1.  File trọng số mô hình (`best.pt`) đã tối ưu hóa metric mAP@0.5 và Recall.
    2.  Source code huấn luyện và ứng dụng demo hoàn chỉnh.
    3.  Báo cáo phân tích so sánh hiệu năng giữa YOLOv11 và một baseline model (ví dụ: YOLOv8 hoặc SSD).
