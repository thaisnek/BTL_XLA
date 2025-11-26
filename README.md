# Phần mềm chuyển ảnh thành tranh vẽ

**Đề tài 4 - Xử lý ảnh và ứng dụng**

## Mô tả

Phần mềm chuyển đổi ảnh thành tranh vẽ sử dụng các kỹ thuật:
- Phát hiện biên (Edge Detection)
- Chuyển đổi mức xám (Grayscale Conversion)
- Kỹ thuật làm mịn (Smoothing Techniques)
- Bilateral Filter

## Tính năng

✅ Tải/lưu ảnh (hỗ trợ ảnh y tế, ảnh tự nhiên, ảnh công nghiệp...)  
✅ Xem kết quả trực quan  
✅ Nhiều phương pháp phát hiện biên: Canny, Sobel, Laplacian  
✅ Bilateral Filter để làm mịn giữ nguyên biên  
✅ Điều chỉnh độ tương phản và độ sáng  
✅ Tải xuống kết quả  

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Chạy ứng dụng:
```bash
streamlit run app.py
```

3. Mở trình duyệt và truy cập: `http://localhost:8501`

## Sử dụng

1. Tải ảnh lên bằng nút "Tải ảnh lên"
2. Điều chỉnh các tham số ở thanh bên trái:
   - Chọn phương pháp phát hiện biên
   - Điều chỉnh kích thước làm mịn
   - Bật/tắt Bilateral Filter
   - Điều chỉnh độ tương phản và độ sáng
3. Xem kết quả và tải xuống nếu hài lòng

## Công nghệ sử dụng

- **Streamlit**: Giao diện web

## Thuật toán

### 1. Phát hiện biên
- **Canny Edge Detection**: Phát hiện biên tối ưu với ngưỡng kép
- **Sobel Operator**: Phát hiện biên dựa trên gradient
- **Laplacian Operator**: Phát hiện biên dựa trên đạo hàm bậc 2

### 2. Làm mịn
- **Gaussian Blur**: Làm mịn ảnh trước khi phát hiện biên
- **Bilateral Filter**: Làm mịn giữ nguyên biên

### 3. Xử lý sau
- Chuyển đổi mức xám
- Điều chỉnh độ tương phản và độ sáng
- Đảo ngược màu để tạo hiệu ứng vẽ tay


