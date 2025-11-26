import streamlit as st
import numpy as np
from PIL import Image
import io
from scipy.signal import convolve2d

st.set_page_config(page_title="Chuyển ảnh thành tranh vẽ", page_icon="🎨", layout="wide")

st.title("🎨 Phần mềm chuyển ảnh thành tranh vẽ")
st.markdown("**Đề tài 4 - Xử lý ảnh và ứng dụng**")


def to_grayscale(image):
    """Chuyển đổi ảnh màu sang ảnh xám (RGB → Grayscale)"""
    if len(image.shape) == 2:
        return image.copy()
    
    if len(image.shape) == 3:
        if image.shape[2] == 3:

            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.astype(np.uint8)
        elif image.shape[2] == 4:
            # RGBA - bỏ qua alpha channel
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.astype(np.uint8) # chuyển về kiểu dữ liệu unit8 (số nguyên dương 0-255255)
    
    return image

def gaussian_kernel(size, sigma):
    """Tạo kernel Gaussian với kích thước và độ lệch chuẩn sigma cho trước"""
    center = size // 2

    y, x = np.ogrid[-center:size-center, -center:size-center]

    coef = 1.0 / (2.0 * np.pi * sigma * sigma) #tính hệ số chuẩn hóa của gauss
    
    kernel = coef * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float64)

def gaussian_blur(image, kernel_size, sigma=None):
    """Làm mịn ảnh bằng Gaussian Blur để giảm nhiễu"""
    if sigma is None:
       
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    # Tạo Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    output = convolve2d(image.astype(np.float64), kernel, mode='same', boundary='symm')
    
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def sobel_operator(image, ksize=3, return_full=False):
    """Phát hiện biên bằng Sobel Operator. Trả về magnitude hoặc dict (Gx, Gy, magnitude, angle)"""
    if ksize == 3:
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float64)
        
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float64)
    else:

        sobel_x = np.zeros((ksize, ksize), dtype=np.float64)
        sobel_y = np.zeros((ksize, ksize), dtype=np.float64)
        center = ksize // 2
        
        for i in range(ksize):
            for j in range(ksize):
                x, y = i - center, j - center
                if y == 0:
                    sobel_x[i, j] = x if x != 0 else 0
                if x == 0:
                    sobel_y[i, j] = y if y != 0 else 0
    
    image_float = image.astype(np.float64)
    gradient_x = convolve2d(image_float, sobel_x, mode='same', boundary='symm')
    gradient_y = convolve2d(image_float, sobel_y, mode='same', boundary='symm')
    
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    if return_full: 
        angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
        
        angle = np.where(angle < 0, angle + 180, angle)
        
        return {
            'gx': gradient_x,
            'gy': gradient_y,
            'magnitude': magnitude,
            'angle': angle
        }
    
    if np.max(magnitude) > 0:
        magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
    else:
        magnitude = magnitude.astype(np.uint8)
    
    return magnitude

def sobel_operator_variable(image, kernel_size=3):
    """Wrapper cho Sobel operator với kernel size có thể thay đổi"""
    return sobel_operator(image, ksize=kernel_size)

def non_maximum_suppression(gradient_data):
    """Loại bỏ các điểm không phải cực đại địa phương theo hướng gradient (cho Canny)"""
    magnitude = gradient_data['magnitude']
    angle = gradient_data['angle']
    height, width = magnitude.shape
    output = np.zeros_like(magnitude)
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            angle_val = angle[y, x]
            
            # Làm tròn góc về 4 hướng chính: 0°, 45°, 90°, 135° 
            if (angle_val >= 0 and angle_val < 22.5) or (angle_val >= 157.5 and angle_val <= 180):
                # 0° (ngang) - so sánh với pixel trái và phải
                q = magnitude[y, x + 1]
                r = magnitude[y, x - 1]
            elif angle_val >= 22.5 and angle_val < 67.5:
                # 45° - so sánh với pixel chéo trên-phải và chéo dưới-trái
                q = magnitude[y - 1, x + 1]
                r = magnitude[y + 1, x - 1]
            elif angle_val >= 67.5 and angle_val < 112.5:
                # 90° (dọc) - so sánh với pixel trên và dưới
                q = magnitude[y + 1, x]
                r = magnitude[y - 1, x]
            else:  # 112.5 <= angle_val < 157.5
                # 135° - so sánh với pixel chéo trên-trái và chéo dưới-phải
                q = magnitude[y - 1, x - 1]
                r = magnitude[y + 1, x + 1]
            
            # Giữ lại pixel nếu nó là cực đại địa phương 
            if magnitude[y, x] >= q and magnitude[y, x] >= r:
                output[y, x] = magnitude[y, x]
    
    return output

def hysteresis_threshold(nms_image, low_threshold, high_threshold):
    """Phân ngưỡng kép để tìm biên mạnh và yếu, sau đó kết nối chúng (cho Canny)"""
    height, width = nms_image.shape
    output = np.zeros_like(nms_image, dtype=np.uint8)
    
    strong = 255
    weak = 75
    
    # Bước 1: Phân loại pixel thành strong, weak, hoặc không phải biên
    for y in range(height):
        for x in range(width):
            if nms_image[y, x] >= high_threshold:
                output[y, x] = strong
            elif nms_image[y, x] >= low_threshold:
                output[y, x] = weak
            else:
                output[y, x] = 0
    
    # Bước 2: Edge tracking - kết nối biên yếu với biên mạnh
    changed = True
    while changed:
        changed = False
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if output[y, x] == weak:
                    # Kiểm tra 8 lân cận xem có biên mạnh không
                    has_strong_neighbor = False
                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            if output[y + ky, x + kx] == strong:
                                has_strong_neighbor = True
                                break
                        if has_strong_neighbor:
                            break
                    
                    if has_strong_neighbor:
                        output[y, x] = strong
                        changed = True
    
    # Bước 3: Loại bỏ các biên yếu còn sót lại (không kết nối với biên mạnh)
    output[output == weak] = 0
    
    return output

def canny_edge_detection(image, blur_kernel=5, low_threshold=50, high_threshold=150):
    """Canny Edge Detection đầy đủ 4 bước: Gaussian Blur → Sobel → Non-max Suppression → Hysteresis Threshold"""
    if blur_kernel > 1:
        blurred = gaussian_blur(image, blur_kernel)
    else:
        blurred = image.copy()
    
    # Bước 2: Tính gradient bằng Sobel 
    gradient_data = sobel_operator(blurred, ksize=3, return_full=True)
    
    # Bước 3: Non-maximum suppression 
    nms_image = non_maximum_suppression(gradient_data)
    
    # Bước 4: Hysteresis thresholding
    edges = hysteresis_threshold(nms_image, low_threshold, high_threshold)
    
    return edges

def laplacian_operator(image, kernel_size=3):
    """Phát hiện biên bằng Laplacian Operator (đạo hàm bậc 2)"""
    if kernel_size == 3:
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float64)
    elif kernel_size == 1:
        laplacian_kernel = np.array([[-1]], dtype=np.float64)
    else:

        laplacian_kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
        center = kernel_size // 2
        laplacian_kernel[center, center] = -(kernel_size * kernel_size - 1)
    
    output = convolve2d(image.astype(np.float64), laplacian_kernel, mode='same', boundary='symm')

    output = np.absolute(output)
    
    if np.max(output) > 0:
        output = (output / np.max(output) * 255).astype(np.uint8)
    else:
        output = output.astype(np.uint8)
    
    return output

def bilateral_filter_custom(image, d, sigma_color, sigma_space):
    """Bilateral Filter: làm mịn ảnh nhưng giữ nguyên biên sắc nét"""
    height, width = image.shape
    image_float = image.astype(np.float64)
    output = np.zeros_like(image_float)
    
    pad = d // 2
    padded = np.pad(image_float, pad, mode='reflect')

    center = d // 2
    y_coords, x_coords = np.mgrid[0:d, 0:d]
    # Tính khoảng cách từ center của kernel
    y_coords = y_coords - center
    x_coords = x_coords - center
    spatial_kernel = np.exp(-(x_coords**2 + y_coords**2) / (2 * sigma_space**2))
    
    for i in range(height):
        for j in range(width):
            # Lấy window xung quanh pixel
            window = padded[i:i+d, j:j+d]
            center_val = padded[i + pad, j + pad]
            
            # Color weights (vectorized)
            color_diff = np.abs(window - center_val)
            color_weights = np.exp(-(color_diff**2) / (2 * sigma_color**2))
            
            # Combined weights
            weights = color_weights * spatial_kernel
            weight_sum = np.sum(weights)
            
            if weight_sum > 0:
                output[i, j] = np.sum(weights * window) / weight_sum
            else:
                output[i, j] = center_val
    
    return output.astype(np.uint8)

# Sidebar cho các tùy chọn
st.sidebar.header("⚙️ Cài đặt")

# Upload ảnh
uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])

# Các tham số điều chỉnh
st.sidebar.subheader("Tham số xử lý")

# Chọn phương pháp phát hiện biên
edge_method = st.sidebar.selectbox(
    "Phương pháp phát hiện biên",
    ["Canny", "Sobel", "Laplacian"]
)

# Tham số Canny
if edge_method == "Canny":
    blur_kernel = st.sidebar.slider("Kích thước làm mịn (Gaussian)", 1, 15, 5, step=2)
    canny_low = st.sidebar.slider("Ngưỡng thấp (T2)", 0, 200, 50)
    canny_high = st.sidebar.slider("Ngưỡng cao (T1)", 0, 300, 150)

# Tham số Sobel
elif edge_method == "Sobel":
    sobel_kernel = st.sidebar.slider("Kích thước kernel Sobel", 3, 7, 3, step=2)
    blur_kernel = st.sidebar.slider("Kích thước làm mịn (Gaussian)", 1, 15, 5, step=2)
    sobel_threshold = st.sidebar.slider("Ngưỡng Sobel", 0, 255, 100)
    use_sobel_threshold = st.sidebar.checkbox("Sử dụng ngưỡng cho Sobel", value=False)

# Tham số Laplacian
else:  # Laplacian
    laplacian_kernel = st.sidebar.slider("Kích thước kernel Laplacian", 3, 7, 3, step=2)
    blur_kernel = st.sidebar.slider("Kích thước làm mịn (Gaussian)", 1, 15, 5, step=2)

# Bilateral filter
use_bilateral = st.sidebar.checkbox("Sử dụng Bilateral Filter", value=True)
if use_bilateral:
    bilateral_d = st.sidebar.slider("Bilateral d", 1, 20, 9)
    bilateral_sigma_color = st.sidebar.slider("Bilateral Sigma Color", 1, 100, 75)
    bilateral_sigma_space = st.sidebar.slider("Bilateral Sigma Space", 1, 100, 75)

# Độ tương phản và độ sáng
contrast = st.sidebar.slider("Độ tương phản", 0.0, 2.0, 1.0, step=0.1)
brightness = st.sidebar.slider("Độ sáng", -50, 50, 0)

def bitwise_not(image):
    """Đảo ngược màu ảnh (255 - pixel) để tạo hiệu ứng tranh vẽ"""
    return 255 - image

def convert_scale_abs(image, alpha=1.0, beta=0):
    """Điều chỉnh độ tương phản (alpha) và độ sáng (beta) của ảnh"""
    output = alpha * image.astype(np.float64) + beta
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def convert_to_sketch(image, edge_method, **params):
    """Chuyển đổi ảnh thành tranh vẽ bằng phương pháp phát hiện biên (Canny/Sobel/Laplacian)"""
    # Chuyển đổi sang mức xám (TỰ IMPLEMENT)
    gray = to_grayscale(image)
    
    # Làm mịn ảnh bằng Gaussian Blur (TỰ IMPLEMENT)
    if params.get('blur_kernel', 5) > 1:
        blurred = gaussian_blur(gray, params['blur_kernel'])
    else:
        blurred = gray.copy()

    # Áp dụng Bilateral Filter nếu được chọn (TỰ IMPLEMENT)
    if params.get('use_bilateral', False):
        bilateral = bilateral_filter_custom(
            blurred,
            params.get('bilateral_d', 9),
            params.get('bilateral_sigma_color', 75),
            params.get('bilateral_sigma_space', 75)
        )
        blurred = bilateral
    
    # Phát hiện biên (TỰ IMPLEMENT)
    if edge_method == "Canny":

        edges = canny_edge_detection(
            blurred,  # Dùng blurred đã qua Bilateral nếu có
            blur_kernel=1,  # Không blur thêm vì đã blur rồi
            low_threshold=params.get('canny_low', 50),
            high_threshold=params.get('canny_high', 150)
        )
        sketch = bitwise_not(edges)
    
    elif edge_method == "Sobel":
        # Sobel Operator 
        if params.get('sobel_kernel', 3) == 3:
            sobel = sobel_operator(blurred)
        else:
            sobel = sobel_operator_variable(blurred, params.get('sobel_kernel', 3))
        
        # Áp dụng threshold nếu được chọn 
        if params.get('use_sobel_threshold', False):
            threshold = params.get('sobel_threshold', 100)
            sobel = np.where(sobel > threshold, 255, 0).astype(np.uint8)
        
        sketch = bitwise_not(sobel)
    
    else:  # Laplacian
        # Laplacian Operator 
        laplacian = laplacian_operator(blurred, params.get('laplacian_kernel', 3))
        sketch = bitwise_not(laplacian)
    
    # Điều chỉnh độ tương phản và độ sáng (TỰ IMPLEMENT)
    sketch = convert_scale_abs(sketch, alpha=params.get('contrast', 1.0), beta=params.get('brightness', 0))
    
    return sketch

def main():
    if uploaded_file is not None:
        # Đọc ảnh
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        params = {
            'blur_kernel': blur_kernel,
            'use_bilateral': use_bilateral,
            'contrast': contrast,
            'brightness': brightness
        }
        
        if edge_method == "Canny":
            params['canny_low'] = canny_low
            params['canny_high'] = canny_high
        elif edge_method == "Sobel":
            params['sobel_kernel'] = sobel_kernel
            params['sobel_threshold'] = sobel_threshold
            params['use_sobel_threshold'] = use_sobel_threshold
        else:  # Laplacian
            params['laplacian_kernel'] = laplacian_kernel
        
        if use_bilateral:
            params['bilateral_d'] = bilateral_d
            params['bilateral_sigma_color'] = bilateral_sigma_color
            params['bilateral_sigma_space'] = bilateral_sigma_space
        
        # Xử lý ảnh
        with st.spinner("Đang xử lý ảnh..."):
            sketch = convert_to_sketch(image_np, edge_method, **params)
        
        # Hiển thị kết quả
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Ảnh gốc")
            st.image(image, width='stretch')
        
        with col2:
            st.subheader("🎨 Tranh vẽ")
            st.image(sketch, width='stretch', channels="GRAY")
        
        # Nút tải xuống
        st.subheader("💾 Tải kết quả")
        
        # Chuyển đổi sketch thành PIL Image
        sketch_pil = Image.fromarray(sketch)
        
        # Tạo buffer để lưu ảnh
        buf = io.BytesIO()
        sketch_pil.save(buf, format='PNG')
        buf.seek(0)
        
        st.download_button(
            label="⬇️ Tải xuống tranh vẽ (PNG)",
            data=buf,
            file_name="sketch_result.png",
            mime="image/png"
        )
        
        # Thông tin về ảnh
        st.sidebar.subheader("📊 Thông tin ảnh")
        st.sidebar.write(f"Kích thước: {image.size[0]} x {image.size[1]}")
        st.sidebar.write(f"Chế độ: {image.mode}")
        
    else:
        st.info("👆 Vui lòng tải ảnh lên để bắt đầu")
        st.markdown("""
        ### Hướng dẫn sử dụng:
        1. **Tải ảnh lên** bằng nút "Tải ảnh lên" ở trên
        2. **Điều chỉnh tham số** ở thanh bên trái:
           - Chọn phương pháp phát hiện biên:
             - **Canny**: Phát hiện biên tối ưu với 4 bước (khuyên dùng)
             - **Sobel**: Phát hiện biên dựa trên gradient bậc 1
             - **Laplacian**: Phát hiện biên dựa trên đạo hàm bậc 2
           - Điều chỉnh các tham số làm mịn
           - Bật/tắt Bilateral Filter (chỉ cho Sobel/Laplacian)
           - Điều chỉnh độ tương phản và độ sáng
        3. **Xem kết quả** và tải xuống nếu hài lòng
        
        ### Hỗ trợ các loại ảnh:
        - Ảnh y tế (X-ray, CT scan, MRI...)
        - Ảnh tự nhiên (phong cảnh, chân dung...)
        - Ảnh công nghiệp (sản phẩm, máy móc...)
        """)

if __name__ == "__main__":
    main()

