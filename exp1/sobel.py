import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 核心算法手动实现
# ==========================================

def manual_convolution(image, kernel):
    """
    手动实现二维卷积操作 (不调用 cv2.filter2D)
    """
    k_h, k_w = kernel.shape
    h, w = image.shape
    
    # Padding (使用0填充边缘)
    pad_h, pad_w = k_h // 2, k_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    output = np.zeros((h, w), dtype=np.float32)
    
    # 遍历每个像素进行卷积运算
    # 为了加速，使用了部分 Numpy 切片操作代替最内层循环
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
            
    return output

def manual_color_histogram(image):
    """
    手动计算彩色图像直方图 (不调用 cv2.calcHist)
    """
    h, w, c = image.shape
    # 初始化直方图数组: 3个通道，每个通道256个灰度级
    histograms = np.zeros((3, 256), dtype=int)
    
    # 遍历三个通道
    for ch in range(3):
        channel_data = image[:, :, ch].flatten()
        # 统计每个像素值的出现次数
        # 使用 numpy 基础操作实现统计，避免 python 原生循环过慢
        for val in range(256):
            histograms[ch, val] = np.sum(channel_data == val)
            
    return histograms

def manual_lbp_texture(image):
    """
    手动提取 LBP (Local Binary Pattern) 纹理特征
    """
    h, w = image.shape
    output = np.zeros((h-2, w-2), dtype=np.uint8)
    
    # LBP 邻域权重
    #   1   2   4
    #  128  0   8
    #  64  32  16
    weights = np.array([[1, 2, 4], [128, 0, 8], [64, 32, 16]], dtype=np.uint8)
    
    # 遍历图像内部像素
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = image[i, j]
            neighborhood = image[i-1:i+2, j-1:j+2]
            
            # 二值化比较：邻域 >= 中心像素 则为1，否则为0
            binary_pattern = (neighborhood >= center).astype(np.uint8)
            
            # 计算 LBP 值
            lbp_val = np.sum(binary_pattern * weights)
            output[i-1, j-1] = lbp_val
            
    return output

# ==========================================
# 2. 主流程
# ==========================================

def main():
    # --- A. 读取图像 ---
    img_path = 'properties/rgb.png'
    
    # 如果没有图片，生成一个简单的测试图
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found. Creating a dummy image.")
        dummy = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(dummy, (100, 100), 50, (255, 255, 255), -1)
        cv2.imwrite(img_path, dummy)

    # 读取彩色图和灰度图
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    print("Processing started...")

    # --- B. 任务1 & 2: 滤波 ---
    
    # 1. 定义 Sobel 核 (X 和 Y 方向)
    # Sobel X (检测垂直边缘)
    sobel_x_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    # Sobel Y (检测水平边缘)
    sobel_y_kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # 2. 题目要求的特定卷积核
    # 注意：题目给出的核其实是一个垂直边缘检测核（类似 Sobel X 的变体或旋转）
    given_kernel = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    # 执行手动卷积
    # Sobel 完整算子通常包含 X 和 Y 的结合
    grad_x = manual_convolution(img_gray, sobel_x_kernel)
    grad_y = manual_convolution(img_gray, sobel_y_kernel)
    sobel_result = np.sqrt(grad_x**2 + grad_y**2) # 幅值
    
    # 执行题目给定核的滤波
    given_kernel_result = manual_convolution(img_gray, given_kernel)

    # 归一化结果以便显示 (0-255)
    sobel_display = cv2.normalize(sobel_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    given_kernel_display = np.clip(np.abs(given_kernel_result), 0, 255).astype(np.uint8)

    # --- C. 任务3: 颜色直方图 ---
    histograms = manual_color_histogram(img_bgr)

    # --- D. 任务4: 纹理特征 ---
    # 使用 LBP 算法
    lbp_feature = manual_lbp_texture(img_gray)
    
    # 保存纹理特征到 npy 文件
    np.save('texture_features.npy', lbp_feature)
    print("Texture features saved to 'texture_features.npy'")

    # ==========================================
    # 3. 可视化结果
    # ==========================================
    plt.figure(figsize=(14, 10))

    # 1. 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # 2. Sobel 算子滤波结果
    plt.subplot(2, 3, 2)
    plt.imshow(sobel_display, cmap='gray')
    plt.title("Filtered by Sobel Operator (Magnitude)")
    plt.axis('off')

    # 3. 给定核滤波结果
    plt.subplot(2, 3, 3)
    plt.imshow(given_kernel_display, cmap='gray')
    plt.title("Filtered by Given Kernel\n(Vertical Edges)")
    plt.axis('off')

    # 4. 颜色直方图可视化
    plt.subplot(2, 3, 4)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        plt.plot(histograms[i], color=color)
    plt.title("Color Histogram (Manual Calc)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])

    # 5. 纹理特征可视化 (LBP)
    plt.subplot(2, 3, 5)
    plt.imshow(lbp_feature, cmap='gray')
    plt.title("Texture Feature (LBP Visualization)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()