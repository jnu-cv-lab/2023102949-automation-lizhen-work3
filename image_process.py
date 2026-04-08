import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. 图像读入与预处理 --------------------------
def load_gray_image(image_path):
    """读取灰度图像"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return img

# -------------------------- 2. 图像下采样（两种方式） --------------------------
def downsample_direct(img, scale=0.5):
    """不做预滤波直接下采样（最近邻插值缩小）"""
    h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    # 直接用最近邻插值缩小，等价于直接采样
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def downsample_gaussian_blur(img, scale=0.5, ksize=(5,5), sigmaX=1.0):
    """先高斯平滑再下采样（抗混叠）"""
    # 高斯模糊做预滤波
    blurred = cv2.GaussianBlur(img, ksize, sigmaX)
    h, w = blurred.shape
    new_h, new_w = int(h * scale), int(w * scale)
    # 双线性插值缩小（更平滑）
    return cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# -------------------------- 3. 图像恢复（三种内插方法） --------------------------
def restore_image(img_small, original_shape, method):
    """
    用指定内插方法恢复到原始尺寸
    method: 'nearest' | 'bilinear' | 'bicubic'
    """
    h, w = original_shape
    if method == 'nearest':
        return cv2.resize(img_small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif method == 'bilinear':
        return cv2.resize(img_small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif method == 'bicubic':
        return cv2.resize(img_small, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError("不支持的内插方法")

# -------------------------- 4. 空间域质量评价（MSE/PSNR） --------------------------
def calculate_mse(img1, img2):
    """计算均方误差 MSE"""
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

def calculate_psnr(img1, img2, max_pixel=255.0):
    """计算峰值信噪比 PSNR"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')  # 完全相同
    return 10 * np.log10((max_pixel ** 2) / mse)

# -------------------------- 5. 二维傅里叶变换与频谱显示 --------------------------
def fft_2d(img):
    """计算二维傅里叶变换，中心化并取对数幅度谱"""
    # 1. 傅里叶变换
    f = np.fft.fft2(img)
    # 2. 频谱中心化（低频移到中心）
    f_shift = np.fft.fftshift(f)
    # 3. 计算幅度谱 + 对数变换（增强显示）
    magnitude = 20 * np.log(np.abs(f_shift) + 1)  # +1避免log(0)
    return magnitude, f_shift

def show_fft_spectrum(original, small, restored_bilinear, titles):
    """显示三张图的傅里叶频谱"""
    mag_ori, _ = fft_2d(original)
    mag_small, _ = fft_2d(small)
    mag_rest, _ = fft_2d(restored_bilinear)

    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(mag_ori, cmap='gray')
    plt.title(titles[0]), plt.axis('off')
    plt.subplot(132), plt.imshow(mag_small, cmap='gray')
    plt.title(titles[1]), plt.axis('off')
    plt.subplot(133), plt.imshow(mag_rest, cmap='gray')
    plt.title(titles[2]), plt.axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------- 6. 二维DCT变换与能量分析 --------------------------
def dct_2d(img):
    """计算二维DCT变换"""
    # 转float32
    img_float = np.float32(img)
    # 二维DCT
    dct = cv2.dct(img_float)
    # 对数变换增强显示
    dct_log = 20 * np.log(np.abs(dct) + 1)
    return dct, dct_log

def calculate_low_freq_energy(dct_coeff, ratio=0.1):
    """
    计算左上角低频区域能量占总能量的比例
    ratio: 低频区域占总尺寸的比例（默认10%）
    """
    h, w = dct_coeff.shape
    # 取左上角 ratio*h x ratio*w 区域
    low_h, low_w = int(h * ratio), int(w * ratio)
    low_freq = dct_coeff[:low_h, :low_w]
    
    # 计算能量（幅度平方和）
    total_energy = np.sum(np.abs(dct_coeff) ** 2)
    low_energy = np.sum(np.abs(low_freq) ** 2)
    return low_energy / total_energy * 100  # 百分比

def show_dct_analysis(original, restored_list, method_names):
    """显示DCT系数图并统计低频能量占比"""
    # 原图DCT
    dct_ori, dct_ori_log = dct_2d(original)
    energy_ori = calculate_low_freq_energy(dct_ori)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(restored_list)+1, 1)
    plt.imshow(dct_ori_log, cmap='gray')
    plt.title(f'原图DCT\n低频能量占比: {energy_ori:.2f}%'), plt.axis('off')

    # 各恢复图DCT
    for i, (restored, name) in enumerate(zip(restored_list, method_names)):
        dct_rest, dct_rest_log = dct_2d(restored)
        energy_rest = calculate_low_freq_energy(dct_rest)
        plt.subplot(1, len(restored_list)+1, i+2)
        plt.imshow(dct_rest_log, cmap='gray')
        plt.title(f'{name}恢复DCT\n低频能量占比: {energy_rest:.2f}%'), plt.axis('off')

    plt.tight_layout()
    plt.show()

# -------------------------- 7. 主函数：完整实验流程 --------------------------
def main(image_path, scale=0.5):
    # 1. 读取图像
    img_original = load_gray_image(image_path)
    h, w = img_original.shape
    print(f"原始图像尺寸: {w}x{h}")

    # 2. 两种下采样
    img_small_direct = downsample_direct(img_original, scale)
    img_small_gauss = downsample_gaussian_blur(img_original, scale)
    print(f"下采样后尺寸: {img_small_direct.shape[1]}x{img_small_direct.shape[0]}")

    # 3. 三种内插方法恢复（以高斯平滑下采样为例，直接下采样同理）
    methods = ['nearest', 'bilinear', 'bicubic']
    method_names = ['最近邻', '双线性', '双三次']
    restored_list = []
    for method in methods:
        restored = restore_image(img_small_gauss, (h, w), method)
        restored_list.append(restored)

    # 4. 空间域显示与质量评价
    # 4.1 显示原图、缩小图、恢复图
    plt.figure(figsize=(20, 10))
    # 原图
    plt.subplot(2, 4, 1), plt.imshow(img_original, cmap='gray')
    plt.title('原图'), plt.axis('off')
    # 直接缩小图
    plt.subplot(2, 4, 2), plt.imshow(img_small_direct, cmap='gray')
    plt.title('直接缩小(1/2)'), plt.axis('off')
    # 高斯平滑后缩小图
    plt.subplot(2, 4, 3), plt.imshow(img_small_gauss, cmap='gray')
    plt.title('高斯平滑后缩小(1/2)'), plt.axis('off')

    # 三种恢复图
    for i, (restored, name) in enumerate(zip(restored_list, method_names)):
        plt.subplot(2, 4, i+4), plt.imshow(restored, cmap='gray')
        plt.title(f'{name}恢复'), plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 4.2 计算MSE和PSNR
    print("\n=== 空间域质量评价（高斯平滑下采样后恢复） ===")
    for restored, name in zip(restored_list, method_names):
        mse = calculate_mse(img_original, restored)
        psnr = calculate_psnr(img_original, restored)
        print(f"{name}内插: MSE = {mse:.2f}, PSNR = {psnr:.2f} dB")

    # 5. 傅里叶变换分析（原图、缩小图、双线性恢复图）
    print("\n=== 傅里叶变换频谱分析 ===")
    show_fft_spectrum(
        original=img_original,
        small=img_small_gauss,
        restored_bilinear=restored_list[1],
        titles=['原图频谱(中心化+对数)', '缩小后图像频谱', '双线性恢复后频谱']
    )

    # 6. DCT分析（原图+三种恢复图）
    print("\n=== DCT变换与能量分析 ===")
    show_dct_analysis(img_original, restored_list, method_names)

# -------------------------- 运行实验 --------------------------
if __name__ == "__main__":
    # 替换为你的图像路径
    IMAGE_PATH = "test.png"  # 例如: "lena.png"
    main(IMAGE_PATH, scale=0.5)  # scale=0.5 缩小为1/2，scale=0.25 缩小为1/4