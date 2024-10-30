import cv2
import pytesseract
import os
import concurrent.futures

# 如果你在Windows上，需要指定Tesseract可执行文件的路径
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 图像路径
image_path = './4.jpg'  # 替换为你的图片路径

# 检查图像是否存在
if not os.path.exists(image_path):
    print("图片不存在")
    exit()

# 读取图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否读取成功
if image is None:
    print("无法读取图像")
    exit()

# 图像尺寸
height, width = image.shape

# 每个格子的大小
cell_height = height // 27  # 根据你的图像设置行数和列数
cell_width = width // 38

# 缩进像素数
inset = 3  # 可以根据实际情况调整

# 创建保存数字的文件夹
output_dir = "digits_dataset"
os.makedirs(output_dir, exist_ok=True)

for i in range(10):
    folder_path = os.path.join(output_dir, str(i))
    os.makedirs(folder_path, exist_ok=True)

# OCR 配置，确保只识别数字
custom_config = r'--oem 3 --psm 10 outputbase digits'

# === 图像预处理步骤 ===
# 应用高斯模糊去除噪声
image = cv2.GaussianBlur(image, (5, 5), 0)

# 应用自适应阈值进行二值化，增强对比度
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 定义单行处理函数，方便多线程调用
def process_row(row):
    for col in range(38):
        # 计算当前格子的坐标
        x = col * cell_width + inset
        y = row * cell_height + inset

        # 调整裁剪范围，提取单个数字区域
        cell = image[y:y + cell_height - 2 * inset, x:x + cell_width - 2 * inset]

        # 使用OCR识别数字
        text = pytesseract.image_to_string(cell, config=custom_config).strip()

        # 检查识别结果是否为数字
        if text.isdigit():
            digit = int(text)
            if 0 <= digit <= 9:
                # 保存切割后的数字图像
                save_path = os.path.join(output_dir, str(digit), f'{row}_{col}.png')
                cv2.imwrite(save_path, cell)
        else:
            print(f"未识别到有效数字，跳过行 {row} 列 {col}")

    print(f"第 {row} 行处理完毕")

# 使用ThreadPoolExecutor进行多线程处理
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 提交每一行的任务
    futures = [executor.submit(process_row, row) for row in range(27)]

    # 确保所有任务执行完毕
    concurrent.futures.wait(futures)

print("数字切割和分类完成！")
